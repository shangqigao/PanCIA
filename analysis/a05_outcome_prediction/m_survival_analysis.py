import os
import sys

# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
relative_path = os.path.join(script_dir, '../../')
sys.path.append(relative_path)

import requests
import argparse
import pathlib
import logging
import warnings
import joblib
import copy
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torchbnn as bnn

from scipy.stats import zscore
from torch_geometric.loader import DataLoader
from tiatoolbox import logger
from tiatoolbox.utils.misc import save_as_json

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis, IPCRidge
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import (
    concordance_index_censored, 
    concordance_index_ipcw,
    integrated_brier_score, 
    cumulative_dynamic_auc,
    as_concordance_index_ipcw_scorer,
    as_cumulative_dynamic_auc_scorer,
    as_integrated_brier_score_scorer
)
from sklearn import set_config
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, VarianceThreshold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression as PlattScaling
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests

from utilities.m_utils import mkdir, select_wsi, load_json, create_pbar, rm_n_mkdir, reset_logging, recur_find_ext, select_checkpoints

from analysis.a04_feature_aggregation.m_gnn_survival_analysis import SurvivalGraphDataset, SurvivalGraphArch, SurvivalBayesGraphArch
from analysis.a04_feature_aggregation.m_gnn_survival_analysis import ScalarMovingAverage, CoxSurvLoss
from analysis.a04_feature_aggregation.m_graph_construction import visualize_radiomic_graph, visualize_pathomic_graph

def plot_survival_curve(data_path):
    df = pd.read_csv(data_path)

    # Prepare the survival data
    df['event'] = df['vital_status'].apply(lambda x: True if x == 'Dead' else False)
    df['duration'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])
    df = df[df['duration'].notna()]
    df = df[df['ajcc_pathologic_stage'].isin(["Stage I", "Stage II"])]
    print("Data strcuture:", df.shape)

    # Fit the Kaplan-Meier estimator
    time, survival_prob, conf_int = kaplan_meier_estimator(
        df["event"], df["duration"], conf_type="log-log"
        )
    plt.step(time, survival_prob, where="post")
    plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.ylim(0, 1)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.savefig(f"{script_dir}/survival_curve.jpg")
    return

def prepare_graph_properties(data_dict, prop_keys=None, subgraphs=False, omics="radiomics"):
    if prop_keys == None: 
        prop_keys = [
            "num_nodes", 
            "num_edges", 
            "num_components", 
            "degree", 
            "closeness", 
            "graph_diameter",
            "graph_assortativity",
            "mean_neighbor_degree"
        ]
    properties = {}
    if subgraphs:
        for subgraph, prop_dict in data_dict.items():
            if subgraph == "MUC": continue
            for k in prop_keys:
                key = f"{omics}.{subgraph}.{k}"
                if prop_dict is None or len(prop_dict[k]) == 0:
                    properties[key] = -1 if k == "graph_assortativity" else 0
                else:
                    if len(prop_dict[k]) == 1:
                        if np.isnan(prop_dict[k][0]):
                            properties[key] = -1 if k == "graph_assortativity" else 0
                        else:
                            properties[key] = prop_dict[k][0]
                    else:
                        properties[key] = np.std(prop_dict[k])
    else:
        prop_dict = data_dict
        for k in prop_keys:
            key = f"{omics}.{k}"
            if prop_dict is None or len(prop_dict[k]) == 0:
                properties[key] = -1 if k == "graph_assortativity" else 0
            else:
                if len(prop_dict[k]) == 1:
                    if np.isnan(prop_dict[k][0]):
                        properties[key] = -1 if k == "graph_assortativity" else 0
                    else:
                        properties[key] = prop_dict[k][0]
                else:
                    properties[key] = np.std(prop_dict[k])
    return properties

def load_radiomic_properties(idx, radiomic_paths, prop_keys=None, pooling="mean"):
    properties_list = []
    for radiomic_path in radiomic_paths:
        suffix = pathlib.Path(radiomic_path).suffix
        if suffix == ".json":
            if prop_keys is None: 
                prop_keys = ["shape", "firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm"]
            data_dict = load_json(radiomic_path)
            properties = {}
            for key, value in data_dict.items():
                selected = [((k in key) and ("diagnostics" not in key)) for k in prop_keys]
                if any(selected): properties[key] = value
        elif suffix == ".npy":
            feature = np.load(radiomic_path)
            feat_list = np.array(feature).squeeze().tolist()
            properties = {}
            for i, feat in enumerate(feat_list):
                k = f"radiomics.feature{i}"
                properties[k] = feat
        properties_list.append(properties)

    # patient-level pooling
    df = pd.DataFrame(properties_list)
    if pooling == "mean":
        properties = df.mean(numeric_only=True).to_dict()
    elif pooling == "max":
        properties = df.max(numeric_only=True).to_dict()
    elif pooling == "min":
        properties = df.min(numeric_only=True).to_dict()
    elif pooling == "std":
        properties = df.std(numeric_only=True).to_dict()
    else:
        raise ValueError(f"Unsupport patient pooling: {pooling}")

    return {f"{idx}": properties}

def prepare_graph_pathomics(
    idx, 
    graph_path, 
    subgraphs=["TUM", "NORM", "DEB"], 
    mode="mean"
    ):
    if subgraphs is None:
        subgraph_ids = None
    else:
        subgraph_dict = {
            "ADI": [0, 4],
            "BACK": [5, 8],
            "DEB": [9, 11],
            "LYM": [12, 16],
            "MUC": [17, 20],
            "MUS": [21, 25],
            "NORM": [26, 26],
            "STR": [27, 31],
            "TUM": [32, 34]
        }
        subgraph_ids = [subgraph_dict[k] for k in subgraphs]
    graph_dict = load_json(graph_path)
    feature = np.array(graph_dict["x"])
    assert feature.ndim == 2
    if subgraph_ids is not None:
        label_path = f"{graph_path}".replace(".json", ".label.npy")
        label = np.load(label_path)
        if label.ndim == 2: label = np.argmax(label, axis=1)
        subset = label < 0
        for ids in subgraph_ids:
            ids_subset = np.logical_and(label >= ids[0], label <= ids[1])
            subset = np.logical_or(subset, ids_subset)
        if subset.sum() < 1:
            feature = np.zeros_like(feature)
        else:
            feature = feature[subset]

    if mode == "mean":
        feat_list = np.mean(feature, axis=0).tolist()
    elif mode == "max":
        feat_list = np.max(feature, axis=0).tolist()
    elif mode == "min":
        feat_list = np.min(feature, axis=0).tolist()
    elif mode == "std":
        feat_list = np.std(feature, axis=0).tolist()
    elif mode == "kmeans":
        kmeans = KMeans(n_clusters=4)
        feat_list = kmeans.fit(feature).cluster_centers_
        feat_list = feat_list.flatten().tolist()
        
    feat_dict = {}
    for i, feat in enumerate(feat_list):
        k = f"pathomics.feature{i}"
        feat_dict[k] = feat
    return {f"{idx}": feat_dict}

def prepare_graph_radiomics(
    idx, 
    graph_path, 
    mode="mean",
    pooling="mean"
    ):
    feat_list = []
    for path in graph_path:
        graph_dict = load_json(path)
        feature = np.array(graph_dict["x"])
        assert feature.ndim == 2

        if mode == "mean":
            feat = np.mean(feature, axis=0)
        elif mode == "max":
            feat = np.max(feature, axis=0)
        elif mode == "min":
            feat = np.min(feature, axis=0)
        elif mode == "std":
            feat = np.std(feature, axis=0)
        elif mode == "kmeans":
            kmeans = KMeans(n_clusters=4)
            feat = kmeans.fit(feature).cluster_centers_
            feat = feat.flatten()
        else:
            raise ValueError(f"Unsupport aggregation mode: {mode}")
        feat_list.append(feat)
    
    # patient-level pooling
    if pooling == "mean":
        feat_list = np.array(feat_list).mean(axis=0).tolist()
    elif pooling == "max":
        feat_list = np.array(feat_list).max(axis=0).tolist()
    elif pooling == "min":
        feat_list = np.array(feat_list).min(axis=0).tolist()
    elif pooling == "std":
        feat_list = np.array(feat_list).std(axis=0).tolist()
    else:
        raise ValueError(f"Unsupport patient pooling: {pooling}")
        
    feat_dict = {}
    for i, feat in enumerate(feat_list):
        k = f"radiomics.feature{i}"
        feat_dict[k] = feat
    return {f"{idx}": feat_dict}

def load_wsi_level_features(idx, wsi_feature_path, pooling="mean"):
    feat_list = []
    for path in wsi_feature_path:
        feat = np.array(np.load(path)).squeeze()
        feat_list.append(feat)

    # patient-level pooling
    if pooling == "mean":
        feat_list = np.array(feat_list).mean(axis=0).tolist()
    elif pooling == "max":
        feat_list = np.array(feat_list).max(axis=0).tolist()
    elif pooling == "min":
        feat_list = np.array(feat_list).min(axis=0).tolist()
    elif pooling == "std":
        feat_list = np.array(feat_list).std(axis=0).tolist()
    else:
        raise ValueError(f"Unsupport patient pooling: {pooling}")
    
    feat_dict = {}
    for i, feat in enumerate(feat_list):
        k = f"pathomics.feature{i}"
        feat_dict[k] = feat
    return {f"{idx}": feat_dict}

def prepare_patient_outcome(outcome_file, dataset="MAMA-MIA", outcome=None):
    if dataset == "MAMA-MIA":
        df = pd.read_excel(outcome_file, sheet_name='dataset_info')
        # Prepare survival data
        if outcome == 'OS':
            df = df[df['days_to_death'].notna()]
            df['event'] = df['days_to_death'].apply(lambda x: True if x > 0 else False)
            df['duration'] = df['days_to_death']
            df.loc[df['days_to_death'] == 0, 'duration'] = df['days_to_follow_up']
        elif outcome == 'Recurrence':
            df = df[df['days_to_recurrence'].notna()]
            df['event'] = df['days_to_recurrence'].apply(lambda x: True if x > 0 else False)
            df['duration'] = df['days_to_recurrence']
            df.loc[df['days_to_recurrence'] == 0, 'duration'] = df['days_to_follow_up']
        else:
            raise ValueError(f'Unsuppored outcome type: {outcome}')
    elif dataset == "TCGA":
        df = pd.read_csv(outcome_file)
        # Prepare the survival data
        if outcome == 'OS':
            df['event'] = df['vital_status'].apply(lambda x: True if x == 'Dead' else False)
            df['duration'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])
            df = df[df['duration'].notna()]
        elif outcome == "DFS":
            df.fillna(float('inf'), inplace=True)
            df['event'] = df.apply(lambda row: True if row['days_to_recurrence'] < float('inf') or 
                                row['days_to_death'] < float('inf') else False, axis=1)
            df['duration'] = df[['days_to_recurrence', 'days_to_death', 'days_to_last_follow_up']].min(axis=1)
            df = df[df['duration'].notna()]
        else:
            raise ValueError(f'Unsuppored outcome type: {outcome}')
    else:
        raise NotImplementedError
    
    logging.info(f"Clinical data strcuture: {df.shape}")

    return df

def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")
    plt.subplots_adjust(left=0.2)
    plt.savefig(f"{script_dir}/coefficients.jpg")

def coxnet(split_idx, tr_X, tr_y, scorer, n_jobs, l1_ratio=0.9, min_ratio=0.1):
    # COX regreession
    print("Selecting the best regularization parameter...")
    cox_elastic_net = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=min_ratio)
    cox_elastic_net.fit(tr_X, tr_y)
    coefficients = pd.DataFrame(cox_elastic_net.coef_, index=tr_X.columns, columns=np.round(cox_elastic_net.alphas_, 5))
    
    plot_coefficients(coefficients, n_highlight=5)

    # choosing penalty strength by cross validation
    cox = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=min_ratio, max_iter=10000)
    coxnet_pipe = make_pipeline(StandardScaler(), cox)
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    coxnet_pipe.fit(tr_X, tr_y)
    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    lower, upper = np.percentile(tr_y["duration"], [20, 80])
    tr_times = np.arange(lower, upper + 1)
    model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, max_iter=10000, fit_baseline_model=True)
    if scorer == "cindex":
        score_name = "C-Index"
        param_grid={"model__alphas": [[v] for v in estimated_alphas]}
    elif scorer == "cindex-ipcw":
        score_name = "C-Index-IPCW"
        model = as_concordance_index_ipcw_scorer(model, tau=upper)
        param_grid={"model__estimator__alphas": [[v] for v in estimated_alphas]}
    elif scorer == "auc":
        score_name = "AUC"
        model = as_cumulative_dynamic_auc_scorer(model, times=tr_times)
        param_grid={"model__estimator__alphas": [[v] for v in estimated_alphas]}
    elif scorer == "ibs":
        score_name = "IBS"
        model = as_integrated_brier_score_scorer(model, times=tr_times)
        param_grid={"model__estimator__alphas": [[v] for v in estimated_alphas]}
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        error_score=0.5,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    if scorer == "cindex":
        alphas = cv_results.param_model__alphas.map(lambda x: x[0])
    else:
        alphas = cv_results.param_model__estimator__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel(score_name)
    ax.set_xlabel("alpha")
    if scorer == "cindex":
        ax.axvline(gcv.best_params_["model__alphas"][0], c="C1")
    else:
        ax.axvline(gcv.best_params_["model__estimator__alphas"][0], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"{script_dir}/cross_validation_fold{split_idx}.jpg")

    # Visualize coefficients of the best estimator
    best_model = gcv.best_estimator_.named_steps["model"]
    if scorer == "cindex":
        best_coefs = pd.DataFrame(best_model.coef_, index=tr_X.columns, columns=["coefficient"])
    else:
        best_coefs = pd.DataFrame(best_model.estimator_.coef_, index=tr_X.columns, columns=["coefficient"])

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(f"Number of non-zero coefficients: {non_zero}")

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index

    _, ax = plt.subplots(figsize=(8, 6))
    non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"{script_dir}/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def rsf(split_idx, tr_X, tr_y, scorer, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    lower, upper = np.percentile(tr_y["duration"], [20, 80])
    tr_times = np.arange(lower, upper + 1)
    model = RandomSurvivalForest(max_depth=2, random_state=1)
    if scorer == "cindex":
        score_name = "C-Index"
        param_grid={"model__max_depth": np.arange(1, 20, dtype=int)}
    elif scorer == "cindex-ipcw":
        score_name = "C-Index-IPCW"
        model = as_concordance_index_ipcw_scorer(model, tau=upper)
        param_grid={"model__estimator__max_depth": np.arange(1, 20, dtype=int)}
    elif scorer == "auc":
        score_name = "AUC"
        model = as_cumulative_dynamic_auc_scorer(model, times=tr_times)
        param_grid={"model__estimator__max_depth": np.arange(1, 20, dtype=int)}
    elif scorer == "ibs":
        score_name = "IBS"
        model = as_integrated_brier_score_scorer(model, times=tr_times)
        param_grid={"model__estimator__max_depth": np.arange(1, 20, dtype=int)}
    else:
        raise NotImplementedError

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        error_score=0.5,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    if scorer == "cindex":
        depths = cv_results.param_model__max_depth
    else:
        depths = cv_results.param_model__estimator__max_depth
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(depths, mean)
    ax.fill_between(depths, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("linear")
    ax.set_ylabel(score_name)
    ax.set_xlabel("max depth")
    if scorer == "cindex":
        ax.axvline(gcv.best_params_["model__max_depth"], c="C1")
    else:
        ax.axvline(gcv.best_params_["model__estimator__max_depth"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"{script_dir}/cross_validation_fold{split_idx}.jpg")

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def gradientboosting(split_idx, tr_X, tr_y, scorer, n_jobs, loss="coxph"):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = GradientBoostingSurvivalAnalysis(loss=loss, max_depth=2, random_state=1)
    lower, upper = np.percentile(tr_y["duration"], [20, 80])
    tr_times = np.arange(lower, upper + 1)
    if scorer == "cindex":
        score_name = "C-Index"
        param_grid={"model__max_depth": np.arange(1, 20, dtype=int)}
    elif scorer == "cindex-ipcw":
        score_name = "C-Index-IPCW"
        model = as_concordance_index_ipcw_scorer(model, tau=upper)
        param_grid={"model__estimator__max_depth": np.arange(1, 20, dtype=int)}
    elif scorer == "auc":
        score_name = "AUC"
        model = as_cumulative_dynamic_auc_scorer(model, times=tr_times)
        param_grid={"model__estimator__max_depth": np.arange(1, 20, dtype=int)}
    elif scorer == "ibs":
        score_name = "IBS"
        model = as_integrated_brier_score_scorer(model, times=tr_times)
        param_grid={"model__estimator__max_depth": np.arange(1, 20, dtype=int)}
    else:
        raise NotImplementedError

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        error_score=0.5,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    if scorer == "cindex":
        depths = cv_results.param_model__max_depth
    else:
        depths = cv_results.param_model__estimator__max_depth
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(depths, mean)
    ax.fill_between(depths, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("linear")
    ax.set_ylabel(score_name)
    ax.set_xlabel("max depth")
    if scorer == "cindex":
        ax.axvline(gcv.best_params_["model__max_depth"], c="C1")
    else:
        ax.axvline(gcv.best_params_["model__estimator__max_depth"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"{script_dir}/cross_validation_fold{split_idx}.jpg")

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe


def coxph(split_idx, tr_X, tr_y, scorer, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    lower, upper = np.percentile(tr_y["duration"], [20, 80])
    tr_times = np.arange(lower, upper + 1)
    model = CoxPHSurvivalAnalysis(alpha=1e-2)
    if scorer == "cindex":
        score_name = "C-Index"
        param_grid={"model__alpha": 10.0 ** np.arange(-2, 5)}
    elif scorer == "cindex-ipcw":
        score_name = "C-Index-IPCW"
        model = as_concordance_index_ipcw_scorer(model, tau=upper)
        param_grid={"model__estimator__alpha": 10.0 ** np.arange(-2, 5)}
    elif scorer == "auc":
        score_name = "AUC"
        model = as_cumulative_dynamic_auc_scorer(model, times=tr_times)
        param_grid={"model__estimator__alpha": 10.0 ** np.arange(-2, 5)}
    elif scorer == "ibs":
        score_name = "IBS"
        model = as_integrated_brier_score_scorer(model, times=tr_times)
        param_grid={"model__estimator__alpha": 10.0 ** np.arange(-2, 5)}
    else:
        raise NotImplementedError

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        error_score=0.5,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    if scorer == "cindex":
        alphas = cv_results.param_model__alpha
    else:
        alphas = cv_results.param_model__estimator__alpha
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel(score_name)
    ax.set_xlabel("alpha")
    if scorer == "cindex":
        ax.axvline(gcv.best_params_["model__alpha"], c="C1")
    else:
        ax.axvline(gcv.best_params_["model__estimator__alpha"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"{script_dir}/cross_validation_fold{split_idx}.jpg")

    # Visualize coefficients of the best estimator
    best_model = gcv.best_estimator_.named_steps["model"]
    if scorer == "cindex":
        best_coefs = pd.DataFrame(best_model.coef_, index=tr_X.columns, columns=["coefficient"])
    else:
        best_coefs = pd.DataFrame(best_model.estimator_.coef_, index=tr_X.columns, columns=["coefficient"])

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(f"Number of non-zero coefficients: {non_zero}")

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index
    top10 = coef_order[:10]

    _, ax = plt.subplots(figsize=(8, 6))
    non_zero_coefs.loc[top10].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"{script_dir}/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def ipcridge(split_idx, tr_X, tr_y, scorer, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = IPCRidge(alpha=1, random_state=1)
    lower, upper = np.percentile(tr_y["duration"], [20, 80])
    tr_times = np.arange(lower, upper + 1)
    if scorer == "cindex":
        score_name = "C-Index"
        param_grid={"model__alpha": 2.0 ** np.arange(0, 26, 2)}
    if scorer == "cindex-ipcw":
        score_name = "C-Index-IPCW"
        model = as_concordance_index_ipcw_scorer(model, tau=upper)
        param_grid={"model__estimator__alpha": 2.0 ** np.arange(0, 26, 2)}
    elif scorer == "auc":
        score_name = "AUC"
        model = as_cumulative_dynamic_auc_scorer(model, times=tr_times)
        param_grid={"model__estimator__alpha": 2.0 ** np.arange(0, 26, 2)}
    elif scorer == "ibs":
        score_name = "IBS"
        model = as_integrated_brier_score_scorer(model, times=tr_times)
        param_grid={"model__estimator__alpha": 2.0 ** np.arange(0, 26, 2)}
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        error_score=0.5,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    if scorer == "cindex":
        alphas = cv_results.param_model__alpha
    else:
        alphas = cv_results.param_model__estimator__alpha
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel(score_name)
    ax.set_xlabel("alpha")
    if scorer == "cindex":
        ax.axvline(gcv.best_params_["model__alpha"], c="C1")
    else:
        ax.axvline(gcv.best_params_["model__estimator__alpha"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"{script_dir}/cross_validation_fold{split_idx}.jpg")

    # Visualize coefficients of the best estimator
    best_model = gcv.best_estimator_.named_steps["model"]
    if scorer == "cindex":
        best_coefs = pd.DataFrame(best_model.coef_, index=tr_X.columns, columns=["coefficient"])
    else:
        best_coefs = pd.DataFrame(best_model.estimator_.coef_, index=tr_X.columns, columns=["coefficient"])

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(f"Number of non-zero coefficients: {non_zero}")

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index
    top10 = coef_order[:10]

    _, ax = plt.subplots(figsize=(8, 6))
    non_zero_coefs.loc[top10].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"{script_dir}/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def fastsvm(split_idx, tr_X, tr_y, scorer, n_jobs, rank_ratio=1):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = FastSurvivalSVM(alpha=1, rank_ratio=rank_ratio)
    lower, upper = np.percentile(tr_y["duration"], [20, 80])
    tr_times = np.arange(lower, upper + 1)
    if scorer == "cindex":
        score_name = "C-Index"
        param_grid={"model__alpha": 2.0 ** np.arange(-26, 0, 2)}
    if scorer == "cindex-ipcw":
        score_name = "C-Index-IPCW"
        model = as_concordance_index_ipcw_scorer(model, tau=upper)
        param_grid={"model__estimator__alpha": 2.0 ** np.arange(-26, 0, 2)}
    elif scorer == "auc":
        score_name = "AUC"
        model = as_cumulative_dynamic_auc_scorer(model, times=tr_times)
        param_grid={"model__estimator__alpha": 2.0 ** np.arange(-26, 0, 2)}
    elif scorer == "ibs":
        score_name = "IBS"
        model = as_integrated_brier_score_scorer(model, times=tr_times)
        param_grid={"model__estimator__alpha": 2.0 ** np.arange(-26, 0, 2)}
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        error_score=0.5,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    if scorer == "cindex":
        alphas = cv_results.param_model__alpha
    else:
        alphas = cv_results.param_model__estimator__alpha
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel(score_name)
    ax.set_xlabel("alpha")
    if scorer == "cindex":
        ax.axvline(gcv.best_params_["model__alpha"], c="C1")
    else:
        ax.axvline(gcv.best_params_["model__estimator__alpha"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"{script_dir}/cross_validation_fold{split_idx}.jpg")

    # Visualize coefficients of the best estimator
    best_model = gcv.best_estimator_.named_steps["model"]
    if scorer == "cindex":
        best_coefs = pd.DataFrame(best_model.coef_, index=tr_X.columns, columns=["coefficient"])
    else:
        best_coefs = pd.DataFrame(best_model.estimator_.coef_, index=tr_X.columns, columns=["coefficient"])

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(f"Number of non-zero coefficients: {non_zero}")

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index
    top10 = coef_order[:10]

    _, ax = plt.subplots(figsize=(8, 6))
    non_zero_coefs.loc[top10].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)
    plt.subplots_adjust(left=0.3)
    plt.savefig(f"{script_dir}/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def load_radiomics(
        split_idx,
        data,
        radiomics_aggregation,
        radiomics_aggregated_mode,
        radiomics_keys,
        use_graph_properties,
        n_jobs
    ):

    if radiomics_aggregated_mode in ["MEAN", "ABMIL", "SPARRA"]:
        radiomics_paths = []
        for p in data:
            path = pathlib.Path(p[0][1]["radiomics"])
            dir = path.parents[1] / radiomics_aggregated_mode / f"0{split_idx}"
            name = path.name.replace(".json", ".npy")
            radiomics_paths.append(dir / path.parent.name / name)
    else:
        radiomics_paths = [p[0][1]["radiomics"] for p in data]

    if radiomics_aggregation:
        dict_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(prepare_graph_radiomics)(idx, graph_path)
            for idx, graph_path in enumerate(radiomics_paths)
        )
    else:
        dict_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(load_radiomic_properties)(idx, graph_path, radiomics_keys)
            for idx, graph_path in enumerate(radiomics_paths)
        )
    radiomics_dict = {}
    for d in dict_list: radiomics_dict.update(d)
    radiomics_X = [radiomics_dict[f"{i}"] for i in range(len(radiomics_paths))]
    radiomics_X = pd.DataFrame(radiomics_X)

    if use_graph_properties:
        prop_paths = [p[0]["radiomics"] for p in data]
        prop_paths = [f"{p}".replace(".json", "_graph_properties.json") for p in prop_paths]
        prop_dict_list = [load_json(p) for p in prop_paths]
        prop_X = [prepare_graph_properties(d, omics="radiomics") for d in prop_dict_list]
        prop_X = pd.DataFrame(prop_X)
        radiomics_X = pd.concat([radiomics_X, prop_X], axis=1)
    return radiomics_X

def load_pathomics(
        split_idx,
        data,
        pathomics_aggregation,
        pathomics_aggregated_mode,
        pathomics_keys,
        use_graph_properties,
        n_jobs
    ):

    if pathomics_aggregated_mode in ["MEAN", "ABMIL", "SPARRA"]:
        pathomics_paths = []
        for p in data:
            path = pathlib.Path(p[0][1]["pathomics"])
            dir = path.parents[0] / pathomics_aggregated_mode / f"0{split_idx}"
            name = path.name.replace(".json", ".npy")
            pathomics_paths.append(dir / name)
    else:
        pathomics_paths = [p[0][1]["pathomics"] for p in data]

    if pathomics_aggregation:
        dict_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(prepare_graph_pathomics)(idx, graph_path, pathomics_keys)
            for idx, graph_path in enumerate(pathomics_paths)
        )
    else:
        dict_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(load_wsi_level_features)(idx, graph_path)
            for idx, graph_path in enumerate(pathomics_paths)
        )
    pathomics_dict = {}
    for d in dict_list: pathomics_dict.update(d)
    pathomics_X = [pathomics_dict[f"{i}"] for i in range(len(pathomics_paths))]
    pathomics_X = pd.DataFrame(pathomics_X)

    if use_graph_properties:
        prop_paths = [p[0]["pathomics"] for p in data]
        prop_paths = [f"{p}".replace(".json", "_graph_properties.json") for p in prop_paths]
        prop_dict_list = [load_json(p) for p in prop_paths]
        prop_X = [prepare_graph_properties(d, omics="pathomics") for d in prop_dict_list]
        prop_X = pd.DataFrame(prop_X)
        pathomics_X = pd.concat([pathomics_X, prop_X], axis=1)
    return pathomics_X

def survival(
        split_path,
        radiomics_keys=None,
        pathomics_keys=None,
        omics="radiopathomics", 
        n_jobs=32,
        radiomics_aggregation=False,
        radiomics_aggregated_mode=None,
        pathomics_aggregation=False,
        pathomics_aggregated_mode=None,
        model="CoxPH",
        scorer="cindex",
        feature_selection=True,
        n_bootstraps=100,
        use_graph_properties=False
        ):
    splits = joblib.load(split_path)
    predict_results = {}
    risk_results = {"risk": [], "event": [], "duration": []}
    for split_idx, split in enumerate(splits):
        print(f"Performing cross-validation on fold {split_idx}...")
        data_tr, data_va, data_te = split["train"], split["valid"], split["test"]
        data_tr = data_tr + data_va

        tr_y = np.array([p[1] for p in data_tr])
        tr_y = pd.DataFrame({'event': tr_y[:, 1].astype(bool), 'duration': tr_y[:, 0]})
        tr_y = tr_y.to_records(index=False)
        te_y = np.array([p[1] for p in data_te])
        te_y = pd.DataFrame({'event': te_y[:, 1].astype(bool), 'duration': te_y[:, 0]})
        te_y = te_y.to_records(index=False)

        # Concatenate multi-omics if required
        if omics == "radiopathomics":
            radiomics_tr_X = load_radiomics(
                split_idx=split_idx,
                data=data_tr,
                radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode,
                radiomics_keys=radiomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs
            )
            print("Selected training radiomics:", radiomics_tr_X.shape)
            print(radiomics_tr_X.head())

            pathomics_tr_X = load_pathomics(
                split_idx=split_idx,
                data=data_tr,
                pathomics_aggregation=pathomics_aggregation,
                pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs
            )
            print("Selected training pathomics:", pathomics_tr_X.shape)
            print(pathomics_tr_X.head())

            tr_X = pd.concat([radiomics_tr_X, pathomics_tr_X], axis=1)

            radiomics_te_X = load_radiomics(
                split_idx=split_idx,
                data=data_te,
                radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode,
                radiomics_keys=radiomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs
            )
            print("Selected testing radiomics:", radiomics_te_X.shape)
            print(radiomics_te_X.head())

            pathomics_te_X = load_pathomics(
                split_idx=split_idx,
                data=data_te,
                pathomics_aggregation=pathomics_aggregation,
                pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs
            )
            print("Selected testing pathomics:", pathomics_te_X.shape)
            print(pathomics_te_X.head())

            te_X = pd.concat([radiomics_te_X, pathomics_te_X], axis=1)
        elif omics == "pathomics":
            pathomics_tr_X = load_pathomics(
                split_idx=split_idx,
                data=data_tr,
                pathomics_aggregation=pathomics_aggregation,
                pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs
            )
            print("Selected training pathomics:", pathomics_tr_X.shape)
            print(pathomics_tr_X.head())

            pathomics_te_X = load_pathomics(
                split_idx=split_idx,
                data=data_te,
                pathomics_aggregation=pathomics_aggregation,
                pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs
            )
            print("Selected testing pathomics:", pathomics_te_X.shape)
            print(pathomics_te_X.head())

            tr_X, te_X = pathomics_tr_X, pathomics_te_X
        elif omics == "radiomics":
            radiomics_tr_X = load_radiomics(
                split_idx=split_idx,
                data=data_tr,
                radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode,
                radiomics_keys=radiomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs
            )
            print("Selected training radiomics:", radiomics_tr_X.shape)
            print(radiomics_tr_X.head())

            radiomics_te_X = load_radiomics(
                split_idx=split_idx,
                data=data_te,
                radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode,
                radiomics_keys=radiomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs
            )
            print("Selected testing radiomics:", radiomics_te_X.shape)
            print(radiomics_te_X.head())

            tr_X, te_X = radiomics_tr_X, radiomics_te_X
        else:
            raise NotImplementedError
        
        # df_prop = df_prop.apply(zscore)
        print("Selected training omics:", tr_X.shape)
        print(tr_X.head())
        print("Selected testing omics:", te_X.shape)
        print(te_X.head())

        # feature selection
        if feature_selection:
            print("Selecting features...")
            selector = VarianceThreshold(threshold=1e-4)
            selector.fit(tr_X[tr_y["event"]])
            selected_names = selector.get_feature_names_out().tolist()
            num_removed = len(tr_X.columns) - len(selected_names)
            print(f"Removing {num_removed} low-variance features...")
            tr_X = tr_X[selected_names]
            te_X = te_X[selected_names]
            print("Selecting univariate feature...")
            univariate_results = []
            for name in list(tr_X.columns):
                cph = CoxPHFitter()
                df = pd.DataFrame(
                    {
                        "duration": tr_y["duration"], 
                        "event": tr_y["event"],
                        name: tr_X[name] 
                    }
                )
                try:
                    cph.fit(df, "duration", "event")
                except Exception as e:
                    print(f"Skipping {name} | Error: {e}")
                    continue
                summary = cph.summary
                univariate_results.append({
                    'name': name,
                    'coef': summary['coef'].values[0],
                    'HR': summary['exp(coef)'].values[0],
                    'p_value': summary['p'].values[0],
                    'CI_low': summary['exp(coef) lower 95%'].values[0],
                    'CI_high': summary['exp(coef) upper 95%'].values[0]
                })
            results_df = pd.DataFrame(univariate_results)
            selected_names = results_df[results_df['p_value'] < 0.2]['name'].tolist()
            print(f"Selected features: {len(selected_names)}")
            tr_X = tr_X[selected_names]
            te_X = te_X[selected_names]

        # model fitting
        print("Fitting survival model...")
        if model == "Coxnet":
            predictor = coxnet(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model == "RSF":
            predictor = rsf(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model == "CoxPH":
            predictor = coxph(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model == "GradientBoost":
            predictor = gradientboosting(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model == "IPCRidge":
            predictor = ipcridge(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model == "FastSVM":
            predictor = fastsvm(split_idx, tr_X, tr_y, scorer, n_jobs, rank_ratio=1)

        # bootstrapping
        if n_bootstraps > 0:
            print("Bootstrapping...")
            stable_coefs = np.zeros(len(selected_names))
            for _ in range(n_bootstraps):
                tr_x_s, tr_y_s = resample(tr_X, tr_y)
                predictor.fit(tr_x_s, tr_y_s)
                if scorer == "cindex":
                    stable_coefs += (predictor.named_steps["model"].coef_ != 0).astype(int)
                else:
                    stable_coefs += (predictor.named_steps["model"].estimator_.coef_ != 0).astype(int)
            stable_coefs = stable_coefs / n_bootstraps
            final_coefs = np.where(stable_coefs > 0.8)[0]
            stable_names = [selected_names[i] for i in final_coefs.tolist()]
            tr_X = tr_X[stable_names]
            te_X = te_X[stable_names]
            predictor.fit(tr_X, tr_y)

        risk_scores = predictor.predict(te_X)
        risk_results["risk"] += risk_scores.tolist()
        risk_results["event"] += te_y["event"].astype(int).tolist()
        risk_results["duration"] += te_y["duration"].tolist()
        C_index = concordance_index_censored(te_y["event"], te_y["duration"], risk_scores)[0]
        C_index_ipcw = concordance_index_ipcw(tr_y, te_y, risk_scores)[0]

        lower, upper = np.percentile(te_y["duration"], [10, 90])
        times = np.arange(lower, upper + 1, 7)
        auc, mean_auc = cumulative_dynamic_auc(tr_y, te_y, risk_scores, times)
        if hasattr(predictor, "predict_survival_function"):
            survs = predictor.predict_survival_function(te_X)
            preds = np.asarray([[fn(t) for t in times] for fn in survs])
            IBS = integrated_brier_score(tr_y, te_y, preds, times)
        else:
            IBS = 0
        scores_dict = {
            "C-index": C_index,
            "C-index-IPCW": C_index_ipcw,
            "Mean AUC": mean_auc,
            "IBS": IBS
        }

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(times, auc)
        ax.set_xscale("linear")
        ax.set_ylabel("time-dependent AUC")
        ax.set_xlabel("days from enrollment")
        ax.axhline(mean_auc, linestyle="--")
        ax.grid(True)
        plt.savefig(f"{script_dir}/AUC_fold{split_idx}.jpg") 

        print(f"Updating regression results on fold {split_idx}")
        predict_results.update({f"Fold {split_idx}": scores_dict})
    print(predict_results)
    for k in scores_dict.keys():
        arr = np.array([v[k] for v in predict_results.values()])
        print(f"CV {k} mean+std", arr.mean(), arr.std())
    # plot survival curve
    pd_risk = pd.DataFrame(risk_results)
    mean_risk = pd_risk["risk"].mean()
    dem = pd_risk["risk"] > mean_risk
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots()
    kmf = KaplanMeierFitter()
    kmf.fit(pd_risk["duration"][dem], event_observed=pd_risk["event"][dem], label="High risk")
    kmf.plot_survival_function(ax=ax)

    kmf.fit(pd_risk["duration"][~dem], event_observed=pd_risk["event"][~dem], label="Low risk")
    kmf.plot_survival_function(ax=ax)

    # logrank test
    test_results = logrank_test(
        pd_risk["duration"][dem], pd_risk["duration"][~dem], 
        pd_risk["event"][dem], pd_risk["event"][~dem], 
        alpha=.99
    )
    test_results.print_summary()
    pvalue = test_results.p_value
    print(f"p-value: {pvalue}")
    ax.set_ylabel("Survival Probability")
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(f"{script_dir}/{omics}_survival_curve.png")
    return

def generate_data_split(
        x: list,
        y: list,
        train: float,
        valid: float,
        test: float,
        num_folds: int,
        seed: int = 5,
):
    """Helper to generate splits
    Args:
        x (list): a list of image paths
        y (list): a list of annotation paths
        train (float): ratio of training samples
        valid (float): ratio of validating samples
        test (float): ratio of testing samples
        num_folds (int): number of folds for cross-validation
        seed (int): random seed
    Returns:
        splits (list): a list of folds, each fold consists of train, valid, and test splits
    """
    assert train + valid + test - 1.0 < 1.0e-10, "Ratios must sum to 1.0"

    outer_splitter = StratifiedShuffleSplit(
        n_splits=num_folds,
        train_size=train + valid,
        random_state=seed,
    )
    inner_splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train / (train + valid),
        random_state=seed,
    )

    l = np.array([status for _, status in y])

    splits = []
    for train_valid_idx, test_idx in outer_splitter.split(x, l):
        test_x = [x[idx] for idx in test_idx]
        test_y = [y[idx] for idx in test_idx]
        x_ = [x[idx] for idx in train_valid_idx]
        y_ = [y[idx] for idx in train_valid_idx]
        l_ = [l[idx] for idx in train_valid_idx]

        if valid > 0:
            train_idx, valid_idx = next(iter(inner_splitter.split(x_, l_)))
            valid_x = [x_[idx] for idx in valid_idx]
            valid_y = [y_[idx] for idx in valid_idx]
            train_x = [x_[idx] for idx in train_idx]
            train_y = [y_[idx] for idx in train_idx]
        else:
            train_x, train_y = x_, y_

        train_x_subjects = []
        for i in train_x: train_x_subjects.append(i[0])
        test_x_subjects = []
        for i in test_x: test_x_subjects.append(i[0])
        if valid > 0:
            valid_x_subjects = []
            for i in valid_x: valid_x_subjects.append(i[0])
            assert len(set(train_x_subjects).intersection(set(valid_x_subjects))) == 0
            assert len(set(valid_x_subjects).intersection(set(test_x_subjects))) == 0
        else:
            assert len(set(train_x_subjects).intersection(set(test_x_subjects))) == 0

        if valid > 0:
            splits.append(
                {
                    "train": list(zip(train_x, train_y)),
                    "valid": list(zip(valid_x, valid_y)),
                    "test": list(zip(test_x, test_y)),
                }
            )
        else:
            splits.append(
                {
                    "train": list(zip(train_x, train_y)),
                    "test": list(zip(test_x, test_y)),
                }
            )
    return splits

def run_once(
        dataset_dict,
        num_epochs,
        save_dir,
        on_gpu=True,
        preproc_func=None,
        pretrained=None,
        loader_kwargs=None,
        arch_kwargs=None,
        optim_kwargs=None,
        BayesGNN=False,
        data_types=["radiomics", "pathomics"],
        sampling_rate=1.0
):
    """running the inference or training loop once"""
    if loader_kwargs is None:
        loader_kwargs = {}

    if arch_kwargs is None:
        arch_kwargs = {}

    if optim_kwargs is None:
        optim_kwargs = {}

    if BayesGNN:
        model = SurvivalBayesGraphArch(**arch_kwargs)
        kl = {"loss": bnn.BKLLoss(), "weight": 0.1}
    else:
        model = SurvivalGraphArch(**arch_kwargs)
        kl = None
    if pretrained is not None:
        model.load(pretrained)
    if on_gpu:
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    loss = CoxSurvLoss()
    optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, **optim_kwargs)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    loader_dict = {}
    for subset_name, subset in dataset_dict.items():
        _loader_kwargs = copy.deepcopy(loader_kwargs)
        if not "train" in subset_name: 
            if _loader_kwargs["batch_size"] > 8:
                _loader_kwargs["batch_size"] = 8
            sampling_rate = 1.0
        ds = SurvivalGraphDataset(
            subset, 
            mode=subset_name, 
            preproc=preproc_func,
            data_types=data_types,
            sampling_rate=sampling_rate
        )
        loader_dict[subset_name] = DataLoader(
            ds,
            drop_last=subset_name == "train",
            shuffle=subset_name == "train",
            **_loader_kwargs,
        )
    best_score = 0
    for epoch in range(num_epochs):
        logger.info("EPOCH: %03d", epoch)
        for loader_name, loader in loader_dict.items():
            step_output = []
            ema = ScalarMovingAverage()
            pbar = create_pbar(loader_name, len(loader))
            for step, batch_data in enumerate(loader):
                if loader_name == "train":
                    output = model.train_batch(model, batch_data, on_gpu, loss, optimizer, kl)
                    ema({"loss": output[0]})
                    pbar.postfix[1]["step"] = step
                    pbar.postfix[1]["EMA"] = ema.tracking_dict["loss"]
                else:
                    output = model.infer_batch(model, batch_data, on_gpu)
                    batch_size = loader_kwargs["batch_size"]
                    batch_size = output[0].shape[0]
                    output = [np.split(v, batch_size, axis=0) for v in output]
                    output = list(zip(*output))
                    step_output += output
                pbar.update()
            pbar.close()

            logging_dict = {}
            if loader_name == "train":
                for val_name, val in ema.tracking_dict.items():
                    logging_dict[f"train-EMA-{val_name}"] = val
            elif "infer" in loader_name and any(v in loader_name for v in ["train", "valid"]):
                output = list(zip(*step_output))
                logit, true = output
                logit = np.array(logit).squeeze()
                hazard = np.exp(logit)
                true = np.array(true).squeeze()
                event_status = true[:, 1] > 0
                event_time = true[:, 0]
                cindex = concordance_index_censored(event_status, event_time, hazard)[0]
                logging_dict[f"{loader_name}-Cindex"] = cindex

                if "valid-A" in loader_name and cindex > best_score: 
                    best_score = cindex
                    model.save(f"{save_dir}/best_model.weights.pth")

                logging_dict[f"{loader_name}-raw-logit"] = logit
                logging_dict[f"{loader_name}-raw-true"] = true

            for val_name, val in logging_dict.items():
                if "raw" not in val_name:
                    logging.info("%s: %0.5f\n", val_name, val)
            
            if "train" not in loader_dict:
                continue

            if (epoch + 1) % 10 == 0:
                new_stats = {}
                if (save_dir / "stats.json").exists():
                    old_stats = load_json(f"{save_dir}/stats.json")
                    save_as_json(old_stats, f"{save_dir}/stats.old.json", exist_ok=True)
                    new_stats = copy.deepcopy(old_stats)
                    new_stats = {int(k): v for k, v in new_stats.items()}

                old_epoch_stats = {}
                if epoch in new_stats:
                    old_epoch_stats = new_stats[epoch]
                old_epoch_stats.update(logging_dict)
                new_stats[epoch] = old_epoch_stats
                save_as_json(new_stats, f"{save_dir}/stats.json", exist_ok=True)
                model.save(
                    f"{save_dir}/epoch={epoch:03d}.weights.pth",
                )
        lr_scheduler.step()
    
    return step_output


def training(
        num_epochs,
        split_path,
        scaler_path,
        num_node_features,
        model_dir,
        conv="GCNConv",
        n_works=32,
        batch_size=32,
        BayesGNN=False,
        pool_ratio={"radiomics": 0.7, "pathomics": 0.2},
        omic_keys=["radiomics", "pathomics"],
        aggregation="SISIR",
        sampling_rate=0.1,
):
    """train node classification neural networks
    Args:
        num_epochs (int): the number of epochs for training
        split_path (str): the path of storing data splits
        scaler_path (str): the path of storing data normalization
        num_node_features (int): the dimension of node feature
        model_dir (str): directory of saving models
    """
    splits = joblib.load(split_path)
    node_scalers = [joblib.load(scaler_path[k]) for k in omic_keys] 
    transform_dict = {k: s.transform for k, s in zip(omic_keys, node_scalers)}
    
    loader_kwargs = {
        "num_workers": n_works, 
        "batch_size": batch_size,
    }
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": 1,
        "layers": [256, 128, 256],
        "dropout": 0.5,
        "pool_ratio": pool_ratio,
        "conv": conv,
        "keys": omic_keys,
        "aggregation": aggregation
    }
    omics_name = "_".join(omic_keys)
    if BayesGNN:
        model_dir = model_dir / f"{omics_name}_Bayes_Survival_Prediction_{conv}_{aggregation}"
    else:
        model_dir = model_dir / f"{omics_name}_Survival_Prediction_{conv}_{aggregation}_1e-1"
    optim_kwargs = {
        "lr": 3e-4,
        "weight_decay": {"ABMIL": 1.0e-5, "SISIR": 0.0}[aggregation],
    }
    for split_idx, split in enumerate(splits):
        # if split_idx < 3: continue
        new_split = {
            "train": split["train"],
            "infer-train": split["train"],
            "infer-valid-A": split["valid"],
            "infer-valid-B": split["test"],
        }
        split_save_dir = pathlib.Path(f"{model_dir}/{split_idx:02d}/")
        rm_n_mkdir(split_save_dir)
        reset_logging(split_save_dir)
        run_once(
            new_split,
            num_epochs,
            save_dir=split_save_dir,
            arch_kwargs=arch_kwargs,
            loader_kwargs=loader_kwargs,
            optim_kwargs=optim_kwargs,
            preproc_func=transform_dict,
            BayesGNN=BayesGNN,
            data_types=omic_keys,
            sampling_rate=sampling_rate
        )
    return

def inference(
        split_path,
        scaler_path,
        num_node_features,
        pretrained_dir,
        n_works=32,
        BayesGNN=False,
        conv="GCNConv",
        pool_ratio={"radiomics": 0.7, "pathomics": 0.2},
        omic_keys=["radiomics", "pathomics"],
        aggregation="SISIR",
        save_features=False
):
    """survival prediction
    """
    splits = joblib.load(split_path)
    node_scalers = [joblib.load(scaler_path[k]) for k in omic_keys] 
    transform_dict = {k: s.transform for k, s in zip(omic_keys, node_scalers)}
    
    loader_kwargs = {
        "num_workers": n_works, 
        "batch_size": 8,
    }
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": 1,
        "layers": [256, 128, 256],
        "dropout": 0.5,
        "pool_ratio": pool_ratio,
        "conv": conv,
        "keys": omic_keys,
        "aggregation": aggregation
    }
    omics_name = "_".join(omic_keys)
    if BayesGNN:
        pretrained_dir = pretrained_dir / f"{omics_name}_Bayes_Survival_Prediction_{conv}_{aggregation}"
    else:
        pretrained_dir = pretrained_dir / f"{omics_name}_Survival_Prediction_{conv}_{aggregation}_1e-1"

    cum_stats = []
    if "radiomics_pathomics" == omics_name: aggregation = f"radiopathomics_{aggregation}"
    for split_idx, split in enumerate(splits):
        if save_features:
            all_samples = splits[split_idx]["train"] + splits[split_idx]["valid"] + splits[split_idx]["test"]
        else:
            all_samples = split["test"]

        new_split = {"infer": [v[0] for v in all_samples]}
        chkpts = pretrained_dir / f"0{split_idx}/epoch=019.weights.pth"
        print(str(chkpts))
        # Perform ensembling by averaging probabilities
        # across checkpoint predictions

        outputs = run_once(
            new_split,
            num_epochs=1,
            on_gpu=True,
            save_dir=None,
            arch_kwargs=arch_kwargs,
            loader_kwargs=loader_kwargs,
            preproc_func=transform_dict,
            pretrained=chkpts,
            BayesGNN=BayesGNN,
            data_types=omic_keys
        )
        logits, features = list(zip(*outputs))
        # saving average features
        if save_features:
            if "radiomics" == omics_name:
                graph_paths = [d["radiomics"] for d in new_split["infer"]]
            elif "pathomics" == omics_name:
                graph_paths = [d["pathomics"] for d in new_split["infer"]]
            elif "radiomics_pathomics" == omics_name:
                graph_paths = [d["pathomics"] for d in new_split["infer"]]

            save_dir = pathlib.Path(graph_paths[0]).parents[0] / aggregation / f"0{split_idx}"
            mkdir(save_dir)
            for i, path in enumerate(graph_paths): 
                save_name = pathlib.Path(path).name.replace(".json", ".npy") 
                save_path = f"{save_dir}/{save_name}"
                np.save(save_path, features[i])
    
        # * Calculate split statistics
        logits = np.array(logits).squeeze()
        hazard = np.exp(logits)
        pred_true = np.array([v[1] for v in all_samples])

        pred_true = np.array(pred_true).squeeze()
        event_status = pred_true[:, 1] > 0
        event_time = pred_true[:, 0]
        cindex = concordance_index_censored(event_status, event_time, hazard)[0]

        cum_stats.append(
            {
                "C-Index": np.array(cindex)
            }
        )
        # print(f"Fold-{split_idx}:", cum_stats[-1])
    cindex_list = [stat["C-Index"] for stat in cum_stats]
    avg_stat = {
        "C-Index mean": np.stack(cindex_list, axis=0).mean(axis=0),
        "C-Index std": np.stack(cindex_list, axis=0).std(axis=0),
    }
    print(f"Avg:", avg_stat)

    return avg_stat

def test(
    graph_path,
    scaler_path,
    num_node_features,
    pretrained_model,
    conv="MLP",
    dropout=0,
    pool_ratio={"radiomics": 0.7, "pathomics": 0.2},
    omic_keys=["radiomics", "pathomics"],
    aggregation="SISIR",
):
    """node classification 
    """
    node_scalers = [joblib.load(scaler_path[k]) for k in omic_keys] 
    transform_dict = {k: s.transform for k, s in zip(omic_keys, node_scalers)}
    
    loader_kwargs = {
        "num_workers": 1,
        "batch_size": 1,
    }
    arch_kwargs = {
        "dim_features": num_node_features,
        "dim_target": 1,
        "layers": [256, 128, 256],
        "dropout": 0.5,
        "pool_ratio": pool_ratio,
        "conv": conv,
        "keys": omic_keys,
        "aggregation": aggregation
    }

    # BayesGNN = True
    new_split = {"infer": [graph_path]}
    outputs = run_once(
        new_split,
        num_epochs=1,
        save_dir=None,
        on_gpu=True,
        pretrained=pretrained_model,
        arch_kwargs=arch_kwargs,
        loader_kwargs=loader_kwargs,
        preproc_func=transform_dict,
        data_types=omic_keys,
    )

    pred_logit, _, attention = list(zip(*outputs))
    hazard = np.exp(np.array(pred_logit).squeeze())
    attention = np.array(attention).squeeze()
    return hazard, attention

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+', required=True, help='Path(s) to the config file(s).')
    args = parser.parse_args()
    
    from utilities.arguments import load_opt_from_config_files
    opt = load_opt_from_config_files(args.config_files)
    radiomics_mode = opt['RADIOMICS']['MODE']['VALUE']
    pathomics_mode = opt['PATHOMICS']['MODE']['VALUE']

    if opt['DATASET'] == "MAMAMIA":
        from analysis.a05_outcome_prediction.m_prepare_omics_info import prepare_MAMAMIA_omics_info
        omics_info = prepare_MAMAMIA_omics_info(
            img_dir=opt['DATA_INFO'],
            save_omics_dir=opt['OMICS_DIR'],
            segmentator=opt['RADIOMICS']['SEGMENTATOR']['VALUE'],
            radiomics_mode=radiomics_mode,
            radiomics_suffix=opt['RADIOMICS']['SUFFIX'],
        )
    elif opt['DATASET'] == "TCGA":
        from analysis.a05_outcome_prediction.m_prepare_omics_info import prepare_TCGA_omics_info
        omics_info = prepare_TCGA_omics_info(
            dataset_json=opt['DATA_INFO'],
            save_omics_dir=opt['OMICS_DIR'],
            radiomics_mode=radiomics_mode,
            segmentator=opt['RADIOMICS']['SEGMENTATOR']['VALUE'],
            radiomics_suffix=opt['RADIOMICS']['SUFFIX'],
            pathomics_mode=pathomics_mode,
            pathomics_suffix=opt['PATHOMICS']['SUFFIX']
        )
    else:
        raise NotImplementedError

    logger.info(f"Found {len(omics_info['omics_paths'])} subjects with omics")

    if not pathlib.Path(opt['SAVE_OUTCOME_FILE']).exists():
        from analysis.utilities.m_request_clinical_data import request_survival_data_by_submitter_new
        save_clinical_dir = pathlib.Path(opt['SAVE_OUTCOME_FILE']).parent
        subject_ids = list(omics_info['omics_paths'].keys())
        request_survival_data_by_submitter_new(
            submitter_ids=subject_ids,
            save_dir=save_clinical_dir,
            dataset=opt['DATASET']
        )
    
    if opt.get('SAVE_MODEL_DIR', False):
        save_model_dir = pathlib.Path(opt['SAVE_MODEL_DIR'])
        save_model_dir = save_model_dir / f"{opt['DATASET']}_survival_{opt['OUTCOME']['VALUE']}"
        save_model_dir = save_model_dir / f"{radiomics_mode}+{pathomics_mode}"
    else:
        save_model_dir = None

    # load patient outcomes
    df_outcome = prepare_patient_outcome(
        outcome_file=opt['SAVE_OUTCOME_FILE'],
        dataset=opt['DATASET'],
        outcome=opt['OUTCOME']['VALUE']
    )
    subject_ids = df_outcome['submitter_id'].to_list()
    logger.info(f"Found {len(subject_ids)} subjects with survival outcomes")

    omics_paths = [[k, omics_info['omics_paths'][k]] for k in subject_ids]
    
    y = df_outcome[['duration', 'event']].to_numpy(dtype=np.float32).tolist()

    # split data set
    splits = generate_data_split(
        x=omics_paths,
        y=y,
        train=opt['SPLIT']['TRAIN_RATIO'],
        valid=opt['SPLIT']['VALID_RATIO'],
        test=opt['SPLIT']['TEST_RATIO'],
        num_folds=opt['SPLIT']['FOLDS']
    )
    mkdir(save_model_dir)
    split_path = f"{save_model_dir}/survival_{radiomics_mode}_{pathomics_mode}_splits.dat"
    joblib.dump(splits, split_path)
    splits = joblib.load(split_path)
    num_train = len(splits[0]["train"])
    logging.info(f"Number of training samples: {num_train}.")
    num_valid = len(splits[0]["valid"])
    logging.info(f"Number of validating samples: {num_valid}.")
    num_test = len(splits[0]["test"])
    logging.info(f"Number of testing samples: {num_test}.")

    # survival analysis from the splits
    if opt['RADIOMICS'].get('KEYS', False):
        radiomics_keys = opt['RADIOMICS']['KEYS']
    else:
        radiomics_keys = None
    if opt['PATHOMICS'].get('KEYS', False):
        pathomics_keys = opt['PATHOMICS']['KEYS']
    else:
        pathomics_keys = None
    survival(
        split_path=split_path,
        omics=opt['USED_OMICS']['VALUE'],
        n_jobs=opt['N_JOBS'],
        radiomics_aggregation=opt['RADIOMICS']['AGGREGATION'],
        radiomics_aggregated_mode=opt['RADIOMICS']['AGGREGATED_MODE']['VALUE'],
        pathomics_aggregation=opt['PATHOMICS']['AGGREGATION'],
        pathomics_aggregated_mode=opt['PATHOMICS']['AGGREGATED_MODE']['VALUE'],
        radiomics_keys=radiomics_keys,
        pathomics_keys=pathomics_keys,
        model=opt['MODEL']['VALUE'],
        scorer=opt['SCORER']['VALUE'],
        feature_selection=opt['FEATURE_SELECTION'],
        n_bootstraps=opt['N_BOOTSTRAPS'],
        use_graph_properties=opt['USE_GRAPH_PROPERTIES']
    )

    # compute mean and std on training data for normalization 
    # splits = joblib.load(split_path)
    # train_graph_paths = [path for path, _ in splits[0]["train"]]
    # loader = SurvivalGraphDataset(train_graph_paths, mode="infer", data_types=data_types)
    # loader = DataLoader(
    #     loader,
    #     num_workers=8,
    #     batch_size=1,
    #     shuffle=False,
    #     drop_last=False,
    # )
    # omic_features = [{k: v.x_dict[k].numpy() for k in data_types} for v in loader]
    # omics_modes = {"radiomics": args.radiomics_mode, "pathomics": args.pathomics_mode}
    # for k, v in omics_modes.items():
    #     node_features = [d[k] for d in omic_features]
    #     node_features = np.concatenate(node_features, axis=0)
    #     node_scaler = StandardScaler(copy=False)
    #     node_scaler.fit(node_features)
    #     scaler_path = f"{save_model_dir}/survival_{k}_{v}_scaler.dat"
    #     joblib.dump(node_scaler, scaler_path)

    # training
    # omics_modes = {"radiomics": args.radiomics_mode, "pathomics": args.pathomics_mode}
    # omics_dims = {"radiomics": args.radiomics_dim, "pathomics": args.pathomics_dim}
    # omics_pool_ratio = {"radiomics": 0.7, "pathomics": 0.2}
    # omics_modes = {"radiomics": args.radiomics_mode}
    # omics_dims = {"radiomics": args.radiomics_dim}
    # omics_pool_ratio = {"radiomics": 0.7}
    # omics_modes = {"pathomics": args.pathomics_mode}
    # omics_dims = {"pathomics": args.pathomics_dim}
    # omics_pool_ratio = {"pathomics": 0.2}
    # split_path = f"{save_model_dir}/survival_radiopathomics_{args.radiomics_mode}_{args.pathomics_mode}_splits.dat"
    # scaler_paths = {k: f"{save_model_dir}/survival_{k}_{v}_scaler.dat" for k, v in omics_modes.items()}
    # training(
    #     num_epochs=args.epochs,
    #     split_path=split_path,
    #     scaler_path=scaler_paths,
    #     num_node_features=omics_dims,
    #     model_dir=save_model_dir,
    #     conv="GCNConv",
    #     n_works=32,
    #     batch_size=32,
    #     BayesGNN=False,
    #     pool_ratio=omics_pool_ratio,
    #     omic_keys=list(omics_modes.keys()),
    #     aggregation=["ABMIL", "SISIR"][1],
    #     sampling_rate=1
    # )

    # inference
    # inference(
    #     split_path=split_path,
    #     scaler_path=scaler_paths,
    #     num_node_features=omics_dims,
    #     pretrained_dir=save_model_dir,
    #     n_works=32,
    #     BayesGNN=False,
    #     conv="GCNConv",
    #     pool_ratio=omics_pool_ratio,
    #     omic_keys=list(omics_modes.keys()),
    #     aggregation=["ABMIL", "SISIR"][1],
    #     save_features=True
    # )

    # visualize radiomics
    # splits = joblib.load(split_path)
    # graph_path = splits[0]["test"][8][0]
    # radiomics_graph_name = pathlib.Path(graph_path["radiomics"]).stem
    # print("Visualizing radiological graph:", radiomics_graph_name)
    # img_path, lab_path = [
    #     (p, v) for p, v in zip(img_paths, lab_paths) 
    #     if pathlib.Path(p).stem == radiomics_graph_name.replace(f"_{class_name}", ".nii")
    # ][0]
    # pathomics_graph_name = pathlib.Path(graph_path["pathomics"]).stem
    # print("Visualizing pathological graph:", pathomics_graph_name)
    # wsi_path = [p for p in wsi_paths if pathlib.Path(p).stem == pathomics_graph_name][0]

    # # visualize radiomic graph
    # visualize_radiomic_graph(
    #     img_path=img_path,
    #     lab_path=lab_path,
    #     save_graph_dir=save_radiomics_dir,
    #     class_name=class_name
    # )

    # # visualize pathomic graph
    # visualize_pathomic_graph(
    #     wsi_path=wsi_path,
    #     graph_path=graph_path["pathomics"],
    #     magnify=True,
    #     # label=label_path,
    #     # show_map=True,
    #     # save_name=f"conch",
    #     # save_title="CONCH classification",
    #     resolution=args.resolution,
    #     units=args.units
    # )


    # visualize attention on WSI
    # splits = joblib.load(split_path)
    # graph_path = splits[0]["test"][15][0] # 
    # graph_name = pathlib.Path(graph_path["pathomics"]).stem
    # print("Visualizing on wsi:", graph_name)
    # wsi_path = [p for p in wsi_paths if p.stem == graph_name][0]
    # label_path = f"{graph_path}".replace(".json", ".label.npy")
    # pretrained_model = save_model_dir / "pathomics_Survival_Prediction_GCNConv_SISIR/00/epoch=019.weights.pth"
    # hazard, attention = test(
    #     graph_path=graph_path,
    #     scaler_path=scaler_paths,
    #     num_node_features=omics_dims,
    #     pretrained_model=pretrained_model,
    #     conv="GCNConv",
    #     dropout=0,
    #     pool_ratio=omics_pool_ratio,
    #     omic_keys=list(omics_modes.keys()),
    #     aggregation=["ABMIL", "SISIR"][1],
    # )
    # visualize_pathomic_graph(
    #     wsi_path=wsi_path,
    #     graph_path=graph_path["pathomics"],
    #     label=attention,
    #     show_map=True,
    #     save_title="Normalized Inverse Precision",
    #     resolution=args.resolution,
    #     units=args.units
    # )

    # visualize attention on CT
    # splits = joblib.load(split_path)
    # graph_path = splits[0]["test"][8][0]
    # radiomics_graph_name = pathlib.Path(graph_path["radiomics"]).stem
    # print("Visualizing radiological graph:", radiomics_graph_name)
    # img_path, lab_path = [
    #     (p, v) for p, v in zip(img_paths, lab_paths) 
    #     if pathlib.Path(p).stem == radiomics_graph_name.replace(f"_{class_name}", ".nii")
    # ][0]
    # pretrained_model = save_model_dir / "radiomics_Survival_Prediction_GCNConv_SISIR_1e-1/00/epoch=019.weights.pth"
    # hazard, attention = test(
    #     graph_path=graph_path,
    #     scaler_path=scaler_paths,
    #     num_node_features=omics_dims,
    #     pretrained_model=pretrained_model,
    #     conv="GCNConv",
    #     dropout=0,
    #     pool_ratio=omics_pool_ratio,
    #     omic_keys=list(omics_modes.keys()),
    #     aggregation=["ABMIL", "SISIR"][1],
    # )
    # visualize_radiomic_graph(
    #     img_path=img_path,
    #     lab_path=lab_path,
    #     save_graph_dir=save_radiomics_dir,
    #     class_name=class_name,
    #     attention=None,
    #     n_jobs=32
    # )
