import os
import sys

# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
relative_path = os.path.join(script_dir, '../../')
sys.path.append(relative_path)

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

from torch_geometric.loader import DataLoader
from tiatoolbox import logger
from tiatoolbox.utils.misc import save_as_json

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from sksurv.metrics import (
    concordance_index_censored, 
    concordance_index_ipcw,
    integrated_brier_score, 
    cumulative_dynamic_auc,
    as_concordance_index_ipcw_scorer,
    as_cumulative_dynamic_auc_scorer,
    as_integrated_brier_score_scorer
)
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample

from utilities.m_utils import mkdir, load_json, create_pbar, rm_n_mkdir, reset_logging

def plot_survival_curve(data_path):
    df = pd.read_csv(data_path)

    # Prepare the survival data
    df['event'] = df['vital_status'].apply(lambda x: True if x == 'Dead' else False)
    df['duration'] = df['days_to_death'].fillna(df['days_to_last_follow_up'])
    df = df[df['duration'].notna()]
    df = df[df['ajcc_pathologic_stage'].isin(["Stage I", "Stage II"])]
    print("Data strcuture:", df.shape)

    # Fit the Kaplan-Meier estimator
    from sksurv.nonparametric import kaplan_meier_estimator
    time, survival_prob, conf_int = kaplan_meier_estimator(
        df["event"], df["duration"], conf_type="log-log"
        )
    plt.step(time, survival_prob, where="post")
    plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.ylim(0, 1)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.savefig(f"{relative_path}/figures/plots/survival_curve.jpg")
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
    for path_dict in radiomic_paths:
        properties_dict = {}
        for radiomic_key, radiomic_path in path_dict.items():
            suffix = pathlib.Path(radiomic_path).suffix
            if suffix == ".json":
                if "/pyradiomics/" in radiomic_path:
                    if prop_keys is None: 
                        prop_keys = ["shape", "firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm"]
                    data_dict = load_json(radiomic_path)
                    properties = {}
                    for key, value in data_dict.items():
                        selected = [((k in key) and ("diagnostics" not in key)) for k in prop_keys]
                        if any(selected): properties[f"{radiomic_key}.{key}"] = value
                    if len(properties) > 0: properties_dict.update(properties)
                elif "/BayesBP/" in radiomic_path:
                    data_dict = load_json(radiomic_path)
                    properties = {}
                    for key, value in data_dict.items():
                        properties[f"{radiomic_key}.{key}"] = value
                    if len(properties) > 0: properties_dict.update(properties)
                else:
                    raise NotImplementedError
            elif suffix == ".npy":
                feature = np.load(radiomic_path)
                feat_list = np.array(feature).squeeze().tolist()
                properties = {}
                for i, feat in enumerate(feat_list):
                    k = f"radiomics.{radiomic_key}.feature{i}"
                    properties[k] = feat
                if len(properties) > 0: properties_dict.update(properties)
        properties_list.append(properties_dict)

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
    graph_paths, 
    subgraphs=["TUM", "NORM", "DEB"], 
    mode="mean",
    pooling="mean"
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
    feat_list = []
    for path_dict in graph_paths:
        feat_stack = []
        for path in path_dict.values():
            graph_dict = load_json(path)
            feature = np.array(graph_dict["x"])
            assert feature.ndim == 2
            if subgraph_ids is not None:
                label_path = f"{path}".replace(".json", ".label.npy")
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
                feature = np.mean(feature, axis=0)
            elif mode == "max":
                feature = np.max(feature, axis=0)
            elif mode == "min":
                feature = np.min(feature, axis=0)
            elif mode == "std":
                feature = np.std(feature, axis=0)
            elif mode == "kmeans":
                kmeans = KMeans(n_clusters=4)
                feature = kmeans.fit(feature).cluster_centers_
                feature = feature.flatten().tolist()
            feat_stack.append(feature)
        feat_list.append(np.hstack(feat_stack))
        
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

def prepare_graph_radiomics(
    idx, 
    graph_paths, 
    mode="mean",
    pooling="mean"
    ):
    feat_list = []
    for path_dict in graph_paths:
        feat_stack = []
        for path in path_dict.values():
            graph_dict = load_json(path)
            feature = np.array(graph_dict["x"])
            assert feature.ndim == 2

            if mode == "mean":
                feature = np.mean(feature, axis=0)
            elif mode == "max":
                feature = np.max(feature, axis=0)
            elif mode == "min":
                feature = np.min(feature, axis=0)
            elif mode == "std":
                feature = np.std(feature, axis=0)
            elif mode == "kmeans":
                kmeans = KMeans(n_clusters=4)
                feature = kmeans.fit(feature).cluster_centers_
                feature = feature.flatten()
            else:
                raise ValueError(f"Unsupport aggregation mode: {mode}")
            feat_stack.append(feature)
        feat_list.append(np.hstack(feat_stack))
    
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

def load_wsi_level_features(idx, wsi_feature_paths, pooling="mean"):
    feat_list = []
    for path_dict in wsi_feature_paths:
        feat_stack = []
        for path in path_dict.values():
            feat = np.array(np.load(path)).squeeze()
            feat_stack.append(feat)
        feat_list.append(np.hstack(feat_stack))

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

def load_subject_level_features(idx, subject_feature_path, outcome=None):
    _, ext = os.path.splitext(subject_feature_path)

    feat_dict = {}

    if ext == ".npy":
        feat_list = np.load(subject_feature_path, allow_pickle=True).squeeze()

        # Ensure iterable
        if np.isscalar(feat_list):
            feat_list = [feat_list]

        for i, feat in enumerate(feat_list):
            k = f"subject.feature{i}"
            feat_dict[k] = feat

    elif ext == ".json":
        with open(subject_feature_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("JSON feature file must contain a dictionary")

        # Use JSON keys and values directly
        for k, feat in data.items():
            if outcome is None:
                feat_dict[k] = feat
            else:
                if outcome == k.split('_')[0]: feat_dict[k] = feat

    else:
        raise ValueError(f"Unsupported feature file format: {ext}")

    return {str(idx): feat_dict}


def prepare_patient_outcome(outcome_file, subject_ids, dataset="MAMA-MIA", outcome=None):
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
        df['SubjectID'] = df['patient_id']
        df = df[df["SubjectID"].isin(subject_ids)]
    elif dataset == "TCGA":
        df = pd.read_csv(outcome_file)
        # Prepare the survival data
        if outcome == 'OS':
            df = df[df['OS'].notna() & df['OS.time'].notna()]
            df['event'] = df['OS'].apply(lambda x: True if x == 1 else False)
            df['duration'] = df['OS.time']
        elif outcome == 'DSS':
            df = df[df['DSS'].notna() & df['DSS.time'].notna()]
            df['event'] = df['DSS'].apply(lambda x: True if x == 1 else False)
            df['duration'] = df['DSS.time']
        elif outcome == 'DFI':
            df = df[df['DFI'].notna() & df['DFI.time'].notna()]
            df['event'] = df['DFI'].apply(lambda x: True if x == 1 else False)
            df['duration'] = df['DFI.time']
        elif outcome == 'PFI':
            df = df[df['PFI'].notna() & df['PFI.time'].notna()]
            df['event'] = df['PFI'].apply(lambda x: True if x == 1 else False)
            df['duration'] = df['PFI.time']
        else:
            raise ValueError(f'Unsuppored outcome type: {outcome}')
        df['SubjectID'] = df["_PATIENT"]
        df = df[df["SubjectID"].isin(subject_ids)]
    else:
        raise NotImplementedError
    
    df = df.drop_duplicates(subset="SubjectID", keep="first")
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
    plt.savefig(f"{relative_path}/figures/plots/coefficients.jpg")

def coxnet(split_idx, tr_X, tr_y, scorer, n_jobs, l1_ratio=0.9, min_ratio=0.1):
    from sksurv.linear_model import CoxnetSurvivalAnalysis
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
    plt.savefig(f"{relative_path}/figures/plots/cross_validation_fold{split_idx}.jpg")

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
    plt.savefig(f"{relative_path}/figures/plots/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def rsf(split_idx, tr_X, tr_y, scorer, n_jobs):
    from sksurv.ensemble import RandomSurvivalForest
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
    plt.savefig(f"{relative_path}/figures/plots/cross_validation_fold{split_idx}.jpg")

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def gradientboosting(split_idx, tr_X, tr_y, scorer, n_jobs, loss="coxph"):
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
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
    plt.savefig(f"{relative_path}/figures/plots/cross_validation_fold{split_idx}.jpg")

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe


def coxph(split_idx, tr_X, tr_y, scorer, n_jobs):
    from sksurv.linear_model import CoxPHSurvivalAnalysis
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
    plt.savefig(f"{relative_path}/figures/plots/cross_validation_fold{split_idx}.jpg")

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
    plt.savefig(f"{relative_path}/figures/plots/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def ipcridge(split_idx, tr_X, tr_y, scorer, n_jobs):
    from sksurv.linear_model import IPCRidge
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
    plt.savefig(f"{relative_path}/figures/plots/cross_validation_fold{split_idx}.jpg")

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
    plt.savefig(f"{relative_path}/figures/plots/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def fastsvm(split_idx, tr_X, tr_y, scorer, n_jobs, rank_ratio=1):
    from sksurv.svm import FastSurvivalSVM
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
    plt.savefig(f"{relative_path}/figures/plots/cross_validation_fold{split_idx}.jpg")

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
    plt.savefig(f"{relative_path}/figures/plots/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def load_radiomics(
        data,
        radiomics_aggregation,
        radiomics_aggregated_mode,
        radiomics_keys,
        use_graph_properties,
        n_jobs,
        save_radiomics_dir=None,
        outcome=None
    ):

    if radiomics_aggregated_mode in ["MEAN", "ABMIL", "SPARRA"]:
        print(f"loading radiomics from {save_radiomics_dir}...")
        radiomics_paths = []
        for p in data:
            subject_id = p[0][0]
            path = pathlib.Path(save_radiomics_dir) / radiomics_aggregated_mode / f"{subject_id}.npy"
            radiomics_paths.append(path)
        dict_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(load_subject_level_features)(idx, graph_path, outcome)
            for idx, graph_path in enumerate(radiomics_paths)
        )
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
        data,
        pathomics_aggregation,
        pathomics_aggregated_mode,
        pathomics_keys,
        use_graph_properties,
        n_jobs,
        save_pathomics_dir=None,
        outcome=None
    ):

    if pathomics_aggregated_mode in ["MEAN", "ABMIL", "SPARRA"]:
        print(f"loading pathomics from {save_pathomics_dir}...")
        pathomics_paths = []
        for p in data:
            subject_id = p[0][0]
            path = pathlib.Path(save_pathomics_dir) / pathomics_aggregated_mode / f"{subject_id}.npy"
            pathomics_paths.append(path)
        dict_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(load_subject_level_features)(idx, graph_path, outcome)
            for idx, graph_path in enumerate(pathomics_paths)
        )
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

def load_radiopathomics(
        data,
        radiomics_aggregation,
        radiomics_aggregated_mode,
        radiomics_keys,
        pathomics_aggregation,
        pathomics_aggregated_mode,
        pathomics_keys,
        use_graph_properties,
        n_jobs,
        save_radiopathomics_dir,
        outcome=None
    ):
    if radiomics_aggregated_mode in ["MEAN", "ABMIL", "SPARRA"]:
        assert radiomics_aggregated_mode == pathomics_aggregated_mode
        print(f"loading radiopathomics from {save_radiopathomics_dir}...")
        radiopathomics_paths = []
        for p in data:
            subject_id = p[0][0]
            path = pathlib.Path(save_radiopathomics_dir) / radiomics_aggregated_mode / f"{subject_id}.npy"
            radiopathomics_paths.append(path)
        dict_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(load_subject_level_features)(idx, graph_path, outcome)
            for idx, graph_path in enumerate(radiopathomics_paths)
        )

        radiopathomics_dict = {}
        for d in dict_list: radiopathomics_dict.update(d)
        radiopathomics_X = [radiopathomics_dict[f"{i}"] for i in range(len(radiopathomics_paths))]
        radiopathomics_X = pd.DataFrame(radiopathomics_X)

        if use_graph_properties:
            prop_paths = [p[0][1]["radiomics"] for p in data]
            prop_paths = [f"{p}".replace(".json", "_graph_properties.json") for p in prop_paths]
            prop_dict_list = [load_json(p) for p in prop_paths]
            prop_X = [prepare_graph_properties(d, omics="radiomics") for d in prop_dict_list]
            prop_X = pd.DataFrame(prop_X)
            radiopathomics_X = pd.concat([radiopathomics_X, prop_X], axis=1)
            prop_paths = [p[0][1]["pathomics"] for p in data]
            prop_paths = [f"{p}".replace(".json", "_graph_properties.json") for p in prop_paths]
            prop_dict_list = [load_json(p) for p in prop_paths]
            prop_X = [prepare_graph_properties(d, omics="pathomics") for d in prop_dict_list]
            prop_X = pd.DataFrame(prop_X)
            radiopathomics_X = pd.concat([radiopathomics_X, prop_X], axis=1)
    else:
        radiomics_X = load_radiomics(
            data=data,
            radiomics_aggregation=radiomics_aggregation,
            radiomics_aggregated_mode=radiomics_aggregated_mode,
            radiomics_keys=radiomics_keys,
            use_graph_properties=use_graph_properties,
            n_jobs=n_jobs
        )
        
        pathomics_X = load_pathomics(
            data=data,
            pathomics_aggregation=pathomics_aggregation,
            pathomics_aggregated_mode=pathomics_aggregated_mode,
            pathomics_keys=pathomics_keys,
            use_graph_properties=use_graph_properties,
            n_jobs=n_jobs
        ) 

        radiopathomics_X = pd.concat([radiomics_X, pathomics_X], axis=1)

    return radiopathomics_X

def survival(
    split_path,
    radiomics_keys=None,
    pathomics_keys=None,
    omics="radiopathomics", 
    save_omics_dir=None,
    outcome=None,
    n_jobs=32,
    radiomics_aggregation=False,
    radiomics_aggregated_mode=None,
    pathomics_aggregation=False,
    pathomics_aggregated_mode=None,
    model="CoxPH",
    scorer="cindex",
    feature_selection=True,
    n_bootstraps=100,
    use_graph_properties=False,
    save_results_dir=None
    ):
    splits = joblib.load(split_path)
    predict_results = {}
    survival_results = {
        "raw_subject": [], "raw_risk": [], 
        "subject": [], "risk": [], 
        "event": [], "duration": []
    }
    for split_idx, split in enumerate(splits):
        print(f"Performing cross-validation on fold {split_idx}...")
        raw_data_tr, raw_data_va, raw_data_te = split["train"], split["valid"], split["test"]
        raw_data_tr = raw_data_tr + raw_data_va

        data_tr = [p for p in raw_data_tr if p[1] is not None]
        data_te = [p for p in raw_data_te if p[1] is not None]

        tr_y = np.array([p[1] for p in data_tr])
        tr_y = pd.DataFrame({'event': tr_y[:, 1].astype(bool), 'duration': np.maximum(tr_y[:, 0], 1e-6)})
        tr_y = tr_y.to_records(index=False)
        te_y = np.array([p[1] for p in data_te])
        te_y = pd.DataFrame({'event': te_y[:, 1].astype(bool), 'duration': np.maximum(te_y[:, 0], 1e-6)})
        te_y = te_y.to_records(index=False)

        # Concatenate multi-omics if required
        if omics == "radiopathomics":
            tr_X = load_radiopathomics(
                data=data_tr,
                radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode,
                radiomics_keys=radiomics_keys,
                pathomics_aggregation=pathomics_aggregation,
                pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs,
                save_radiopathomics_dir=save_omics_dir,
                outcome=outcome
            )
            print("Selected training radiopathomics:", tr_X.shape)
            print(tr_X.head())

            te_X = load_radiopathomics(
                data=data_te,
                radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode,
                radiomics_keys=radiomics_keys,
                pathomics_aggregation=pathomics_aggregation,
                pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs,
                save_radiopathomics_dir=save_omics_dir,
                outcome=outcome
            )
            print("Selected testing radiopathomics:", te_X.shape)
            print(te_X.head())

            raw_te_X = load_radiopathomics(
                data=raw_data_te,
                radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode,
                radiomics_keys=radiomics_keys,
                pathomics_aggregation=pathomics_aggregation,
                pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs,
                save_radiopathomics_dir=save_omics_dir,
                outcome=outcome
            )
            print("Selected raw testing radiopathomics:", raw_te_X.shape)
            print(raw_te_X.head())
        elif omics == "pathomics":
            pathomics_tr_X = load_pathomics(
                data=data_tr,
                pathomics_aggregation=pathomics_aggregation,
                pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs,
                save_pathomics_dir=save_omics_dir,
                outcome=outcome
            )
            print("Selected training pathomics:", pathomics_tr_X.shape)
            print(pathomics_tr_X.head())

            pathomics_te_X = load_pathomics(
                data=data_te,
                pathomics_aggregation=pathomics_aggregation,
                pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs,
                save_pathomics_dir=save_omics_dir,
                outcome=outcome
            )
            print("Selected testing pathomics:", pathomics_te_X.shape)
            print(pathomics_te_X.head())

            tr_X, te_X = pathomics_tr_X, pathomics_te_X

            raw_te_X = load_pathomics(
                data=raw_data_te,
                pathomics_aggregation=pathomics_aggregation,
                pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs,
                save_pathomics_dir=save_omics_dir,
                outcome=outcome
            )
            print("Selected raw testing pathomics:", raw_te_X.shape)
            print(raw_te_X.head())
        elif omics == "radiomics":
            radiomics_tr_X = load_radiomics(
                data=data_tr,
                radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode,
                radiomics_keys=radiomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs,
                save_radiomics_dir=save_omics_dir,
                outcome=outcome
            )
            print("Selected training radiomics:", radiomics_tr_X.shape)
            print(radiomics_tr_X.head())

            radiomics_te_X = load_radiomics(
                data=data_te,
                radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode,
                radiomics_keys=radiomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs,
                save_radiomics_dir=save_omics_dir,
                outcome=outcome
            )
            print("Selected testing radiomics:", radiomics_te_X.shape)
            print(radiomics_te_X.head())

            tr_X, te_X = radiomics_tr_X, radiomics_te_X
            raw_te_X = load_radiomics(
                data=raw_data_te,
                radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode,
                radiomics_keys=radiomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs,
                save_radiomics_dir=save_omics_dir,
                outcome=outcome
            )
            print("Selected raw testing radiomics:", raw_te_X.shape)
            print(raw_te_X.head())
        else:
            raise NotImplementedError
        
        # df_prop = df_prop.apply(zscore)
        print("Selected training omics:", tr_X.shape)
        print(tr_X.head())
        print("Selected testing omics:", te_X.shape)
        print(te_X.head())
        print("Selected raw testing omics:", raw_te_X.shape)
        print(raw_te_X.head())

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
            raw_te_X = raw_te_X[selected_names]
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
            raw_te_X = raw_te_X[selected_names]

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
            raw_te_X = raw_te_X[stable_names]
            predictor.fit(tr_X, tr_y)

        raw_subject_ids = [p[0][0] for p in raw_data_te]
        survival_results["raw_subject"] += raw_subject_ids
        raw_risk_scores = predictor.predict(raw_te_X)
        survival_results["raw_risk"] += raw_risk_scores.tolist()

        subject_ids = [p[0][0] for p in data_te]
        survival_results["subject"] += subject_ids
        risk_scores = predictor.predict(te_X)
        survival_results["risk"] += risk_scores.tolist()
        survival_results["event"] += te_y["event"].astype(int).tolist()
        survival_results["duration"] += te_y["duration"].tolist()
        C_index = concordance_index_censored(te_y["event"], te_y["duration"], risk_scores)[0]
        C_index_ipcw = concordance_index_ipcw(tr_y, te_y, risk_scores)[0]

        lower, upper = np.percentile(te_y["duration"], [10, 90])
        times = np.arange(lower, upper + 1, 7)
        auc, mean_auc = cumulative_dynamic_auc(tr_y, te_y, risk_scores, times)
        if hasattr(predictor, "predict_survival_function"):
            try:
                survs = predictor.predict_survival_function(te_X)
                preds = np.asarray([[fn(t) for t in times] for fn in survs])
                IBS = integrated_brier_score(tr_y, te_y, preds, times)
            except Exception as e:
                IBS = 0
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
        plt.savefig(f"{relative_path}/figures/plots/AUC_fold{split_idx}.jpg") 

        print(f"Updating regression results on fold {split_idx}")
        predict_results.update({f"Fold {split_idx}": scores_dict})
    print(predict_results)
    for k in scores_dict.keys():
        arr = np.array([v[k] for v in predict_results.values() if v[k] != 0])
        print(f"CV {k} mean+std", arr.mean(), arr.std())
    # plot survival curve
    pd_risk = pd.DataFrame({k: survival_results[k] for k in ["risk", "event", "duration"]})
    mean_risk = pd_risk["risk"].mean()
    dem = pd_risk["risk"] > mean_risk

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})
    kmf1 = KaplanMeierFitter()
    kmf1.fit(pd_risk["duration"][dem], event_observed=pd_risk["event"][dem], label="High risk")
    kmf1.plot_survival_function(ax=ax)

    kmf2 = KaplanMeierFitter()
    kmf2.fit(pd_risk["duration"][~dem], event_observed=pd_risk["event"][~dem], label="Low risk")
    kmf2.plot_survival_function(ax=ax)
    add_at_risk_counts(kmf1, kmf2, ax=ax)
    plt.tight_layout()

    # logrank test
    test_results = logrank_test(
        pd_risk["duration"][dem], pd_risk["duration"][~dem], 
        pd_risk["event"][dem], pd_risk["event"][~dem], 
        alpha=.99
    )
    test_results.print_summary()
    pvalue = test_results.p_value
    print(f"p-value: {pvalue}")
    predict_results.update({'p-value': pvalue})
    ax.set_ylabel("Survival Probability")
    # plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(f"{relative_path}/figures/plots/{omics}_survival_curve.png")

    #save predicted results
    save_path = f"{save_results_dir}/{omics}_" + \
        f"radio+{radiomics_aggregated_mode}_" + \
        f"patho+{pathomics_aggregated_mode}_" + \
        f"model+{model}_scorer+{scorer}_results.json"
    with open(save_path, "w") as f:
        json.dump(survival_results, f, indent=4)

    # save metrics
    save_path = f"{save_results_dir}/{omics}_" + \
        f"radio+{radiomics_aggregated_mode}_" + \
        f"patho+{pathomics_aggregated_mode}_" + \
        f"model+{model}_scorer+{scorer}_metrics.json"
    with open(save_path, "w") as f:
        json.dump(predict_results, f, indent=4)

    return

def generate_data_split(
        x: list,
        y: list,
        train: float,
        valid: float,
        test: float,
        num_folds: int,
        seed: int = 42
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
    from sklearn.model_selection import train_test_split
    assert train + valid <= 1.0, "train + valid ratio must be <= 1.0"

    outer_splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = []

    for train_valid_idx, test_idx in outer_splitter.split(x):
        # Split test set
        test_x = [x[i] for i in test_idx]
        test_y = [y[i] for i in test_idx]

        train_valid_x = [x[i] for i in train_valid_idx]
        train_valid_y = [y[i] for i in train_valid_idx]

        # Optional validation split
        if valid > 0:
            ratio = valid / (train + valid)
            train_x, valid_x, train_y, valid_y = train_test_split(
                train_valid_x, train_valid_y, test_size=ratio, random_state=seed, shuffle=True
            )
        else:
            train_x, train_y = train_valid_x, train_valid_y
            valid_x, valid_y = None, None

        split_dict = {
            "train": list(zip(train_x, train_y)),
            "test": list(zip(test_x, test_y))
        }
        if valid > 0:
            split_dict["valid"] = list(zip(valid_x, valid_y))

        splits.append(split_dict)

    return splits

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_files',
        nargs='+',
        required=True,
        help='Path(s) to the config file(s).'
    )
    parser.add_argument(
        '--override',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Override config entries'
    )

    args = parser.parse_args()
    
    from utilities.arguments import load_opt_from_config_files_overrides
    opt = load_opt_from_config_files_overrides(args.config_files, overrides=args.override)

    radiomics_mode = opt['RADIOMICS']['MODE']['VALUE']
    pathomics_mode = opt['PATHOMICS']['MODE']['VALUE']

    if opt['DATASET'] == "MAMAMIA":
        from analysis.a05_outcome_prediction.m_prepare_omics_info import prepare_MAMAMIA_omics_info
        from analysis.a05_outcome_prediction.m_prepare_omics_info import radiomics_suffix
        omics_info = prepare_MAMAMIA_omics_info(
            img_dir=opt['DATA_INFO'],
            save_omics_dir=opt['OMICS_DIR'],
            segmentator=opt['RADIOMICS']['SEGMENTATOR']['VALUE'],
            radiomics_mode=radiomics_mode,
            radiomics_suffix=radiomics_suffix[radiomics_mode],
        )
    elif opt['DATASET'] == "TCGA":
        from analysis.a05_outcome_prediction.m_prepare_omics_info import prepare_TCGA_omics_info
        from analysis.a05_outcome_prediction.m_prepare_omics_info import radiomics_suffix
        from analysis.a05_outcome_prediction.m_prepare_omics_info import pathomics_suffix
        omics_info = prepare_TCGA_omics_info(
            dataset_json=opt['DATA_INFO'],
            save_omics_dir=opt['OMICS_DIR'],
            radiomics_mode=radiomics_mode,
            segmentator=opt['RADIOMICS']['SEGMENTATOR']['VALUE'],
            radiomics_suffix=radiomics_suffix[radiomics_mode],
            pathomics_mode=pathomics_mode,
            pathomics_suffix=pathomics_suffix[pathomics_mode]
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

    # split data set based on subject ids
    subject_ids=list(omics_info['omics_paths'].keys())
    splits = generate_data_split(
        x=subject_ids,
        y=[None]*len(subject_ids),
        train=opt['SPLIT']['TRAIN_RATIO'],
        valid=opt['SPLIT']['VALID_RATIO'],
        test=opt['SPLIT']['TEST_RATIO'],
        num_folds=opt['SPLIT']['FOLDS']
    )

    # load patient outcomes
    df_outcome = prepare_patient_outcome(
        outcome_file=opt['SAVE_OUTCOME_FILE'],
        subject_ids=subject_ids,
        dataset=opt['DATASET'],
        outcome=opt['OUTCOME']['VALUE']
    )
    subject_ids = df_outcome['SubjectID'].to_list()
    logger.info(f"Found {len(subject_ids)} subjects with survival outcomes")

    omics_paths = [[k, omics_info['omics_paths'][k]] for k in subject_ids]
    outcomes = df_outcome[['duration', 'event']].to_numpy(dtype=np.float32).tolist()

    # Build mapping: subject_id → (omics_path_info, outcome)
    matched_samples = {x[0]: (x, y) for x, y in zip(omics_paths, outcomes)}
    all_samples = {k: ([k, v], None) for k, v in omics_info['omics_paths'].items()}

    # Filter splits based on available subject_ids
    if opt['SPLIT']['FILTER_SUBJECTS']:
        new_splits = []
        for split in splits:
            new_split = {
                key: [matched_samples[k] for (k, _) in subjects if k in subject_ids]
                for key, subjects in split.items()
            }
            new_splits.append(new_split)
    else:
        new_splits = []
        for split in splits:
            new_split = {
                key: [matched_samples[k] if k in subject_ids else all_samples[k] for (k, _) in subjects]
                for key, subjects in split.items()
            }
            new_splits.append(new_split)
    
    mkdir(save_model_dir)
    split_path = f"{save_model_dir}/survival_{radiomics_mode}_{pathomics_mode}_splits.dat"
    joblib.dump(new_splits, split_path)
    splits = joblib.load(split_path)
    num_train = len(splits[0]["train"])
    logging.info(f"Number of training samples: {num_train}.")
    num_valid = len(splits[0]["valid"])
    logging.info(f"Number of validating samples: {num_valid}.")
    num_test = len(splits[0]["test"])
    logging.info(f"Number of testing samples: {num_test}.")

    if opt['TASKS']['PREDICTION']:
        # survival analysis from the splits
        if opt['RADIOMICS'].get('KEYS', False):
            radiomics_keys = opt['RADIOMICS']['KEYS']
        else:
            radiomics_keys = None
        if opt['PATHOMICS'].get('KEYS', False):
            pathomics_keys = opt['PATHOMICS']['KEYS']
        else:
            pathomics_keys = None

        save_omics_dir = opt['PREDICTION']['OMICS_DIR'] + f"/{radiomics_mode}+{pathomics_mode}"
        if opt['PREDICTION']['USED_OMICS']['VALUE'] == "radiomics":
            save_omics_dir = save_omics_dir + f"/radiomics_GCNConv_" \
                + opt['RADIOMICS']['AGGREGATED_MODE']['VALUE']
        elif opt['PREDICTION']['USED_OMICS']['VALUE'] == "pathomics":
            save_omics_dir = save_omics_dir + f"/pathomics_GCNConv_" \
                + opt['PATHOMICS']['AGGREGATED_MODE']['VALUE']
        elif opt['PREDICTION']['USED_OMICS']['VALUE'] == "radiopathomics":
            save_omics_dir = save_omics_dir + f"/radiomics_pathomics_GCNConv_" \
                + opt['PATHOMICS']['AGGREGATED_MODE']['VALUE']

        survival(
            split_path=split_path,
            omics=opt['PREDICTION']['USED_OMICS']['VALUE'],
            save_omics_dir=save_omics_dir,
            n_jobs=opt['PREDICTION']['N_JOBS'],
            radiomics_aggregation=opt['RADIOMICS']['AGGREGATION'],
            radiomics_aggregated_mode=opt['RADIOMICS']['AGGREGATED_MODE']['VALUE'],
            pathomics_aggregation=opt['PATHOMICS']['AGGREGATION'],
            pathomics_aggregated_mode=opt['PATHOMICS']['AGGREGATED_MODE']['VALUE'],
            radiomics_keys=radiomics_keys,
            pathomics_keys=pathomics_keys,
            model=opt['PREDICTION']['MODEL']['VALUE'],
            scorer=opt['PREDICTION']['SCORER']['VALUE'],
            feature_selection=opt['PREDICTION']['FEATURE_SELECTION'],
            n_bootstraps=opt['PREDICTION']['N_BOOTSTRAPS'],
            use_graph_properties=opt['PREDICTION']['USE_GRAPH_PROPERTIES'],
            save_results_dir=save_model_dir
        )

    # visualize radiomics
    # from analysis.a04_feature_aggregation.m_graph_construction import visualize_radiomic_graph, visualize_pathomic_graph
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
