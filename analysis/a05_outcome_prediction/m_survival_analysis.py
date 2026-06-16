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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin

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
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.decomposition import PCA

from utilities.m_utils import mkdir, load_json, create_pbar, rm_n_mkdir, reset_logging

from collections import Counter

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


def get_voted_embedding(data, method="average", return_all=False):
    """
    Select the final embedding based on majority voting of structured_reports.raw_text.

    Parameters
    ----------
    data : dict
        Input dictionary with keys:
        - "structured_reports": list of dicts containing "raw_text"
        - "embeddings": list of embedding vectors
    method : str, optional
        How to combine embeddings for the winning text:
        - "average": average all embeddings corresponding to the voted text
        - "first": use the first matching embedding
    return_all : bool, optional
        If True, return additional information.

    Returns
    -------
    list or dict
        If return_all=False:
            Final embedding as a list.
        If return_all=True:
            Dictionary containing:
            - "voted_text"
            - "matched_indices"
            - "matched_embeddings"
            - "final_embedding"
            - "vote_count"
    """
    # Extract texts
    texts = [report["raw_text"] for report in data["structured_reports"]]

    # Basic validation
    if len(texts) != len(data["embeddings"]):
        raise ValueError(
            "Length of structured_reports and embeddings must be the same."
        )

    if len(texts) == 0:
        raise ValueError("No structured reports found.")

    # Majority vote
    text_counts = Counter(texts)
    voted_text, vote_count = text_counts.most_common(1)[0]

    # Find all matching indices
    matched_indices = [i for i, text in enumerate(texts) if text == voted_text]

    # Get corresponding embeddings
    matched_embeddings = [data["embeddings"][i] for i in matched_indices]

    # Select final embedding
    if method == "first":
        final_embedding = matched_embeddings[0]
    elif method == "average":
        final_embedding = np.mean(matched_embeddings, axis=0).tolist()
    else:
        raise ValueError("method must be either 'average' or 'first'")

    if return_all:
        return {
            "voted_text": voted_text,
            "matched_indices": matched_indices,
            "matched_embeddings": matched_embeddings,
            "final_embedding": final_embedding,
            "vote_count": vote_count,
        }

    return final_embedding

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
                elif "/LLaVA-Med/" in radiomic_path:
                    data_dict = load_json(radiomic_path)
                    feat_list = get_voted_embedding(data_dict)
                    properties = {}
                    for i, feat in enumerate(feat_list):
                        k = f"radiomics.{radiomic_key}.feature{i}"
                        properties[k] = feat
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

def load_subject_level_features(idx, subject_feature_path, omics='', outcome=None):
    _, ext = os.path.splitext(subject_feature_path)

    feat_dict = {}

    if ext == ".npy":
        feat_list = np.load(subject_feature_path, allow_pickle=True).squeeze()

        # Ensure iterable
        if np.isscalar(feat_list):
            feat_list = [feat_list]

        for i, feat in enumerate(feat_list):
            k = f"subject.{omics}.feature{i}"
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
    fig, ax = plt.subplots(figsize=(9, 6))

    alphas = coefs.columns.to_numpy()

    # plot coefficient paths
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", linewidth=1)

    # use smallest alpha (least regularized model)
    alpha_min = alphas.min()

    # select column safely
    coef_column = coefs.loc[:, alpha_min]

    top_coefs = coef_column.abs().sort_values().tail(n_highlight)

    for name in top_coefs.index:
        coef_val = float(coef_column.loc[name])
        ax.text(alpha_min, coef_val, name,
                ha="right", va="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")

    plt.subplots_adjust(left=0.2)
    plt.savefig(f"{relative_path}/figures/plots/coefficients.jpg", dpi=150)
    plt.close(fig)

def coxnet(split_idx, tr_X, tr_y, scorer, n_jobs, l1_ratio=0.9, min_ratio=0.1):
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    # COX regreession
    print("Selecting the best regularization parameter...")
    cox_elastic_net = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=min_ratio)
    cox_elastic_net.fit(tr_X, tr_y)
    coefficients = pd.DataFrame(
        cox_elastic_net.coef_,
        index=tr_X.columns,
        columns=cox_elastic_net.alphas_,
    )

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
            joblib.delayed(load_subject_level_features)(idx, graph_path, 'radiomics', outcome)
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
            joblib.delayed(load_subject_level_features)(idx, graph_path, 'pathomics', outcome)
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
    if isinstance(save_radiopathomics_dir, str):
        assert radiomics_aggregated_mode == pathomics_aggregated_mode
        print(f"loading radiopathomics from {save_radiopathomics_dir}...")
        radiopathomics_paths = []
        for p in data:
            subject_id = p[0][0]
            path = pathlib.Path(save_radiopathomics_dir) / radiomics_aggregated_mode / f"{subject_id}.npy"
            radiopathomics_paths.append(path)
        dict_list = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(load_subject_level_features)(idx, graph_path, 'radiopathomics', outcome)
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
            # radiopathomics_X = [radiopathomics_X, prop_X]
            radiopathomics_X = pd.concat([radiopathomics_X, prop_X], axis=1)
    elif isinstance(save_radiopathomics_dir, dict):
        radiomics_X = load_radiomics(
            data=data,
            radiomics_aggregation=radiomics_aggregation,
            radiomics_aggregated_mode=radiomics_aggregated_mode,
            radiomics_keys=radiomics_keys,
            use_graph_properties=use_graph_properties,
            n_jobs=n_jobs,
            save_radiomics_dir=save_radiopathomics_dir['radiomics']
        )
        
        pathomics_X = load_pathomics(
            data=data,
            pathomics_aggregation=pathomics_aggregation,
            pathomics_aggregated_mode=pathomics_aggregated_mode,
            pathomics_keys=pathomics_keys,
            use_graph_properties=use_graph_properties,
            n_jobs=n_jobs,
            save_pathomics_dir=save_radiopathomics_dir['pathomics']
        ) 

        radiopathomics_X = pd.concat([radiomics_X, pathomics_X], axis=1)
        # radiopathomics_X = [radiomics_X, pathomics_X]

    return radiopathomics_X

def select_multivariate_cox_features(
    tr_X,
    tr_y,
    variance_threshold=1e-4,
    alpha=0.01,          # regularization strength (λ)
    l1_ratio=1.0,        # 1.0 = LASSO, <1 = elastic net
    max_features=30,
    coef_threshold=1e-6,
    verbose=True,
):
    """
    Fast multivariate feature selection using Coxnet (glmnet-style).

    Parameters
    ----------
    tr_X : pd.DataFrame
    tr_y : pd.DataFrame with columns ['duration','event']
    variance_threshold : float
    alpha : float
        Regularization strength (larger -> more sparsity)
    l1_ratio : float
        1.0 = LASSO, 0.5 = elastic net
    max_features : int
    coef_threshold : float
    verbose : bool

    Returns
    -------
    selected_names : list
    """

    from sksurv.linear_model import CoxnetSurvivalAnalysis

    X = tr_X.copy()

    if verbose:
        print("Starting feature selection...")
        print(f"Initial feature count: {X.shape[1]}")

    # --------------------------------------------------
    # 1️⃣ Remove low-variance features
    # --------------------------------------------------
    var_selector = VarianceThreshold(threshold=variance_threshold)
    X = X.loc[:, var_selector.fit(X).get_support()]

    if verbose:
        print(f"Remaining after variance filter: {X.shape[1]}")

    # --------------------------------------------------
    # 2️⃣ Standardize features
    # --------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------
    # 3️⃣ Prepare survival target
    # --------------------------------------------------

    if verbose:
        print("Fitting Coxnet model...")

    # --------------------------------------------------
    # 4️⃣ Fit Coxnet (fast)
    # --------------------------------------------------
    model = CoxnetSurvivalAnalysis(
        alphas=[alpha],   # single penalty strength
        l1_ratio=l1_ratio
    )
    model.fit(X_scaled, tr_y)

    # coefficients shape: (n_alphas, n_features)
    coefs = np.abs(model.coef_).ravel()

    coef_series = pd.Series(coefs, index=X.columns)

    # threshold filtering
    selected = coef_series[coef_series > coef_threshold]

    # limit number of features
    if max_features is not None:
        selected = selected.sort_values(ascending=False).head(max_features)

    selected_names = selected.index.tolist()

    if verbose:
        print(f"Selected features: {len(selected_names)}")

    return selected_names

class DomainAdaptationVAE(nn.Module):
    """VAE for domain adaptation with dual encoders and separate decoders"""
    
    def __init__(self, radiomics_dim, pathomics_dim, hidden_dim=128, latent_dim=32):
        super(DomainAdaptationVAE, self).__init__()
        
        # Encoders for each domain
        self.radiomics_encoder = nn.Sequential(
            nn.Linear(radiomics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.pathomics_encoder = nn.Sequential(
            nn.Linear(pathomics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Mean and variance layers for latent space
        self.radiomics_mu = nn.Linear(hidden_dim, latent_dim)
        self.radiomics_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.pathomics_mu = nn.Linear(hidden_dim, latent_dim)
        self.pathomics_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Separate decoders for each modality
        self.radiomics_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, radiomics_dim)
        )
        
        self.pathomics_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pathomics_dim)
        )
        
    def encode_radiomics(self, x):
        h = self.radiomics_encoder(x)
        return self.radiomics_mu(h), self.radiomics_logvar(h)
    
    def encode_pathomics(self, x):
        h = self.pathomics_encoder(x)
        return self.pathomics_mu(h), self.pathomics_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode_radiomics(self, z):
        return self.radiomics_decoder(z)
    
    def decode_pathomics(self, z):
        return self.pathomics_decoder(z)
    
    def forward(self, radiomics, pathomics):
        # Encode radiomics
        radio_mu, radio_logvar = self.encode_radiomics(radiomics)
        radio_z = self.reparameterize(radio_mu, radio_logvar)
        
        # Encode pathomics
        patho_mu, patho_logvar = self.encode_pathomics(pathomics)
        patho_z = self.reparameterize(patho_mu, patho_logvar)
        
        # Decode from radiomics latent space
        radio_recon_from_radio = self.decode_radiomics(radio_z)
        patho_recon_from_radio = self.decode_pathomics(radio_z)
        
        # Decode from pathomics latent space
        radio_recon_from_patho = self.decode_radiomics(patho_z)
        patho_recon_from_patho = self.decode_pathomics(patho_z)
        
        return {
            'radio_mu': radio_mu, 'radio_logvar': radio_logvar,
            'patho_mu': patho_mu, 'patho_logvar': patho_logvar,
            'radio_z': radio_z, 'patho_z': patho_z,
            'radio_recon_from_radio': radio_recon_from_radio,
            'patho_recon_from_patho': patho_recon_from_patho,
            'radio_recon_from_patho': radio_recon_from_patho,
            'patho_recon_from_radio': patho_recon_from_radio
        }

class DomainAdaptationTransformer(BaseEstimator, TransformerMixin):
    """Domain adaptation using dual VAE encoders with separate decoders"""
    
    def __init__(self, radiomics_dim, pathomics_dim, hidden_dim=128, latent_dim=32,
                 epochs=100, batch_size=64, learning_rate=1e-3, beta_kl=1.0,
                 beta_domain=1.0, beta_recon=1.0, beta_cross=0.5, device='cuda'):
        self.radiomics_dim = radiomics_dim
        self.pathomics_dim = pathomics_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_kl = beta_kl
        self.beta_domain = beta_domain
        self.beta_recon = beta_recon
        self.beta_cross = beta_cross
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.model = None
        self.radiomics_scaler = StandardScaler()
        self.pathomics_scaler = StandardScaler()
        
    def kl_divergence(self, mu, logvar):
        """KL divergence for a single Gaussian distribution (VAE regularization)"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    def kl_divergence_between_distributions(self, mu1, logvar1, mu2, logvar2):
        """KL divergence between two Gaussian distributions (domain alignment)"""
        var1 = logvar1.exp()
        var2 = logvar2.exp()
        
        kl = 0.5 * torch.sum(
            logvar2 - logvar1 + 
            (var1 + (mu1 - mu2).pow(2)) / var2 - 1,
            dim=1
        ).mean()
        return kl
    
    def reconstruction_loss(self, recon_x, x):
        """MSE reconstruction loss"""
        return nn.MSELoss()(recon_x, x)
    
    def fit(self, radiomics_data, pathomics_data, verbose=True):
        """Train the domain adaptation model"""
        
        # Normalize data
        radiomics_norm = self.radiomics_scaler.fit_transform(radiomics_data)
        pathomics_norm = self.pathomics_scaler.fit_transform(pathomics_data)
        
        # Convert to tensors
        radiomics_tensor = torch.FloatTensor(radiomics_norm)
        pathomics_tensor = torch.FloatTensor(pathomics_norm)
        
        dataset = TensorDataset(radiomics_tensor, pathomics_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = DomainAdaptationVAE(
            self.radiomics_dim, self.pathomics_dim,
            self.hidden_dim, self.latent_dim
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            total_recon_loss = 0
            total_cross_recon_loss = 0
            total_kl_loss = 0
            total_domain_kl_loss = 0
            
            for batch_radio, batch_patho in dataloader:
                batch_radio = batch_radio.to(self.device)
                batch_patho = batch_patho.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_radio, batch_patho)
                
                # Self-reconstruction losses (reconstruct from own latent space)
                recon_loss_radio_self = self.reconstruction_loss(
                    outputs['radio_recon_from_radio'], batch_radio
                )
                recon_loss_patho_self = self.reconstruction_loss(
                    outputs['patho_recon_from_patho'], batch_patho
                )
                
                # Cross-reconstruction losses (reconstruct from other domain's latent space)
                recon_loss_radio_cross = self.reconstruction_loss(
                    outputs['radio_recon_from_patho'], batch_radio
                )
                recon_loss_patho_cross = self.reconstruction_loss(
                    outputs['patho_recon_from_radio'], batch_patho
                )
                
                # Total reconstruction losses
                recon_loss = (recon_loss_radio_self + recon_loss_patho_self) / 2
                cross_recon_loss = (recon_loss_radio_cross + recon_loss_patho_cross) / 2
                
                # KL divergence for each distribution (VAE regularization)
                kl_radio = self.kl_divergence(outputs['radio_mu'], outputs['radio_logvar'])
                kl_patho = self.kl_divergence(outputs['patho_mu'], outputs['patho_logvar'])
                kl_loss = (kl_radio + kl_patho) / 2
                
                # KL divergence between the two distributions (domain adaptation)
                domain_kl_loss = self.kl_divergence_between_distributions(
                    outputs['radio_mu'], outputs['radio_logvar'],
                    outputs['patho_mu'], outputs['patho_logvar']
                )
                
                # Total loss with separate weights for self and cross reconstruction
                loss = (self.beta_recon * recon_loss + 
                       self.beta_cross * cross_recon_loss +
                       self.beta_kl * kl_loss + 
                       self.beta_domain * domain_kl_loss)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_cross_recon_loss += cross_recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_domain_kl_loss += domain_kl_loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Loss: {total_loss/len(dataloader):.4f}, "
                      f"Recon: {total_recon_loss/len(dataloader):.4f}, "
                      f"CrossRecon: {total_cross_recon_loss/len(dataloader):.4f}, "
                      f"KL: {total_kl_loss/len(dataloader):.4f}, "
                      f"DomainKL: {total_domain_kl_loss/len(dataloader):.4f}")
        
        return self
    
    def transform(self, radiomics_data, pathomics_data, return_embeddings='concatenated',
                  return_dataframe=True, index=None):
        """Transform data to aligned latent representations"""
        
        # Normalize
        radiomics_norm = self.radiomics_scaler.transform(radiomics_data)
        pathomics_norm = self.pathomics_scaler.transform(pathomics_data)
        
        # Convert to tensors
        radiomics_tensor = torch.FloatTensor(radiomics_norm).to(self.device)
        pathomics_tensor = torch.FloatTensor(pathomics_norm).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # Get latent representations
            radio_mu, radio_logvar = self.model.encode_radiomics(radiomics_tensor)
            patho_mu, patho_logvar = self.model.encode_pathomics(pathomics_tensor)
            
            # Use means as embeddings (deterministic)
            radio_z = radio_mu
            patho_z = patho_mu
            
            if return_embeddings == 'concatenated':
                # Concatenate latent representations
                embeddings = torch.cat([radio_z, patho_z], dim=1)
                
            elif return_embeddings == 'mean':
                # Simple mean of the two latent representations
                embeddings = (radio_z + patho_z) / 2
                
            elif return_embeddings == 'radiomics':
                # Use only radiomics latent space
                embeddings = radio_z
                
            elif return_embeddings == 'pathomics':
                # Use only pathomics latent space
                embeddings = patho_z
                
            elif return_embeddings == 'bayesian':
                # Bayesian fusion using product of experts
                # Convert log variance to precision (inverse variance)
                radio_precision = torch.exp(-radio_logvar)
                patho_precision = torch.exp(-patho_logvar)
                
                # Product of experts: precision = sum of precisions, mean = weighted sum by precision
                fused_precision = radio_precision + patho_precision
                fused_mean = (radio_precision * radio_z + patho_precision * patho_z) / (fused_precision + 1e-8)
                
                embeddings = fused_mean
                
            else:
                raise ValueError(f"Unknown return_embeddings: {return_embeddings}")
            
        embeddings_np = embeddings.cpu().numpy()
        
        # Convert to DataFrame if requested
        if return_dataframe:
            if return_embeddings == 'concatenated':
                columns = [f'latent_radio_{i}' for i in range(self.latent_dim)] + \
                         [f'latent_patho_{i}' for i in range(self.latent_dim)]
            elif return_embeddings in ['mean', 'bayesian']:
                columns = [f'fused_latent_{i}' for i in range(self.latent_dim)]
            elif return_embeddings == 'radiomics':
                columns = [f'radio_latent_{i}' for i in range(self.latent_dim)]
            elif return_embeddings == 'pathomics':
                columns = [f'patho_latent_{i}' for i in range(self.latent_dim)]
            else:
                columns = [f'embedding_{i}' for i in range(embeddings_np.shape[1])]
            
            embeddings_np = pd.DataFrame(embeddings_np, columns=columns)
            if index is not None:
                embeddings_np.index = index
        
        return embeddings_np
    
    def fit_transform(self, radiomics_data, pathomics_data, return_embeddings='concatenated'):
        """Fit and transform in one step"""
        self.fit(radiomics_data, pathomics_data)
        return self.transform(radiomics_data, pathomics_data, return_embeddings)
    
    def save(self, filepath):
        """Save the trained transformer"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'radiomics_scaler': self.radiomics_scaler,
            'pathomics_scaler': self.pathomics_scaler,
            'radiomics_dim': self.radiomics_dim,
            'pathomics_dim': self.pathomics_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim
        }
        torch.save(save_dict, filepath)
    
    def load(self, filepath):
        """Load a trained transformer"""
        save_dict = torch.load(filepath, map_location=self.device)
        self.radiomics_dim = save_dict['radiomics_dim']
        self.pathomics_dim = save_dict['pathomics_dim']
        self.hidden_dim = save_dict['hidden_dim']
        self.latent_dim = save_dict['latent_dim']
        
        self.model = DomainAdaptationVAE(
            self.radiomics_dim, self.pathomics_dim,
            self.hidden_dim, self.latent_dim
        ).to(self.device)
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.radiomics_scaler = save_dict['radiomics_scaler']
        self.pathomics_scaler = save_dict['pathomics_scaler']


class DomainAdaptedSurvivalPipeline:
    """Pipeline that combines domain adaptation with survival prediction"""
    
    def __init__(self, domain_adaptor, survival_model, embedding_type='concatenated'):
        self.domain_adaptor = domain_adaptor
        self.survival_model = survival_model
        self.embedding_type = embedding_type
        self.is_fitted = False
    
    def fit(self, radiomics_train, pathomics_train, y_train):
        """Fit domain adaptation and survival model"""
        
        # Fit domain adaptation and get embeddings
        train_embeddings = self.domain_adaptor.fit_transform(
            radiomics_train, pathomics_train, 
            return_embeddings=self.embedding_type
        )
        
        # Train survival model on embeddings
        self.survival_model.fit(train_embeddings, y_train)
        self.is_fitted = True
        
        return self
    
    def predict(self, radiomics_test, pathomics_test):
        """Predict risk scores"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Transform test data
        test_embeddings = self.domain_adaptor.transform(
            radiomics_test, pathomics_test, 
            return_embeddings=self.embedding_type
        )
        
        # Predict with survival model
        return self.survival_model.predict(test_embeddings)
    
    def predict_proba(self, radiomics_test, pathomics_test):
        """Predict survival probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        test_embeddings = self.domain_adaptor.transform(
            radiomics_test, pathomics_test, 
            return_embeddings=self.embedding_type
        )
        
        if hasattr(self.survival_model, 'predict_proba'):
            return self.survival_model.predict_proba(test_embeddings)
        else:
            raise AttributeError("Survival model does not have predict_proba method")
        
    def predict_with_uncertainty(self, radiomics_test, pathomics_test, n_samples=100):
        """Predict with uncertainty estimation using Monte Carlo sampling"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Normalize
        radiomics_norm = self.domain_adaptor.radiomics_scaler.transform(radiomics_test)
        pathomics_norm = self.domain_adaptor.pathomics_scaler.transform(pathomics_test)
        
        # Convert to tensors
        radiomics_tensor = torch.FloatTensor(radiomics_norm).to(self.domain_adaptor.device)
        pathomics_tensor = torch.FloatTensor(pathomics_norm).to(self.domain_adaptor.device)
        
        self.domain_adaptor.model.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Sample from latent distributions
                radio_mu, radio_logvar = self.domain_adaptor.model.encode_radiomics(radiomics_tensor)
                patho_mu, patho_logvar = self.domain_adaptor.model.encode_pathomics(pathomics_tensor)
                
                # Reparameterize to get samples
                radio_z = self.domain_adaptor.model.reparameterize(radio_mu, radio_logvar)
                patho_z = self.domain_adaptor.model.reparameterize(patho_mu, patho_logvar)
                
                # Fuse based on embedding type
                if self.embedding_type == 'concatenated':
                    embeddings = torch.cat([radio_z, patho_z], dim=1)
                elif self.embedding_type == 'mean':
                    embeddings = (radio_z + patho_z) / 2
                elif self.embedding_type == 'bayesian':
                    radio_precision = torch.exp(-radio_logvar)
                    patho_precision = torch.exp(-patho_logvar)
                    
                    # Product of experts: precision = sum of precisions, mean = weighted sum by precision
                    fused_precision = radio_precision + patho_precision
                    embeddings = (radio_precision * radio_z + patho_precision * patho_z) / (fused_precision + 1e-8)
                else:
                    embeddings = radio_z if self.embedding_type == 'radiomics' else patho_z
                
                # Predict
                pred = self.survival_model.predict(embeddings.cpu().numpy())
                predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred
    
    def get_latent_representations(self, radiomics_data, pathomics_data):
        """Get the aligned latent representations without survival prediction"""
        return self.domain_adaptor.transform(
            radiomics_data, pathomics_data, 
            return_embeddings=self.embedding_type
        )

class SurvivalPipeline:
    """Unified pipeline that combines preprocessing, PCA, and survival model"""
    
    def __init__(self, scaler=None, pca=None, predictor=None, feature_names=None, 
                 fusion_weights=None, fusion_method='average'):
        self.scaler = scaler
        self.pca = pca
        self.predictor = predictor
        self.feature_names = feature_names
        self.fusion_weights = fusion_weights
        self.fusion_method = fusion_method
        
    def transform(self, X):
        """Apply preprocessing and PCA"""
        # Scale
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            
        # Apply PCA if exists
        if self.pca is not None:
            X_transformed = self.pca.transform(X_scaled)
        else:
            X_transformed = X_scaled
            
        return X_transformed
    
    def predict(self, X):
        """Predict risk scores"""
        X_transformed = self.transform(X)
        return self.predictor.predict(X_transformed)
    
    def predict_single(self, X):
        """Predict for single patient"""
        return self.predict(X)[0] if hasattr(self.predict(X), '__len__') else self.predict(X)

class MultiModalPipeline:
    """Pipeline for multi-modal fusion (Strategy 3 & 4)"""
    
    def __init__(self, radio_pipeline=None, patho_pipeline=None, 
                 fusion_method='average', fusion_weights=None):
        self.radio_pipeline = radio_pipeline
        self.patho_pipeline = patho_pipeline
        self.fusion_method = fusion_method
        self.fusion_weights = fusion_weights
        
    def predict(self, radio_X, patho_X):
        """Predict using both modalities"""
        # Get individual predictions
        radio_risk = self.radio_pipeline.predict(radio_X)
        patho_risk = self.patho_pipeline.predict(patho_X)
        
        # Fuse
        return (radio_risk * self.fusion_weights['radiomics'] + 
                patho_risk * self.fusion_weights['pathomics'])
    
    def predict_single(self, radio_X, patho_X):
        """Predict for single patient"""
        result = self.predict(radio_X, patho_X)
        return result[0] if hasattr(result, '__len__') else result

class SurvivalAnalyzer:
    def __init__(self, save_results_dir, relative_path="."):
        self.save_results_dir = save_results_dir
        self.relative_path = relative_path
        
    def load_data_for_fold(self, split, omics, radiomics_aggregation, radiomics_aggregated_mode,
                           radiomics_keys, pathomics_aggregation, pathomics_aggregated_mode,
                           pathomics_keys, use_graph_properties, n_jobs, save_omics_dir, outcome):
        """Load data for a specific fold based on omics type"""
        split_idx, split_data = split
        raw_data_tr, raw_data_va, raw_data_te = split_data["train"], split_data["valid"], split_data["test"]
        raw_data_tr = raw_data_tr + raw_data_va
        
        data_tr = [p for p in raw_data_tr if p[1] is not None]
        data_te = [p for p in raw_data_te if p[1] is not None]
        
        tr_y = np.array([p[1] for p in data_tr])
        tr_y = pd.DataFrame({'event': tr_y[:, 1].astype(bool), 'duration': np.maximum(tr_y[:, 0], 1e-6)})
        tr_y = tr_y.to_records(index=False)
        
        te_y = np.array([p[1] for p in data_te])
        te_y = pd.DataFrame({'event': te_y[:, 1].astype(bool), 'duration': np.maximum(te_y[:, 0], 1e-6)})
        te_y = te_y.to_records(index=False)
        
        if omics == "radiopathomics":
            tr_X = load_radiopathomics(
                data=data_tr, radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode, radiomics_keys=radiomics_keys,
                pathomics_aggregation=pathomics_aggregation, pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys, use_graph_properties=use_graph_properties,
                n_jobs=n_jobs, save_radiopathomics_dir=save_omics_dir, outcome=outcome
            )
            te_X = load_radiopathomics(
                data=data_te, radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode, radiomics_keys=radiomics_keys,
                pathomics_aggregation=pathomics_aggregation, pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys, use_graph_properties=use_graph_properties,
                n_jobs=n_jobs, save_radiopathomics_dir=save_omics_dir, outcome=outcome
            )
            raw_te_X = load_radiopathomics(
                data=raw_data_te, radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode, radiomics_keys=radiomics_keys,
                pathomics_aggregation=pathomics_aggregation, pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys, use_graph_properties=use_graph_properties,
                n_jobs=n_jobs, save_radiopathomics_dir=save_omics_dir, outcome=outcome
            )
        elif omics == "pathomics":
            tr_X = load_pathomics(
                data=data_tr, pathomics_aggregation=pathomics_aggregation,
                pathomics_aggregated_mode=pathomics_aggregated_mode, pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties, n_jobs=n_jobs,
                save_pathomics_dir=save_omics_dir, outcome=outcome
            )
            te_X = load_pathomics(
                data=data_te, pathomics_aggregation=pathomics_aggregation,
                pathomics_aggregated_mode=pathomics_aggregated_mode, pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties, n_jobs=n_jobs,
                save_pathomics_dir=save_omics_dir, outcome=outcome
            )
            raw_te_X = load_pathomics(
                data=raw_data_te, pathomics_aggregation=pathomics_aggregation,
                pathomics_aggregated_mode=pathomics_aggregated_mode, pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties, n_jobs=n_jobs,
                save_pathomics_dir=save_omics_dir, outcome=outcome
            )
        elif omics == "radiomics":
            tr_X = load_radiomics(
                data=data_tr, radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode, radiomics_keys=radiomics_keys,
                use_graph_properties=use_graph_properties, n_jobs=n_jobs,
                save_radiomics_dir=save_omics_dir, outcome=outcome
            )
            te_X = load_radiomics(
                data=data_te, radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode, radiomics_keys=radiomics_keys,
                use_graph_properties=use_graph_properties, n_jobs=n_jobs,
                save_radiomics_dir=save_omics_dir, outcome=outcome
            )
            raw_te_X = load_radiomics(
                data=raw_data_te, radiomics_aggregation=radiomics_aggregation,
                radiomics_aggregated_mode=radiomics_aggregated_mode, radiomics_keys=radiomics_keys,
                use_graph_properties=use_graph_properties, n_jobs=n_jobs,
                save_radiomics_dir=save_omics_dir, outcome=outcome
            )
        else:
            raise NotImplementedError
            
        return data_tr, data_te, raw_data_te, tr_X, te_X, raw_te_X, tr_y, te_y
    
    def apply_pca(self, X_train, X_test, X_raw, n_components=None, variance_ratio=0.95):
        """Apply PCA for dimensionality reduction"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_raw_scaled = scaler.transform(X_raw)
        
        if n_components is None:
            pca = PCA(n_components=variance_ratio)
        else:
            pca = PCA(n_components=n_components)
            
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        X_raw_pca = pca.transform(X_raw_scaled)
        
        # Convert to DataFrame with column names
        X_train_pca = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
        X_test_pca = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])
        X_raw_pca = pd.DataFrame(X_raw_pca, columns=[f'PC{i+1}' for i in range(X_raw_pca.shape[1])])
        
        return X_train_pca, X_test_pca, X_raw_pca, pca, scaler
    
    def train_model(self, split_idx, tr_X, tr_y, scorer, n_jobs, model_name="CoxPH"):
        """Train a survival model"""
        if model_name == "Coxnet":
            predictor = coxnet(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model_name == "RSF":
            predictor = rsf(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model_name == "CoxPH":
            predictor = coxph(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model_name == "GradientBoost":
            predictor = gradientboosting(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model_name == "IPCRidge":
            predictor = ipcridge(split_idx, tr_X, tr_y, scorer, n_jobs)
        elif model_name == "FastSVM":
            predictor = fastsvm(split_idx, tr_X, tr_y, scorer, n_jobs, rank_ratio=1)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        return predictor
    
    def feature_selection(self, tr_X, te_X, raw_te_X, tr_y, feature_var_threshold=1e-4, p_threshold=0.2):
        """Perform feature selection"""
        print("Selecting features...")
        selector = VarianceThreshold(threshold=feature_var_threshold)
        selector.fit(tr_X)
        selected_names = selector.get_feature_names_out().tolist()
        num_removed = len(tr_X.columns) - len(selected_names)
        print(f"Removing {num_removed} low-variance features...")
        tr_X = tr_X[selected_names]
        te_X = te_X[selected_names]
        raw_te_X = raw_te_X[selected_names]
        
        print("Selecting univariate features...")
        univariate_results = []
        for name in list(tr_X.columns):
            cph = CoxPHFitter()
            df = pd.DataFrame({
                "duration": tr_y["duration"], 
                "event": tr_y["event"],
                name: tr_X[name] 
            })
            try:
                cph.fit(df, "duration", "event")
                summary = cph.summary
                univariate_results.append({
                    'name': name,
                    'coef': summary['coef'].values[0],
                    'p_value': summary['p'].values[0],
                })
            except Exception as e:
                print(f"Skipping {name} | Error: {e}")
                continue
                
        results_df = pd.DataFrame(univariate_results)
        selected_names = results_df[results_df['p_value'] < p_threshold]['name'].tolist()
        print(f"Selected features: {len(selected_names)}")
        tr_X = tr_X[selected_names]
        te_X = te_X[selected_names]
        raw_te_X = raw_te_X[selected_names]
        
        return tr_X, te_X, raw_te_X
    
    def bootstrap_stabilization(self, tr_X, tr_y, predictor, selected_names, n_bootstraps=100, stability_threshold=0.8):
        """Perform bootstrap feature stabilization"""
        if n_bootstraps <= 0:
            return tr_X, te_X, raw_te_X, predictor
            
        print("Bootstrapping...")
        stable_coefs = np.zeros(len(selected_names))
        for _ in range(n_bootstraps):
            tr_x_s, tr_y_s = resample(tr_X, tr_y)
            temp_predictor = self.train_model(0, tr_x_s, tr_y_s, "cindex", 1, predictor.named_steps["model"].__class__.__name__)
            if hasattr(temp_predictor.named_steps["model"], "coef_"):
                stable_coefs += (temp_predictor.named_steps["model"].coef_ != 0).astype(int)
            else:
                stable_coefs += (temp_predictor.named_steps["model"].estimator_.coef_ != 0).astype(int)
                
        stable_coefs = stable_coefs / n_bootstraps
        final_coefs = np.where(stable_coefs > stability_threshold)[0]
        stable_names = [selected_names[i] for i in final_coefs.tolist()]
        tr_X = tr_X[stable_names]
        te_X = te_X[stable_names]
        raw_te_X = raw_te_X[stable_names]
        predictor.fit(tr_X, tr_y)
        
        return tr_X, te_X, raw_te_X, predictor
    
    def evaluate_predictions(self, tr_y, te_X, te_y, risk_scores, predictor=None, times=None):
        """Evaluate survival predictions"""
        C_index = concordance_index_censored(te_y["event"], te_y["duration"], risk_scores)[0]
        C_index_ipcw = concordance_index_ipcw(tr_y, te_y, risk_scores)[0]
        
        if times is None:
            lower, upper = np.percentile(te_y["duration"], [10, 90])
            times = np.arange(lower, upper + 1, 7)
            
        auc, mean_auc = cumulative_dynamic_auc(tr_y, te_y, risk_scores, times)
        
        if predictor is not None and hasattr(predictor, "predict_survival_function"):
            try:
                survs = predictor.predict_survival_function(te_X)
                preds = np.asarray([[fn(t) for t in times] for fn in survs])
                IBS = integrated_brier_score(tr_y, te_y, preds, times)
            except Exception as e:
                print(f"Error computing IBS: {e}")
                IBS = 0
        else:
            IBS = 0
            
        return {
            "C-index": C_index,
            "C-index-IPCW": C_index_ipcw,
            "Mean AUC": mean_auc,
            "IBS": IBS
        }, times
    
    def strategy_1_direct_concat(self, split, split_idx, omics_params, model_params):
        """Strategy 1: Direct concatenation of radiomics and pathomics"""
        print(f"\n=== Strategy 1: Direct Concatenation ===")
        
        # Load data
        data_tr, data_te, raw_data_te, tr_X, te_X, raw_te_X, tr_y, te_y = self.load_data_for_fold(
            split, **omics_params
        )
        
        # Feature selection
        if model_params['feature_selection']:
            tr_X, te_X, raw_te_X = self.feature_selection(
                tr_X, te_X, raw_te_X, tr_y,
                model_params['feature_var_threshold'],
                model_params.get('p_threshold', 0.2)
            )
        
        # Train model
        predictor = self.train_model(
            split_idx, tr_X, tr_y, model_params['scorer'],
            model_params['n_jobs'], model_params['model']
        )

        # Create pipeline
        pipeline = SurvivalPipeline(
            scaler=None,
            pca=None,
            predictor=predictor,
            feature_names=list(tr_X.columns),
            fusion_method='direct'
        )
            
        # Bootstrap stabilization
        if model_params['n_bootstraps'] > 0:
            tr_X, te_X, raw_te_X, predictor = self.bootstrap_stabilization(
                tr_X, te_X, predictor, list(tr_X.columns),
                model_params['n_bootstraps']
            )
        
        # Predict and evaluate
        risk_scores = predictor.predict(te_X)
        raw_risk_scores = predictor.predict(raw_te_X)
        scores_dict, times = self.evaluate_predictions(tr_y, te_X, te_y, risk_scores, predictor)
        
        # Return as dictionary for consistency
        return {
            'pipeline': pipeline,
            'risk_scores': risk_scores,
            'raw_risk_scores': raw_risk_scores,
            'scores': scores_dict,
            'times': times,
            'subject_ids': [p[0][0] for p in data_te],
            'raw_subject_ids': [p[0][0] for p in raw_data_te],
            'event': te_y["event"].astype(int).tolist(),
            'duration': te_y["duration"].tolist()
        }


    def strategy_2_pca_concat(self, split, split_idx, omics_params, model_params, n_pca_components=None):
        """Strategy 2: PCA on concatenated radiomics and pathomics"""
        print(f"\n=== Strategy 2: PCA on Concatenated Features ===")
        
        # Load data
        data_tr, data_te, raw_data_te, tr_X, te_X, raw_te_X, tr_y, te_y = self.load_data_for_fold(
            split, **omics_params
        )
        
        # Apply PCA
        tr_X_pca, te_X_pca, raw_te_X_pca, pca_model, scaler = self.apply_pca(
            tr_X, te_X, raw_te_X, n_components=n_pca_components
        )
        print(f"PCA reduced dimensions from {tr_X.shape[1]} to {tr_X_pca.shape[1]}")
        
        # Train model on PCA features
        predictor = self.train_model(
            split_idx, tr_X_pca, tr_y, model_params['scorer'],
            model_params['n_jobs'], model_params['model']
        )

        # Create pipeline
        pipeline = SurvivalPipeline(
            scaler=scaler,
            pca=pca_model,
            predictor=predictor,
            feature_names=list(tr_X_pca.columns),
            fusion_method='pca_concat'
        )
        
        # Predict and evaluate
        risk_scores = predictor.predict(te_X_pca)
        raw_risk_scores = predictor.predict(raw_te_X_pca)
        scores_dict, times = self.evaluate_predictions(tr_y, te_X_pca, te_y, risk_scores, predictor)
        
        # Return as dictionary for consistency
        return {
            'pipeline': pipeline,
            'risk_scores': risk_scores,
            'raw_risk_scores': raw_risk_scores,
            'scores': scores_dict,
            'times': times,
            'subject_ids': [p[0][0] for p in data_te],
            'raw_subject_ids': [p[0][0] for p in raw_data_te],
            'event': te_y["event"].astype(int).tolist(),
            'duration': te_y["duration"].tolist()
        }


    def strategy_3_separate_fusion(self, split, split_idx, omics_params, model_params, fusion_method='average'):
        """Strategy 3: Separate ML models for radiomics and pathomics with result fusion"""
        print(f"\n=== Strategy 3: Separate Models with Result Fusion ===")
        
        # Load radiomics data
        radiomics_params = omics_params.copy()
        radiomics_params['omics'] = 'radiomics'
        radiomics_params['save_omics_dir'] = omics_params['save_omics_dir']['radiomics']
        data_tr, data_te, raw_data_te, tr_X_radio, te_X_radio, raw_te_X_radio, tr_y, te_y = self.load_data_for_fold(
            split, **radiomics_params
        )
        
        # Load pathomics data
        pathomics_params = omics_params.copy()
        pathomics_params['omics'] = 'pathomics'
        pathomics_params['save_omics_dir'] = omics_params['save_omics_dir']['pathomics']
        _, _, _, tr_X_patho, te_X_patho, raw_te_X_patho, _, _ = self.load_data_for_fold(
            split, **pathomics_params
        )
        
        # Feature selection for each modality
        if model_params['feature_selection']:
            tr_X_radio, te_X_radio, raw_te_X_radio = self.feature_selection(
                tr_X_radio, te_X_radio, raw_te_X_radio, tr_y,
                model_params['feature_var_threshold']
            )
            tr_X_patho, te_X_patho, raw_te_X_patho = self.feature_selection(
                tr_X_patho, te_X_patho, raw_te_X_patho, tr_y,
                model_params['feature_var_threshold']
            )
        
        # Train separate models
        predictor_radio = self.train_model(
            split_idx, tr_X_radio, tr_y, model_params['scorer'],
            model_params['n_jobs'], model_params['model']
        )
        predictor_patho = self.train_model(
            split_idx, tr_X_patho, tr_y, model_params['scorer'],
            model_params['n_jobs'], model_params['model']
        )

        # Create individual pipelines
        radio_pipeline = SurvivalPipeline(
            scaler=None, pca=None, predictor=predictor_radio,
            feature_names=list(tr_X_radio.columns)
        )
        patho_pipeline = SurvivalPipeline(
            scaler=None, pca=None, predictor=predictor_patho,
            feature_names=list(tr_X_patho.columns)
        )
        
        # Get predictions
        risk_scores_radio = predictor_radio.predict(te_X_radio)
        raw_risk_scores_radio = predictor_radio.predict(raw_te_X_radio)
        risk_scores_patho = predictor_patho.predict(te_X_patho)
        raw_risk_scores_patho = predictor_patho.predict(raw_te_X_patho)
        
        # Fuse predictions
        fusion_weights = {}
        if fusion_method == 'average':
            fusion_weights['radiomics'] = 0.5
            fusion_weights['pathomics'] = 0.5
        elif fusion_method == 'weighted':
            # Use C-index as weight
            tr_risk_scores_radio = predictor_radio.predict(tr_X_radio)
            tr_risk_scores_patho = predictor_patho.predict(tr_X_patho)
            c_index_radio = concordance_index_censored(tr_y["event"], tr_y["duration"], tr_risk_scores_radio)[0]
            c_index_patho = concordance_index_censored(tr_y["event"], tr_y["duration"], tr_risk_scores_patho)[0]
            total = c_index_radio + c_index_patho
            if total > 0:
                fusion_weights['radiomics'] = c_index_radio / total
                fusion_weights['pathomics'] = c_index_patho / total
            else:
                fusion_weights['radiomics'] = 0.5
                fusion_weights['pathomics'] = 0.5
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        risk_scores = fusion_weights['radiomics'] * risk_scores_radio + fusion_weights['pathomics'] * risk_scores_patho
        raw_risk_scores = fusion_weights['radiomics'] * raw_risk_scores_radio + fusion_weights['pathomics'] * raw_risk_scores_patho
        
        # Create multi-modal pipeline
        multi_pipeline = MultiModalPipeline(
            radio_pipeline=radio_pipeline,
            patho_pipeline=patho_pipeline,
            fusion_method=fusion_method,
            fusion_weights=fusion_weights
        )
        
        # Evaluate fused predictions
        scores_dict, times = self.evaluate_predictions(tr_y, None, te_y, risk_scores, None)
        
        # Return as dictionary for consistency
        return {
            'pipeline': multi_pipeline,
            'risk_scores': risk_scores,
            'raw_risk_scores': raw_risk_scores,
            'scores': scores_dict,
            'times': times,
            'subject_ids': [p[0][0] for p in data_te],
            'raw_subject_ids': [p[0][0] for p in raw_data_te],
            'event': te_y["event"].astype(int).tolist(),
            'duration': te_y["duration"].tolist()
        }


    def strategy_4_pca_separate_fusion(self, split, split_idx, omics_params, model_params, 
                                        n_pca_components=None, fusion_method='average'):
        """Strategy 4: PCA on each modality separately, then separate models, then fusion"""
        print(f"\n=== Strategy 4: Separate PCA + Separate Models + Fusion ===")
        
        # Load radiomics data
        radiomics_params = omics_params.copy()
        radiomics_params['omics'] = 'radiomics'
        radiomics_params['save_omics_dir'] = omics_params['save_omics_dir']['radiomics']
        data_tr, data_te, raw_data_te, tr_X_radio, te_X_radio, raw_te_X_radio, tr_y, te_y = self.load_data_for_fold(
            split, **radiomics_params
        )
        
        # Load pathomics data
        pathomics_params = omics_params.copy()
        pathomics_params['omics'] = 'pathomics'
        pathomics_params['save_omics_dir'] = omics_params['save_omics_dir']['pathomics']
        _, _, _, tr_X_patho, te_X_patho, raw_te_X_patho, _, _ = self.load_data_for_fold(
            split, **pathomics_params
        )
        
        # Apply PCA separately
        tr_X_radio_pca, te_X_radio_pca, raw_te_X_radio_pca, pca_radio, scaler_radio = self.apply_pca(
            tr_X_radio, te_X_radio, raw_te_X_radio, n_components=n_pca_components
        )
        tr_X_patho_pca, te_X_patho_pca, raw_te_X_patho_pca, pca_patho, scaler_patho = self.apply_pca(
            tr_X_patho, te_X_patho, raw_te_X_patho, n_components=n_pca_components
        )
        
        print(f"Radiomics PCA: {tr_X_radio.shape[1]} -> {tr_X_radio_pca.shape[1]}")
        print(f"Pathomics PCA: {tr_X_patho.shape[1]} -> {tr_X_patho_pca.shape[1]}")
        
        # Train separate models on PCA features
        predictor_radio = self.train_model(
            split_idx, tr_X_radio_pca, tr_y, model_params['scorer'],
            model_params['n_jobs'], model_params['model']
        )
        predictor_patho = self.train_model(
            split_idx, tr_X_patho_pca, tr_y, model_params['scorer'],
            model_params['n_jobs'], model_params['model']
        )

        # Create individual pipelines
        radio_pipeline = SurvivalPipeline(
            scaler=scaler_radio,
            pca=pca_radio,
            predictor=predictor_radio,
            feature_names=list(tr_X_radio_pca.columns)
        )
        patho_pipeline = SurvivalPipeline(
            scaler=scaler_patho,
            pca=pca_patho,
            predictor=predictor_patho,
            feature_names=list(tr_X_patho_pca.columns)
        )
        
        # Get predictions
        risk_scores_radio = predictor_radio.predict(te_X_radio_pca)
        raw_risk_scores_radio = predictor_radio.predict(raw_te_X_radio_pca)
        risk_scores_patho = predictor_patho.predict(te_X_patho_pca)
        raw_risk_scores_patho = predictor_patho.predict(raw_te_X_patho_pca)
        
        # Fuse predictions
        fusion_weights = {}
        if fusion_method == 'average':
            fusion_weights['radiomics'] = 0.5
            fusion_weights['pathomics'] = 0.5
        elif fusion_method == 'weighted':
            # Use C-index as weight
            tr_risk_scores_radio = predictor_radio.predict(tr_X_radio_pca)
            tr_risk_scores_patho = predictor_patho.predict(tr_X_patho_pca)
            c_index_radio = concordance_index_censored(tr_y["event"], tr_y["duration"], tr_risk_scores_radio)[0]
            c_index_patho = concordance_index_censored(tr_y["event"], tr_y["duration"], tr_risk_scores_patho)[0]
            total = c_index_radio + c_index_patho
            if total > 0:
                fusion_weights['radiomics'] = c_index_radio / total
                fusion_weights['pathomics'] = c_index_patho / total
            else:
                fusion_weights['radiomics'] = 0.5
                fusion_weights['pathomics'] = 0.5
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        risk_scores = fusion_weights['radiomics'] * risk_scores_radio + fusion_weights['pathomics'] * risk_scores_patho
        raw_risk_scores = fusion_weights['radiomics'] * raw_risk_scores_radio + fusion_weights['pathomics'] * raw_risk_scores_patho
        
        # Create multi-modal pipeline
        multi_pipeline = MultiModalPipeline(
            radio_pipeline=radio_pipeline,
            patho_pipeline=patho_pipeline,
            fusion_method=fusion_method,
            fusion_weights=fusion_weights
        )
        
        # Evaluate
        scores_dict, times = self.evaluate_predictions(tr_y, None, te_y, risk_scores, None)
        
        # Return as dictionary for consistency
        return {
            'pipeline': multi_pipeline,
            'risk_scores': risk_scores,
            'raw_risk_scores': raw_risk_scores,
            'scores': scores_dict,
            'times': times,
            'subject_ids': [p[0][0] for p in data_te],
            'raw_subject_ids': [p[0][0] for p in raw_data_te],
            'event': te_y["event"].astype(int).tolist(),
            'duration': te_y["duration"].tolist()
        }

    def strategy_5_domain_adaptation_fusion(self, split, split_idx, omics_params, model_params, 
                                            embedding_type='bayesian'):
        """Strategy 5 with domain adaptation using separate decoders"""
        print(f"\n=== Strategy 5: Domain Adaptation with Separate Decoders ===")
        
        # Load radiomics data
        radiomics_params = omics_params.copy()
        radiomics_params['omics'] = 'radiomics'
        radiomics_params['save_omics_dir'] = omics_params['save_omics_dir']['radiomics']
        data_tr, data_te, raw_data_te, tr_X_radio, te_X_radio, raw_te_X_radio, tr_y, te_y = self.load_data_for_fold(
            split, **radiomics_params
        )
        
        # Load pathomics data
        pathomics_params = omics_params.copy()
        pathomics_params['omics'] = 'pathomics'
        pathomics_params['save_omics_dir'] = omics_params['save_omics_dir']['pathomics']
        _, _, _, tr_X_patho, te_X_patho, raw_te_X_patho, _, _ = self.load_data_for_fold(
            split, **pathomics_params
        )
        
        print("Applying domain adaptation with separate decoder VAE...")
        
        # Initialize domain adaptation transformer
        domain_adaptor = DomainAdaptationTransformer(
            radiomics_dim=tr_X_radio.shape[1],
            pathomics_dim=tr_X_patho.shape[1],
            hidden_dim=model_params.get('vae_hidden_dim', 512),
            latent_dim=model_params.get('vae_latent_dim', 128),
            epochs=model_params.get('vae_epochs', 30),
            batch_size=model_params.get('vae_batch_size', 64),
            learning_rate=model_params.get('vae_learning_rate', 1e-3),
            beta_kl=model_params.get('vae_beta_kl', 1e-2),
            beta_domain=model_params.get('vae_beta_domain', 1e-2),
            beta_recon=model_params.get('vae_beta_recon', 1.0),
            beta_cross=model_params.get('vae_beta_cross', 1.0),
            device=model_params.get('device', 'cuda')
        )
        

        # Get aligned embeddings
        train_embeddings = domain_adaptor.fit_transform(
            tr_X_radio, tr_X_patho, 
            return_embeddings=embedding_type
        )
        te_embeddings = domain_adaptor.transform(
            te_X_radio, te_X_patho, 
            return_embeddings=embedding_type
        )
        raw_te_embeddings = domain_adaptor.transform(
            raw_te_X_radio, raw_te_X_patho, 
            return_embeddings=embedding_type
        )
        
        # Train survival model on embeddings
        predictor = self.train_model(
            split_idx, train_embeddings, tr_y, model_params['scorer'],
            model_params['n_jobs'], model_params['model']
        )
        
        # Create pipeline
        pipeline = DomainAdaptedSurvivalPipeline(
            domain_adaptor, predictor, embedding_type=embedding_type
        )
        
        # Get predictions
        risk_scores = predictor.predict(te_embeddings)
        raw_risk_scores = predictor.predict(raw_te_embeddings)
        
        # Also get latent representations for analysis
        train_latent = domain_adaptor.transform(tr_X_radio, tr_X_patho, return_embeddings=embedding_type)
        
        # Evaluate predictions
        scores_dict, times = self.evaluate_predictions(tr_y, None, te_y, risk_scores, None)
        
        # Return results
        return {
            'pipeline': pipeline,
            'risk_scores': risk_scores,
            'raw_risk_scores': raw_risk_scores,
            'scores': scores_dict,
            'times': times,
            'subject_ids': [p[0][0] for p in data_te],
            'raw_subject_ids': [p[0][0] for p in raw_data_te],
            'event': te_y["event"].astype(int).tolist(),
            'duration': te_y["duration"].tolist(),
            'latent_embeddings': train_latent  # For analysis
        }

    def plot_survival_curve(self, survival_results, omics, strategy_name, pvalue):
        """Plot survival curve"""
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
        ax.set_ylabel("Survival Probability")
        ax.set_title(f"{omics} - {strategy_name}\np-value: {pvalue:.4f}")
        
        # Create directory if it doesn't exist
        os.makedirs(f"{self.relative_path}/figures/plots", exist_ok=True)
        plt.savefig(f"{self.relative_path}/figures/plots/{omics}_{strategy_name}_survival_curve.png")
        plt.close()


    def run_all_strategies(self, split_path, omics="radiopathomics", save_omics_dir=None,
                        radiomics_aggregation=False, radiomics_aggregated_mode=None,
                        radiomics_keys=None, pathomics_aggregation=False,
                        pathomics_aggregated_mode=None, pathomics_keys=None,
                        use_graph_properties=False, n_jobs=32, outcome=None,
                        model="CoxPH", scorer="cindex", feature_selection=True,
                        feature_var_threshold=1e-4, n_selected_features=64,
                        n_bootstraps=100, n_pca_components=None, fusion_method='weighted',
                        save_results_dir=None):
        """Run all four strategies and save results"""
        
        splits = joblib.load(split_path)
        
        # Prepare parameters
        omics_params = {
            'omics': omics,
            'radiomics_aggregation': radiomics_aggregation,
            'radiomics_aggregated_mode': radiomics_aggregated_mode,
            'radiomics_keys': radiomics_keys,
            'pathomics_aggregation': pathomics_aggregation,
            'pathomics_aggregated_mode': pathomics_aggregated_mode,
            'pathomics_keys': pathomics_keys,
            'use_graph_properties': use_graph_properties,
            'n_jobs': n_jobs,
            'save_omics_dir': save_omics_dir,
            'outcome': outcome
        }
        
        model_params = {
            'model': model,
            'scorer': scorer,
            'feature_selection': feature_selection,
            'feature_var_threshold': feature_var_threshold,
            'n_selected_features': n_selected_features,
            'n_bootstraps': n_bootstraps,
            'n_jobs': n_jobs,
            'p_threshold': 0.2  # Default p-value threshold for feature selection
        }
        
        if omics in ['radiomics', 'pathomics']:
            strategies = {
                'Strategy1_DirectConcat': self.strategy_1_direct_concat,
                'Strategy2_PCA_Concat': lambda s, idx, op, mp: self.strategy_2_pca_concat(s, idx, op, mp, n_pca_components)
            }
        elif omics == 'radiopathomics':
            strategies = {
                'Strategy1_DirectConcat': self.strategy_1_direct_concat,
                'Strategy2_PCA_Concat': lambda s, idx, op, mp: self.strategy_2_pca_concat(s, idx, op, mp, n_pca_components),
                'Strategy3_Separate_Fusion': lambda s, idx, op, mp: self.strategy_3_separate_fusion(s, idx, op, mp, fusion_method),
                'Strategy4_PCA_Separate_Fusion': lambda s, idx, op, mp: self.strategy_4_pca_separate_fusion(s, idx, op, mp, n_pca_components, fusion_method),
                'Strategy5_Domain_Adaptation_Fusion': lambda s, idx, op, mp: self.strategy_5_domain_adaptation_fusion(s, idx, op, mp),
            }
        else:
            raise ValueError(f"{omics} is not supported yet")
        
        ml_model_name = f"{omics}_" + \
            f"radio+{radiomics_aggregated_mode}_" + \
            f"patho+{pathomics_aggregated_mode}_" + \
            f"model+{model}_scorer+{scorer}"
        
        # Create save directory if it doesn't exist
        os.makedirs(save_results_dir, exist_ok=True)
        
        all_results = {}
        
        for strategy_name, strategy_func in strategies.items():
            print(f"\n{'='*60}")
            print(f"Running {strategy_name}")
            print(f"{'='*60}")
            
            strategy_results = {
                "predict_results": {},
                "survival_results": {"raw_subject": [], "raw_risk": [], "subject": [], "risk": [], "event": [], "duration": []},
                "cv_scores": [],
                "pvalues": []
            }
            
            model_dir = os.path.join(save_results_dir, f"{ml_model_name}_{strategy_name}")
            os.makedirs(model_dir, exist_ok=True)
            
            for split_idx, split in enumerate(splits):
                print(f"\n--- Fold {split_idx} ---")
                
                # Run strategy
                try:
                    results = strategy_func(
                        (split_idx, split), split_idx, omics_params, model_params
                    )
                    
                    # Store results
                    strategy_results["predict_results"][f"Fold {split_idx}"] = results['scores']
                    strategy_results["cv_scores"].append(results['scores'])
                    
                    # Store survival data
                    strategy_results["survival_results"]["raw_subject"] += results['raw_subject_ids']
                    strategy_results["survival_results"]["raw_risk"] += results['raw_risk_scores'].tolist()
                    strategy_results["survival_results"]["subject"] += results['subject_ids']
                    strategy_results["survival_results"]["risk"] += results['risk_scores'].tolist()
                    strategy_results["survival_results"]["event"] += results['event']
                    strategy_results["survival_results"]["duration"] += results['duration']
                    
                    # Save model
                    model_path = os.path.join(model_dir, f"fold{split_idx}_model.joblib")
                    joblib.dump({"pipeline": results['pipeline']}, model_path)
                    
                except Exception as e:
                    print(f"Error in fold {split_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Compute cross-validation statistics
            if strategy_results["cv_scores"]:
                cv_stats = {}
                for metric in strategy_results["cv_scores"][0].keys():
                    values = [score[metric] for score in strategy_results["cv_scores"] if score.get(metric, 0) != 0]
                    if values:
                        cv_stats[metric] = {"mean": np.mean(values), "std": np.std(values)}
                
                # Log-rank test on aggregated data
                pd_risk = pd.DataFrame({
                    k: strategy_results["survival_results"][k] 
                    for k in ["risk", "event", "duration"]
                })
                mean_risk = pd_risk["risk"].mean()
                dem = pd_risk["risk"] > mean_risk
                
                test_results = logrank_test(
                    pd_risk["duration"][dem], pd_risk["duration"][~dem],
                    pd_risk["event"][dem], pd_risk["event"][~dem],
                    alpha=.99
                )
                pvalue = test_results.p_value
                strategy_results["pvalues"].append(pvalue)
                
                # Save plot
                self.plot_survival_curve(strategy_results["survival_results"], omics, strategy_name, pvalue)
                
                #save predicted results
                save_path = f"{save_results_dir}/{ml_model_name}_{strategy_name}_results.json"
                with open(save_path, "w") as f:
                    json.dump(strategy_results["survival_results"], f, indent=4)

                # save metrics
                save_path = f"{save_results_dir}/{ml_model_name}_{strategy_name}_metrics.json"
                with open(save_path, "w") as f:
                    json.dump({
                        "cv_results": strategy_results["predict_results"],
                        "cv_statistics": cv_stats,
                        "logrank_pvalue": pvalue
                    }, f, indent=4)
                
                all_results[strategy_name] = {
                    "cv_statistics": cv_stats,
                    "logrank_pvalue": pvalue
                }
                
                print(f"\n{strategy_name} Results:")
                for metric, stats in cv_stats.items():
                    print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Log-rank p-value: {pvalue:.4f}")
        
        # Save comparison summary
        comparison_path = f"{save_results_dir}/{ml_model_name}_strategies.json"
        with open(comparison_path, "w") as f:
            json.dump(all_results, f, indent=4)
        
        return all_results


# Modified main function for backward compatibility
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
    feature_var_threshold=1e-4,
    n_selected_features=64,
    n_bootstraps=100,
    use_graph_properties=False,
    save_results_dir=None,
    n_pca_components=None,
    fusion_method='weighted'
):
    """
    Enhanced survival analysis with four different strategies:
    1. Direct concatenation of radiomics and pathomics
    2. PCA on concatenated features
    3. Separate models with result fusion
    4. Separate PCA + separate models + result fusion
    
    Parameters:
    -----------
    split_path : str
        Path to the cross-validation splits file
    radiomics_keys : list, optional
        Keys for radiomics features to extract
    pathomics_keys : list, optional
        Keys for pathomics features to extract
    omics : str
        Type of omics data ('radiomics', 'pathomics', or 'radiopathomics')
    save_omics_dir : dict or str
        Directory to save/load omics features
    outcome : str
        Outcome variable name
    n_jobs : int
        Number of parallel jobs
    radiomics_aggregation : bool
        Whether to aggregate radiomics features
    radiomics_aggregated_mode : str
        Aggregation mode for radiomics ('mean', 'median', etc.)
    pathomics_aggregation : bool
        Whether to aggregate pathomics features
    pathomics_aggregated_mode : str
        Aggregation mode for pathomics ('mean', 'median', etc.)
    model : str
        Survival model type ('CoxPH', 'Coxnet', 'RSF', 'GradientBoost', 'IPCRidge', 'FastSVM')
    scorer : str
        Metric for model selection ('cindex', etc.)
    feature_selection : bool
        Whether to perform feature selection
    feature_var_threshold : float
        Variance threshold for feature selection
    n_selected_features : int
        Maximum number of features to select
    n_bootstraps : int
        Number of bootstrap samples for feature stabilization
    use_graph_properties : bool
        Whether to use graph properties for pathomics
    save_results_dir : str
        Directory to save results
    n_pca_components : int or None
        Number of PCA components (None for automatic based on variance)
    fusion_method : str
        Fusion method for multi-modal strategies ('average', 'weighted')
    
    Returns:
    --------
    dict
        Results for all strategies
    """
    
    analyzer = SurvivalAnalyzer(save_results_dir)
    
    results = analyzer.run_all_strategies(
        split_path=split_path,
        omics=omics,
        save_omics_dir=save_omics_dir,
        radiomics_aggregation=radiomics_aggregation,
        radiomics_aggregated_mode=radiomics_aggregated_mode,
        radiomics_keys=radiomics_keys,
        pathomics_aggregation=pathomics_aggregation,
        pathomics_aggregated_mode=pathomics_aggregated_mode,
        pathomics_keys=pathomics_keys,
        use_graph_properties=use_graph_properties,
        n_jobs=n_jobs,
        outcome=outcome,
        model=model,
        scorer=scorer,
        feature_selection=feature_selection,
        feature_var_threshold=feature_var_threshold,
        n_selected_features=n_selected_features,
        n_bootstraps=n_bootstraps,
        n_pca_components=n_pca_components,
        fusion_method=fusion_method,
        save_results_dir=save_results_dir
    )
    
    return results

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

        save_omics_dir = opt['PREDICTION']['OMICS_DIR']
        if opt['PREDICTION']['USED_OMICS']['VALUE'] == "radiomics":
            save_omics_dir = save_omics_dir + f"/{radiomics_mode}" + f"/radiomics_GCNConv_" \
                + opt['RADIOMICS']['AGGREGATED_MODE']['VALUE']
        elif opt['PREDICTION']['USED_OMICS']['VALUE'] == "pathomics":
            save_omics_dir = save_omics_dir + f"/{pathomics_mode}" + f"/pathomics_GCNConv_" \
                + opt['PATHOMICS']['AGGREGATED_MODE']['VALUE']
        elif opt['PREDICTION']['USED_OMICS']['VALUE'] == "radiopathomics":
            if opt['PREDICTION']['USED_OMICS']['CONCAT']:
                save_omics_dir = {
                    "radiomics": save_omics_dir + f"/{radiomics_mode}" + f"/radiomics_GCNConv_" \
                        + opt['RADIOMICS']['AGGREGATED_MODE']['VALUE'],
                    "pathomics": save_omics_dir + f"/{pathomics_mode}" + f"/pathomics_GCNConv_" \
                        + opt['PATHOMICS']['AGGREGATED_MODE']['VALUE']
                }
            else:
                save_omics_dir = save_omics_dir + f"/{radiomics_mode}+{pathomics_mode}" + f"/radiomics_pathomics_GCNConv_" \
                    + opt['PATHOMICS']['AGGREGATED_MODE']['VALUE']
        # save_omics_dir = opt['PREDICTION']['OMICS_DIR'] + f"/{radiomics_mode}+{pathomics_mode}" + f"/pathomics_GCNConv_SPARRA_homo+heter_vi0.1ae10_noDS_x_enc_sample0.1"

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
            feature_selection=opt['PREDICTION']['FEATURE_SELECTION']['VALUE'],
            feature_var_threshold=opt['PREDICTION']['FEATURE_SELECTION']['VAR_THRESHOLD'],
            n_selected_features=opt['PREDICTION']['FEATURE_SELECTION']['NUM_FEATURES'],
            n_bootstraps=opt['PREDICTION']['N_BOOTSTRAPS'],
            use_graph_properties=opt['PREDICTION']['USE_GRAPH_PROPERTIES'],
            n_pca_components=opt['PREDICTION']['N_PCA_COMPONENTS'],
            fusion_method=opt['PREDICTION']['FUSION_METHOD'],
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
