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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin

from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

from torch_geometric.loader import DataLoader
from tiatoolbox import logger

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
                 beta_domain=1.0, beta_recon=1.0, beta_cross=0.5, kl_annealing_steps=50,
                 device='cuda'):
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
        self.kl_annealing_steps = kl_annealing_steps
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
    
    def mmd_between_distributions(self, z1, z2, sigma=None):
        """
        Maximum Mean Discrepancy between two sets of hidden representations.
        
        Args:
            z1: Source domain features [batch_size, feature_dim]
            z2: Target domain features [batch_size, feature_dim]
            sigma: Bandwidth for RBF kernel (if None, use median heuristic)
        
        Returns:
            MMD loss (scalar)
        """
        batch_size = z1.size(0)
        features = torch.cat([z1, z2], dim=0)
        
        # Compute pairwise distances
        xx = torch.mm(z1, z1.t())
        xy = torch.mm(z1, z2.t())
        yy = torch.mm(z2, z2.t())
        
        # Squared L2 distances
        diag_xx = torch.diag(xx).unsqueeze(1).expand_as(xx)
        diag_yy = torch.diag(yy).unsqueeze(1).expand_as(yy)
        
        dist_xx = diag_xx + diag_xx.t() - 2 * xx
        dist_xy = diag_xx + diag_yy.t() - 2 * xy
        dist_yy = diag_yy + diag_yy.t() - 2 * yy
        
        # Clamp for numerical stability
        dist_xx = torch.clamp(dist_xx, min=0)
        dist_xy = torch.clamp(dist_xy, min=0)
        dist_yy = torch.clamp(dist_yy, min=0)
        
        # Median heuristic for bandwidth (if not provided)
        if sigma is None:
            with torch.no_grad():
                all_dists = torch.cat([
                    dist_xx.flatten(), 
                    dist_xy.flatten(), 
                    dist_yy.flatten()
                ])
                sigma = torch.median(all_dists[all_dists > 0]).sqrt()
                sigma = sigma.clamp(min=1e-5)
        
        # RBF kernel: K(x,y) = exp(-||x-y||^2 / (2*sigma^2))
        gamma = 1.0 / (2 * sigma**2 + 1e-8)
        
        k_xx = torch.exp(-gamma * dist_xx)
        k_xy = torch.exp(-gamma * dist_xy)
        k_yy = torch.exp(-gamma * dist_yy)
        
        # MMD: E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
        mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        
        return mmd

    def mmd_multi_kernel(self, z1, z2, sigmas=None):
        """
        Multi-Kernel MMD with multiple bandwidths.
        """
        if sigmas is None:
            # Common practice: use multiple scales
            sigmas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        mmd_total = 0
        for sigma in sigmas:
            mmd_total += self.mmd_between_distributions(z1, z2, sigma)
        
        return mmd_total / len(sigmas)
    
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
            total_domain_loss = 0

            progress = epoch / self.kl_annealing_steps
            kl_weight = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            
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
                # domain_loss = self.kl_divergence_between_distributions(
                #     outputs['radio_mu'], outputs['radio_logvar'],
                #     outputs['patho_mu'], outputs['patho_logvar']
                # )

                domain_loss = self.mmd_multi_kernel(
                    outputs['radio_z'], outputs['patho_z']
                )
                
                # Total loss with separate weights for self and cross reconstruction
                loss = (self.beta_recon * recon_loss + 
                       self.beta_cross * cross_recon_loss +
                       self.beta_kl * kl_weight * kl_loss + 
                       self.beta_domain * domain_loss)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_cross_recon_loss += cross_recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_domain_loss += domain_loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Loss: {total_loss/len(dataloader):.4f}, "
                      f"Recon: {total_recon_loss/len(dataloader):.4f}, "
                      f"CrossRecon: {total_cross_recon_loss/len(dataloader):.4f}, "
                      f"KL: {total_kl_loss/len(dataloader):.4f}, "
                      f"DomainLoss: {total_domain_loss/len(dataloader):.4f}")
        
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

class PolicyNetwork(nn.Module):
    """
    Neural network policy that outputs probabilities for each action.
    
    Input: State vector [R, P, |R-P|] normalized
    Output: Softmax probabilities for actions [Rad, Path, RP]
    """
    
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=3, dropout_rate=0.1):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        logits = self.network(x)
        return torch.softmax(logits, dim=1)
    
    def get_logits(self, x):
        """Get raw logits for gradient computation."""
        return self.network(x)


class WeightedCoxPLLoss(nn.Module):
    """
    Negative Weighted Cox Partial Likelihood Loss with Exploration Mechanisms.
    
    This loss function combines:
    1. Weighted Cox PL (exploitation - maximize survival ranking)
    2. Entropy bonus (exploration - encourage action diversity)
    3. Uncertainty bonus (exploration - reward high-variance decisions)
    4. Temperature annealing (gradual shift from exploration to exploitation)
    """
    
    def __init__(self, 
                 entropy_weight=0.1,
                 uncertainty_weight=0.05,
                 min_entropy_weight=0.01,
                 max_entropy_weight=0.5,
                 temperature=1.0,
                 min_temperature=0.1,
                 annealing_rate=0.95):
        """
        Parameters
        ----------
        entropy_weight : float
            Initial weight for entropy bonus (higher = more exploration)
        uncertainty_weight : float
            Weight for uncertainty bonus (higher = explore uncertain actions)
        min_entropy_weight : float
            Minimum entropy weight (prevents complete exploitation)
        max_entropy_weight : float
            Maximum entropy weight (caps exploration)
        temperature : float
            Initial softmax temperature (higher = more uniform exploration)
        min_temperature : float
            Minimum temperature (sharpens policy over time)
        annealing_rate : float
            Rate at which temperature and entropy weight decrease
        """
        super(WeightedCoxPLLoss, self).__init__()
        
        self.entropy_weight = entropy_weight
        self.uncertainty_weight = uncertainty_weight
        self.min_entropy_weight = min_entropy_weight
        self.max_entropy_weight = max_entropy_weight
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.annealing_rate = annealing_rate
        
        # Store for adaptive adjustment
        self.step_count = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def update_parameters(self, step_count=None):
        """
        Update temperature and entropy weight based on training progress.
        This implements exploration annealing.
        """
        if step_count is not None:
            self.step_count = step_count
        
        # Anneal temperature (gradually sharpen policy)
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.annealing_rate
        )
        
        # Anneal entropy weight (gradually reduce exploration)
        self.entropy_weight = max(
            self.min_entropy_weight,
            self.entropy_weight * self.annealing_rate
        )
        
        return self.temperature, self.entropy_weight
    
    def compute_entropy(self, probs):
        """
        Compute entropy of policy distribution.
        Higher entropy = more uniform action selection (exploration).
        """
        # Add small epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return entropy.mean()
    
    def compute_uncertainty(self, probs, R, P, RP):
        """
        Compute uncertainty based on variance of risk scores across actions.
        Higher variance = policy is uncertain about which action is best.
        """
        # Weighted risk scores per action
        risk_all = torch.stack([R, P, RP], dim=1)  # (n_samples, 3)
        
        # Expected risk (already computed in weighted Cox PL)
        expected_risk = (probs * risk_all).sum(dim=1, keepdim=True)
        
        # Variance of risk across actions
        variance = torch.sum(probs * (risk_all - expected_risk)**2, dim=1)
        
        # Encourage exploration when variance is high (uncertainty bonus)
        return variance.mean()
    
    def forward(self, probs, R, P, RP, E, T,
                return_components=False, regularization_probs=None):
        """
        Compute loss with exploration bonuses.
        
        Parameters
        ----------
        probs : torch.Tensor
            Policy probabilities for each action (n_samples, 3)
        R, P, RP : torch.Tensor
            Risk scores from each modality (n_samples,)
        E : torch.Tensor
            Event indicators (n_samples,)
        T : torch.Tensor
            Survival times (n_samples,)
        return_components : bool
            If True, return individual loss components for monitoring
            
        Returns
        -------
        loss : torch.Tensor
            Total loss (negative weighted Cox PL + exploration bonuses)
        """
        # ============================================================
        # COMPONENT 1: WEIGHTED COX PARTIAL LIKELIHOOD (Exploitation)
        # ============================================================
        
        reg_probs = probs if regularization_probs is None else regularization_probs

        # Compute weighted risk scores
        h_weighted = (probs[:, 0] * R + 
                      probs[:, 1] * P + 
                      probs[:, 2] * RP)
        
        # Apply temperature scaling for sharper/softer decisions
        # Higher temperature = softer (more exploration)
        # Lower temperature = sharper (more exploitation)
        h_weighted = h_weighted / self.temperature
        
        # Sort by time (descending)
        idx = torch.argsort(T, descending=True)
        h_sorted = h_weighted[idx]
        E_sorted = E[idx]
        
        # Stable log risk-set sums. Unlike subtracting a global maximum,
        # logcumsumexp retains the corresponding additive offset.
        log_risk_sums = torch.logcumsumexp(h_sorted, dim=0)

        # Cox partial likelihood (negative for minimization)
        loglik = torch.sum(E_sorted * (h_sorted - log_risk_sums))
        cox_loss = -loglik / E_sorted.sum().clamp_min(1.0)
        
        # ============================================================
        # COMPONENT 2: ENTROPY BONUS (Exploration)
        # ============================================================
        
        # Compute entropy of policy distribution
        entropy = self.compute_entropy(reg_probs)
        
        # Entropy bonus: encourage exploration when entropy is low
        # We want to maximize entropy (minimize -entropy)
        entropy_bonus = -self.entropy_weight * entropy
        
        # ============================================================
        # COMPONENT 3: UNCERTAINTY BONUS (Exploration)
        # ============================================================
        
        # Encourage exploration when uncertainty is high
        uncertainty = self.compute_uncertainty(reg_probs, R, P, RP)
        uncertainty_bonus = self.uncertainty_weight * uncertainty
        
        # ============================================================
        # COMPONENT 4: ACTION DIVERSITY REGULARIZATION (Exploration)
        # ============================================================
        
        # Penalize if policy assigns near-zero probability to any action
        # This ensures all actions remain possible
        min_prob = torch.min(reg_probs, dim=1)[0]
        diversity_penalty = -torch.log(min_prob + 1e-8).mean()
        diversity_weight = 0.01  # Small weight to avoid over-regularization
        
        # ============================================================
        # COMBINE LOSS COMPONENTS
        # ============================================================
        
        total_loss = (cox_loss + 
                      entropy_bonus + 
                      uncertainty_bonus + 
                      diversity_weight * diversity_penalty)
        
        if return_components:
            return {
                'total_loss': total_loss,
                'cox_loss': cox_loss,
                'entropy_bonus': entropy_bonus,
                'entropy_value': entropy,
                'uncertainty_bonus': uncertainty_bonus,
                'uncertainty_value': uncertainty,
                'diversity_penalty': diversity_penalty,
                'temperature': self.temperature,
                'entropy_weight': self.entropy_weight
            }
        
        return total_loss


class BayesianWeightedCoxPLLoss(nn.Module):
    """
    Bayesian-inspired loss with Thompson sampling style exploration.
    
    This loss adds noise to the risk scores based on policy uncertainty,
    encouraging the policy to explore actions with high epistemic uncertainty.
    """
    
    def __init__(self, 
                 noise_scale=0.1,
                 min_noise_scale=0.01,
                 exploration_bonus_weight=0.1,
                 annealing_rate=0.95):
        """
        Parameters
        ----------
        noise_scale : float
            Initial noise scale for risk scores
        min_noise_scale : float
            Minimum noise scale (prevents complete exploitation)
        exploration_bonus_weight : float
            Weight for exploration bonus based on uncertainty
        annealing_rate : float
            Rate at which noise scale decreases
        """
        super(BayesianWeightedCoxPLLoss, self).__init__()
        
        self.noise_scale = noise_scale
        self.min_noise_scale = min_noise_scale
        self.exploration_bonus_weight = exploration_bonus_weight
        self.annealing_rate = annealing_rate
        
        self.step_count = 0
        
    def update_parameters(self, step_count=None):
        """Annealing: gradually reduce exploration noise."""
        if step_count is not None:
            self.step_count = step_count
        
        self.noise_scale = max(
            self.min_noise_scale,
            self.noise_scale * self.annealing_rate
        )
        
        return self.noise_scale
    
    def compute_epistemic_uncertainty(self, probs, R, P, RP):
        """
        Compute epistemic uncertainty using dropout-like variance.
        Higher uncertainty = encourage more exploration.
        """
        # Standard deviation of risk scores across actions
        risk_all = torch.stack([R, P, RP], dim=1)
        std_risk = torch.std(risk_all, dim=1)
        return std_risk.mean()
    
    def forward(self, probs, R, P, RP, E, T,
                return_components=False, regularization_probs=None):
        """
        Forward pass with Bayesian exploration.
        """
        reg_probs = probs if regularization_probs is None else regularization_probs

        # ============================================================
        # COMPONENT 1: BAYESIAN RISK SCORES (Exploration via noise)
        # ============================================================
        
        # Add noise to risk scores proportional to uncertainty
        # This implements a simple form of Thompson sampling
        noise = torch.randn_like(R) * self.noise_scale
        R_noisy = R + noise * (probs[:, 0] * 0.1 + 0.1)  # More noise when action prob is low
        
        noise = torch.randn_like(P) * self.noise_scale
        P_noisy = P + noise * (probs[:, 1] * 0.1 + 0.1)
        
        noise = torch.randn_like(RP) * self.noise_scale
        RP_noisy = RP + noise * (probs[:, 2] * 0.1 + 0.1)
        
        # Weighted risk with noisy scores
        h_weighted = (probs[:, 0] * R_noisy + 
                      probs[:, 1] * P_noisy + 
                      probs[:, 2] * RP_noisy)
        
        # Sort by time
        idx = torch.argsort(T, descending=True)
        h_sorted = h_weighted[idx]
        E_sorted = E[idx]
        
        # Cox PL
        log_risk_sums = torch.logcumsumexp(h_sorted, dim=0)
        loglik = torch.sum(E_sorted * (h_sorted - log_risk_sums))
        cox_loss = -loglik / E_sorted.sum().clamp_min(1.0)
        
        # ============================================================
        # COMPONENT 2: EXPLORATION BONUS (epistemic uncertainty)
        # ============================================================
        
        # Encourage exploration when epistemic uncertainty is high
        epistemic_uncertainty = self.compute_epistemic_uncertainty(probs, R, P, RP)
        exploration_bonus = -self.exploration_bonus_weight * epistemic_uncertainty
        
        # ============================================================
        # COMPONENT 3: STANDARD ENTROPY BONUS
        # ============================================================
        
        entropy = -torch.sum(
            reg_probs * torch.log(reg_probs + 1e-8), dim=1
        ).mean()
        entropy_bonus = -0.05 * entropy
        
        # ============================================================
        # COMBINE
        # ============================================================
        
        total_loss = cox_loss + exploration_bonus + entropy_bonus
        
        if return_components:
            return {
                'total_loss': total_loss,
                'cox_loss': cox_loss,
                'exploration_bonus': exploration_bonus,
                'epistemic_uncertainty': epistemic_uncertainty,
                'entropy_bonus': entropy_bonus,
                'entropy_value': entropy,
                'noise_scale': self.noise_scale
            }
        
        return total_loss


class AdaptiveWeightedCoxPLLoss(nn.Module):
    """
    Adaptive loss with automatic exploration-exploitation balancing.
    
    This loss adaptively adjusts the exploration weight based on:
    1. Training progress (more exploration early, more exploitation late)
    2. Performance plateaus (increase exploration when stuck)
    3. Action diversity (ensure all actions are explored)
    """
    
    def __init__(self,
                 initial_exploration_weight=0.3,
                 min_exploration_weight=0.01,
                 max_exploration_weight=0.5,
                 plateau_threshold=0.001,
                 plateau_patience=5):
        """
        Parameters
        ----------
        initial_exploration_weight : float
            Initial weight for exploration bonuses
        min_exploration_weight : float
            Minimum exploration weight
        max_exploration_weight : float
            Maximum exploration weight
        plateau_threshold : float
            Minimum loss improvement to detect plateau
        plateau_patience : int
            Number of steps before considering plateau
        """
        super(AdaptiveWeightedCoxPLLoss, self).__init__()
        
        self.exploration_weight = initial_exploration_weight
        self.min_exploration_weight = min_exploration_weight
        self.max_exploration_weight = max_exploration_weight
        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience
        
        # Tracking for adaptive adjustment
        self.loss_history = []
        self.plateau_counter = 0
        self.step_count = 0
        self.best_loss = float('inf')
        
    def update_exploration_weight(self, loss_value):
        """
        Adaptive adjustment of exploration weight.
        - Increase exploration if performance plateaus
        - Decrease exploration if performance improves
        """
        self.step_count += 1
        self.loss_history.append(loss_value.item() if torch.is_tensor(loss_value) else loss_value)
        
        if len(self.loss_history) > 1:
            improvement = self.loss_history[-2] - self.loss_history[-1]
            
            # Check for plateau
            if abs(improvement) < self.plateau_threshold:
                self.plateau_counter += 1
            else:
                self.plateau_counter = 0
            
            # If plateau detected, increase exploration
            if self.plateau_counter >= self.plateau_patience:
                self.exploration_weight = min(
                    self.max_exploration_weight,
                    self.exploration_weight * 1.2
                )
                self.plateau_counter = 0
            else:
                # Gradual decrease of exploration
                self.exploration_weight = max(
                    self.min_exploration_weight,
                    self.exploration_weight * 0.99
                )
        
        return self.exploration_weight
    
    def compute_action_diversity(self, probs):
        """
        Compute diversity metric: how evenly distributed are actions?
        """
        # Average probability per action across batch
        avg_probs = probs.mean(dim=0)
        
        # Uniform distribution
        uniform = torch.ones_like(avg_probs) / avg_probs.shape[0]
        
        # KL divergence between average and uniform
        # Higher KL = more focused (less exploration)
        # Lower KL = more uniform (more exploration)
        kl_div = torch.sum(avg_probs * (torch.log(avg_probs + 1e-8) - 
                                        torch.log(uniform + 1e-8)))
        
        return kl_div
    
    def forward(self, probs, R, P, RP, E, T,
                return_components=False, regularization_probs=None):
        """
        Forward pass with adaptive exploration.
        """
        reg_probs = probs if regularization_probs is None else regularization_probs

        # ============================================================
        # COMPONENT 1: WEIGHTED COX PL (Exploitation)
        # ============================================================
        
        h_weighted = (probs[:, 0] * R + 
                      probs[:, 1] * P + 
                      probs[:, 2] * RP)
        
        idx = torch.argsort(T, descending=True)
        h_sorted = h_weighted[idx]
        E_sorted = E[idx]
        
        log_risk_sums = torch.logcumsumexp(h_sorted, dim=0)
        loglik = torch.sum(E_sorted * (h_sorted - log_risk_sums))
        cox_loss = -loglik / E_sorted.sum().clamp_min(1.0)
        
        # ============================================================
        # COMPONENT 2: EXPLORATION BONUS (Adaptive)
        # ============================================================
        
        # Entropy bonus
        entropy = -torch.sum(
            reg_probs * torch.log(reg_probs + 1e-8), dim=1
        ).mean()
        entropy_bonus = -self.exploration_weight * entropy
        
        # Diversity bonus: encourage exploration when actions are imbalanced
        diversity = self.compute_action_diversity(reg_probs)
        diversity_bonus = self.exploration_weight * 0.1 * diversity
        
        # Uncertainty bonus
        risk_all = torch.stack([R, P, RP], dim=1)
        expected_risk = (reg_probs * risk_all).sum(dim=1, keepdim=True)
        variance = torch.sum(
            reg_probs * (risk_all - expected_risk)**2, dim=1
        )
        uncertainty_bonus = self.exploration_weight * 0.1 * variance.mean()
        
        # ============================================================
        # COMBINE
        # ============================================================
        
        total_loss = cox_loss + entropy_bonus + diversity_bonus + uncertainty_bonus
        
        if return_components:
            return {
                'total_loss': total_loss,
                'cox_loss': cox_loss,
                'entropy_bonus': entropy_bonus,
                'entropy_value': entropy,
                'diversity_bonus': diversity_bonus,
                'diversity_value': diversity,
                'uncertainty_bonus': uncertainty_bonus,
                'uncertainty_value': variance.mean(),
                'exploration_weight': self.exploration_weight
            }
        
        return total_loss


class EnsembleWeightedCoxPLLoss(nn.Module):
    """
    Ensemble-based loss with exploration via multiple hypotheses.
    
    Maintains multiple risk estimates and encourages the policy to
    explore actions that are favored by different ensemble members.
    """
    
    def __init__(self, 
                 ensemble_size=5,
                 exploration_weight=0.1,
                 ensemble_noise_scale=0.05):
        """
        Parameters
        ----------
        ensemble_size : int
            Number of ensemble members (different risk estimates)
        exploration_weight : float
            Weight for exploration bonus
        ensemble_noise_scale : float
            Noise scale for ensemble diversity
        """
        super(EnsembleWeightedCoxPLLoss, self).__init__()
        
        self.ensemble_size = ensemble_size
        self.exploration_weight = exploration_weight
        self.ensemble_noise_scale = ensemble_noise_scale
        
    def forward(self, probs, R, P, RP, E, T,
                return_components=False, regularization_probs=None):
        """
        Forward pass with ensemble exploration.
        """
        reg_probs = probs if regularization_probs is None else regularization_probs

        # ============================================================
        # COMPONENT 1: ENSEMBLE RISK ESTIMATES
        # ============================================================
        
        # Create ensemble of perturbed risk scores
        ensemble_losses = []
        
        for e in range(self.ensemble_size):
            # Add noise to risk scores (different for each ensemble member)
            noise_scale = self.ensemble_noise_scale * (1 + 0.1 * e / self.ensemble_size)
            
            R_e = R + torch.randn_like(R) * noise_scale
            P_e = P + torch.randn_like(P) * noise_scale
            RP_e = RP + torch.randn_like(RP) * noise_scale
            
            # Compute weighted risk
            h_e = (probs[:, 0] * R_e + 
                   probs[:, 1] * P_e + 
                   probs[:, 2] * RP_e)
            
            # Cox PL for this ensemble member
            idx = torch.argsort(T, descending=True)
            h_sorted = h_e[idx]
            E_sorted = E[idx]
            
            log_risk_sums = torch.logcumsumexp(h_sorted, dim=0)
            loglik = torch.sum(E_sorted * (h_sorted - log_risk_sums))
            ensemble_losses.append(-loglik / E_sorted.sum().clamp_min(1.0))
        
        # Average over ensemble
        cox_loss = torch.stack(ensemble_losses).mean()
        
        # ============================================================
        # COMPONENT 2: ENSEMBLE UNCERTAINTY (Exploration)
        # ============================================================
        
        # Variance across ensemble members
        ensemble_losses_tensor = torch.stack(ensemble_losses)
        ensemble_variance = torch.var(ensemble_losses_tensor)
        
        # Encourage exploration when ensemble members disagree
        exploration_bonus = -self.exploration_weight * ensemble_variance
        
        # ============================================================
        # COMPONENT 3: STANDARD ENTROPY BONUS
        # ============================================================
        
        entropy = -torch.sum(
            reg_probs * torch.log(reg_probs + 1e-8), dim=1
        ).mean()
        entropy_bonus = -0.05 * entropy
        
        # ============================================================
        # COMBINE
        # ============================================================
        
        total_loss = cox_loss + exploration_bonus + entropy_bonus
        
        if return_components:
            return {
                'total_loss': total_loss,
                'cox_loss': cox_loss,
                'exploration_bonus': exploration_bonus,
                'ensemble_variance': ensemble_variance,
                'entropy_bonus': entropy_bonus,
                'entropy_value': entropy
            }
        
        return total_loss


class TorchCoxPH:
    """Linear Cox proportional-hazards model optimized with PyTorch.

    The implementation supports case weights, Breslow handling of tied event
    times, elastic-net regularization, warm starts, and CUDA execution.  It is
    intentionally prediction-focused: unlike ``lifelines.CoxPHFitter``, it does
    not calculate standard errors or a robust covariance matrix.
    """

    def __init__(self, penalizer=0.0, l1_ratio=0.0, learning_rate=0.05,
                 max_epochs=500, tolerance=1e-6, patience=20,
                 gradient_clip=10.0, device='cpu'):
        if penalizer < 0:
            raise ValueError("penalizer must be non-negative")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be between 0 and 1")

        self.penalizer = float(penalizer)
        self.l1_ratio = float(l1_ratio)
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.tolerance = float(tolerance)
        self.patience = int(patience)
        self.gradient_clip = float(gradient_clip)
        self.device = torch.device(
            device if str(device).startswith('cuda') and torch.cuda.is_available()
            else 'cpu'
        )
        self.coef_ = None
        self.n_features_in_ = None
        self.n_iter_ = 0
        self.loss_ = None

    @staticmethod
    def _as_tensor(values, device, dtype=torch.float32):
        if torch.is_tensor(values):
            return values.detach().to(device=device, dtype=dtype)
        return torch.as_tensor(values, device=device, dtype=dtype)

    @staticmethod
    def _validate_inputs(X, T, E, weights):
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        n_samples = X.shape[0]
        if T.ndim != 1 or E.ndim != 1 or weights.ndim != 1:
            raise ValueError("T, E, and weights must be one-dimensional")
        if not (len(T) == len(E) == len(weights) == n_samples):
            raise ValueError("X, T, E, and weights must contain the same samples")
        if n_samples < 2:
            raise ValueError("At least two samples are required")
        if not torch.isfinite(X).all() or not torch.isfinite(T).all():
            raise ValueError("X and T must contain only finite values")
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("weights must be finite and non-negative")
        if torch.sum(weights * E) <= 0:
            raise ValueError("At least one event must have a positive weight")

    @staticmethod
    def _prepare_risk_sets(X, T, E, weights):
        order = torch.argsort(T, descending=True, stable=True)
        X_sorted = X[order]
        T_sorted = T[order]
        E_sorted = E[order]
        W_sorted = weights[order]

        _, group_ids, group_counts = torch.unique_consecutive(
            T_sorted, return_inverse=True, return_counts=True
        )
        group_end = torch.cumsum(group_counts, dim=0) - 1
        return X_sorted, E_sorted, W_sorted, group_ids, group_end

    @staticmethod
    def _breslow_loss_from_risk_sets(beta, risk_sets):
        """Return mean weighted negative partial log-likelihood."""
        X_sorted, E_sorted, W_sorted, group_ids, group_end = risk_sets

        eta = X_sorted.mv(beta)
        log_weights = torch.where(
            W_sorted > 0,
            torch.log(W_sorted),
            torch.full_like(W_sorted, -torch.inf),
        )
        log_risk_sums = torch.logcumsumexp(eta + log_weights, dim=0)

        n_groups = group_end.numel()
        event_weights = W_sorted * E_sorted

        weighted_event_eta = torch.zeros(
            n_groups, device=X_sorted.device, dtype=X_sorted.dtype
        ).scatter_add_(0, group_ids, event_weights * eta)
        event_weight_sum = torch.zeros(
            n_groups, device=X_sorted.device, dtype=X_sorted.dtype
        ).scatter_add_(0, group_ids, event_weights)

        group_log_risk = log_risk_sums[group_end]
        # Groups with no weighted events contribute exactly zero. Mask them
        # before multiplication so an empty early risk set cannot form 0 * -inf.
        event_groups = event_weight_sum > 0
        log_likelihood = (
            weighted_event_eta[event_groups]
            - event_weight_sum[event_groups] * group_log_risk[event_groups]
        )
        total_event_weight = event_weight_sum[event_groups].sum()
        return -log_likelihood.sum() / total_event_weight

    @classmethod
    def _breslow_negative_log_likelihood(cls, beta, X, T, E, weights):
        """Convenience entry point that prepares and evaluates risk sets."""
        risk_sets = cls._prepare_risk_sets(X, T, E, weights)
        return cls._breslow_loss_from_risk_sets(beta, risk_sets)

    def _objective(self, beta, risk_sets, include_l1=True):
        loss = self._breslow_loss_from_risk_sets(beta, risk_sets)
        l2_strength = self.penalizer * (1.0 - self.l1_ratio)
        if l2_strength:
            loss = loss + 0.5 * l2_strength * torch.sum(beta.square())
        if include_l1:
            l1_strength = self.penalizer * self.l1_ratio
            if l1_strength:
                loss = loss + l1_strength * torch.sum(torch.abs(beta))
        return loss

    def fit(self, X, T, E, weights=None, initial_coef=None):
        X_tensor = self._as_tensor(X, self.device)
        T_tensor = self._as_tensor(T, self.device).reshape(-1)
        E_tensor = self._as_tensor(E, self.device).reshape(-1)
        if weights is None:
            W_tensor = torch.ones_like(T_tensor)
        else:
            W_tensor = self._as_tensor(weights, self.device).reshape(-1)

        self._validate_inputs(X_tensor, T_tensor, E_tensor, W_tensor)
        self.n_features_in_ = X_tensor.shape[1]
        risk_sets = self._prepare_risk_sets(
            X_tensor, T_tensor, E_tensor, W_tensor
        )

        if initial_coef is None:
            beta = torch.zeros(
                self.n_features_in_, device=self.device, dtype=X_tensor.dtype
            )
        else:
            beta = self._as_tensor(initial_coef, self.device).reshape(-1).clone()
            if beta.numel() != self.n_features_in_:
                raise ValueError("initial_coef has the wrong number of features")
        beta.requires_grad_(True)

        optimizer = optim.Adam([beta], lr=self.learning_rate)
        best_coef = beta.detach().clone()
        best_loss = float('inf')
        previous_loss = None
        stale_epochs = 0
        l1_strength = self.penalizer * self.l1_ratio

        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            smooth_loss = self._objective(
                beta, risk_sets, include_l1=False
            )
            if not torch.isfinite(smooth_loss):
                raise FloatingPointError("Non-finite TorchCoxPH loss")
            smooth_loss.backward()
            torch.nn.utils.clip_grad_norm_([beta], self.gradient_clip)
            optimizer.step()

            # Proximal step for the non-smooth L1 component.
            if l1_strength:
                threshold = self.learning_rate * l1_strength
                with torch.no_grad():
                    beta.copy_(
                        torch.sign(beta) * torch.clamp(torch.abs(beta) - threshold, min=0)
                    )

            with torch.no_grad():
                current_loss = self._objective(
                    beta, risk_sets
                ).item()
            if not np.isfinite(current_loss):
                raise FloatingPointError("Non-finite TorchCoxPH objective")

            if current_loss < best_loss:
                best_loss = current_loss
                best_coef = beta.detach().clone()

            if previous_loss is not None:
                relative_change = abs(previous_loss - current_loss) / max(
                    1.0, abs(previous_loss)
                )
                stale_epochs = stale_epochs + 1 if relative_change < self.tolerance else 0
                if stale_epochs >= self.patience:
                    self.n_iter_ = epoch + 1
                    break
            previous_loss = current_loss
            self.n_iter_ = epoch + 1

        self.coef_ = best_coef.detach().cpu().numpy().copy()
        self.loss_ = best_loss
        return self

    def predict_log_partial_hazard(self, X):
        if self.coef_ is None:
            raise RuntimeError("TorchCoxPH must be fitted before prediction")
        X_tensor = self._as_tensor(X, self.device)
        if X_tensor.ndim != 2 or X_tensor.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible feature dimension")
        coef = self._as_tensor(self.coef_, self.device)
        with torch.no_grad():
            return X_tensor.mv(coef).cpu().numpy()


class ContextualBandit:
    """
    Contextual Bandit with PyTorch policy network.
    
    Implements the EM framework with direct policy optimization:
    - E-Step: Update policy network by minimizing negative weighted Cox PL
    - M-Step: Update subgroup-specific Cox models with sample weights
    
    Parameters
    ----------
    alpha_range : list, default=[0.001, 0.01, 0.1, 1.0, 10.0]
        Elastic-net penalty range for Cox models
    max_iterations : int, default=10
        Maximum number of EM iterations
    convergence_threshold : float, default=0.001
        Minimum improvement in C-index for convergence
    hidden_dim : int, default=16
        Hidden layer dimension for policy network
    learning_rate : float, default=0.01
        Learning rate for policy network
    batch_size : int, default=32
        Retained for compatibility; policy Cox training uses full risk sets
    policy_epochs : int, default=50
        Number of epochs for policy training per EM iteration
    cv_folds : int, default=5
        Number of cross-validation folds for Cox model selection
    cox_learning_rate : float, default=0.05
        Learning rate for TorchCoxPH optimization
    cox_max_epochs : int, default=500
        Maximum optimizer steps for each TorchCoxPH fit
    cox_tolerance : float, default=1e-6
        Relative objective-change threshold for Cox early stopping
    cox_patience : int, default=20
        Consecutive low-change steps before stopping Cox optimization
    cox_l1_ratio : float, default=0.9
        Elastic-net mixing parameter used by TorchCoxPH
    min_expert_weight : float, default=0.01
        Minimum M-step sample weight assigned to each survival expert
    rp_cost_weight : float, default=1.0
        Strength of the evidence-based penalty on RP policy probability
    rp_minimum_gain : float, default=0.01
        Required lower-confidence-bound C-index gain for cost-free RP use
    rp_bootstrap_samples : int, default=500
        Paired bootstrap samples used to estimate RP performance evidence
    hard_policy : bool, default=False
        Train Cox risk with straight-through one-hot Gumbel-Softmax actions
    gumbel_temperature : float, default=1.0
        Initial Gumbel-Softmax temperature
    policy_horizon_days : float, default=1826.25
        Common survival-calibration horizon (five years for day-based outcomes)
    m_step_momentum : float, default=0.3
        Fraction of newly proposed policy weights used in each M-step
    min_expert_ess : float, default=50
        Absolute lower bound for each expert's effective sample size
    min_expert_ess_fraction : float, default=0.2
        Dataset-relative lower bound for each expert's effective sample size
    expert_cindex_tolerance : float, default=0.01
        Maximum accepted OOF C-index loss relative to the previous expert
    expert_anchor_tolerance : float, default=0.03
        Maximum accepted OOF C-index loss relative to the frozen anchor
    device : str, default='cuda'
        Device for PyTorch ('cuda' or 'cpu')
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, 
                 alpha_range=None,
                 max_iterations=10,
                 convergence_threshold=0.001,
                 hidden_dim=16,
                 learning_rate=0.01,
                 batch_size=32,
                 policy_epochs=50,
                 cv_folds=5,
                 cox_learning_rate=0.05,
                 cox_max_epochs=500,
                 cox_tolerance=1e-6,
                 cox_patience=20,
                 cox_l1_ratio=0.9,
                 cox_gradient_clip=10.0,
                 min_expert_weight=0.01,
                 rp_cost_weight=1.0,
                 rp_minimum_gain=0.01,
                 rp_bootstrap_samples=500,
                 rp_confidence=0.95,
                 hard_policy=False,
                 gumbel_temperature=1.0,
                 gumbel_min_temperature=0.1,
                 gumbel_anneal_rate=0.95,
                 policy_horizon_days=5 * 365.25,
                 m_step_momentum=0.3,
                 min_expert_ess=50,
                 min_expert_ess_fraction=0.2,
                 expert_cindex_tolerance=0.01,
                 expert_anchor_tolerance=0.03,
                 loss_type='adaptive',  # 'weighted', 'bayesian', 'adaptive', 'ensemble'
                 exploration_weight=0.1,
                 entropy_weight=0.05,
                 uncertainty_weight=0.05,
                 temperature=1.0,
                 device='cuda',
                 random_state=None):
        
        self.alpha_range = (
            [0.001, 0.01, 0.1, 1.0, 10.0]
            if alpha_range is None else list(alpha_range)
        )
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.policy_epochs = policy_epochs
        self.cv_folds = cv_folds
        self.cox_learning_rate = cox_learning_rate
        self.cox_max_epochs = cox_max_epochs
        self.cox_tolerance = cox_tolerance
        self.cox_patience = cox_patience
        self.cox_l1_ratio = cox_l1_ratio
        self.cox_gradient_clip = cox_gradient_clip
        if not 0.0 <= min_expert_weight < 1.0 / 3.0:
            raise ValueError("min_expert_weight must be in [0, 1/3)")
        self.min_expert_weight = min_expert_weight
        if rp_cost_weight < 0:
            raise ValueError("rp_cost_weight must be non-negative")
        if rp_bootstrap_samples < 0:
            raise ValueError("rp_bootstrap_samples must be non-negative")
        if not 0.0 < rp_confidence < 1.0:
            raise ValueError("rp_confidence must be between 0 and 1")
        self.rp_cost_weight = rp_cost_weight
        self.rp_minimum_gain = rp_minimum_gain
        self.rp_bootstrap_samples = rp_bootstrap_samples
        self.rp_confidence = rp_confidence
        if gumbel_temperature <= 0 or gumbel_min_temperature <= 0:
            raise ValueError("Gumbel temperatures must be positive")
        if not 0.0 < gumbel_anneal_rate <= 1.0:
            raise ValueError("gumbel_anneal_rate must be in (0, 1]")
        self.hard_policy = hard_policy
        self.gumbel_initial_temperature = gumbel_temperature
        self.gumbel_temperature = gumbel_temperature
        self.gumbel_min_temperature = gumbel_min_temperature
        self.gumbel_anneal_rate = gumbel_anneal_rate
        if policy_horizon_days <= 0:
            raise ValueError("policy_horizon_days must be positive")
        self.policy_horizon_days = float(policy_horizon_days)
        if not 0.0 < m_step_momentum <= 1.0:
            raise ValueError("m_step_momentum must be in (0, 1]")
        if min_expert_ess < 1 or not 0.0 <= min_expert_ess_fraction <= 1.0:
            raise ValueError("Invalid minimum expert ESS configuration")
        if expert_cindex_tolerance < 0 or expert_anchor_tolerance < 0:
            raise ValueError("Expert C-index tolerances must be non-negative")
        self.m_step_momentum = float(m_step_momentum)
        self.min_expert_ess = float(min_expert_ess)
        self.min_expert_ess_fraction = float(min_expert_ess_fraction)
        self.expert_cindex_tolerance = float(expert_cindex_tolerance)
        self.expert_anchor_tolerance = float(expert_anchor_tolerance)
        self.loss_type = loss_type
        self.exploration_weight = exploration_weight
        self.entropy_weight = entropy_weight
        self.uncertainty_weight = uncertainty_weight
        self.temperature = temperature
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.random_state = random_state
        self._cox_warm_starts = {}
        
    def _init_loss_function(self):
        """Initialize the appropriate loss function."""
        if self.loss_type == 'weighted':
            # Original weighted Cox PL with entropy bonus
            self.policy_loss_fn = WeightedCoxPLLoss(
                entropy_weight=self.entropy_weight,
                uncertainty_weight=self.uncertainty_weight,
                temperature=self.temperature
            )
        elif self.loss_type == 'bayesian':
            # Bayesian exploration with noise
            self.policy_loss_fn = BayesianWeightedCoxPLLoss(
                noise_scale=0.1,
                exploration_bonus_weight=self.exploration_weight
            )
        elif self.loss_type == 'ensemble':
            # Ensemble-based exploration
            self.policy_loss_fn = EnsembleWeightedCoxPLLoss(
                ensemble_size=5,
                exploration_weight=self.exploration_weight
            )
        else:  # 'adaptive' (default)
            # Fully adaptive exploration-exploitation balancing
            self.policy_loss_fn = AdaptiveWeightedCoxPLLoss(
                initial_exploration_weight=self.exploration_weight,
                min_exploration_weight=0.01,
                max_exploration_weight=0.5
            )
    
    def _fit_horizon_calibrator(self, risk, T, E, fit_indices):
        """Fit an OOF Cox recalibrator and its five-year baseline hazard."""
        risk = np.asarray(risk, dtype=np.float32).reshape(-1)
        T = np.asarray(T, dtype=np.float32).reshape(-1)
        E = np.asarray(E, dtype=np.float32).reshape(-1)
        fit_indices = np.asarray(fit_indices, dtype=np.int64)
        risk_fit = risk[fit_indices]
        T_fit = T[fit_indices]
        E_fit = E[fit_indices]

        center = float(risk_fit.mean())
        scale = max(float(risk_fit.std()), 1e-6)
        normalized_risk = ((risk_fit - center) / scale).reshape(-1, 1)
        calibration_model = TorchCoxPH(
            penalizer=0.01,
            l1_ratio=0.0,
            learning_rate=self.cox_learning_rate,
            max_epochs=self.cox_max_epochs,
            tolerance=self.cox_tolerance,
            patience=self.cox_patience,
            gradient_clip=self.cox_gradient_clip,
            device=self.device,
        ).fit(normalized_risk, T_fit, E_fit)

        # A negative calibration slope would reverse an expert based on a
        # noisy calibration sample. Treat such an expert as uninformative;
        # the upper cap prevents extreme OOF slopes from destabilizing routing.
        slope = float(np.clip(calibration_model.coef_[0], 0.0, 5.0))
        calibrated_eta = slope * normalized_risk[:, 0]

        event_times = np.unique(
            T_fit[(E_fit > 0) & (T_fit <= self.policy_horizon_days)]
        )
        if event_times.size == 0:
            raise ValueError(
                "No calibration events occur at or before the policy horizon"
            )
        exp_eta = np.exp(np.clip(calibrated_eta, -50.0, 50.0))
        baseline_hazard = 0.0
        for event_time in event_times:
            n_events = float(np.sum((T_fit == event_time) & (E_fit > 0)))
            risk_set_sum = float(exp_eta[T_fit >= event_time].sum())
            if risk_set_sum > 0:
                baseline_hazard += n_events / risk_set_sum
        baseline_hazard = max(baseline_hazard, 1e-12)
        return {
            'center': center,
            'scale': scale,
            'slope': slope,
            'baseline_cumulative_hazard': baseline_hazard,
            'horizon_days': self.policy_horizon_days,
        }

    @staticmethod
    def _apply_horizon_calibrator(risk, calibrator):
        """Return calibrated log cumulative hazard: log(-log(S(tau)))."""
        risk = np.asarray(risk, dtype=np.float32)
        normalized = (risk - calibrator['center']) / calibrator['scale']
        normalized = np.clip(normalized, -8.0, 8.0)
        return (
            np.log(calibrator['baseline_cumulative_hazard'])
            + calibrator['slope'] * normalized
        ).astype(np.float32)

    @classmethod
    def _predict_horizon_survival(cls, risk, calibrator):
        log_cumulative_hazard = cls._apply_horizon_calibrator(risk, calibrator)
        cumulative_hazard = np.exp(np.clip(log_cumulative_hazard, -50.0, 50.0))
        return np.exp(-cumulative_hazard).astype(np.float32)

    def _calibrate_expert_risks(self, R, P, RP):
        return (
            self._apply_horizon_calibrator(R, self.policy_calibrators['R']),
            self._apply_horizon_calibrator(P, self.policy_calibrators['P']),
            self._apply_horizon_calibrator(RP, self.policy_calibrators['RP']),
        )

    @staticmethod
    def _effective_sample_size(weights):
        weights = np.asarray(weights, dtype=np.float64)
        denominator = np.sum(weights ** 2)
        return 0.0 if denominator <= 0 else float(weights.sum() ** 2 / denominator)

    def _stabilize_expert_weights(self, previous, proposed, minimum_ess):
        """Momentum-smooth weights and blend toward uniform to satisfy ESS."""
        previous = np.asarray(previous, dtype=np.float64)
        proposed = np.asarray(proposed, dtype=np.float64)
        smoothed = (
            (1.0 - self.m_step_momentum) * previous
            + self.m_step_momentum * proposed
        )
        if self._effective_sample_size(smoothed) >= minimum_ess:
            return smoothed.astype(np.float32)

        uniform = np.full_like(smoothed, smoothed.mean())
        low, high = 0.0, 1.0
        for _ in range(30):
            blend = 0.5 * (low + high)
            candidate = (1.0 - blend) * smoothed + blend * uniform
            if self._effective_sample_size(candidate) >= minimum_ess:
                high = blend
            else:
                low = blend
        stabilized = (1.0 - high) * smoothed + high * uniform
        return stabilized.astype(np.float32)

    def _expert_candidate_is_acceptable(self, candidate_cindex,
                                         previous_cindex, anchor_cindex,
                                         calibration_is_finite=True):
        threshold = max(
            previous_cindex - self.expert_cindex_tolerance,
            anchor_cindex - self.expert_anchor_tolerance,
        )
        accepted = bool(
            np.isfinite(candidate_cindex)
            and calibration_is_finite
            and candidate_cindex >= threshold
        )
        return accepted, threshold

    @staticmethod
    def _make_policy_state(R, P, RP):
        """Build the compact Version-B state: R, P, RP, and signed R-P."""
        # return np.column_stack([R, P, RP, R - P]).astype(np.float32)
        return np.column_stack([R, P, R - P]).astype(np.float32)

    def _make_routing_state_from_features(self, X_rad, X_path):
        """Build the stationary state from frozen global routing experts."""
        X_rp = np.concatenate([X_rad, X_path], axis=1)
        R = self._apply_horizon_calibrator(
            self._predict_risk(self.routing_cox_rad, X_rad),
            self.routing_calibrators['R'],
        )
        P = self._apply_horizon_calibrator(
            self._predict_risk(self.routing_cox_path, X_path),
            self.routing_calibrators['P'],
        )
        RP = self._apply_horizon_calibrator(
            self._predict_risk(self.routing_cox_rp, X_rp),
            self.routing_calibrators['RP'],
        )
        return self._make_policy_state(R, P, RP)

    @staticmethod
    def _risk_cindex(risk, E, T):
        return concordance_index(T, -risk, E.astype(bool))

    def _compute_rp_cost(self, R, P, RP, E, T, bootstrap_indices=None,
                         seed_offset=0):
        """Estimate whether RP reliably improves on both unimodal experts.

        The same patient indices are used for all three experts in every
        bootstrap replicate. RP is cost-free only when the lower confidence
        bounds of both paired C-index gains exceed ``rp_minimum_gain``.
        """
        R = np.asarray(R, dtype=np.float32)
        P = np.asarray(P, dtype=np.float32)
        RP = np.asarray(RP, dtype=np.float32)
        E = np.asarray(E, dtype=bool)
        T = np.asarray(T, dtype=np.float32)

        point_scores = {
            'rad': self._risk_cindex(R, E, T),
            'path': self._risk_cindex(P, E, T),
            'rp': self._risk_cindex(RP, E, T),
        }
        point_gains = np.array([
            point_scores['rp'] - point_scores['rad'],
            point_scores['rp'] - point_scores['path'],
        ])

        gains = []
        if bootstrap_indices is not None:
            bootstrap_indices = np.asarray(bootstrap_indices, dtype=np.int64)
        elif self.rp_bootstrap_samples > 0:
            base_seed = 0 if self.random_state is None else self.random_state
            rng = np.random.default_rng(base_seed + seed_offset)
            n_samples = len(T)
            bootstrap_indices = rng.integers(
                0, n_samples, size=(self.rp_bootstrap_samples, n_samples)
            )

        if bootstrap_indices is not None:
            for idx in bootstrap_indices:
                try:
                    c_rad = self._risk_cindex(R[idx], E[idx], T[idx])
                    c_path = self._risk_cindex(P[idx], E[idx], T[idx])
                    c_rp = self._risk_cindex(RP[idx], E[idx], T[idx])
                    gains.append([c_rp - c_rad, c_rp - c_path])
                except Exception:
                    # Some small/censored resamples contain no comparable pair.
                    continue

        if gains:
            lower_percentile = 100.0 * (1.0 - self.rp_confidence) / 2.0
            lower_gains = np.percentile(np.asarray(gains), lower_percentile, axis=0)
        else:
            lower_gains = point_gains

        rp_evidence = float(np.min(lower_gains))
        rp_cost = max(0.0, self.rp_minimum_gain - rp_evidence)
        return rp_cost, {
            'cindex_rad': point_scores['rad'],
            'cindex_path': point_scores['path'],
            'cindex_rp': point_scores['rp'],
            'lower_gain_vs_rad': float(lower_gains[0]),
            'lower_gain_vs_path': float(lower_gains[1]),
            'valid_bootstraps': len(gains),
        }
    
    def train_survival_model(self, X, T, E, weights=None, alpha_range=None,
                             model_key='cox'):
        """
        Train a CoxPH model with cross-validation for regularization parameter selection.
        
        Parameters
        ----------
        X : ndarray
            Feature matrix
        T : ndarray
            Survival times
        E : ndarray
            Event indicators (1=event, 0=censored)
        weights : ndarray, optional
            Sample weights
        alpha_range : list, optional
            Regularization parameter range
            
        Returns
        -------
        model : TorchCoxPH
            Fitted GPU-native CoxPH model with the best regularization parameter
        best_alpha : float
            Best regularization parameter
        """
        if alpha_range is None:
            alpha_range = self.alpha_range

        X = np.ascontiguousarray(X, dtype=np.float32)
        T = np.ascontiguousarray(T, dtype=np.float32).reshape(-1)
        E = np.ascontiguousarray(E, dtype=np.float32).reshape(-1)
        if weights is not None:
            weights = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1)
        
        # Cross-validation to select best alpha
        best_alpha = None
        best_concordance = -1
        best_oof_risk = None
        n_samples = len(T)
        indices = np.arange(n_samples)

        def make_model(alpha):
            return TorchCoxPH(
                penalizer=alpha,
                l1_ratio=self.cox_l1_ratio,
                learning_rate=self.cox_learning_rate,
                max_epochs=self.cox_max_epochs,
                tolerance=self.cox_tolerance,
                patience=self.cox_patience,
                gradient_clip=self.cox_gradient_clip,
                device=self.device,
            )
        
        for alpha in alpha_range:
            try:
                cv_scores = []
                oof_risk = np.full(n_samples, np.nan, dtype=np.float32)
                
                # Preserve the existing contiguous-fold CV construction.
                for fold in range(self.cv_folds):
                    # Split indices
                    fold_size = n_samples // self.cv_folds
                    start = fold * fold_size
                    end = start + fold_size if fold < self.cv_folds - 1 else n_samples
                    
                    val_idx = indices[start:end]
                    train_idx = np.concatenate([indices[:start], indices[end:]])

                    if len(val_idx) == 0 or len(train_idx) < 2:
                        continue

                    warm_key = (model_key, float(alpha), 'fold', fold)
                    model = make_model(alpha)
                    fold_weights = None if weights is None else weights[train_idx]
                    model.fit(
                        X[train_idx], T[train_idx], E[train_idx],
                        weights=fold_weights,
                        initial_coef=self._cox_warm_starts.get(warm_key),
                    )
                    self._cox_warm_starts[warm_key] = model.coef_.copy()
                    
                    # Validate
                    try:
                        risk_scores = model.predict_log_partial_hazard(X[val_idx])
                        oof_risk[val_idx] = risk_scores
                        c_index = concordance_index(
                            T[val_idx],
                            -risk_scores,
                            E[val_idx].astype(bool)
                        )
                        cv_scores.append(c_index)
                    except Exception:
                        cv_scores.append(0.0)

                if not cv_scores:
                    continue
                mean_cv_score = np.mean(cv_scores)
                
                if mean_cv_score > best_concordance:
                    best_concordance = mean_cv_score
                    best_alpha = alpha
                    best_oof_risk = oof_risk.copy()
                    
            except Exception as e:
                print(f"  Alpha={alpha} failed: {e}")
                continue
        
        # Fit final model with best alpha on full data
        if best_alpha is None:
            print(f"  All alphas failed, fitting with default alpha=0.01")
            best_alpha = 0.01

        full_key = (model_key, float(best_alpha), 'full')
        model = make_model(best_alpha)
        model.fit(
            X, T, E, weights=weights,
            initial_coef=self._cox_warm_starts.get(full_key),
        )
        self._cox_warm_starts[full_key] = model.coef_.copy()
        model.oof_risk_ = best_oof_risk
        model.cv_concordance_ = best_concordance
        
        return model, best_alpha
    
    def _predict_risk(self, model, X):
        """
        Predict log-risk scores using TorchCoxPH.
        
        Returns log partial hazard (higher = higher risk).
        """
        return model.predict_log_partial_hazard(X)
    
    def _policy_outputs(self, S, stochastic=False):
        """Return action weights used for risk and soft policy probabilities."""
        logits = self.policy_network.get_logits(S)
        soft_probs = torch.softmax(logits / self.gumbel_temperature, dim=1)
        if not self.hard_policy:
            return soft_probs, soft_probs
        if stochastic:
            action_weights = F.gumbel_softmax(
                logits, tau=self.gumbel_temperature, hard=True, dim=1
            )
        else:
            actions = torch.argmax(logits, dim=1)
            action_weights = F.one_hot(
                actions, num_classes=logits.shape[1]
            ).to(dtype=logits.dtype)
        return action_weights, soft_probs

    def _get_policy_probs(self, S, hard=False):
        """
        Get policy probabilities for state vectors.
        
        Parameters
        ----------
        S : ndarray
            State matrix (n_samples, 3)
            
        Returns
        -------
        probs : ndarray
            Policy probabilities (n_samples, 3)
        """
        self.policy_network.eval()
        with torch.no_grad():
            S_tensor = torch.as_tensor(
                np.ascontiguousarray(S), dtype=torch.float32, device=self.device
            )
            action_weights, soft_probs = self._policy_outputs(
                S_tensor, stochastic=False
            )
            output = action_weights if hard else soft_probs
            return output.detach().cpu().numpy()
    
    def _train_policy_epoch(self, S, R, P, RP, E, T, rp_cost=0.0):
        """Train one full-risk-set policy epoch."""
        self.policy_network.train()
        action_weights, soft_probs = self._policy_outputs(S, stochastic=True)
        base_loss = self.policy_loss_fn(
            action_weights, R, P, RP, E, T,
            regularization_probs=soft_probs
        )
        rp_penalty = (
            self.rp_cost_weight * rp_cost * action_weights[:, 2].mean()
        )
        loss = base_loss + rp_penalty

        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # Update stochastic/adaptive schedules from training only. Validation
        # and objective reporting must be side-effect free.
        if hasattr(self.policy_loss_fn, 'update_exploration_weight'):
            self.policy_loss_fn.update_exploration_weight(loss.detach())
        if hasattr(self.policy_loss_fn, 'update_parameters'):
            self.policy_loss_fn.update_parameters()
        if self.hard_policy:
            self.gumbel_temperature = max(
                self.gumbel_min_temperature,
                self.gumbel_temperature * self.gumbel_anneal_rate,
            )

        return loss.item()

    def _fit_policy_network(self, S, R, P, RP, E, T, train_idx, val_idx,
                            rp_cost, verbose=True):
        """Fit policy with a fixed validation set and restore one checkpoint."""
        train_tensors = [
            torch.as_tensor(np.ascontiguousarray(values[train_idx]),
                            dtype=torch.float32, device=self.device)
            for values in (S, R, P, RP, E, T)
        ]
        val_tensors = [
            torch.as_tensor(np.ascontiguousarray(values[val_idx]),
                            dtype=torch.float32, device=self.device)
            for values in (S, R, P, RP, E, T)
        ]
        best_val_loss = float('inf')
        best_checkpoint = None
        patience_counter = 0
        patience = 10

        for epoch in range(self.policy_epochs):
            train_loss = self._train_policy_epoch(
                *train_tensors, rp_cost=rp_cost
            )

            self.policy_network.eval()
            with torch.no_grad():
                S_val, R_val, P_val, RP_val, E_val, T_val = val_tensors
                action_val, soft_val = self._policy_outputs(
                    S_val, stochastic=False
                )
                components = self.policy_loss_fn(
                    action_val, R_val, P_val, RP_val, E_val, T_val,
                    return_components=True, regularization_probs=soft_val
                )
                rp_penalty = (
                    self.rp_cost_weight * rp_cost * action_val[:, 2].mean()
                )
                val_loss = (components['total_loss'] + rp_penalty).item()
                exploitation_loss = components['cox_loss'].item()
                exploration_loss = (
                    components['total_loss'].item() - exploitation_loss
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = {
                    'policy': {
                        key: value.detach().cpu().clone()
                        for key, value in self.policy_network.state_dict().items()
                    },
                    'optimizer': copy.deepcopy(self.policy_optimizer.state_dict()),
                    'loss_fn': copy.deepcopy(self.policy_loss_fn),
                    'gumbel_temperature': self.gumbel_temperature,
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(
                    f"  Epoch {epoch + 1}/{self.policy_epochs}: "
                    f"Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                    f"Exploitation Loss = {exploitation_loss:.4f}, "
                    f"Exploration Loss = {exploration_loss:.4f}, "
                    f"RP Penalty = {rp_penalty.item():.4f}, "
                    f"Gumbel T = {self.gumbel_temperature:.3f}"
                )

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        if best_checkpoint is None:
            raise RuntimeError("Policy training did not produce a valid checkpoint")
        self.policy_network.load_state_dict(best_checkpoint['policy'])
        self.policy_optimizer.load_state_dict(best_checkpoint['optimizer'])
        self.policy_loss_fn = best_checkpoint['loss_fn']
        self.gumbel_temperature = best_checkpoint['gumbel_temperature']
        return best_val_loss
    
    def _init_policy_network(self):
        """Initialize the policy network and optimizer."""
        self.policy_network = PolicyNetwork(
            input_dim=3,
            hidden_dim=self.hidden_dim,
            output_dim=3,
            dropout_rate=0.1
        ).to(self.device)
        
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        self._init_loss_function()
    
    def fit(self, X_rad, X_path, y):
        """
        Fit the Contextual Bandit with EM algorithm.
        
        Parameters
        ----------
        X_rad : ndarray
            Radiomic features
        X_path : ndarray
            Pathomic features
        y : structured array or DataFrame
            Survival data with fields 'event' and 'duration'
            
        Returns
        -------
        self : ContextualBandit
            Fitted instance
        """
        # Reset fit-specific state while retaining warm starts within this fit.
        self._cox_warm_starts = {}
        self.gumbel_temperature = self.gumbel_initial_temperature
        self.objective_history = []
        self.cindex_history = []
        self.rp_cost_history = []
        self.policies = []

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        X_rad = np.ascontiguousarray(X_rad, dtype=np.float32)
        X_path = np.ascontiguousarray(X_path, dtype=np.float32)

        # Extract survival data
        if isinstance(y, pd.DataFrame):
            T_train = y["duration"].values
            E_train = y["event"].values.astype(bool)
        else:
            T_train = y["duration"]
            E_train = y["event"].astype(bool)

        # Structured-array fields are often strided views whose strides are
        # incompatible with torch.FloatTensor. Materialize compact arrays once
        # before policy and Cox training.
        T_train = np.ascontiguousarray(T_train, dtype=np.float32)
        E_train = np.ascontiguousarray(E_train, dtype=np.float32)
        
        N_train = len(T_train)
        
        # ============================================================
        # STEP 1: INITIALIZATION - Train Global Cox Models
        # ============================================================
        
        print("Initializing global Cox models...")
        print(f"Training Radiomic model...")
        self.cox_rad, _ = self.train_survival_model(
            X_rad, T_train, E_train, alpha_range=self.alpha_range,
            model_key='radiomics'
        )
        R_train = self._predict_risk(self.cox_rad, X_rad)
        
        print(f"Training Pathomic model...")
        self.cox_path, _ = self.train_survival_model(
            X_path, T_train, E_train, alpha_range=self.alpha_range,
            model_key='pathomics'
        )
        P_train = self._predict_risk(self.cox_path, X_path)
        
        print(f"Training Radiopathomics model...")
        X_rp = np.concatenate([X_rad, X_path], axis=1)
        self.cox_rp, _ = self.train_survival_model(
            X_rp, T_train, E_train, alpha_range=self.alpha_range,
            model_key='radiopathomics'
        )
        RP_train = self._predict_risk(self.cox_rp, X_rp)
        
        # Store initial models
        self.models_rad = [self.cox_rad]
        self.models_path = [self.cox_path]
        self.models_rp = [self.cox_rp]
        
        # Current risk scores
        self.R_curr = R_train.copy()
        self.P_curr = P_train.copy()
        self.RP_curr = RP_train.copy()
        
        # Initialize policy network
        self._init_policy_network()

        # One fixed split supports comparable early stopping, RP evidence, and
        # EM checkpoint selection. Expert OOF risks keep RP evidence independent
        # of the samples used to fit each corresponding Cox fold model.
        indices = np.arange(N_train)
        policy_train_idx, policy_val_idx = train_test_split(
            indices, test_size=0.2, random_state=self.random_state
        )
        bootstrap_seed = 0 if self.random_state is None else self.random_state
        bootstrap_rng = np.random.default_rng(bootstrap_seed)
        if self.rp_bootstrap_samples > 0:
            fixed_bootstrap_indices = bootstrap_rng.integers(
                0, N_train,
                size=(self.rp_bootstrap_samples, N_train)
            )
        else:
            fixed_bootstrap_indices = None

        self.policy_train_indices_ = policy_train_idx.copy()
        self.policy_val_indices_ = policy_val_idx.copy()

        # Frozen global experts provide a stationary routing context. Their OOF
        # predictions and calibration never change during EM.
        self.routing_cox_rad = copy.deepcopy(self.cox_rad)
        self.routing_cox_path = copy.deepcopy(self.cox_path)
        self.routing_cox_rp = copy.deepcopy(self.cox_rp)
        routing_raw_oof = {
            'R': self.routing_cox_rad.oof_risk_.copy(),
            'P': self.routing_cox_path.oof_risk_.copy(),
            'RP': self.routing_cox_rp.oof_risk_.copy(),
        }
        self.routing_calibrators = {
            name: self._fit_horizon_calibrator(
                risk, T_train, E_train, policy_train_idx
            )
            for name, risk in routing_raw_oof.items()
        }
        routing_oof = {
            name: self._apply_horizon_calibrator(
                risk, self.routing_calibrators[name]
            )
            for name, risk in routing_raw_oof.items()
        }
        routing_full = {
            'R': self._apply_horizon_calibrator(
                R_train, self.routing_calibrators['R']
            ),
            'P': self._apply_horizon_calibrator(
                P_train, self.routing_calibrators['P']
            ),
            'RP': self._apply_horizon_calibrator(
                RP_train, self.routing_calibrators['RP']
            ),
        }
        fixed_oof_policy_state = self._make_policy_state(
            routing_oof['R'], routing_oof['P'], routing_oof['RP']
        )
        fixed_full_policy_state = self._make_policy_state(
            routing_full['R'], routing_full['P'], routing_full['RP']
        )
        self.anchor_expert_cindices_ = {
            name: self._risk_cindex(
                risk[policy_train_idx], E_train[policy_train_idx],
                T_train[policy_train_idx]
            )
            for name, risk in routing_raw_oof.items()
        }
        self.current_expert_cindices_ = self.anchor_expert_cindices_.copy()
        self.w_rad = np.full(N_train, 1.0 / 3.0, dtype=np.float32)
        self.w_path = np.full(N_train, 1.0 / 3.0, dtype=np.float32)
        self.w_rp = np.full(N_train, 1.0 / 3.0, dtype=np.float32)
        minimum_expert_ess = min(
            float(N_train),
            max(self.min_expert_ess, self.min_expert_ess_fraction * N_train),
        )
        self.training_cindex_history = []
        self.validation_cindex_history = []
        self.expert_diagnostics_history = []
        best_em_checkpoint = None
        best_em_cindex = -np.inf
        no_improvement = 0
        em_patience = 2
        experts_updated_since_policy = True

        def capture_checkpoint(validation_cindex):
            return {
                'cox_rad': copy.deepcopy(self.cox_rad),
                'cox_path': copy.deepcopy(self.cox_path),
                'cox_rp': copy.deepcopy(self.cox_rp),
                'policy': {
                    key: value.detach().cpu().clone()
                    for key, value in self.policy_network.state_dict().items()
                },
                'optimizer': copy.deepcopy(self.policy_optimizer.state_dict()),
                'loss_fn': copy.deepcopy(self.policy_loss_fn),
                'calibrators': copy.deepcopy(self.policy_calibrators),
                'routing_cox_rad': copy.deepcopy(self.routing_cox_rad),
                'routing_cox_path': copy.deepcopy(self.routing_cox_path),
                'routing_cox_rp': copy.deepcopy(self.routing_cox_rp),
                'routing_calibrators': copy.deepcopy(self.routing_calibrators),
                'expert_cindices': copy.deepcopy(self.current_expert_cindices_),
                'gumbel_temperature': self.gumbel_temperature,
                'w_rad': self.w_rad.copy(),
                'w_path': self.w_path.copy(),
                'w_rp': self.w_rp.copy(),
                'validation_cindex': validation_cindex,
            }

        def experts_meet_quality_guard():
            return all(
                self.current_expert_cindices_[name]
                >= self.anchor_expert_cindices_[name] - self.expert_anchor_tolerance
                for name in ('R', 'P', 'RP')
            )

        def prepare_rp_cost():
            oof_risks = (
                self.cox_rad.oof_risk_,
                self.cox_path.oof_risk_,
                self.cox_rp.oof_risk_,
            )
            if any(risk is None for risk in oof_risks):
                raise RuntimeError("OOF expert risks are required for RP cost")
            R_oof, P_oof, RP_oof = oof_risks
            if not all(np.isfinite(risk).all() for risk in oof_risks):
                raise RuntimeError("OOF expert risks contain non-finite values")
            return self._compute_rp_cost(
                R_oof, P_oof, RP_oof, E_train, T_train,
                bootstrap_indices=fixed_bootstrap_indices
            )

        def fit_aligned_policy(verbose=True):
            rp_cost, rp_cost_info = prepare_rp_cost()

            raw_oof = {
                'R': self.cox_rad.oof_risk_,
                'P': self.cox_path.oof_risk_,
                'RP': self.cox_rp.oof_risk_,
            }
            self.policy_calibrators = {
                name: self._fit_horizon_calibrator(
                    risk, T_train, E_train, policy_train_idx
                )
                for name, risk in raw_oof.items()
            }

            # The policy is trained and validated entirely on OOF expert risks.
            # Full-fit risks are used only to obtain EM assignment weights and,
            # after fitting, to make predictions for new patients.
            R_for_fit = self._apply_horizon_calibrator(
                raw_oof['R'], self.policy_calibrators['R']
            )
            P_for_fit = self._apply_horizon_calibrator(
                raw_oof['P'], self.policy_calibrators['P']
            )
            RP_for_fit = self._apply_horizon_calibrator(
                raw_oof['RP'], self.policy_calibrators['RP']
            )
            S_for_fit = fixed_oof_policy_state

            R_policy = self._apply_horizon_calibrator(
                self.R_curr, self.policy_calibrators['R']
            )
            P_policy = self._apply_horizon_calibrator(
                self.P_curr, self.policy_calibrators['P']
            )
            RP_policy = self._apply_horizon_calibrator(
                self.RP_curr, self.policy_calibrators['RP']
            )
            S_policy = fixed_full_policy_state

            self.rp_cost_history.append(rp_cost)
            if verbose:
                print(
                    f"RP evidence cost: {rp_cost:.4f} "
                    f"(OOF C-index R={rp_cost_info['cindex_rad']:.4f}, "
                    f"P={rp_cost_info['cindex_path']:.4f}, "
                    f"RP={rp_cost_info['cindex_rp']:.4f}; "
                    f"lower gains RP-R={rp_cost_info['lower_gain_vs_rad']:.4f}, "
                    f"RP-P={rp_cost_info['lower_gain_vs_path']:.4f})"
                )
                print(
                    "5-year calibration slopes - "
                    f"R: {self.policy_calibrators['R']['slope']:.3f}, "
                    f"P: {self.policy_calibrators['P']['slope']:.3f}, "
                    f"RP: {self.policy_calibrators['RP']['slope']:.3f}"
                )
            best_val_loss = self._fit_policy_network(
                S_for_fit, R_for_fit, P_for_fit, RP_for_fit, E_train, T_train,
                policy_train_idx, policy_val_idx, rp_cost, verbose=verbose
            )
            action_weights = self._get_policy_probs(
                S_policy, hard=self.hard_policy
            )
            soft_probs = self._get_policy_probs(S_policy, hard=False)
            action_val = self._get_policy_probs(
                S_for_fit[policy_val_idx], hard=self.hard_policy
            )
            calibrated_val_risk = (
                action_val[:, 0] * R_for_fit[policy_val_idx]
                + action_val[:, 1] * P_for_fit[policy_val_idx]
                + action_val[:, 2] * RP_for_fit[policy_val_idx]
            )
            validation_cindex = concordance_index(
                T_train[policy_val_idx], -calibrated_val_risk,
                E_train[policy_val_idx].astype(bool)
            )
            return {
                'S': S_policy,
                'R': R_policy,
                'P': P_policy,
                'RP': RP_policy,
                'probs': action_weights,
                'soft_probs': soft_probs,
                'rp_cost': rp_cost,
                'val_loss': best_val_loss,
                'val_cindex': validation_cindex,
            }

        def fit_guarded_expert(name, X, weights, current_model, model_key):
            """Fit an M-step candidate and reject material OOF degradation."""
            warm_starts_before = copy.deepcopy(self._cox_warm_starts)
            previous_cindex = self.current_expert_cindices_[name]
            anchor_cindex = self.anchor_expert_cindices_[name]
            try:
                candidate, _ = self.train_survival_model(
                    X, T_train, E_train, weights=weights,
                    alpha_range=self.alpha_range, model_key=model_key
                )
                candidate_cindex = self._risk_cindex(
                    candidate.oof_risk_[policy_train_idx],
                    E_train[policy_train_idx], T_train[policy_train_idx]
                )
                candidate_calibrator = self._fit_horizon_calibrator(
                    candidate.oof_risk_, T_train, E_train, policy_train_idx
                )
                calibration_is_finite = all(np.isfinite([
                    candidate_calibrator['slope'],
                    candidate_calibrator['baseline_cumulative_hazard'],
                ]))
                accepted, threshold = self._expert_candidate_is_acceptable(
                    candidate_cindex, previous_cindex, anchor_cindex,
                    calibration_is_finite,
                )
            except Exception as error:
                candidate = None
                candidate_cindex = float('nan')
                accepted = False
                _, threshold = self._expert_candidate_is_acceptable(
                    candidate_cindex, previous_cindex, anchor_cindex, False
                )
                print(f"    {name} candidate failed quality evaluation: {error}")

            if accepted:
                selected_model = candidate
                selected_cindex = candidate_cindex
            else:
                self._cox_warm_starts = warm_starts_before
                selected_model = current_model
                selected_cindex = previous_cindex

            selected_risk = self._predict_risk(selected_model, X)
            return selected_model, selected_risk, {
                'accepted': accepted,
                'previous_cindex': previous_cindex,
                'candidate_cindex': candidate_cindex,
                'selected_cindex': selected_cindex,
                'threshold': threshold,
                'ess': self._effective_sample_size(weights),
            }
        
        # ============================================================
        # STEP 2: EM LOOP
        # ============================================================
        
        print("\nStarting EM iterations...")
        
        for iteration in range(self.max_iterations):
            print(f"\n--- EM Iteration {iteration + 1} ---")
            print(f"Training policy network for {self.policy_epochs} epochs...")
            aligned = fit_aligned_policy(verbose=True)
            experts_updated_since_policy = False
            policy_probs = aligned['probs']
            floor = self.min_expert_weight
            m_step_probs = policy_probs * (1.0 - 3.0 * floor) + floor
            next_w_rad = self._stabilize_expert_weights(
                self.w_rad, m_step_probs[:, 0], minimum_expert_ess
            )
            next_w_path = self._stabilize_expert_weights(
                self.w_path, m_step_probs[:, 1], minimum_expert_ess
            )
            next_w_rp = self._stabilize_expert_weights(
                self.w_rp, m_step_probs[:, 2], minimum_expert_ess
            )

            self.validation_cindex_history.append(aligned['val_cindex'])
            self.cindex_history.append(aligned['val_cindex'])
            self.objective_history.append(-aligned['val_loss'])
            self.policies.append({
                key: value.detach().cpu().clone()
                for key, value in self.policy_network.state_dict().items()
            })
            self.models_rad.append(self.cox_rad)
            self.models_path.append(self.cox_path)
            self.models_rp.append(self.cox_rp)
            print(f"Aligned validation C-index: {aligned['val_cindex']:.4f}")

            previous_best = best_em_cindex
            if (
                aligned['val_cindex'] > best_em_cindex
                and experts_meet_quality_guard()
            ):
                best_em_cindex = aligned['val_cindex']
                best_em_checkpoint = capture_checkpoint(aligned['val_cindex'])
            if aligned['val_cindex'] > previous_best + self.convergence_threshold:
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= em_patience:
                print("EM convergence reached on fixed validation C-index")
                break

            # Commit proposed weights only when an M-step will actually use
            # them. Checkpoints above therefore remain synchronized with the
            # weights that fitted their stored expert models.
            previous_expert_weights = {
                'R': self.w_rad.copy(),
                'P': self.w_path.copy(),
                'RP': self.w_rp.copy(),
            }
            self.w_rad = next_w_rad
            self.w_path = next_w_path
            self.w_rp = next_w_rp
            
            # ============================================================
            # M-STEP: Train Weighted Cox Models
            # ============================================================
            
            print(f"Training weighted Cox models...")
            
            # Radiomic model with weights
            print(f"  Weighted Radiomic model...")
            self.cox_rad, R_new, rad_diagnostics = fit_guarded_expert(
                'R', X_rad, self.w_rad, self.cox_rad, 'radiomics'
            )
            
            # Pathomic model with weights
            print(f"  Weighted Pathomic model...")
            self.cox_path, P_new, path_diagnostics = fit_guarded_expert(
                'P', X_path, self.w_path, self.cox_path, 'pathomics'
            )
            
            # Fusion model with weights
            print(f"  Weighted Fusion (RP) model...")
            self.cox_rp, RP_new, rp_diagnostics = fit_guarded_expert(
                'RP', X_rp, self.w_rp, self.cox_rp, 'radiopathomics'
            )
            iteration_diagnostics = {
                'iteration': iteration + 1,
                'R': rad_diagnostics,
                'P': path_diagnostics,
                'RP': rp_diagnostics,
            }
            self.expert_diagnostics_history.append(iteration_diagnostics)
            self.current_expert_cindices_ = {
                name: diagnostics['selected_cindex']
                for name, diagnostics in (
                    ('R', rad_diagnostics),
                    ('P', path_diagnostics),
                    ('RP', rp_diagnostics),
                )
            }
            if not rad_diagnostics['accepted']:
                self.w_rad = previous_expert_weights['R']
            if not path_diagnostics['accepted']:
                self.w_path = previous_expert_weights['P']
            if not rp_diagnostics['accepted']:
                self.w_rp = previous_expert_weights['RP']
            for name, diagnostics in (
                ('R', rad_diagnostics),
                ('P', path_diagnostics),
                ('RP', rp_diagnostics),
            ):
                status = "accepted" if diagnostics['accepted'] else "rejected"
                print(
                    f"    {name}: {status}; candidate OOF C-index="
                    f"{diagnostics['candidate_cindex']:.4f}, "
                    f"selected={diagnostics['selected_cindex']:.4f}, "
                    f"ESS={diagnostics['ess']:.1f}"
                )
            
            # ============================================================
            # UPDATE RISK SCORES
            # ============================================================
            
            self.R_curr = R_new
            self.P_curr = P_new
            self.RP_curr = RP_new
            experts_updated_since_policy = True
            
            # ============================================================
            # EVALUATE AND CHECK CONVERGENCE
            # ============================================================
            
            self.policy_calibrators = {
                name: self._fit_horizon_calibrator(
                    model.oof_risk_, T_train, E_train, policy_train_idx
                )
                for name, model in (
                    ('R', self.cox_rad),
                    ('P', self.cox_path),
                    ('RP', self.cox_rp),
                )
            }
            R_eval = self._apply_horizon_calibrator(
                self.R_curr, self.policy_calibrators['R']
            )
            P_eval = self._apply_horizon_calibrator(
                self.P_curr, self.policy_calibrators['P']
            )
            RP_eval = self._apply_horizon_calibrator(
                self.RP_curr, self.policy_calibrators['RP']
            )
            h_weighted = (
                self.w_rad * R_eval + self.w_path * P_eval + self.w_rp * RP_eval
            )
            training_cindex = concordance_index(T_train, -h_weighted, E_train)
            self.training_cindex_history.append(training_cindex)
            print(f"Training C-index after M-step: {training_cindex:.4f}")
            
            # Print mutually exclusive policy assignments. The M-step still
            # uses the complete soft weights above.
            expert_weights = np.column_stack([
                self.w_rad, self.w_path, self.w_rp
            ])
            subgroup_counts = np.bincount(
                np.argmax(expert_weights, axis=1), minlength=3
            )
            print(f"Argmax subgroup sizes - Rad: {subgroup_counts[0]}, "
                  f"Path: {subgroup_counts[1]}, RP: {subgroup_counts[2]}")
            print(f"Mean expert weights - Rad: {self.w_rad.mean():.3f}, "
                  f"Path: {self.w_path.mean():.3f}, RP: {self.w_rp.mean():.3f}")
            

        # A final E-step aligns the policy with experts produced by the last
        # M-step. It is skipped when convergence stopped before another M-step.
        if experts_updated_since_policy:
            print("\nFinal policy calibration on the last Cox experts...")
            aligned = fit_aligned_policy(verbose=False)
            self.validation_cindex_history.append(aligned['val_cindex'])
            self.cindex_history.append(aligned['val_cindex'])
            self.objective_history.append(-aligned['val_loss'])
            if (
                aligned['val_cindex'] > best_em_cindex
                and experts_meet_quality_guard()
            ):
                best_em_cindex = aligned['val_cindex']
                best_em_checkpoint = capture_checkpoint(aligned['val_cindex'])

        if best_em_checkpoint is None:
            raise RuntimeError("EM did not produce a valid synchronized checkpoint")
        self.cox_rad = best_em_checkpoint['cox_rad']
        self.cox_path = best_em_checkpoint['cox_path']
        self.cox_rp = best_em_checkpoint['cox_rp']
        self.policy_network.load_state_dict(best_em_checkpoint['policy'])
        self.policy_optimizer.load_state_dict(best_em_checkpoint['optimizer'])
        self.policy_loss_fn = best_em_checkpoint['loss_fn']
        self.policy_calibrators = best_em_checkpoint['calibrators']
        self.routing_cox_rad = best_em_checkpoint['routing_cox_rad']
        self.routing_cox_path = best_em_checkpoint['routing_cox_path']
        self.routing_cox_rp = best_em_checkpoint['routing_cox_rp']
        self.routing_calibrators = best_em_checkpoint['routing_calibrators']
        self.current_expert_cindices_ = best_em_checkpoint['expert_cindices']
        self.gumbel_temperature = best_em_checkpoint['gumbel_temperature']
        self.w_rad = best_em_checkpoint['w_rad']
        self.w_path = best_em_checkpoint['w_path']
        self.w_rp = best_em_checkpoint['w_rp']

        print(f"\nEM completed. Best validation C-index: {best_em_cindex:.4f}")
        print(f"Aligned policy evaluations: {len(self.validation_cindex_history)}")
        
        return self
    
    def predict_risk(self, X_rad, X_path):
        """
        Predict risk scores for new patients using the learned policy.
        
        Parameters
        ----------
        X_rad : ndarray
            Radiomic features
        X_path : ndarray
            Pathomic features
            
        Returns
        -------
        risk_scores : ndarray
            Selected calibrated log cumulative hazards at five years
        actions : ndarray
            Selected actions for each patient
        probs : ndarray
            Policy probabilities for each action
        """
        # Get risk scores from each model
        R = self._predict_risk(self.cox_rad, X_rad)
        P = self._predict_risk(self.cox_path, X_path)
        X_rp = np.concatenate([X_rad, X_path], axis=1)
        RP = self._predict_risk(self.cox_rp, X_rp)

        R, P, RP = self._calibrate_expert_risks(R, P, RP)
        S = self._make_routing_state_from_features(X_rad, X_path)
        
        # Get policy probabilities
        probs = self._get_policy_probs(S)
        actions = np.argmax(probs, axis=1)
        
        # Compute final risk scores
        N = len(R)
        risk_scores = np.zeros(N)
        for i in range(N):
            if actions[i] == 0:
                risk_scores[i] = R[i]
            elif actions[i] == 1:
                risk_scores[i] = P[i]
            else:
                risk_scores[i] = RP[i]
        
        return risk_scores, actions, probs

    def predict_survival_probability(self, X_rad, X_path):
        """Predict calibrated five-year survival under the selected expert."""
        risk_scores, actions, probs = self.predict_risk(X_rad, X_path)
        cumulative_hazard = np.exp(np.clip(risk_scores, -50.0, 50.0))
        survival = np.exp(-cumulative_hazard).astype(np.float32)
        return survival, actions, probs
    
    def get_subgroup_probabilities(self, X_rad, X_path):
        """
        Get soft subgroup assignment probabilities for new patients.
        """
        S = self._make_routing_state_from_features(X_rad, X_path)
        return self._get_policy_probs(S)
    
    def get_weighted_risk(self, X_rad, X_path):
        """
        Get weighted risk scores using policy probabilities directly (soft ensemble).
        """
        R = self._predict_risk(self.cox_rad, X_rad)
        P = self._predict_risk(self.cox_path, X_path)
        X_rp = np.concatenate([X_rad, X_path], axis=1)
        RP = self._predict_risk(self.cox_rp, X_rp)

        R, P, RP = self._calibrate_expert_risks(R, P, RP)
        S = self._make_routing_state_from_features(X_rad, X_path)
        probs = self._get_policy_probs(S)
        
        # Weighted risk = sum(prob * risk)
        risk_scores = probs[:, 0] * R + probs[:, 1] * P + probs[:, 2] * RP
        
        return risk_scores, probs


class ContextualBanditPipeline:
    """
    Pipeline wrapper for Contextual Bandit with survival prediction.
    """
    
    def __init__(self, bandit, use_soft_ensemble=False,
                 radiomics_scaler=None, pathomics_scaler=None):
        self.bandit = bandit
        self.use_soft_ensemble = use_soft_ensemble
        self.radiomics_scaler = (
            StandardScaler() if radiomics_scaler is None else radiomics_scaler
        )
        self.pathomics_scaler = (
            StandardScaler() if pathomics_scaler is None else pathomics_scaler
        )
        self.risk_scores_ = None
        self.actions_ = None
        self.probs_ = None

    @staticmethod
    def _as_feature_matrix(X):
        values = X.values if hasattr(X, 'values') else X
        return np.ascontiguousarray(values, dtype=np.float32)

    def _transform_inputs(self, X_rad, X_path):
        if not hasattr(self.radiomics_scaler, 'mean_'):
            raise RuntimeError("ContextualBanditPipeline must be fitted first")
        X_rad = self._as_feature_matrix(X_rad)
        X_path = self._as_feature_matrix(X_path)
        return (
            np.ascontiguousarray(
                self.radiomics_scaler.transform(X_rad), dtype=np.float32
            ),
            np.ascontiguousarray(
                self.pathomics_scaler.transform(X_path), dtype=np.float32
            ),
        )
    
    def fit(self, X_rad, X_path, y):
        X_rad = self._as_feature_matrix(X_rad)
        X_path = self._as_feature_matrix(X_path)
        X_rad_scaled = np.ascontiguousarray(
            self.radiomics_scaler.fit_transform(X_rad), dtype=np.float32
        )
        X_path_scaled = np.ascontiguousarray(
            self.pathomics_scaler.fit_transform(X_path), dtype=np.float32
        )
        self.bandit.fit(X_rad_scaled, X_path_scaled, y)
        return self
    
    def transform(self, X_rad, X_path):
        X_rad, X_path = self._transform_inputs(X_rad, X_path)
        if self.use_soft_ensemble:
            risk_scores, probs = self.bandit.get_weighted_risk(X_rad, X_path)
            self.probs_ = probs
            # Diagnostic hard assignments remain available even though risk
            # prediction uses the complete soft probability distribution.
            self.actions_ = np.argmax(probs, axis=1)
        else:
            risk_scores, actions, probs = self.bandit.predict_risk(X_rad, X_path)
            self.actions_ = actions
            self.probs_ = probs
        
        self.risk_scores_ = risk_scores
        return risk_scores
    
    def fit_transform(self, X_rad, X_path, y):
        self.fit(X_rad, X_path, y)
        return self.transform(X_rad, X_path)
    
    def get_subgroup_probs(self, X_rad, X_path):
        X_rad, X_path = self._transform_inputs(X_rad, X_path)
        return self.bandit.get_subgroup_probabilities(X_rad, X_path)
    
    def get_cindex_history(self):
        return self.bandit.cindex_history
    
    def get_objective_history(self):
        return self.bandit.objective_history

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
            epochs=model_params.get('vae_epochs', 200),
            batch_size=model_params.get('vae_batch_size', 64),
            learning_rate=model_params.get('vae_learning_rate', 1e-3),
            beta_kl=model_params.get('vae_beta_kl', 1e-3),
            beta_domain=model_params.get('vae_beta_domain', 10),
            beta_recon=model_params.get('vae_beta_recon', 1.0),
            beta_cross=model_params.get('vae_beta_cross', 1.0),
            kl_annealing_steps=model_params.get('kl_annealing_steps', 100),
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
    
    def strategy_6_EM_Contextual_Bandit(self, split, split_idx, omics_params, model_params):
        """
        Strategy 6: EM + Contextual Bandit for Dynamic Subgroup Discovery and Fusion.
        
        This strategy implements the iterative EM framework with a contextual bandit policy
        to learn soft subgroup assignments (Radiomics-Best, Pathomics-Best, Fusion-Best)
        and train specialized weighted Cox models for each subgroup.
        
        Parameters
        ----------
        split : object
            Data split object
        split_idx : int
            Split index for reproducibility
        omics_params : dict
            Parameters for loading omics data
        model_params : dict
            Model parameters including:
            - alpha : float, regularization for Cox models
            - em_max_iterations : int, maximum EM iterations
            - convergence_threshold : float, convergence tolerance
            - feature_selection : bool, whether to apply feature selection
            - feature_var_threshold : float, variance threshold
        fusion_method : str, default='average'
            Fusion method (unused, kept for compatibility)
            
        Returns
        -------
        dict : Results containing predictions, scores, and metadata
        """
        print(f"\n=== Strategy 6: EM + Contextual Bandit ===")
        
        # ============================================================
        # STEP 1: LOAD DATA
        # ============================================================
        
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
        
        X_rad_train = tr_X_radio.values if hasattr(tr_X_radio, 'values') else tr_X_radio
        X_path_train = tr_X_patho.values if hasattr(tr_X_patho, 'values') else tr_X_patho
        X_rad_test = te_X_radio.values if hasattr(te_X_radio, 'values') else te_X_radio
        X_path_test = te_X_patho.values if hasattr(te_X_patho, 'values') else te_X_patho
        X_rad_raw = raw_te_X_radio.values if hasattr(raw_te_X_radio, 'values') else raw_te_X_radio
        X_path_raw = raw_te_X_patho.values if hasattr(raw_te_X_patho, 'values') else raw_te_X_patho
        
        # Initialize bandit
        bandit = ContextualBandit(
            alpha_range=[0.001, 0.01, 0.1, 1.0],
            max_iterations=20,
            hidden_dim=8,
            learning_rate=0.01,
            batch_size=64,
            policy_epochs=50,
            loss_type='adaptive',  # 'weighted', 'bayesian', 'adaptive', 'ensemble'
            exploration_weight=0.2,
            entropy_weight=0.05,
            rp_cost_weight=0.0,
            temperature=1.0,
            hard_policy=True,
            gumbel_temperature=1.0,
            gumbel_min_temperature=0.1,
            gumbel_anneal_rate=0.95,
            device='cuda',
            random_state=42
        )

        # Create pipeline
        pipeline = ContextualBanditPipeline(bandit, use_soft_ensemble=False)

        # Fit
        pipeline.fit(X_rad_train, X_path_train, tr_y)

        # Test set predictions
        risk_scores = pipeline.transform(X_rad_test, X_path_test)
        actions = pipeline.actions_
        probs = pipeline.probs_
        if actions is None:
            if probs is None:
                raise RuntimeError("Contextual-bandit predictions returned no policy probabilities")
            actions = np.argmax(probs, axis=1)
        
        # Raw test set predictions
        raw_risk_scores = pipeline.transform(X_rad_raw, X_path_raw)
        # Keep the returned pipeline diagnostics aligned with the primary test
        # cohort rather than the auxiliary raw cohort predicted last.
        pipeline.risk_scores_ = risk_scores
        pipeline.actions_ = actions
        pipeline.probs_ = probs
        
        scores_dict, times = self.evaluate_predictions(tr_y, None, te_y, risk_scores, None)
        
        print(f"\nFinal Results:")
        print(f"  C-index: {scores_dict.get('C-index', 0):.4f}")
        print(f"  Policy action distribution: "
            f"Rad: {np.sum(actions == 0)}, "
            f"Path: {np.sum(actions == 1)}, "
            f"RP: {np.sum(actions == 2)}")
        print(f"  EM iterations: {len(bandit.cindex_history)}")
        
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
            'cindex_history': bandit.cindex_history,
            'actions': actions.tolist(),
            'policy_probs': probs.tolist(),
            'subgroup_weights': {
                'radiomics': np.mean(probs[:, 0]) if probs.shape[1] >= 1 else 0,
                'pathomics': np.mean(probs[:, 1]) if probs.shape[1] >= 2 else 0,
                'rp': np.mean(probs[:, 2]) if probs.shape[1] >= 3 else 0
            },
            'w_rad': bandit.w_rad.tolist() if bandit.w_rad is not None else None,
            'w_path': bandit.w_path.tolist() if bandit.w_path is not None else None,
            'w_rp': bandit.w_rp.tolist() if bandit.w_rp is not None else None
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
                # 'Strategy1_DirectConcat': self.strategy_1_direct_concat,
                # 'Strategy2_PCA_Concat': lambda s, idx, op, mp: self.strategy_2_pca_concat(s, idx, op, mp, n_pca_components),
                # 'Strategy3_Separate_Fusion': lambda s, idx, op, mp: self.strategy_3_separate_fusion(s, idx, op, mp, fusion_method),
                # 'Strategy4_PCA_Separate_Fusion': lambda s, idx, op, mp: self.strategy_4_pca_separate_fusion(s, idx, op, mp, n_pca_components, fusion_method),
                # 'Strategy5_Domain_Adaptation_Fusion': lambda s, idx, op, mp: self.strategy_5_domain_adaptation_fusion(s, idx, op, mp),
                'Strategy6_Contextual_Bandit_Fusion': lambda s, idx, op, mp: self.strategy_6_EM_Contextual_Bandit(s, idx, op, mp),
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

    if opt['TASKS']['TRAIN']:
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
