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

from scipy.special import softmax
from scipy.stats import zscore, ttest_ind
from torch_geometric.loader import DataLoader
from tiatoolbox import logger
from tiatoolbox.utils.misc import save_as_json

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold, SelectFromModel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer
from sklearn.metrics import balanced_accuracy_score as acc_scorer
from sklearn.metrics import f1_score as f1_scorer
from sklearn.decomposition import PCA
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

from utilities.m_utils import mkdir, load_json, create_pbar, rm_n_mkdir, reset_logging

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

def prepare_patient_outcome(outcome_dir, subject_ids, outcome=None, signature_ids=None, pooling='mean'):
    if outcome == "GeneProgrames":
        outcome_file = f"{outcome_dir}/signatures/gene_programs/gene_programs.csv"
        df = pd.read_csv(outcome_file)
        df["SubjectID"] = df["SampleID"].str.extract(r'^(TCGA-\w\w-\w{4})')
        df_matched = df[df["SubjectID"].isin(subject_ids)]
    elif outcome == "HRDscore":
        outcome_file = f"{outcome_dir}/signatures/HRD_score/HRD_score.csv"
        df = pd.read_csv(outcome_file)
        df["SubjectID"] = df["SampleID"].str.extract(r'^(TCGA-\w\w-\w{4})')
        df_matched = df[df["SubjectID"].isin(subject_ids)]
        df_matched = df_matched.applymap(
            lambda x: x / 100.0 if isinstance(x, (int, float)) else x
        )
    elif outcome == "ImmuneSignatureScore":
        outcome_file = f"{outcome_dir}/signatures/immune_signature_score/immune_signature_score.csv"
        df = pd.read_csv(outcome_file)
        df["SubjectID"] = df["SampleID"].str.extract(r'^(TCGA-\w\w-\w{4})')
        df_matched = df[df["SubjectID"].isin(subject_ids)]
    elif outcome == "StemnessScoreDNA":
        outcome_file = f"{outcome_dir}/signatures/stemness_score_DNA/stemness_scores_DNAmeth.csv"
        df = pd.read_csv(outcome_file)
        df["SubjectID"] = df["SampleID"].str.extract(r'^(TCGA-\w\w-\w{4})')
        df_matched = df[df["SubjectID"].isin(subject_ids)]
    elif outcome == "StemScoreRNA":
        outcome_file = f"{outcome_dir}/signatures/stemness_score_RNA/stemness_scores_RNAexp.csv"
        df = pd.read_csv(outcome_file)
        df["SubjectID"] = df["SampleID"].str.extract(r'^(TCGA-\w\w-\w{4})')
        df_matched = df[df["SubjectID"].isin(subject_ids)]
    else:
        raise NotImplementedError
    
    if signature_ids is not None:
        assert isinstance(signature_ids, list), "signature ids should be list"
        df_matched = df_matched[["SubjectID"] + signature_ids]

    if pooling == 'mean':
        df_matched = df_matched.groupby("SubjectID", as_index=False).mean(numeric_only=True)
    elif pooling == 'max':
        df_matched = df_matched.groupby("SubjectID", as_index=False).max(numeric_only=True)
    elif pooling == 'min':
        df_matched = df_matched.groupby("SubjectID", as_index=False).min(numeric_only=True)
    else:
        raise NotImplementedError
    logging.info(f"Phenotype data strcuture: {df_matched.shape}")

    clinical_file = f"{outcome_dir}/phenotypes/clinical_data/survival_data.csv"
    df = pd.read_csv(clinical_file)
    df_clinical = df[df["_PATIENT"].isin(df_matched["SubjectID"])]
    df_clinical = df_clinical.drop_duplicates(subset="_PATIENT", keep="first")
    df_clinical = df_clinical.set_index("_PATIENT").loc[df_matched["SubjectID"]].reset_index()
    logging.info(f"Clinical data strcuture: {df_clinical.shape}")

    return df_matched, df_clinical

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

def randomforest_regression(split_idx, tr_X, tr_y, refit, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = RandomForestRegressor(max_depth=2, random_state=42)

    # Grid search parameters
    param_grid = {"model__max_depth": list(range(1, 6))}

    # Scoring metrics for regression
    scoring = {
        "r2": "r2",
        "neg_mse": "neg_mean_squared_error",
    }

    assert scoring.get(refit, False), f"{refit} is unsupported"

    # Pipeline: scaling + model
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )

    # Grid search with cross validation
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=refit,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # Plot cross-validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    depths = cv_results["param_model__max_depth"]
    mean = cv_results[f"mean_test_{refit}"]
    std = cv_results[f"std_test_{refit}"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(depths, mean, marker="o")
    ax.fill_between(depths, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("linear")
    ax.set_ylabel(refit)
    ax.set_xlabel("max depth")
    ax.axvline(gcv.best_params_["model__max_depth"], color="C1", linestyle="--", label="Best depth")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{relative_path}/figures/plots/cross_validation_fold{split_idx}.jpg")
    plt.close(fig)

    # Fit final model with best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def linearregression(split_idx, tr_X, tr_y, refit, n_jobs, model_type="ridge"):
    """
    Cross-validated linear regression with support for Ridge, Lasso, or ElasticNet,
    including coefficient visualization for the best model.
    """
    # Validate model_type
    assert model_type in ["ridge", "lasso", "elasticnet"], "model_type must be one of ['ridge', 'lasso', 'elasticnet']"

    # 5-fold CV setup
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    # Model selection
    if model_type == "ridge":
        model = Ridge(random_state=42)
        param_grid = {"model__alpha": np.logspace(-3, 3, 7)}
    elif model_type == "lasso":
        model = Lasso(random_state=42, max_iter=5000)
        param_grid = {"model__alpha": np.logspace(-3, 3, 7)}
    else:  # ElasticNet
        model = ElasticNet(random_state=42, max_iter=5000)
        param_grid = {
            "model__alpha": np.logspace(-3, 3, 7),
            "model__l1_ratio": [0.2, 0.5, 0.8]
        }

    # Pipeline
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", model)
    ])

    # Scoring metrics
    scoring = {
        "r2": "r2",
        "neg_mse": "neg_mean_squared_error"
    }
    assert refit in scoring, f"{refit} is unsupported"

    # Grid search
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=refit,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # --- Plot CV results ---
    cv_results = pd.DataFrame(gcv.cv_results_)
    mean = cv_results[f"mean_test_{refit}"]
    std = cv_results[f"std_test_{refit}"]

    fig, ax = plt.subplots(figsize=(9, 6))
    if model_type == "elasticnet":
        for ratio in sorted(cv_results["param_model__l1_ratio"].unique()):
            subset = cv_results[cv_results["param_model__l1_ratio"] == ratio]
            ax.plot(subset["param_model__alpha"], subset[f"mean_test_{refit}"], label=f"l1_ratio={ratio}")
            ax.fill_between(subset["param_model__alpha"],
                            subset[f"mean_test_{refit}"] - subset[f"std_test_{refit}"],
                            subset[f"mean_test_{refit}"] + subset[f"std_test_{refit}"], alpha=0.1)
    else:
        ax.plot(cv_results["param_model__alpha"], mean, marker="o")
        ax.fill_between(cv_results["param_model__alpha"], mean - std, mean + std, alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("alpha (regularization strength)")
    ax.set_ylabel(refit)
    ax.axvline(gcv.best_params_["model__alpha"], color="C1", linestyle="--", label="Best alpha")
    ax.legend()
    ax.grid(True)
    ax.set_title(f"{model_type.capitalize()} Regression CV ({refit})")

    plt.tight_layout()
    plt.savefig(f"{relative_path}/figures/plots/cross_validation_fold{split_idx}.jpg")
    plt.close(fig)

    # --- Fit final model with best parameters ---
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    # --- Visualize coefficients of best model ---
    best_model = pipe.named_steps["model"]

    if best_model.coef_.ndim == 1:
        best_coefs = pd.DataFrame(best_model.coef_, index=tr_X.columns, columns=["target"])
    else:
        target_names = tr_y.columns if hasattr(tr_y, "columns") else [f"target_{i}" for i in range(best_model.coef_.shape[0])]
        best_coefs = pd.DataFrame(best_model.coef_.T, index=tr_X.columns, columns=target_names)

    # Select only features with any non-zero coefficient
    non_zero_mask = (best_coefs != 0).any(axis=1)
    non_zero_coefs = best_coefs[non_zero_mask]

    # Sort features by max absolute coefficient across targets and take top 10
    coef_order = non_zero_coefs.abs().max(axis=1).sort_values(ascending=False).head(10).index
    top10_coefs = non_zero_coefs.loc[coef_order]

    # Plot setup
    y = np.arange(len(top10_coefs))[::-1]
    num_targets = len(top10_coefs.columns)
    bar_width = 0.8 / num_targets

    fig, ax = plt.subplots(figsize=(10, max(6, len(top10_coefs) * 0.5)))

    # Use a diverging colormap
    cmap = plt.get_cmap("seismic")

    for i, target in enumerate(top10_coefs.columns):
        coef_values = top10_coefs[target]

        # Map target index to colormap: i=0 -> 0.2, i=max -> 0.8
        norm_i = 0.2 + 0.6 * i / max(1, num_targets-1)  # avoid 0/0 if one target
        colors = [cmap(norm_i) for _ in coef_values]
        ax.barh(y + i*bar_width, coef_values, height=bar_width, color=colors, edgecolor="k", label=target)

    ax.set_yticks(y + bar_width*(num_targets-1)/2)
    ax.set_yticklabels(top10_coefs.index)
    ax.set_xlabel("Coefficient value")
    ax.set_title(f"{model_type.capitalize()} Top10 Coefficients (Fold {split_idx})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{relative_path}/figures/plots/best_coefficients_fold{split_idx}.jpg")
    plt.close(fig)

    return pipe

def svr(split_idx, tr_X, tr_y, refit, n_jobs):
    """
    Cross-validated Support Vector Regression (SVR) with RBF kernel.
    Performs parameter tuning over C and visualizes CV performance.
    """
    # --- Cross-validation setup ---
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    # --- Define model and parameter grid ---
    model = SVR(C=1.0, kernel='rbf')
    param_grid = {"model__C": np.logspace(-3, 3, 7)}

    # --- Build pipeline ---
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", model),
    ])

    # --- Define scoring metrics ---
    scoring = {
        "r2": "r2",
        "neg_mse": "neg_mean_squared_error"
    }
    assert refit in scoring, f"{refit} is unsupported (choose from {list(scoring.keys())})"

    # --- Grid search cross-validation ---
    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=refit,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # --- Plot cross-validation results ---
    cv_results = pd.DataFrame(gcv.cv_results_)
    Cs = cv_results['param_model__C']
    mean = cv_results[f'mean_test_{refit}']
    std = cv_results[f'std_test_{refit}']

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(Cs, mean, marker="o")
    ax.fill_between(Cs, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel(refit)
    ax.set_xlabel("C (Regularization Strength)")
    ax.axvline(gcv.best_params_["model__C"], c="C1", linestyle="--", label="Best C")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_title(f"SVR Cross-Validation ({refit})")

    plt.tight_layout()
    plt.savefig(f"{relative_path}/figures/plots/cross_validation_fold{split_idx}.jpg")
    plt.close(fig)

    # --- Fit final model using best parameters ---
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)
    print(f"[Fold {split_idx}] Best SVR params: {gcv.best_params_}")

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

    if radiomics_aggregated_mode in ["ABMIL", "SPARRA"]:
        print(f"loading radiomics from {save_radiomics_dir}...")
        radiomics_paths = []
        for p in data:
            subject_id = p[0][0]
            path = pathlib.Path(save_radiomics_dir) / radiomics_aggregated_mode / f"{subject_id}.json"
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

    if pathomics_aggregated_mode in ["ABMIL", "SPARRA"]:
        print(f"loading pathomics from {save_pathomics_dir}...")
        pathomics_paths = []
        for p in data:
            subject_id = p[0][0]
            path = pathlib.Path(save_pathomics_dir) / pathomics_aggregated_mode / f"{subject_id}.json"
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
    if radiomics_aggregated_mode in ["ABMIL", "SPARRA"]:
        assert radiomics_aggregated_mode == pathomics_aggregated_mode
        print(f"loading radiopathomics from {save_radiopathomics_dir}...")
        radiopathomics_paths = []
        for p in data:
            subject_id = p[0][0]
            path = pathlib.Path(save_radiopathomics_dir) / radiomics_aggregated_mode / f"{subject_id}.json"
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

def multioutput_f_regression(X, Y):
    """
    Computes F-scores for each feature by averaging F-scores across all outputs.
    Y can be (n_samples,) or (n_samples, n_outputs).
    """
    if Y.ndim == 1:
        # Single target → standard f_regression
        return f_regression(X, Y)

    # Multi-output case
    n_outputs = Y.shape[1]
    f_scores = []
    p_values = []

    for i in range(n_outputs):
        f_i, p_i = f_regression(X, Y[:, i])
        f_scores.append(f_i)
        p_values.append(p_i)

    # Aggregate scores across outputs
    f_scores = np.mean(f_scores, axis=0)     # or np.max(...)
    p_values = np.mean(p_values, axis=0)

    return f_scores, p_values

def signature_regression(
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
    model="ElasticNet",
    refit="r2",
    feature_selection=True,
    feature_var_threshold=1e-4,
    n_selected_features=64,
    target_selection=True,
    target_selection_cv=5,
    pvalue_threshold=0.05,
    use_graph_properties=False,
    survival_outcome='OS',
    save_results_dir=None,
    ):
    splits = joblib.load(split_path)
    predict_results = {}
    target_has_selected = False
    predictable_targets = []
    regression_results = {
        "raw_subject": [], "raw_pred": [],
        "subject": [], "pred": [],
        "label": [],
    }
    risk_results = {"risk": [], "event": [], "duration": []}
    for split_idx, split in enumerate(splits):
        print(f"Performing cross-validation on fold {split_idx}...")
        raw_data_tr, raw_data_va, raw_data_te = split["train"], split["valid"], split["test"]
        raw_data_tr = raw_data_tr + raw_data_va

        data_tr = [p for p in raw_data_tr if p[1] is not None]
        data_te = [p for p in raw_data_te if p[1] is not None]

        tr_y = [p[1]['signature'] for p in data_tr]
        tr_y = pd.DataFrame(tr_y)
        tr_y_clinical = [p[1]['clinical'] for p in data_tr]
        tr_y_clinical = pd.DataFrame(tr_y_clinical)
        te_y = [p[1]['signature'] for p in data_te]
        te_y = pd.DataFrame(te_y)
        te_y_clinical = [p[1]['clinical'] for p in data_te]
        te_y_clinical = pd.DataFrame(te_y_clinical)

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

        # target selection
        if target_selection:
            if not target_has_selected:
                print("Selecting targets...")
                for col in tr_y.columns:
                    np.random.seed(42)
                    y_true = tr_y[col].values
                    model_pred = make_pipeline(PCA(n_components=64), Ridge(random_state=42))
                    cv_splitter = KFold(n_splits=target_selection_cv, shuffle=False)
                    y_pred = cross_val_predict(model_pred, tr_X, y_true, cv=cv_splitter)
                    r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
                    # logrank test
                    pd_risk = pd.DataFrame({'risk': y_pred.tolist()})
                    pd_risk['event'] = tr_y_clinical[survival_outcome]
                    pd_risk['duration'] = tr_y_clinical[f'{survival_outcome}.time']
                    mean_risk = pd_risk["risk"].mean()
                    dem = pd_risk["risk"] > mean_risk
                    test_results = logrank_test(
                        pd_risk["duration"][dem], pd_risk["duration"][~dem], 
                        pd_risk["event"][dem], pd_risk["event"][~dem], 
                        alpha=.99
                    )
                    pval = test_results.p_value
                    if pval < pvalue_threshold:
                        predictable_targets.append(col)
                        print(f"Target '{col}' selected (R²={r2:.3f}, p={pval:.3f})")
                    else:
                        print(f"Target '{col}' removed (R²={r2:.3f}, p={pval:.3f})")
                target_has_selected = True

            print(f'Selected targets: {len(predictable_targets)}')
            print(predictable_targets)
            tr_y = tr_y[predictable_targets].copy()
            te_y = te_y[predictable_targets].copy()

        # feature selection
        if feature_selection:
            print("Selecting features using Lasso (L1)...")

            # Stronger alpha → more sparsity
            lasso = Lasso(
                alpha=0.001,         # tune this
                max_iter=5000,
                random_state=42
            )

            lasso.fit(tr_X, tr_y)

            selector = SelectFromModel(
                lasso,
                prefit=True,
                max_features=n_selected_features
            )

            selected_mask = selector.get_support()
            selected_names = tr_X.columns[selected_mask]

            tr_X = tr_X[selected_names]
            te_X = te_X[selected_names]
            raw_te_X = raw_te_X[selected_names]
            print(f"Selected features: {len(tr_X.columns)}")

        # model selection
        print("Selecting regressor...")
        if model == "RF":
            predictor = randomforest_regression(split_idx, tr_X, tr_y, refit, n_jobs)
        elif model == "Ridge":
            predictor = linearregression(split_idx, tr_X, tr_y, refit, n_jobs, model_type='ridge')
        elif model == "LASSO":
            predictor = linearregression(split_idx, tr_X, tr_y, refit, n_jobs, model_type='lasso')
        elif model == "ElasticNet":
            predictor = linearregression(split_idx, tr_X, tr_y, refit, n_jobs, model_type='elasticnet')
        elif model == "SVR":
            predictor = svr(split_idx, tr_X, tr_y, refit, n_jobs)
        else:
            raise NotImplementedError

        # Predictions
        raw_subject_ids = [p[0][0] for p in raw_data_te]
        regression_results["raw_subject"] += raw_subject_ids
        raw_pred = predictor.predict(raw_te_X)
        if raw_pred.ndim == 1:
            raw_pred = raw_pred.reshape(-1, 1)
        regression_results["raw_pred"] += raw_pred.tolist()

        # Perform prediction
        subject_ids = [p[0][0] for p in data_te]
        regression_results["subject"] += subject_ids
        pred = predictor.predict(te_X)
        label = te_y.to_numpy(dtype=float)  # te_y can be multi-target (n_samples, n_targets)

        # Ensure 2D arrays
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if label.ndim == 1:
            label = label.reshape(-1, 1)

        regression_results["pred"] += pred.tolist()
        regression_results["label"] += label.tolist()

        num_targets = pred.shape[1]

        risk_results["event"] += te_y_clinical['OS'].astype(int).tolist()
        risk_results["duration"] += te_y_clinical['OS.time'].tolist()

        # Compute metrics per target
        r2_list = []
        rmse_list = []
        mae_list = []
        var_explained_list = []

        # Containers for classification metrics
        acc_list = []
        f1_list = []

        for i in range(num_targets):
            r2 = r2_score(label[:, i], pred[:, i])
            rmse = np.sqrt(mean_squared_error(label[:, i], pred[:, i]))
            mae = mean_absolute_error(label[:, i], pred[:, i])
            var_explained = 1 - np.var(label[:, i] - pred[:, i]) / np.var(label[:, i])

            r2_list.append(r2)
            rmse_list.append(rmse)
            mae_list.append(mae)
            var_explained_list.append(var_explained)

            # Convert to binary classes based on each target’s own mean
            label_bin = (label[:, i] > np.mean(label[:, i])).astype(int)
            pred_bin = (pred[:, i] > np.mean(pred[:, i])).astype(int)
            acc = acc_scorer(label_bin, pred_bin)
            f1 = f1_scorer(label_bin, pred_bin)

            acc_list.append(acc)
            f1_list.append(f1)

        # Optionally, average across targets
        scores_dict = {
            "r2": r2_list,
            "rmse": rmse_list,
            "mae": mae_list,
            "var_explained": var_explained_list,
            "acc": acc_list,
            "f1": f1_list
        }
        predict_results.update({f"Fold {split_idx}": scores_dict})
    
    # print average results across folds per class
    print(predict_results)
    for k in scores_dict.keys():
        arr = np.array([v[k] for v in predict_results.values()])
        print(f"CV {k} mean+std", arr.mean(axis=0), arr.std(axis=0))

    # select the most predictable signature
    pvalues = []
    for i, column in enumerate(te_y.columns):
        risk_results['score'] = [v[i] for v in regression_results['pred']]
        pd_risk = pd.DataFrame(risk_results)
        pd_risk = pd_risk[pd_risk["duration"].notna() & pd_risk["event"].notna()]
        mean_risk = pd_risk["score"].mean()
        dem = pd_risk["score"] > mean_risk
        test_results = logrank_test(
            pd_risk["duration"][dem], pd_risk["duration"][~dem], 
            pd_risk["event"][dem], pd_risk["event"][~dem], 
            alpha=.99
        )
        pvalue = test_results.p_value
        pvalues.append(pvalue)
    min_index = np.argmin(np.array(pvalues))
    signature_name = te_y.columns[min_index]
    risk_results['risk'] = [v[min_index] for v in risk_results['risk']]
    pvalues = {k : v for k, v in zip(te_y.columns, pvalues)}
    print("Signature p-values:", pvalues)
    predict_results.update({'p-value': pvalues})

    # plot survival curves
    pd_risk = pd.DataFrame(risk_results)
    pd_risk = pd_risk[pd_risk["duration"].notna() & pd_risk["event"].notna()]
    mean_risk = pd_risk["risk"].mean()
    dem = pd_risk["risk"] > mean_risk

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({'font.size': 12})
    kmf1 = KaplanMeierFitter()
    kmf1.fit(pd_risk["duration"][dem], event_observed=pd_risk["event"][dem], label="High score")
    kmf1.plot_survival_function(ax=ax)

    kmf2 = KaplanMeierFitter()
    kmf2.fit(pd_risk["duration"][~dem], event_observed=pd_risk["event"][~dem], label="Low score")
    kmf2.plot_survival_function(ax=ax)
    add_at_risk_counts(kmf1, kmf2, ax=ax)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

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
    ax.set_title(f"Signature: {signature_name}")
    # plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(f"{relative_path}/figures/plots/{omics}_survival_curve.png")

    #save predicted results
    save_path = f"{save_results_dir}/{omics}_" + \
        f"radio+{radiomics_aggregated_mode}_" + \
        f"patho+{pathomics_aggregated_mode}_" + \
        f"model+{model}_scorer+{refit}_results.json"
    with open(save_path, "w") as f:
        json.dump(regression_results, f, indent=4)

    # save metrics
    save_path = f"{save_results_dir}/{omics}_" + \
        f"radio+{radiomics_aggregated_mode}_" + \
        f"patho+{pathomics_aggregated_mode}_" + \
        f"model+{model}_scorer+{refit}_" + \
        f"survival+{survival_outcome}_metrics.json"
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
    
    if opt.get('SAVE_MODEL_DIR', False):
        save_model_dir = pathlib.Path(opt['SAVE_MODEL_DIR'])
        save_model_dir = save_model_dir / f"{opt['DATASET']}_signature_{opt['OUTCOME']['VALUE']}"
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
    from analysis.a05_outcome_prediction.m_prepare_omics_info import signatures_predictable
    if opt['PREDICTION']['TARGET_SELECTION']['VALUE']:
        signature_ids = None
    else:
        signature_ids = signatures_predictable[opt['OUTCOME']['VALUE']]
    df_outcome, df_clinical = prepare_patient_outcome(
        outcome_dir=opt['SAVE_OUTCOME_dir'],
        subject_ids=list(omics_info['omics_paths'].keys()),
        outcome=opt['OUTCOME']['VALUE'],
        signature_ids=signature_ids,
        pooling=opt['OUTCOME']['POOLING']
    )
    subject_ids = df_outcome['SubjectID'].to_list()
    logger.info(f"Found {len(subject_ids)} subjects with signature scores")

    omics_paths = [[k, omics_info['omics_paths'][k]] for k in subject_ids]
    
    signature_data = df_outcome.drop(columns=['SubjectID']).to_dict(orient='records')
    clinical_data = df_clinical.to_dict(orient='records')
    outcomes = [{'signature': s, 'clinical': c} for s, c in zip(signature_data, clinical_data)]

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
    split_path = f"{save_model_dir}/signature_{radiomics_mode}_{pathomics_mode}_splits.dat"
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
            save_omics_dir = save_omics_dir + f"/radiomics_Survival_Prediction_GCNConv_" \
                + opt['RADIOMICS']['AGGREGATED_MODE']['VALUE']
        elif opt['PREDICTION']['USED_OMICS']['VALUE'] == "pathomics":
            save_omics_dir = save_omics_dir + f"/pathomics_Survival_Prediction_GCNConv_" \
                + opt['PATHOMICS']['AGGREGATED_MODE']['VALUE']
        elif opt['PREDICTION']['USED_OMICS']['VALUE'] == "radiopathomics":
            save_omics_dir = save_omics_dir + f"/radiomics_pathomics_Survival_Prediction_GCNConv_" \
                + opt['PATHOMICS']['AGGREGATED_MODE']['VALUE']
            
        signature_regression(
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
            refit=opt['PREDICTION']['REFIT']['VALUE'],
            feature_selection=opt['PREDICTION']['FEATURE_SELECTION']['VALUE'],
            feature_var_threshold=opt['PREDICTION']['FEATURE_SELECTION']['VAR_THRESHOLD'],
            n_selected_features=opt['PREDICTION']['FEATURE_SELECTION']['NUM_FEATURES'],
            target_selection=opt['PREDICTION']['TARGET_SELECTION']['VALUE'],
            target_selection_cv=opt['PREDICTION']['TARGET_SELECTION']['CROSS_VALID'],
            pvalue_threshold=opt['PREDICTION']['TARGET_SELECTION']['PVALUE_THRESHOLD'],
            use_graph_properties=opt['PREDICTION']['USE_GRAPH_PROPERTIES'],
            survival_outcome=opt['PREDICTION']['OUTCOME_LOGRANK_TEST']['VALUE'],
            save_results_dir=save_model_dir
        )

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
