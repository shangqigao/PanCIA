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

from analysis.a04_feature_aggregation.m_gnn_phenotype_prediction import PhenotypeGraphDataset, PhenotypeGraphArch, PhenotypeBayesGraphArch
from analysis.a04_feature_aggregation.m_gnn_therapy_response import ScalarMovingAverage, R2TLoss

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
                if prop_keys is None: 
                    prop_keys = ["shape", "firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm"]
                data_dict = load_json(radiomic_path)
                properties = {}
                for key, value in data_dict.items():
                    selected = [((k in key) and ("diagnostics" not in key)) for k in prop_keys]
                    if any(selected): properties[f"{radiomic_key}.{key}"] = value
                if len(properties) > 0: properties_dict.update(properties)
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
            path_list = pathlib.Path(p[0][1]["radiomics"])
            new_path_list = []
            for path_dict in path_list:
                new_path_dict = {}
                for k, path in path_dict.items():
                    dir = path.parents[1] / radiomics_aggregated_mode / f"0{split_idx}"
                    name = path.name.replace(".json", ".npy")
                    new_path_dict[k] = dir / path.parent.name / name
                new_path_list.append(new_path_dict)
            radiomics_paths.append(new_path_list)
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
            path_list = pathlib.Path(p[0][1]["pathomics"])
            new_path_list = []
            for path_dict in path_list:
                new_path_dict = {}
                for k, path in path_dict.items():
                    dir = path.parents[1] / pathomics_aggregated_mode / f"0{split_idx}"
                    name = path.name.replace(".json", ".npy")
                    new_path_dict[k] = dir / path.parent.name / name
                new_path_list.append(new_path_dict)
            pathomics_paths.append(new_path_list)
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
    omics="all", 
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
    use_graph_properties=False
    ):
    splits = joblib.load(split_path)
    predict_results = {}
    target_has_selected = False
    predictable_targets = []
    risk_results = {"risk": [], "event": [], "duration": []}
    for split_idx, split in enumerate(splits):
        print(f"Performing cross-validation on fold {split_idx}...")
        data_tr, data_va, data_te = split["train"], split["valid"], split["test"]
        data_tr = data_tr + data_va

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

        # Normalization
        scaler = StandardScaler()
        tr_X = pd.DataFrame(scaler.fit_transform(tr_X), columns=tr_X.columns,index=tr_X.index)
        te_X = pd.DataFrame(scaler.transform(te_X), columns=te_X.columns, index=te_X.index)

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
                    pd_risk['event'] = tr_y_clinical['OS']
                    pd_risk['duration'] = tr_y_clinical['OS.time']
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

        # Perform prediction
        pred = predictor.predict(te_X)
        label = te_y.to_numpy(dtype=float)  # te_y can be multi-target (n_samples, n_targets)

        # Ensure 2D arrays
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if label.ndim == 1:
            label = label.reshape(-1, 1)

        num_targets = pred.shape[1]

        # risk_results["risk"] += pred.tolist()
        risk_results["risk"] += label.tolist()
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
        risk_results['score'] = [v[i] for v in risk_results['risk']]
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

    return

def generate_data_split(
        x: list,
        y: list,
        train: float,
        valid: float,
        test: float,
        num_folds: int,
        seed: int = 5
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
            n_train_valid = len(train_valid_x)
            n_valid = int(n_train_valid * valid / (train + valid))
            valid_x, valid_y = train_valid_x[:n_valid], train_valid_y[:n_valid]
            train_x, train_y = train_valid_x[n_valid:], train_valid_y[n_valid:]
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
        model = PhenotypeBayesGraphArch(**arch_kwargs)
        kl = {"loss": bnn.BKLLoss(), "weight": 0.1}
    else:
        model = PhenotypeGraphArch(**arch_kwargs)
        kl = None
    if pretrained is not None:
        model.load(pretrained)
    if on_gpu:
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    loss = R2TLoss(binary_cls=arch_kwargs['binary_cls'])
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
        ds = PhenotypeGraphDataset(
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
                preds, label = output
                preds = np.array(preds).squeeze()
                label = np.array(label).squeeze()

                # Ensure 2D
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                if label.ndim == 1:
                    label = label.reshape(-1, 1)

                num_targets = preds.shape[1]

                r2_list = []
                rmse_list = []
                mae_list = []

                for i in range(num_targets):
                    r2 = r2_score(label[:, i], preds[:, i])
                    rmse = np.sqrt(mean_squared_error(label[:, i], preds[:, i]))
                    mae = mean_absolute_error(label[:, i], preds[:, i])

                    r2_list.append(r2)
                    rmse_list.append(rmse)
                    mae_list.append(mae)

                # Average over multiple targets if necessary
                logging_dict[f"{loader_name}-r2"] = np.mean(r2_list)
                logging_dict[f"{loader_name}-rmse"] = np.mean(rmse_list)
                logging_dict[f"{loader_name}-mae"] = np.mean(mae_list)

                # Optional: Pearson correlation
                if num_targets == 1:
                    corr = np.corrcoef(label[:, 0], preds[:, 0])[0, 1]
                    logging_dict[f"{loader_name}-corr"] = corr


                logging_dict[f"{loader_name}-raw-pred"] = preds
                logging_dict[f"{loader_name}-raw-label"] = label
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
        split_path,
        scaler_path,
        model_dir,
        omics_dims,
        arch_opt,
        train_opt
):
    """train node classification neural networks
    Args:
        num_epochs (int): the number of epochs for training
        split_path (str): the path of storing data splits
        scaler_path (str): the path of storing data normalization
        num_node_features (int): the dimension of node feature
        model_dir (str): directory of saving models
    """
    omics_pool_ratio = {
        "radiomics": arch_opt['POOL_RATIO']['RAIOMICS'], 
        "pathomics": arch_opt['POOL_RATIO']['PATHOMICS']
    }
    omics_pool_ratio = {k: omics_pool_ratio[k] for k in arch_opt['OMICS']}

    splits = joblib.load(split_path)
    node_scalers = [joblib.load(scaler_path[k]) for k in arch_opt['OMICS']] 
    transform_dict = {k: s.transform for k, s in zip(arch_opt['OMICS'], node_scalers)}
    
    loader_kwargs = {
        "num_workers": train_opt['N_WORKS'], 
        "batch_size": train_opt['BATCH_SIZE'],
    }
    arch_kwargs = {
        "dim_features": omics_dims,
        "dim_target": arch_opt['DIM_TARGET'],
        "layers": arch_opt['LAYERS'],
        "dropout": arch_opt['DROPOUT'],
        "pool_ratio": omics_pool_ratio,
        "conv": arch_opt['GNN'],
        "keys": arch_opt['OMICS'],
        "aggregation": arch_opt['AGGREGATION']
    }
    omics_name = "_".join(arch_opt['OMICS'])
    if arch_opt['BayesGNN']:
        model_dir = model_dir / f"{omics_name}_Bayes_Phenotype_Prediction_{arch_opt['GNN']}_{arch_opt['AGGREGATION']}"
    else:
        model_dir = model_dir / f"{omics_name}_Phenotype_Prediction_{arch_opt['GNN']}_{arch_opt['AGGREGATION']}"
    optim_kwargs = {
        "lr": 3e-4,
        "weight_decay": {
            "ABMIL": train_opt['WIEGHT_DECAY']['ABMIL'], 
            "SISIR": train_opt['WIEGHT_DECAY']['SPARRA']
        }[arch_opt['AGGREGATION']],
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
            train_opt['EPOCHS'],
            save_dir=split_save_dir,
            arch_kwargs=arch_kwargs,
            loader_kwargs=loader_kwargs,
            optim_kwargs=optim_kwargs,
            preproc_func=transform_dict,
            BayesGNN=arch_opt['BayesGNN'],
            data_types=arch_opt['OMICS'],
            sampling_rate=train_opt['SAMPLING_RATE']
        )
    return

def inference(
        split_path,
        scaler_path,
        omics_dims,
        arch_opt,
        infer_opt
):
    """survival prediction
    """
    omics_pool_ratio = {
        "radiomics": arch_opt['POOL_RATIO']['RAIOMICS'], 
        "pathomics": arch_opt['POOL_RATIO']['PATHOMICS']
    }
    omics_pool_ratio = {k: omics_pool_ratio[k] for k in arch_opt['OMICS']}

    splits = joblib.load(split_path)
    node_scalers = [joblib.load(scaler_path[k]) for k in arch_opt['OMICS']] 
    transform_dict = {k: s.transform for k, s in zip(arch_opt['OMICS'], node_scalers)}
    
    loader_kwargs = {
        "num_workers": infer_opt['N_WORKS'], 
        "batch_size": infer_opt['BATCH_SIZE'],
    }
    arch_kwargs = {
        "dim_features": omics_dims,
        "dim_target": arch_opt['DIM_TARGET'],
        "layers": arch_opt['LAYERS'],
        "dropout": arch_opt['DROPOUT'],
        "pool_ratio": omics_pool_ratio,
        "conv": arch_opt['GNN'],
        "keys": arch_opt['OMICS'],
        "aggregation": arch_opt['AGGREGATION']
    }
    omics_name = "_".join(arch_opt['OMICS'])

    cum_stats = []
    if "radiomics_pathomics" == omics_name: aggregation = f"radiopathomics_{aggregation}"
    for split_idx, split in enumerate(splits):
        if infer_opt['SAVE_OMICS']:
            all_samples = splits[split_idx]["train"] + splits[split_idx]["valid"] + splits[split_idx]["test"]
        else:
            all_samples = split["test"]

        new_split = {"infer": [v[0] for v in all_samples]}
        chkpts = pathlib.Path(infer_opt['PRE_TRAINED']) / f"0{split_idx}/epoch=019.weights.pth"
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
            BayesGNN=arch_opt['BayesGNN'],
            data_types=arch_opt['OMICS'],
        )
        logits, features = list(zip(*outputs))
        # saving average features
        if infer_opt['SAVE_OMICS']:
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
    
        # Convert logits to predictions
        preds = np.array(logits).squeeze()
        labels = np.array([v[1] for v in split["test"]]).squeeze()

        # Ensure 2D arrays for multi-output regression
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        num_targets = preds.shape[1]

        # Compute per-target metrics
        r2_scores = []
        rmse_scores = []
        mae_scores = []

        for i in range(num_targets):
            r2 = r2_score(labels[:, i], preds[:, i])
            rmse = np.sqrt(mean_squared_error(labels[:, i], preds[:, i]))
            mae = mean_absolute_error(labels[:, i], preds[:, i])

            r2_scores.append(r2)
            rmse_scores.append(rmse)
            mae_scores.append(mae)

        # Save cumulative stats
        cum_stats.append(
            {
                "r2": np.array(r2_scores),       # per-target R²
                "rmse": np.array(rmse_scores),   # per-target RMSE
                "mae": np.array(mae_scores),     # per-target MAE
            }
        )

        print(f"Fold-{split_idx}:", cum_stats[-1])
    acc_list = [stat["acc"] for stat in cum_stats]
    auroc_list = [stat["auroc"] for stat in cum_stats]
    auprc_list = [stat["auprc"] for stat in cum_stats]
    avg_stat = {
        "acc-mean": np.stack(acc_list, axis=0).mean(axis=0),
        "acc-std": np.stack(acc_list, axis=0).std(axis=0),
        "auroc-mean": np.stack(auroc_list, axis=0).mean(axis=0),
        "auroc-std": np.stack(auroc_list, axis=0).std(axis=0),
        "auprc-mean": np.stack(auprc_list, axis=0).mean(axis=0),
        "auprc-std": np.stack(auprc_list, axis=0).std(axis=0)
    }
    print(f"Avg:", avg_stat)
    return avg_stat

def test(
    graph_path,
    scaler_path,
    omics_dims,
    arch_opt,
    test_opt
):
    """node classification 
    """
    omics_pool_ratio = {
        "radiomics": arch_opt['POOL_RATIO']['RAIOMICS'], 
        "pathomics": arch_opt['POOL_RATIO']['PATHOMICS']
    }
    omics_pool_ratio = {k: omics_pool_ratio[k] for k in arch_opt['OMICS']}

    node_scalers = [joblib.load(scaler_path[k]) for k in arch_opt['OMICS']] 
    transform_dict = {k: s.transform for k, s in zip(arch_opt['OMICS'], node_scalers)}
    
    loader_kwargs = {
        "num_workers": test_opt['N_WORKS'],
        "batch_size": test_opt['BATCH_SIZE'],
    }
    arch_kwargs = {
        "dim_features": omics_dims,
        "dim_target": arch_opt['DIM_TARGET'],
        "layers": arch_opt['LAYERS'],
        "dropout": arch_opt['DROPOUT'],
        "pool_ratio": omics_pool_ratio,
        "conv": arch_opt['GNN'],
        "keys": arch_opt['OMICS'],
        "aggregation": arch_opt['AGGREGATION']
    }

    # BayesGNN = True
    new_split = {"infer": [graph_path]}
    outputs = run_once(
        new_split,
        num_epochs=1,
        save_dir=None,
        on_gpu=True,
        pretrained=test_opt['PRE_TRAINED'],
        arch_kwargs=arch_kwargs,
        loader_kwargs=loader_kwargs,
        preproc_func=transform_dict,
        BayesGNN=arch_opt['BayesGNN'],
        data_types=arch_opt['OMICS'],
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
            radiomics_suffix=opt['RADIOMICS']['MODE']['SUFFIX'],
        )
    elif opt['DATASET'] == "TCGA":
        from analysis.a05_outcome_prediction.m_prepare_omics_info import prepare_TCGA_omics_info
        omics_info = prepare_TCGA_omics_info(
            dataset_json=opt['DATA_INFO'],
            save_omics_dir=opt['OMICS_DIR'],
            radiomics_mode=radiomics_mode,
            segmentator=opt['RADIOMICS']['SEGMENTATOR']['VALUE'],
            radiomics_suffix=opt['RADIOMICS']['MODE']['SUFFIX'],
            pathomics_mode=pathomics_mode,
            pathomics_suffix=opt['PATHOMICS']['MODE']['SUFFIX']
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
    y = [{'signature': s, 'clinical': c} for s, c in zip(signature_data, clinical_data)]

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

    from analysis.a05_outcome_prediction.m_prepare_omics_info import radiomics_dims
    from analysis.a05_outcome_prediction.m_prepare_omics_info import pathomics_dims
    omics_dims = {"radiomics": radiomics_dims[radiomics_mode], "pathomics": pathomics_dims[pathomics_mode]}
    omics_keys = opt['ARCH']['OMICS']
    omics_dims = {k: omics_dims[k] for k in omics_keys}

    if opt['TASKS']['TRAIN']:
        # compute mean and std on training data for normalization 
        splits = joblib.load(split_path)
        train_graph_paths = [path for path, _ in splits[0]["train"]]
        loader = PhenotypeGraphDataset(train_graph_paths, mode="infer", data_types=omics_keys)
        loader = DataLoader(
            loader,
            num_workers=8,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        if len(omics_keys) > 1:
            omic_features = [{k: v.x_dict[k].numpy() for k in omics_keys} for v in loader]
        else:
            omic_features = [{k: v.x.numpy() for k in omics_keys} for v in loader]
        omics_modes = {"radiomics": radiomics_mode, "pathomics": pathomics_mode}
        omics_modes = {k: omics_modes[k] for k in omics_keys}
        for k, v in omics_modes.items():
            node_features = [d[k] for d in omic_features]
            node_features = np.concatenate(node_features, axis=0)
            node_scaler = StandardScaler(copy=False)
            node_scaler.fit(node_features)
            scaler_path = f"{save_model_dir}/survival_{k}_{v}_scaler.dat"
            joblib.dump(node_scaler, scaler_path)

        # training
        split_path = f"{save_model_dir}/survival_radiopathomics_{radiomics_mode}_{pathomics_mode}_splits.dat"
        scaler_paths = {k: f"{save_model_dir}/survival_{k}_{v}_scaler.dat" for k, v in omics_modes.items()}
        training(
            split_path=split_path,
            scaler_path=scaler_paths,
            model_dir=save_model_dir,
            omics_dims=omics_dims,
            arch_opt=opt['ARCH'],
            train_opt=opt['TRAIN']
        )

    if opt['TASKS']['INFERENCE']:
        # inference
        inference(
            split_path=split_path,
            scaler_path=scaler_paths,
            omics_dims=omics_dims,
            arch_opt=opt['ARCH'],
            infer_opt=opt['INFERENCE']
        )

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
        signature_regression(
            split_path=split_path,
            omics=opt['PREDICTION']['USED_OMICS']['VALUE'],
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
            use_graph_properties=opt['PREDICTION']['USE_GRAPH_PROPERTIES']
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
