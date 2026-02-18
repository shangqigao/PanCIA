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
from scipy.stats import zscore
from torch_geometric.loader import DataLoader
from tiatoolbox import logger
from tiatoolbox.utils.misc import save_as_json

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer
from sklearn.metrics import balanced_accuracy_score as acc_scorer
from sklearn.metrics import f1_score as f1_scorer
from xgboost import XGBClassifier

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

def prepare_patient_outcome(outcome_dir, subject_ids, outcome=None, minimum_per_class=20):
    if outcome == "ImmuneSubtype":
        outcome_file = f"{outcome_dir}/phenotypes/immune_subtype/immune_subtype.csv"
        df = pd.read_csv(outcome_file)
        df["SubjectID"] = df["SampleID"].str.extract(r'^(TCGA-\w\w-\w{4})')
        df_matched = df[df["SubjectID"].isin(subject_ids)]
        df_matched = df_matched[df_matched["Subtype_Immune_Model_Based"].notna()]
        counts = df_matched["Subtype_Immune_Model_Based"].value_counts()
        print("Before filtering:", counts)
        valid_classes = counts[counts >= minimum_per_class].index
        df_matched = df_matched[df_matched["Subtype_Immune_Model_Based"].isin(valid_classes)]
        print("After filtering:", df_matched["Subtype_Immune_Model_Based"].value_counts())
        df_matched["PhenotypeClass"], uniques = pd.factorize(df_matched["Subtype_Immune_Model_Based"])
        print("Immune subtype mapping:", dict(enumerate(uniques)))
    elif outcome == "MolecularSubtype":
        outcome_file = f"{outcome_dir}/phenotypes/molecular_subtype/molecular_subtype.csv"
        df = pd.read_csv(outcome_file)
        df["SubjectID"] = df["SampleID"].str.extract(r'^(TCGA-\w\w-\w{4})')
        df_matched = df[df["SubjectID"].isin(subject_ids)]
        df_matched = df_matched[df_matched["Subtype_Selected"].notna()]
        counts = df_matched["Subtype_Selected"].value_counts()
        print("Before filtering:", counts)
        valid_classes = counts[counts >= minimum_per_class].index
        df_matched = df_matched[df_matched["Subtype_Selected"].isin(valid_classes)]
        print(df_matched["Subtype_Selected"].value_counts())
        df_matched["PhenotypeClass"], uniques = pd.factorize(df_matched["Subtype_Selected"])
        print("Molecular subtype mapping:", dict(enumerate(uniques)))
    elif outcome == "PrimaryDisease":
        outcome_file = f"{outcome_dir}/phenotypes/sample_type_and_primary_disease/sample_type_primary_disease.csv"
        df = pd.read_csv(outcome_file)
        df["SubjectID"] = df["SampleID"].str.extract(r'^(TCGA-\w\w-\w{4})')
        df_matched = df[df["SubjectID"].isin(subject_ids)]
        df_matched = df_matched[df_matched["_primary_disease"].notna()]
        df_matched = df_matched.drop_duplicates(subset=["SubjectID"], keep="first")
        counts = df_matched["_primary_disease"].value_counts()
        print("Before filtering:", counts)
        valid_classes = counts[counts >= minimum_per_class].index
        df_matched = df_matched[df_matched["_primary_disease"].isin(valid_classes)]
        print("After filtering:", df_matched["_primary_disease"].value_counts())
        df_matched["PhenotypeClass"], uniques = pd.factorize(df_matched["_primary_disease"])
        print("Primary disease mapping:", dict(enumerate(uniques)))
    else:
        raise NotImplementedError

    df_matched = df_matched.drop_duplicates(subset=["SubjectID"], keep="first")

    logging.info(f"Phenotype data strcuture: {df_matched.shape}")

    return df_matched

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

def randomforest(split_idx, tr_X, tr_y, refit, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = RandomForestClassifier(max_depth=2, class_weight="balanced", random_state=42)
    param_grid={"model__max_depth": list(range(1, 6))}
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1_macro"
    }
    assert scoring.get(refit, False), f"{refit} is unsupported"
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
        scoring=scoring,
        refit=refit,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    depths = cv_results['param_model__max_depth']
    mean = cv_results[f'mean_test_{refit}']
    std = cv_results[f'std_test_{refit}']
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(depths, mean)
    ax.fill_between(depths, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("linear")
    ax.set_ylabel(refit)
    ax.set_xlabel("max depth")
    ax.axvline(gcv.best_params_["model__max_depth"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"{relative_path}/figures/plots/cross_validation_fold{split_idx}.jpg")

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def xgboost(split_idx, tr_X, tr_y, refit, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    neg, pos = np.bincount(tr_y.to_numpy(dtype=int))
    model = XGBClassifier(max_depth=2, scale_pos_weight=neg / pos, random_state=42)
    param_grid={"model__max_depth": list(range(1, 6))}
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1_macro"
    }
    assert scoring.get(refit, False), f"{refit} is unsupported"
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
        scoring=scoring,
        refit=refit,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    depths = cv_results['param_model__max_depth']
    mean = cv_results[f'mean_test_{refit}']
    std = cv_results[f'std_test_{refit}']
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(depths, mean)
    ax.fill_between(depths, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("linear")
    ax.set_ylabel(refit)
    ax.set_xlabel("max depth")
    ax.axvline(gcv.best_params_["model__max_depth"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"{relative_path}/figures/plots/cross_validation_fold{split_idx}.jpg")

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def logisticregression(split_idx, tr_X, tr_y, refit, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.5, 
        C=1.0, max_iter=1000, class_weight="balanced", random_state=42
    )
    param_grid={"model__C": np.logspace(-3, 3, 7)}
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1_macro"
    }
    assert scoring.get(refit, False), f"{refit} is unsupported"

    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=refit,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    Cs = cv_results['param_model__C']
    mean = cv_results[f'mean_test_{refit}']
    std = cv_results[f'std_test_{refit}']
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(Cs, mean)
    ax.fill_between(Cs, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel(refit)
    ax.set_xlabel("C")
    ax.axvline(gcv.best_params_["model__C"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"{relative_path}/figures/plots/cross_validation_fold{split_idx}.jpg")

    # Visualize coefficients of the best estimator
    best_model = gcv.best_estimator_.named_steps["model"]

    # coef_ shape: (n_classes, n_features)
    best_coefs = pd.DataFrame(
        best_model.coef_.T,  # shape: (n_features, n_classes)
        index=tr_X.columns,
        columns=[f"class_{cls}" for cls in best_model.classes_]
    )

    # Select only features with any non-zero coefficient
    non_zero_mask = (best_coefs != 0).any(axis=1)
    non_zero_coefs = best_coefs[non_zero_mask]

    # Compute max absolute coefficient across targets for each feature
    max_abs_coefs = non_zero_coefs.abs().max(axis=1)

    # Select top 10 features overall (not per target)
    top10_features = max_abs_coefs.sort_values(ascending=False).head(10).index
    top10_coefs = non_zero_coefs.loc[top10_features]

    # Plot setup
    y = np.arange(len(top10_coefs))[::-1]  # for horizontal bars
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
    ax.set_yticklabels(top10_features)
    ax.set_xlabel("Coefficient value")
    ax.set_title(f"Top10 Coefficients (Fold {split_idx})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(f"{relative_path}/figures/plots/best_coefficients_fold{split_idx}.jpg")
    plt.close(fig)

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)
    return pipe

def svc(split_idx, tr_X, tr_y, refit, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = SVC(C=1, kernel='rbf', class_weight="balanced", probability=True, random_state=42)
    param_grid={"model__C": np.logspace(-3, 3, 7)}
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1_macro"
    }
    assert scoring.get(refit, False), f"{refit} is unsupported"

    gcv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=refit,
        n_jobs=n_jobs,
    ).fit(tr_X, tr_y)

    # plot cross validation results
    cv_results = pd.DataFrame(gcv.cv_results_)
    Cs = cv_results['param_model__C']
    mean = cv_results[f'mean_test_{refit}']
    std = cv_results[f'std_test_{refit}']
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(Cs, mean)
    ax.fill_between(Cs, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel(refit)
    ax.set_xlabel("C")
    ax.axvline(gcv.best_params_["model__C"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"{relative_path}/figures/plots/cross_validation_fold{split_idx}.jpg")

    # Visualize coefficients of the best estimator
    # best_model = gcv.best_estimator_.named_steps["model"]
    # best_coefs = pd.DataFrame(best_model.coef_.ravel(), index=tr_X.columns, columns=["coefficient"])
    # non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    # print(f"Number of non-zero coefficients: {non_zero}")

    # non_zero_coefs = best_coefs.query("coefficient != 0")
    # coef_order = non_zero_coefs.abs().sort_values("coefficient").index
    # top10 = coef_order[:10]

    # _, ax = plt.subplots(figsize=(8, 6))
    # non_zero_coefs.loc[top10].plot.barh(ax=ax, legend=False)
    # ax.set_xlabel("coefficient")
    # ax.grid(True)
    # plt.subplots_adjust(left=0.3)
    # plt.savefig(f"analysis/a05_outcome_prediction/best_coefficients_fold{split_idx}.jpg") 

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

def phenotype_classification(
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
    model="LR",
    refit="roc_auc",
    feature_selection=True,
    feature_var_threshold=1e-4,
    n_selected_features=64,
    use_graph_properties=False,
    save_results_dir=None
    ):
    splits = joblib.load(split_path)
    predict_results = {}
    classification_results = {
        "raw_subject": [], "raw_pred": [], "raw_prob": [],
        "subject": [], "pred": [],  "prob": [],
        "label": [],
    }
    for split_idx, split in enumerate(splits):
        print(f"Performing cross-validation on fold {split_idx}...")
        raw_data_tr, raw_data_va, raw_data_te = split["train"], split["valid"], split["test"]
        raw_data_tr = raw_data_tr + raw_data_va

        data_tr = [p for p in raw_data_tr if p[1] is not None]
        data_te = [p for p in raw_data_te if p[1] is not None]

        tr_y = np.array([p[1] for p in data_tr])
        tr_y = pd.DataFrame({'label': tr_y})
        te_y = np.array([p[1] for p in data_te])
        te_y = pd.DataFrame({'label': te_y})

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
            print("Selecting features using LinearSVC + L1...")
                
            # Initialize LinearSVC with L1 penalty
            svc = LinearSVC(
                penalty='l1',
                dual=False,       # required for L1
                C=1.0,            # can tune C for sparsity
                class_weight='balanced',
                max_iter=5000,
                random_state=42
            )
            
            # Fit to training data
            svc.fit(tr_X, tr_y["label"])
            
            # Select features with non-zero coefficients
            selector = SelectFromModel(svc, prefit=True, max_features=n_selected_features)
            selected_mask = selector.get_support()
            selected_names = tr_X.columns[selected_mask]
            
            tr_X = tr_X[selected_names] 
            te_X = te_X[selected_names]
            raw_te_X = raw_te_X[selected_names]
            print(f"Selected features: {len(tr_X.columns)}")

        # model selection
        print("Selecting classifier...")
        if model == "RF":
            predictor = randomforest(split_idx, tr_X, tr_y['label'], refit, n_jobs)
        elif model == "XG":
            predictor = xgboost(split_idx, tr_X, tr_y['label'], refit, n_jobs)
        elif model == "LR":
            predictor = logisticregression(split_idx, tr_X, tr_y['label'], refit, n_jobs)
        elif model == "SVC":
            predictor = svc(split_idx, tr_X, tr_y['label'], refit, n_jobs)
        else:
            raise NotImplementedError

        # Predictions
        raw_subject_ids = [p[0][0] for p in raw_data_te]
        classification_results["raw_subject"] += raw_subject_ids
        raw_pred = predictor.predict(raw_te_X)
        classification_results["raw_pred"] += raw_pred.tolist()
        raw_prob = predictor.predict_proba(raw_te_X)
        classification_results["raw_prob"] += raw_prob.tolist()

        subject_ids = [p[0][0] for p in data_te]
        classification_results["subject"] += subject_ids
        pred = predictor.predict(te_X)
        classification_results["pred"] += pred.tolist()
        prob = predictor.predict_proba(te_X)
        classification_results["prob"] += prob.tolist()
        num_class = prob.shape[1]

        # True labels
        label = te_y["label"].to_numpy(dtype=int)
        classification_results["label"] += label.tolist()

        # Initialize lists to store metrics per class
        acc_list, f1_list, auroc_list, auprc_list = [], [], [], []

        for cls in range(num_class):
            # Binary one-vs-rest labels for this class
            y_true_cls = (label == cls).astype(int)
            y_pred_cls = (pred == cls).astype(int)
            y_prob_cls = prob[:, cls]

            # Compute metrics using existing scorer functions
            acc_list.append(acc_scorer(y_true_cls, y_pred_cls))
            f1_list.append(f1_scorer(y_true_cls, y_pred_cls))
            auroc_list.append(auroc_scorer(y_true_cls, y_prob_cls))
            auprc_list.append(auprc_scorer(y_true_cls, y_prob_cls))

        # Optional: organize as dictionary
        scores_dict = {
            "acc": acc_list,
            "f1": f1_list,
            "auroc": auroc_list,
            "auprc": auprc_list
        }

        predict_results.update({f"Fold {split_idx}": scores_dict})

    # print average results across folds per class
    print(predict_results)
    for k in scores_dict.keys():
        arr = np.array([v[k] for v in predict_results.values()])
        print(f"CV {k} mean+std", arr.mean(axis=0), arr.std(axis=0))
    
    #save predicted results
    save_path = f"{save_results_dir}/{omics}_" + \
        f"radio+{radiomics_aggregated_mode}_" + \
        f"patho+{pathomics_aggregated_mode}_" + \
        f"model+{model}_scorer+{refit}_results.json"
    with open(save_path, "w") as f:
        json.dump(classification_results, f, indent=4)

    # save metrics
    save_path = f"{save_results_dir}/{omics}_" + \
        f"radio+{radiomics_aggregated_mode}_" + \
        f"patho+{pathomics_aggregated_mode}_" + \
        f"model+{model}_scorer+{refit}_metrics.json"
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
        save_model_dir = save_model_dir / f"{opt['DATASET']}_phenotype_{opt['OUTCOME']['VALUE']}"
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
        outcome_dir=opt['SAVE_OUTCOME_dir'],
        subject_ids=list(omics_info['omics_paths'].keys()),
        outcome=opt['OUTCOME']['VALUE']
    )
    subject_ids = df_outcome['SubjectID'].to_list()
    logger.info(f"Found {len(subject_ids)} subjects with phenotypes")

    omics_paths = [[k, omics_info['omics_paths'][k]] for k in subject_ids]
    outcomes = df_outcome['PhenotypeClass'].to_numpy(dtype=np.float32).tolist()

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
    split_path = f"{save_model_dir}/phenotype_{radiomics_mode}_{pathomics_mode}_splits.dat"
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
        # save_omics_dir = opt['PREDICTION']['OMICS_DIR'] + f"/{radiomics_mode}+{pathomics_mode}" + f"/pathomics_GCNConv_SPARRA_heter_vi0.1ae1.0_noDS"
            
        phenotype_classification(
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
            use_graph_properties=opt['PREDICTION']['USE_GRAPH_PROPERTIES'],
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
