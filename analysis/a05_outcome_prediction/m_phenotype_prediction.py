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
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer
from sklearn.metrics import balanced_accuracy_score as acc_scorer
from sklearn.metrics import f1_score as f1_scorer
from xgboost import XGBClassifier

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

def phenotype_classification(
    split_path,
    radiomics_keys=None,
    pathomics_keys=None,
    omics="all", 
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
    use_graph_properties=False
    ):
    splits = joblib.load(split_path)
    predict_results = {}
    for split_idx, split in enumerate(splits):
        print(f"Performing cross-validation on fold {split_idx}...")
        data_tr, data_va, data_te = split["train"], split["valid"], split["test"]
        data_tr = data_tr + data_va

        tr_y = np.array([p[1] for p in data_tr])
        tr_y = pd.DataFrame({'label': tr_y})
        te_y = np.array([p[1] for p in data_te])
        te_y = pd.DataFrame({'label': te_y})

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
        
        # feature selection
        if feature_selection:
            print("Selecting features...")
            selector = VarianceThreshold(threshold=feature_var_threshold)
            selector.fit(tr_X)
            selected_names = selector.get_feature_names_out().tolist()
            num_removed = len(tr_X.columns) - len(selected_names)
            print(f"Removing {num_removed} low-variance features...")
            tr_X = tr_X[selected_names]
            te_X = te_X[selected_names]
            if n_selected_features is not None:
                print("Selecting univariate feature...")
                selector = SelectKBest(score_func=f_classif, k=n_selected_features)
                _ = selector.fit_transform(tr_X, tr_y["label"])
                selected_mask = selector.get_support()
                selected_names = tr_X.columns[selected_mask]
                tr_X = tr_X[selected_names]
                te_X = te_X[selected_names]
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

        pred = predictor.predict(te_X)
        prob = predictor.predict_proba(te_X)
        num_class = prob.shape[1]

        label = te_y["label"].to_numpy(dtype=int)
        acc = acc_scorer(label, pred)
        f1 = f1_scorer(label, pred, average="macro")

        prob = prob[:, 1] if prob.shape[1] == 2 else prob
        auroc = auroc_scorer(label, prob, multi_class="ovr")

        if num_class == 2:
            auprc = auprc_scorer(label, prob)
        else:
            onehot = np.eye(num_class)[label]
            auprc = auprc_scorer(onehot, prob)
        
        scores_dict = {
            "acc": acc,
            "f1": f1,
            "auroc": auroc,
            "auprc": auprc
        }

        print(f"Updating predicted results on fold {split_idx}")
        predict_results.update({f"Fold {split_idx}": scores_dict})
    print(predict_results)
    for k in scores_dict.keys():
        arr = np.array([v[k] for v in predict_results.values()])
        print(f"CV {k} mean+std", arr.mean(), arr.std())
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

    l = np.array(y)

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
                logit, label = output
                logit = np.array(logit).squeeze()
                label = np.array(label).squeeze()
                assert logit.ndim == 2

                if not arch_kwargs['binary_cls']:
                    prob = softmax(logit, axis=1)
                    num_class = prob.shape[1]
                    pred = np.argmax(prob, axis=1)

                    val = acc_scorer(label, pred)
                    logging_dict[f"{loader_name}-acc"] = val

                    val = f1_scorer(label, pred)
                    logging_dict[f"{loader_name}-f1"] = val

                    prob = prob[:, 1] if num_class == 2 else prob
                    val = auroc_scorer(label, prob, multi_class="ovr")
                    logging_dict[f"{loader_name}-auroc"] = val

                    if num_class == 2:
                        val = auprc_scorer(label, prob)
                    else:
                        onehot = np.eye(num_class)[label]
                        val = auprc_scorer(onehot, prob)
                    logging_dict[f"{loader_name}-auprc"] = val
                else:
                    prob = 1 / (1 + np.exp(-logit))
                    num_class = prob.shape[1]
                    pred = (prob > 0.5).astype(np.int8)

                    acc_list = []
                    for i in range(num_class):
                        acc_list.append(acc_scorer(label[:, i], pred[:, i]))
                    logging_dict[f"{loader_name}-acc"] = sum(acc_list) / num_class

                    f1_list = []
                    for i in range(num_class):
                        f1_list.append(f1_scorer(label[:, i], pred[:, i]))
                    logging_dict[f"{loader_name}-f1"] = sum(f1_list) / num_class

                    auroc_list = []
                    for i in range(num_class):
                        auroc_list.append(auroc_scorer(label[:, i], pred[:, i]))
                    logging_dict[f"{loader_name}-auroc"] = sum(auroc_list) / num_class

                    auprc_list = []
                    for i in range(num_class):
                        auprc_list.append(auprc_scorer(label[:, i], pred[:, i]))
                    logging_dict[f"{loader_name}-auprc"] = sum(auroc_list) / num_class

                logging_dict[f"{loader_name}-raw-logit"] = logit
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
        "aggregation": arch_opt['AGGREGATION'],
        'binary_cls': arch_opt['BINARY_CLS']
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
        "aggregation": arch_opt['AGGREGATION'],
        'binary_cls': arch_opt['BINARY_CLS']
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
    
        # * Calculate split statistics
        logits = np.array(logits).squeeze()
        prob = softmax(logits, axis=1)
        num_class = prob.shape[1]
        label = np.array([v[1] for v in split["test"]])
        if num_class <= 2:
            label = (label > 0).astype(np.int32)
            onehot = label
        else:
            onehot = np.eye(num_class)[label]  
        
        ## compute per-class accuracy
        pred = np.argmax(prob, axis=1)
        uids = np.unique(label)
        acc_scores = []
        for i in range(len(uids)):
            indices = label == uids[i]
            score = acc_scorer(label[indices], pred[indices])
            acc_scores.append(score)

        cum_stats.append(
            {
                "acc": np.array(acc_scores),
                "auroc": auroc_scorer(label, prob, average=None, multi_class="ovr"), 
                "auprc": auprc_scorer(onehot, prob, average=None),
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
        "aggregation": arch_opt['AGGREGATION'],
        'binary_cls': arch_opt['BINARY_CLS']
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
        save_model_dir = save_model_dir / f"{opt['DATASET']}_phenotype_{opt['OUTCOME']['VALUE']}"
        save_model_dir = save_model_dir / f"{radiomics_mode}+{pathomics_mode}"
    else:
        save_model_dir = None

    # load patient outcomes
    df_outcome = prepare_patient_outcome(
        outcome_dir=opt['SAVE_OUTCOME_dir'],
        subject_ids=list(omics_info['omics_paths'].keys()),
        outcome=opt['OUTCOME']['VALUE']
    )
    subject_ids = df_outcome['SubjectID'].to_list()
    logger.info(f"Found {len(subject_ids)} subjects with phenotypes")

    omics_paths = [[k, omics_info['omics_paths'][k]] for k in subject_ids]
    
    y = df_outcome['PhenotypeClass'].to_numpy(dtype=np.float32).tolist()

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
        phenotype_classification(
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
