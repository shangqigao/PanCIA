import sys
sys.path.append('../')

import requests
import argparse
import pathlib
import logging
import warnings
import joblib
import copy
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torchbnn as bnn

from scipy.stats import zscore
from scipy.special import softmax
from torch_geometric.loader import DataLoader
from tiatoolbox import logger
from tiatoolbox.utils.misc import save_as_json

from sklearn import set_config
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer
from sklearn.metrics import balanced_accuracy_score as acc_scorer
from sklearn.metrics import f1_score as f1_scorer
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

from utilities.m_utils import mkdir, select_wsi, load_json, create_pbar, rm_n_mkdir, reset_logging, recur_find_ext, select_checkpoints

from analysis.feature_aggregation.m_gnn_therapy_response import TherapyGraphDataset, TherapyGraphArch, TherapyBayesGraphArch
from analysis.feature_aggregation.m_gnn_therapy_response import ScalarMovingAverage, R2TLoss
from analysis.feature_aggregation.m_graph_construction import visualize_radiomic_graph, visualize_pathomic_graph

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

def load_radiomic_properties(idx, radiomic_path, prop_keys=None):
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
    label_path = f"{graph_path}".replace(".json", ".label.npy")
    label = np.load(label_path)
    if label.ndim == 2: label = np.argmax(label, axis=1)
    feature = np.array(graph_dict["x"])
    assert feature.ndim == 2
    if subgraph_ids is not None:
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
    mode="mean"
    ):
    graph_dict = load_json(graph_path)
    feature = np.array(graph_dict["x"])
    assert feature.ndim == 2

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
        k = f"radiomics.feature{i}"
        feat_dict[k] = feat
    return {f"{idx}": feat_dict}

def load_wsi_level_features(idx, wsi_feature_path):
    feat_list = np.array(np.load(wsi_feature_path)).squeeze().tolist()
    feat_dict = {}
    for i, feat in enumerate(feat_list):
        k = f"pathomics.feature{i}"
        feat_dict[k] = feat
    return {f"{idx}": feat_dict}

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
    plt.savefig("a_07explainable_AI/coefficients.jpg")

def matched_therapy_graph(save_clinical_dir, save_graph_paths, dataset="MAMA-MIA"):
    clinical_info_path = f"{save_clinical_dir}/{dataset}_clinical_and_imaging_info.xlsx"
    df = pd.read_excel(clinical_info_path, sheet_name='dataset_info')
    
    # Prepare the response to therapy data
    df = df[df['pcr'].notna()]
    print("Clinical data strcuture:", df.shape)

    # filter graph properties 
    graph_names = [pathlib.Path(p).stem for p in save_graph_paths]
    graph_ids = [f"{n}".split("-")[0:2] for n in graph_names]
    graph_ids = ["-".join(d) for d in graph_ids]
    df = df[df["patient_id"].isin(graph_ids)]
    matched_indices = [graph_ids.index(d) for d in df["patient_id"]]
    logging.info(f"The number of matched clinical samples are {len(matched_indices)}")
    return df, matched_indices

def matched_pathomics_radiomics(save_pathomics_paths, save_radiomics_paths, save_clinical_dir, dataset="TCGA-RCC", project_ids=None):
    df = pd.read_csv(f"{save_clinical_dir}/TCIA_{dataset}_mappings.csv")
    if project_ids is not None: df = df[df["Collection Name"].isin(project_ids)]

    df = df[["Subject ID", "Series ID"]]
    pathomics_names = [pathlib.Path(p).stem for p in save_pathomics_paths]
    pathomics_ids = [f"{n}".split("-")[0:3] for n in pathomics_names]
    pathomics_ids = ["-".join(d) for d in pathomics_ids]
    radiomics_names = [pathlib.Path(p).stem for p in save_radiomics_paths]
    radiomics_all_ids = [f"{n}".split(".")[0:13] for n in radiomics_names]
    radiomics_end_ids = [f"{d[-1]}".split("_")[0] for d in radiomics_all_ids]
    radiomics_ids = [id1[0:12] + [id2] for id1, id2 in zip(radiomics_all_ids, radiomics_end_ids)]
    radiomics_ids = [".".join(d) for d in radiomics_ids]

    df = df[df["Subject ID"].isin(pathomics_ids) & df["Series ID"].isin(radiomics_ids)] 
    matched_pathomics_indices, matched_radiomics_indices = [], []
    for subject_id, series_id in zip(df["Subject ID"], df["Series ID"]):
        matched_pathomics_indices.append(pathomics_ids.index(subject_id))
        matched_radiomics_indices.append(radiomics_ids.index(series_id))
    logging.info(f"The number of matched pathomic and radiomic cases are {len(matched_pathomics_indices)}")
    return matched_pathomics_indices, matched_radiomics_indices

def randomforest(split_idx, tr_X, tr_y, refit, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = RandomForestClassifier(max_depth=2, random_state=42)
    param_grid={"model_max_depth": [None] + list(range(1, 21))}
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc"
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
    depths = cv_results.param_model__max_depth
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(depths, mean)
    ax.fill_between(depths, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("linear")
    ax.set_ylabel(refit)
    ax.set_xlabel("max depth")
    ax.axvline(gcv.best_params_["model__max_depth"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"analysis/outcome_prediction/cross_validation_fold{split_idx}.jpg")

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def xgboost(split_idx, tr_X, tr_y, refit, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = XGBClassifier(max_depth=2, random_state=42)
    param_grid={"model_max_depth": [None] + list(range(1, 21))}
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc"
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
    depths = cv_results.param_model__max_depth
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(depths, mean)
    ax.fill_between(depths, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("linear")
    ax.set_ylabel(refit)
    ax.set_xlabel("max depth")
    ax.axvline(gcv.best_params_["model__max_depth"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"analysis/outcome_prediction/cross_validation_fold{split_idx}.jpg")

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)

    return pipe

def logisticregression(split_idx, tr_X, tr_y, refit, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.5, 
        C=1.0, max_iter=1000, random_state=42
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
        "f1": "f1",
        "roc_auc": "roc_auc"
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
    alphas = cv_results.param_model__C
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel(refit)
    ax.set_xlabel("alpha")
    ax.axvline(gcv.best_params_["model__C"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"analysis/outcome_prediction/cross_validation_fold{split_idx}.jpg")

    # Visualize coefficients of the best estimator
    best_model = gcv.best_estimator_.named_steps["model"]
    best_coefs = pd.DataFrame(best_model.coef_, index=tr_X.columns, columns=["coefficient"])
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
    plt.savefig(f"analysis/outcome_prediction/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)
    return pipe

def svc(split_idx, tr_X, tr_y, refit, n_jobs):
    # choosing parameters by cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = SVC(C=1, kernel='rbf', random_state=42)
    param_grid={"model__C": np.logspace(-3, 3, 7)}
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    scoring = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc"
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
    alphas = cv_results.param_model__C
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel(refit)
    ax.set_xlabel("alpha")
    ax.axvline(gcv.best_params_["model__C"], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    plt.savefig(f"analysis/outcome_prediction/cross_validation_fold{split_idx}.jpg")

    # Visualize coefficients of the best estimator
    best_model = gcv.best_estimator_.named_steps["model"]
    best_coefs = pd.DataFrame(best_model.coef_, index=tr_X.columns, columns=["coefficient"])
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
    plt.savefig(f"analysis/outcome_prediction/best_coefficients_fold{split_idx}.jpg") 

    # perform prediction using the best params
    pipe.set_params(**gcv.best_params_)
    pipe.fit(tr_X, tr_y)
    return pipe

def load_radiomics(
        split_idx,
        data,
        radiomics_aggregated_mode,
        radiomics_keys,
        use_graph_properties,
        n_jobs
    ):

    if radiomics_aggregated_mode in ["ABMIL", "SPARRA"]:
        radiomics_paths = []
        for p in data:
            path = pathlib.Path(p[0]["radiomics"])
            dir = path.parents[0] / radiomics_aggregated_mode / f"0{split_idx}"
            name = path.name.replace(".json", ".npy")
            radiomics_paths.append(dir / name)
    else:
        radiomics_paths = [p[0]["radiomics"] for p in data]

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
        if radiomics_aggregated_mode == "Mean":
            prop_paths = [f"{p}".replace("_aggr_mean.npy", "_graph_properties.json") for p in prop_paths]
        else:
            prop_paths = [f"{p}".replace(".json", "_graph_properties.json") for p in prop_paths]
        prop_dict_list = [load_json(p) for p in prop_paths]
        prop_X = [prepare_graph_properties(d, omics="radiomics") for d in prop_dict_list]
        prop_X = pd.DataFrame(prop_X)
        radiomics_X = pd.concat([radiomics_X, prop_X], axis=1)
    return radiomics_X

def load_pathomics(
        split_idx,
        data,
        pathomics_aggregated_mode,
        pathomics_keys,
        use_graph_properties,
        n_jobs
    ):

    if pathomics_aggregated_mode in ["ABMIL", "SPARRA"]:
        pathomics_paths = []
        for p in data:
            path = pathlib.Path(p[0]["pathomics"])
            dir = path.parents[0] / pathomics_aggregated_mode / f"0{split_idx}"
            name = path.name.replace(".json", ".npy")
            pathomics_paths.append(dir / name)
    else:
        pathomics_paths = [p[0]["pathomics"] for p in data]

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

def response2therapy(
    split_path,
    radiomics_keys=None,
    pathomics_keys=None,
    used="all", 
    n_jobs=32,
    radiomics_aggregation=False,
    radiomics_aggregated_mode=None,
    pathomics_aggregation=False,
    pathomics_aggregated_mode=None,
    model="LR",
    refit="roc_auc",
    feature_selection=True,
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
        if used == "radiopathomics":
            radiomics_tr_X = load_radiomics(
                split_idx=split_idx,
                data=data_tr,
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
                pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs
            )
            print("Selected testing pathomics:", pathomics_te_X.shape)
            print(pathomics_te_X.head())

            te_X = pd.concat([radiomics_te_X, pathomics_te_X], axis=1)
        elif used == "pathomics":
            pathomics_tr_X = load_pathomics(
                split_idx=split_idx,
                data=data_tr,
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
                pathomics_aggregated_mode=pathomics_aggregated_mode,
                pathomics_keys=pathomics_keys,
                use_graph_properties=use_graph_properties,
                n_jobs=n_jobs
            )
            print("Selected testing pathomics:", pathomics_te_X.shape)
            print(pathomics_te_X.head())

            tr_X, te_X = pathomics_tr_X, pathomics_te_X
        elif used == "radiomics":
            radiomics_tr_X = load_radiomics(
                split_idx=split_idx,
                data=data_tr,
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
            selector.fit(tr_X[tr_y["label"]])
            selected_names = selector.get_feature_names_out().tolist()
            num_removed = len(tr_X.columns) - len(selected_names)
            print(f"Removing {num_removed} low-variance features...")
            tr_X = tr_X[selected_names]
            te_X = te_X[selected_names]
            print("Selecting univariate feature...")
            selector = SelectKBest(score_func=f_classif, k=n_selected_features)
            _ = selector.fit_transform(tr_X, tr_y["label"])
            selected_mask = selector.get_support()
            selected_names = tr_X.columns[selected_mask]
            print(f"Selected features: {len(selected_names)}")
            tr_X = tr_X[selected_names]
            te_X = te_X[selected_names]

        # model selection
        print("Selecting classifier...")
        if model == "RF":
            predictor = randomforest(split_idx, tr_X, tr_y, refit, n_jobs)
        elif model == "XG":
            predictor = xgboost(split_idx, tr_X, tr_y, refit, n_jobs)
        elif model == "LR":
            predictor = logisticregression(split_idx, tr_X, tr_y, refit, n_jobs)
        elif model == "SVC":
            predictor = svc(split_idx, tr_X, tr_y, refit, n_jobs)

        pred = predictor.predict(te_X)
        prob = predictor.predict_proba(te_X)
        num_class = prob.shape[1]

        label = te_y["label"].to_numpy()
        label = np.argmax(prob, axis=1)
        acc = acc_scorer(label, pred)
        f1 = f1_scorer(label, pred)

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

        print(f"Updating regression results on fold {split_idx}")
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
        balanced=False
):
    """Helper to generate splits
    Args:
        x (list): a list of image paths
        y (list): a list of annotation paths
        train (float or int): if int, number of training samples per class
        if float, ratio of training samples
        test (float): ratio of testing samples
        num_folds (int): number of folds for cross-validation
        seed (int): random seed
        balanced (bool): if true, sampling equal number of samples for each class
    Returns:
        splits (list): a list of folds, each fold consists of train, valid, and test splits
    """
    if balanced:
        assert train >= 1, "The number training samples should be no less than 1"
        assert test < 1, "The ratio of testing samples should be less than 1"
        outer_splitter = StratifiedShuffleSplit(
            n_splits=num_folds,
            test_size=test,
            random_state=seed,
        )
    else:
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
    random_seed = seed
    for train_valid_idx, test_idx in outer_splitter.split(x, l):
        test_x = [x[idx] for idx in test_idx]
        test_y = [y[idx] for idx in test_idx]
        x_ = [x[idx] for idx in train_valid_idx]
        y_ = [y[idx] for idx in train_valid_idx]
        l_ = [l[idx] for idx in train_valid_idx]

        l_ = np.array(l_)
        if balanced:
            train_idx, valid_idx = BalancedShuffleSplitter(l_, train, random_seed)
            random_seed += 1
        else:
            train_idx, valid_idx = next(iter(inner_splitter.split(x_, l_)))

        valid_x = [x_[idx] for idx in valid_idx]
        valid_y = [y_[idx] for idx in valid_idx]
        train_x = [x_[idx] for idx in train_idx]
        train_y = [y_[idx] for idx in train_idx]

        assert len(set(train_x).intersection(set(valid_x))) == 0
        assert len(set(valid_x).intersection(set(test_x))) == 0
        assert len(set(train_x).intersection(set(test_x))) == 0

        splits.append(
            {
                "train": list(zip(train_x, train_y)),
                "valid": list(zip(valid_x, valid_y)),
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
        model = TherapyBayesGraphArch(**arch_kwargs)
        kl = {"loss": bnn.BKLLoss(), "weight": 0.1}
    else:
        model = TherapyGraphArch(**arch_kwargs)
        kl = None
    if pretrained is not None:
        model.load(pretrained)
    if on_gpu:
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    loss = R2TLoss()
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
        ds = TherapyGraphDataset(
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
                    f"{save_dir}/epoch={epoch:03d}.aux.dat",
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
        aggregation="SPARRA",
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
        "dim_target": 2,
        "layers": [256, 128, 256],
        "dropout": 0.5,
        "pool_ratio": pool_ratio,
        "conv": conv,
        "keys": omic_keys,
        "aggregation": aggregation
    }
    omics_name = "_".join(omic_keys)
    if BayesGNN:
        model_dir = model_dir / f"{omics_name}_Bayes_Response2Therapy_{conv}_{aggregation}"
    else:
        model_dir = model_dir / f"{omics_name}_Response2Therapy_{conv}_{aggregation}_1e-1"
    optim_kwargs = {
        "lr": 3e-4,
        "weight_decay": {"ABMIL": 1.0e-5, "SPARRA": 0.0}[aggregation],
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
        aggregation="SPARRA",
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
        "dim_target": 2,
        "layers": [256, 128, 256],
        "dropout": 0.5,
        "pool_ratio": pool_ratio,
        "conv": conv,
        "keys": omic_keys,
        "aggregation": aggregation
    }
    omics_name = "_".join(omic_keys)
    if BayesGNN:
        pretrained_dir = pretrained_dir / f"{omics_name}_Bayes_Response2Therapy_{conv}_{aggregation}"
    else:
        pretrained_dir = pretrained_dir / f"{omics_name}_Response2Therapy_{conv}_{aggregation}_1e-1"

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
    num_node_features,
    pretrained_model,
    conv="MLP",
    dropout=0,
    pool_ratio={"radiomics": 0.7, "pathomics": 0.2},
    omic_keys=["radiomics", "pathomics"],
    aggregation="SPARRA",
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

    logit, _, attention = list(zip(*outputs))
    logit = np.array(logit).squeeze()
    prob = softmax(logit, axis=1)[:, 1]
    attention = np.array(attention).squeeze()
    return prob, attention

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', default=None)
    parser.add_argument('--img_dir', default=None)
    parser.add_argument('--lab_mode', default="nnUNet", choices=["expert", "nnUNet", "BiomedParse"], type=str)
    parser.add_argument('--dataset', default="TCGA-RCC", type=str)
    parser.add_argument('--modality', default="CT", type=str)
    parser.add_argument('--format', default="nifti", choices=["dicom", "nifti"], type=str)
    parser.add_argument('--phase', default="pre-contrast", choices=["pre-contrast", "1st-contrast", "2nd-contrast", "multiple"], type=str)
    parser.add_argument('--site', default="breast", type=str)
    parser.add_argument('--target', default="tumor", type=str)
    parser.add_argument('--save_pathomics_dir', default=None)
    parser.add_argument('--save_radiomics_dir', default=None)
    parser.add_argument('--save_clinical_dir', default=None)
    parser.add_argument('--save_model_dir', default=None)
    parser.add_argument('--slide_mode', default="wsi", choices=["tile", "wsi"], type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--pathomics_mode', default="None", choices=["None", "cnn", "vit", "uni", "conch", "chief"], type=str)
    parser.add_argument('--pathomics_dim', default=1024, choices=[2048, 384, 1024, 35, 768], type=int)
    parser.add_argument('--radiomics_mode', default="pyradiomics", choices=["None", "pyradiomics", "SegVol", "M3D-CLIP"], type=str)
    parser.add_argument('--radiomics_dim', default=768, choices=[107, 768, 768], type=int)
    parser.add_argument('--radiomics_aggr_mode', default="SPARRA", choices=["None", "Mean", "ABMIL", "SPARRA"], type=str, 
                        help="if graph has been aggregated, specify which mode, defaut is none"
                        )
    parser.add_argument('--pathomics_aggr_mode', default="SPARRA", choices=["None", "Mean", "ABMIL", "SPARRA", "radiopathomics_SPARRA"], type=str, 
                        help="if graph has been aggregated, specify which mode, defaut is none"
                        )
    parser.add_argument('--resolution', default=20, type=float)
    parser.add_argument('--units', default="power", type=str)
    args = parser.parse_args()

    if args.wsi_dir is None and args.img_dir is None:
        raise ValueError("Neither radiology nor pathology provided, please provide at least one!")
    
    ## get wsi path
    if args.wsi_dir is not None:
        wsi_dir = pathlib.Path(args.wsi_dir) / args.dataset
        all_paths = sorted(pathlib.Path(wsi_dir).rglob("*.svs"))
        excluded_wsi = ["TCGA-5P-A9KC-01Z-00-DX1", "TCGA-5P-A9KA-01Z-00-DX1", "TCGA-UZ-A9PQ-01Z-00-DX1"]
        wsi_paths = []
        for path in all_paths:
            wsi_name = f"{path}".split("/")[-1].split(".")[0]
            if wsi_name not in excluded_wsi: wsi_paths.append(path)
        logging.info(f"The number of WSIs on {args.dataset}: {len(wsi_paths)}")
    else:
        wsi_paths = None

    ## get image and label paths
    if args.img_dir is not None:
        if args.format == 'dicom':
            dicom_cases = pathlib.Path(args.img_dir).glob('*')
            dicom_cases = [p for p in dicom_cases if p.is_dir()]
            img_paths = [sorted(p.glob('*.dcm')) for p in dicom_cases]
        elif args.format == 'nifti':
            if args.phase == "pre-contrast":
                img_paths = sorted(pathlib.Path(args.img_dir).rglob('*_0000.nii.gz'))
            elif args.phase == "1st-contrast":
                img_paths = sorted(pathlib.Path(args.img_dir).rglob('*_0001.nii.gz'))
            elif args.phase == "2nd-contrast":
                img_paths = sorted(pathlib.Path(args.img_dir).rglob('*_0002.nii.gz'))
            else:
                case_paths = sorted(pathlib.Path(args.img_dir).glob('*'))
                case_paths = [p for p in case_paths if p.is_dir()]
                img_paths = []
                for path in case_paths:
                    nii_paths = path.glob("*.nii.gz")
                    multiphase_keys = ["_0000.nii.gz", "_0001.nii.gz", "_0002.nii.gz"]
                    nii_paths = [p for p in nii_paths if any(k in p.name for k in multiphase_keys)]
                    img_paths.append(sorted(nii_paths))
        logging.info(f"The number of {args.modality} images on {args.site}: {len(img_paths)}")
    else:
        img_paths = None
    
    ## set save dir
    if args.save_pathomics_dir is not None:
        save_pathomics_dir = pathlib.Path(f"{args.save_pathomics_dir}/{args.dataset}_{args.slide_mode}_pathomic_features/{args.pathomics_mode}")
    else:
        save_pathomics_dir = None
    if args.save_radiomics_dir is not None:
        save_radiomics_dir = pathlib.Path(f"{args.save_radiomics_dir}/{args.site}_{args.modality}_radiomic_features/{args.radiomics_mode}/{args.phase}/{args.lab_mode}")
    else:
        save_radiomics_dir = None
    if args.save_clinical_dir is not None:
        save_clinical_dir = pathlib.Path(f"{args.save_clinical_dir}")
    else:
        save_clinical_dir = None
    if args.save_model_dir is not None:
        save_model_dir = pathlib.Path(f"{args.save_model_dir}/{args.dataset}_response2therapy/{args.radiomics_mode}+{args.pathomics_mode}")
    else:
        save_model_dir = None

    # predict response to therapy
    if None not in (wsi_paths, save_pathomics_dir):
        pathomics_aggregation = args.pathomics_aggr_mode == "None" # false if load aggregated features else true
        pathomics_paths = sorted([save_pathomics_dir / f"{p.stem}.json" for p in wsi_paths])
    else:
        pathomics_paths = None

    if None not in (img_paths, save_radiomics_dir):
        radiomics_aggregation = args.radiomics_aggr_mode == "None" # false if load image-level features else true
        if radiomics_aggregation:
            radiomics_paths = sorted([save_radiomics_dir / f"{p.stem}_{args.target}.json" for p in img_paths])
        else:
            if args.radiomics_mode == "pyradiomics":
                radiomics_paths = sorted([save_radiomics_dir / f"{p.stem}_{args.target}_radiomics.json" for p in img_paths])
            elif args.radiomics_mode == "SegVol":
                if args.radiomics_aggregated_mode == "Mean":
                    radiomics_paths = sorted([save_radiomics_dir / f"{p.stem}_{args.target}_aggr_mean.json" for p in img_paths])
                else:
                    radiomics_paths = sorted([save_radiomics_dir / f"{p.stem}_{args.target}.json" for p in img_paths])
            elif args.radiomics_mode == "M3D-CLIP":
                radiomics_paths = sorted([save_radiomics_dir / f"{p.stem}_{args.target}.json" for p in img_paths])
    else:
        radiomics_paths = None

    # matching radiomics and pathomics
    if None not in (pathomics_paths, radiomics_paths):
        matched_pathomics_indices, matched_radiomics_indices = matched_pathomics_radiomics(
            save_pathomics_paths=pathomics_paths,
            save_radiomics_paths=radiomics_paths,
            save_clinical_dir=save_clinical_dir,
            dataset=args.dataset,
            project_ids=None #["TCGA-KIRC"]
        )
        matched_pathomics_paths = [pathomics_paths[i] for i in matched_pathomics_indices]
        matched_radiomics_paths = [radiomics_paths[i] for i in matched_radiomics_indices]
    else:
        matched_pathomics_paths = pathomics_paths
        matched_radiomics_paths = radiomics_paths

    # split data set
    num_folds = 5
    test_ratio = 0.2
    train_ratio = 0.8 * 0.8
    valid_ratio = 0.8 * 0.2
    data_types = ["radiomics", "pathomics"]

    if None not in (matched_pathomics_paths, matched_radiomics_paths):
        df, matched_i = matched_therapy_graph(save_clinical_dir, matched_pathomics_paths)
        matched_pathomics_paths = [matched_pathomics_paths[i] for i in matched_i]
        matched_radiomics_paths = [matched_radiomics_paths[i] for i in matched_i]
        kr, kp = data_types[0], data_types[1]
        matched_graph_paths = [{kr : r, kp : p} for r, p in zip(matched_radiomics_paths, matched_pathomics_paths)]
    elif matched_pathomics_paths is not None and matched_radiomics_paths is None:
        df, matched_i = matched_therapy_graph(save_clinical_dir, matched_pathomics_paths)
        matched_pathomics_paths = [matched_pathomics_paths[i] for i in matched_i]
        kr, kp = data_types[0], data_types[1]
        matched_graph_paths = [{kr : None, kp : p} for p in matched_pathomics_paths]
    elif matched_radiomics_paths is not None and matched_pathomics_paths is None:
        df, matched_i = matched_therapy_graph(save_clinical_dir, matched_radiomics_paths)
        matched_radiomics_paths = [matched_radiomics_paths[i] for i in matched_i]
        kr, kp = data_types[0], data_types[1]
        matched_graph_paths = [{kr : r, kp : None} for r in matched_radiomics_paths]
    else:
        raise ValueError("Cannot find matched radiomics or pathomics!")

    y = df[['pcr']].to_numpy(dtype=np.float32).tolist()
    splits = generate_data_split(
        x=matched_graph_paths,
        y=y,
        train=train_ratio,
        valid=valid_ratio,
        test=test_ratio,
        num_folds=num_folds
    )
    mkdir(save_model_dir)
    split_path = f"{save_model_dir}/response2therapy_{args.radiomics_mode}_{args.pathomics_mode}_splits.dat"
    joblib.dump(splits, split_path)
    splits = joblib.load(split_path)
    num_train = len(splits[0]["train"])
    logging.info(f"Number of training samples: {num_train}.")
    num_valid = len(splits[0]["valid"])
    logging.info(f"Number of validating samples: {num_valid}.")
    num_test = len(splits[0]["test"])
    logging.info(f"Number of testing samples: {num_test}.")

    # response prediction from the splits
    response2therapy(
        split_path=split_path,
        used=["radiomics", "pathomics", "radiopathomics"][0],
        n_jobs=8,
        radiomics_aggregation=radiomics_aggregation,
        radiomics_aggregated_mode=args.radiomics_aggregated_mode,
        pathomics_aggregation=pathomics_aggregation,
        pathomics_aggregated_mode=args.pathomics_aggregated_mode,
        radiomics_keys=None, #radiomic_propereties,
        pathomics_keys=None, #["TUM", "NORM", "DEB"],
        model=["RF", "XG", "LR", "SVC"][0],
        refit=["accuracy", "f1", "roc_auc"][0],
        feature_selection=True,
        n_selected_features=64,
        use_graph_properties=False
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
    #     aggregation=["ABMIL", "SPARRA"][1],
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
    #     aggregation=["ABMIL", "SPARRA"][1],
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
    # pretrained_model = save_model_dir / "pathomics_Survival_Prediction_GCNConv_SPARRA/00/epoch=019.weights.pth"
    # hazard, attention = test(
    #     graph_path=graph_path,
    #     scaler_path=scaler_paths,
    #     num_node_features=omics_dims,
    #     pretrained_model=pretrained_model,
    #     conv="GCNConv",
    #     dropout=0,
    #     pool_ratio=omics_pool_ratio,
    #     omic_keys=list(omics_modes.keys()),
    #     aggregation=["ABMIL", "SPARRA"][1],
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
    # pretrained_model = save_model_dir / "radiomics_Survival_Prediction_GCNConv_SPARRA_1e-1/00/epoch=019.weights.pth"
    # hazard, attention = test(
    #     graph_path=graph_path,
    #     scaler_path=scaler_paths,
    #     num_node_features=omics_dims,
    #     pretrained_model=pretrained_model,
    #     conv="GCNConv",
    #     dropout=0,
    #     pool_ratio=omics_pool_ratio,
    #     omic_keys=list(omics_modes.keys()),
    #     aggregation=["ABMIL", "SPARRA"][1],
    # )
    # visualize_radiomic_graph(
    #     img_path=img_path,
    #     lab_path=lab_path,
    #     save_graph_dir=save_radiomics_dir,
    #     class_name=class_name,
    #     attention=None,
    #     n_jobs=32
    # )
