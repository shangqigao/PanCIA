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
from functools import reduce
from collections import OrderedDict

from torch_geometric.loader import DataLoader
from tiatoolbox import logger
from tiatoolbox.utils.misc import save_as_json

from sklearn.metrics import f1_score, r2_score
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from utilities.m_utils import mkdir, load_json, create_pbar, rm_n_mkdir, reset_logging

from analysis.a04_feature_aggregation.m_gnn_multitask_learning import MultiTaskGraphDataset, MultiTaskGraphArch, MultiTaskBayesGraphArch
from analysis.a04_feature_aggregation.m_gnn_multitask_learning import ScalarMovingAverage, MultiPredEnsembleLoss

def prepare_multitask_outcomes(
    outcome_dir,
    subject_ids,
    outcomes,
    signature_ids=None,
    pooling="mean",
    minimum_per_class=20,
):
    """
    Multi-task outcome loader producing a unified table by SubjectID.
    Supports survival, phenotype, and signature outcomes (multi-column).
    """

    # --------------------- helpers ---------------------
    def add_subject_id(df):
        df["SubjectID"] = df["SampleID"].str.extract(r"^(TCGA-\w\w-\w{4})")
        return df[df["SubjectID"].isin(subject_ids)]

    def pool(df):
        if pooling == "mean":
            return df.groupby("SubjectID", as_index=False).mean(numeric_only=True)
        elif pooling == "max":
            return df.groupby("SubjectID", as_index=False).max(numeric_only=True)
        elif pooling == "min":
            return df.groupby("SubjectID", as_index=False).min(numeric_only=True)
        else:
            raise ValueError("Invalid pooling mode")

    def prefix_columns(df, prefix):
        cols = [c for c in df.columns if c != "SubjectID"]
        df = df.rename(columns={c: f"{prefix}_{c}" for c in cols})
        return df

    # --------------------- collector ---------------------
    tables = {}

    # =====================================================
    # 1. SURVIVAL TASKS
    # =====================================================
    survival_specs = {
        "OS": ("OS", "OS.time"),
        "DSS": ("DSS", "DSS.time"),
        "DFI": ("DFI", "DFI.time"),
        "PFI": ("PFI", "PFI.time"),
    }

    surv_df = pd.read_csv(f"{outcome_dir}/phenotypes/clinical_data/survival_data.csv")
    surv_df = surv_df.drop_duplicates(subset="_PATIENT", keep="first")

    for task in outcomes:
        if task in survival_specs:
            event_col, time_col = survival_specs[task]
            df = surv_df.copy()

            df = df[df[event_col].notna() & df[time_col].notna()]
            df["SubjectID"] = df["_PATIENT"]

            df[f"{task}_duration"] = df[time_col]
            df[f"{task}_event"] = (df[event_col].astype(int) == 1).astype(int)

            df = df[df["SubjectID"].isin(subject_ids)]

            tables[task] = df[["SubjectID", f"{task}_duration", f"{task}_event"]]

    # =====================================================
    # 2. PHENOTYPE (CLASSIFICATION) TASKS
    # =====================================================
    if "ImmuneSubtype" in outcomes:
        df = pd.read_csv(f"{outcome_dir}/phenotypes/immune_subtype/immune_subtype.csv")
        df = add_subject_id(df)
        df = df[df["Subtype_Immune_Model_Based"].notna()]
        df = df.drop_duplicates(subset="SubjectID", keep="first")

        counts = df["Subtype_Immune_Model_Based"].value_counts()
        valid = counts[counts >= minimum_per_class].index
        df = df[df["Subtype_Immune_Model_Based"].isin(valid)]

        df["ImmuneSubtype_class"], _ = pd.factorize(df["Subtype_Immune_Model_Based"])
        tables["ImmuneSubtype"] = df[[
            "SubjectID", "Subtype_Immune_Model_Based", "ImmuneSubtype_class"
        ]]

    if "MolecularSubtype" in outcomes:
        df = pd.read_csv(f"{outcome_dir}/phenotypes/molecular_subtype/molecular_subtype.csv")
        df = add_subject_id(df)
        df = df[df["Subtype_Selected"].notna()]
        df = df.drop_duplicates(subset="SubjectID", keep="first")

        counts = df["Subtype_Selected"].value_counts()
        valid = counts[counts >= minimum_per_class].index
        df = df[df["Subtype_Selected"].isin(valid)]

        df["MolecularSubtype_class"], _ = pd.factorize(df["Subtype_Selected"])
        tables["MolecularSubtype"] = df[[
            "SubjectID", "Subtype_Selected", "MolecularSubtype_class"
        ]]

    if "PrimaryDisease" in outcomes:
        df = pd.read_csv(
            f"{outcome_dir}/phenotypes/sample_type_and_primary_disease/sample_type_primary_disease.csv")
        df = add_subject_id(df)
        df = df[df["_primary_disease"].notna()]
        df = df.drop_duplicates(subset="SubjectID", keep="first")

        counts = df["_primary_disease"].value_counts()
        valid = counts[counts >= minimum_per_class].index
        df = df[df["_primary_disease"].isin(valid)]

        df["PrimaryDisease_class"], _ = pd.factorize(df["_primary_disease"])
        tables["PrimaryDisease"] = df[[
            "SubjectID", "_primary_disease", "PrimaryDisease_class"
        ]]

    # =====================================================
    # 3. SIGNATURE TASKS (MANY COLUMNS)
    # =====================================================
    signature_paths = {
        "GeneProgrames": f"{outcome_dir}/signatures/gene_programs/gene_programs.csv",
        "HRDscore": f"{outcome_dir}/signatures/HRD_score/HRD_score.csv",
        "ImmuneSignatureScore": f"{outcome_dir}/signatures/immune_signature_score/immune_signature_score.csv",
        "StemnessScoreDNA": f"{outcome_dir}/signatures/stemness_score_DNA/stemness_scores_DNAmeth.csv",
        "StemScoreRNA": f"{outcome_dir}/signatures/stemness_score_RNA/stemness_scores_RNAexp.csv",
    }

    for task in outcomes:
        if task in signature_paths:
            df = pd.read_csv(signature_paths[task])
            df = add_subject_id(df)

            # Special normalization case
            if task == "HRDscore":
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols] / 100


            # Filter only selected signature columns
            if signature_ids is not None:
                df = df[["SubjectID"] + signature_ids]

            # Apply pooling
            df = pool(df)

            # Rename with prefix to avoid column collisions
            df = prefix_columns(df, prefix=task)

            tables[task] = df

    # =====================================================
    # 4. PROTEIN EXPRESSION TASKS (MANY COLUMNS)
    # =====================================================
    if "ProteinExpression" in outcomes:
        df = pd.read_csv(f"{outcome_dir}/protein_expression/protein_expression.csv")
        df = add_subject_id(df)

        # Apply pooling
        df = pool(df)

        # Rename with prefix to avoid column collisions
        df = prefix_columns(df, prefix="ProteinExpression")

        # Keep only columns with >= minimum_per_class non-NaN values
        df = df.loc[:, df.notna().sum() >= minimum_per_class]

        tables["ProteinExpression"] = df

    # =====================================================
    # Final merge (outer join keeps missing outcomes)
    # =====================================================
    merged = reduce(
        lambda left, right: pd.merge(left, right, on="SubjectID", how="outer"),
        tables.values()
    )

    # keep only requested subjects
    merged = merged[merged["SubjectID"].isin(subject_ids)].reset_index(drop=True)

    logging.info(f"Final multi-task table shape: {merged.shape}")

    return merged


def extract_class_info_from_outcomes(outcomes):
    """
    Extract number of classes and class-name mappings
    for each classification task.

    Robust to extra columns.
    """

    class_info = {}

    for class_col in outcomes.columns:
        if not class_col.endswith("_class"):
            continue

        task = class_col.replace("_class", "")

        # Candidate name columns:
        candidate_cols = [
            c for c in outcomes.columns
            if c not in ("SubjectID", class_col)
        ]

        name_col = None

        for c in candidate_cols:
            tmp = outcomes[[c, class_col]].dropna()

            # Check one-to-one mapping: each class id → exactly one name
            mapping = tmp.groupby(class_col)[c].nunique()

            if (mapping == 1).all():
                name_col = c
                break

        if name_col is None:
            raise ValueError(
                f"Could not infer class-name column for task '{task}'. "
                f"Candidates tried: {candidate_cols}"
            )

        df_valid = outcomes[[name_col, class_col]].dropna()

        pairs = (
            df_valid
            .drop_duplicates(subset=class_col)
            .sort_values(class_col)
        )

        class_to_name = {
            int(row[class_col]): row[name_col]
            for _, row in pairs.iterrows()
        }

        name_to_class = {v: k for k, v in class_to_name.items()}

        class_info[task] = {
            "num_classes": len(class_to_name),
            "class_to_name": class_to_name,
            "name_to_class": name_to_class,
            "class_name_column": name_col,
        }

    return class_info


def generate_task_to_idx_grouped(outcomes, classes_dict):
    """
    Generate mapping from task type -> indices.
    Classification tasks are intervals returned as (start, end) where `end` is exclusive.
    Survival/regression single-output tasks are ints.

    Args:
        outcomes: list of dicts (length = batch size). Used only to discover task names.
        classes_dict: dict mapping classification task name -> dict{number of classes (K)}

    Returns:
        task_to_idx: dict with keys "survival","classification","regression" as described above.
    """
    task_to_idx = {"survival": {}, "classification": {}, "regression": {}}
    outcome_to_idx = {}
    if len(outcomes) == 0:
        return task_to_idx, outcome_to_idx

    idx = 0
    sample = outcomes[0]

    # ---------------- Survival ----------------
    # assume each survival task uses a single logit (hazard) -- change if you use 2-per-task
    for task in sample.get("survival", {}).keys():
        task_to_idx["survival"][task] = idx
        outcome_to_idx[task] = idx
        idx += 1

    # ---------------- Classification ----------------
    # allocate K logits per classification task; store interval [start, end) (end exclusive)
    for task in sample.get("classification", {}).keys():
        K = int(classes_dict[task].get("num_classes", 1))
        start = idx
        end = idx + K   # exclusive
        task_to_idx["classification"][task] = (start, end)
        for i in range(K): 
            name = classes_dict[task]["class_to_name"][i]
            outcome_to_idx[f"{task}_{name}"] = idx + i
        idx = end

    # ---------------- Regression ----------------
    # each regression subcolumn gets a single logit
    for task, subdict in sample.get("regression", {}).items():
        for subcol in subdict.keys():
            name = f"{task}_{subcol}"
            task_to_idx["regression"][name] = idx
            outcome_to_idx[subcol] = idx
            idx += 1

    return task_to_idx, outcome_to_idx


def count_subtasks_and_predictions(task_to_idx):
    """
    Count subtasks and predictions for the structure returned above.

    Subtask: each survival task, each classification task, each regression subcol (i.e. number of small tasks)
    Predictions: total number of logits (model output dim)
    """
    num_subtasks = 0
    num_predictions = 0

    for k, v in task_to_idx.items():
        if isinstance(v, dict):
            for name, entry in v.items():
                num_subtasks += 1
                if isinstance(entry, int):
                    num_predictions += 1
                elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                    start, end = entry
                    if not (isinstance(start, int) and isinstance(end, int)):
                        raise ValueError(f"Interval indices must be ints: {entry}")
                    if end < start:
                        raise ValueError(f"Invalid interval: start {start} >= end {end}")
                    num_predictions += (end - start)   # end is exclusive
                else:
                    raise ValueError(f"Unsupported task_to_idx entry: {entry}")
        else:
            raise ValueError("task_to_idx must be dict of dicts")

    return num_subtasks, num_predictions

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

def split_batch_output(output):
    """
    Split batch outputs into sample-wise outputs.

    Args:
        output: list of arrays, dicts, or list of dicts. Examples:
            [outputs, features, atte_dict]
            [outputs, labels]  (labels can be list of dicts)
            [outputs, features]
            [outputs]
    Returns:
        step_output: list of length batch_size, each element is a tuple
                     containing the i-th sample from each output
    """
    # --- Determine batch size from the first array or list of dicts ---
    batch_size = None
    for v in output:
        if hasattr(v, "shape"):  # array
            batch_size = v.shape[0]
            break
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            batch_size = len(v)
            break
    if batch_size is None:
        raise ValueError("No array or list-of-dict found to infer batch size.")

    # --- Split each element ---
    split_items = []
    for v in output:
        if hasattr(v, "shape"):  # numpy array
            split_items.append(np.split(v, batch_size, axis=0))
        elif isinstance(v, dict):  # dict of arrays
            split_items.append({k: np.split(val, batch_size, axis=0) for k, val in v.items()})
        else:
            raise TypeError(f"Unsupported output type: {type(v)}")

    # --- Build sample-wise outputs ---
    step_output = []
    for i in range(batch_size):
        sample_items = []
        for item in split_items:
            if isinstance(item, list):
                sample_items.append(item[i])
            elif isinstance(item, dict):
                sample_items.append({k: item[k][i] for k in item})
        step_output.append(tuple(sample_items))

    return step_output


def evaluate_multitask(logits, y_list, mask_list, task_to_idx):
    """
    Evaluate multi-task outputs considering masks, for batch-wise list of dicts.

    ```
    Args:
        logits: np.array, [B, N_total_outcomes]
        y_list: list of dicts (length B) with keys 'survival', 'classification', 'regression'
        mask_list: list of dicts (length B), same structure as y_list
        task_to_idx: dict mapping tasks to idx or index intervals

    Returns:
        metrics: dict of per-task metrics
        avg_metrics: dict of average metrics per big task
    """
    B = len(y_list)
    metrics = {"survival": [], "classification": [], "regression": []}

    # ---------------- Survival ----------------
    if task_to_idx.get("survival"):
        for i, task in enumerate(task_to_idx["survival"].keys()):
            dur_list, ev_list, hazard_list = [], [], []

            for b in range(B):
                y_surv = y_list[b]["survival"].squeeze(0)   # [num_tasks, 2]
                mask_surv = mask_list[b]["survival"].squeeze(0)  # [num_tasks]
                if mask_surv[i] == 0:
                    continue
                dur_list.append(y_surv[i, 0])
                ev_list.append(y_surv[i, 1])
                hazard_list.append(logits[b, task_to_idx["survival"][task]])

            if dur_list:
                dur = np.array(dur_list, dtype=np.float32)
                ev = np.array(ev_list, dtype=np.int32)
                hazard = np.array(hazard_list, dtype=np.float32)
                cidx = concordance_index(dur, -hazard, ev)
                metrics["survival"].append((task, cidx))

    # ---------------- Classification ----------------
    if task_to_idx.get("classification"):
        for i, task in enumerate(task_to_idx["classification"].keys()):
            idx_info = task_to_idx["classification"][task]

            labels_list, preds_list = [], []

            for b in range(B):
                y_cls = y_list[b]["classification"].squeeze(0)  # [num_tasks] or [num_tasks, ...]
                mask_cls = mask_list[b]["classification"].squeeze(0)  # [num_tasks]
                if mask_cls[i] == 0:
                    continue
                labels = np.array([y_cls[i]], dtype=int)

                if isinstance(idx_info, tuple):  # multi-class
                    start, end = idx_info
                    pred_logits = logits[b, start:end]
                    preds = np.array([np.argmax(pred_logits)], dtype=int)
                else:  # binary
                    preds = np.array([int(logits[b, idx_info] > 0)])

                labels_list.append(labels)
                preds_list.append(preds)

            if labels_list:
                labels_all = np.concatenate(labels_list)
                preds_all = np.concatenate(preds_list)
                f1 = f1_score(labels_all, preds_all, average="macro")
                metrics["classification"].append((task, f1))

    # ---------------- Regression ----------------
    if task_to_idx.get("regression"):
        for i, task in enumerate(task_to_idx["regression"].keys()):
            target_list, pred_list = [], []
            idx = task_to_idx["regression"][task]

            for b in range(B):
                y_reg = y_list[b]["regression"].squeeze(0)  # [num_tasks, n_subcols]
                mask_reg = mask_list[b]["regression"].squeeze(0)  # [num_tasks]
                if mask_reg[i] == 0:
                    continue
                target_list.append(np.array([y_reg[i]]))
                pred_list.append(np.array([logits[b, idx]]))

            if target_list:
                target_all = np.concatenate(target_list)
                pred_all = np.concatenate(pred_list)
                r2 = r2_score(target_all, pred_all)
                metrics["regression"].append((task, r2))

    # ---------------- Average metrics ----------------
    avg_metrics = {}
    for k in metrics:
        vals = [v for _, v in metrics[k]]
        avg_metrics[k] = np.mean(vals) if vals else np.nan

    return metrics, avg_metrics

import torch
import json

def export_all_weights(model, outcome_to_idx, save_dir):
    """
    Export ALL linear weights per modality.

    Output format:
    {
        modality: {
            outcome_name: {
                feature_name: weight_value
            }
        }
    }
    """

    idx_to_outcome = {v: k for k, v in outcome_to_idx.items()}
    output_dict = {}

    for modality, linear_layer in model.predictor_dict.items():

        # shape: [dim_target, input_dim]
        weight = linear_layer.weight.detach().cpu()
        dim_target, input_dim = weight.shape

        modality_dict = {}

        for outcome_idx in range(dim_target):

            outcome_name = idx_to_outcome[outcome_idx]
            weights_row = weight[outcome_idx]

            feature_dict = {}

            for feature_idx in range(input_dim):
                feature_name = f"feature_{feature_idx}"  # replace if real feature names exist
                feature_dict[feature_name] = float(weights_row[feature_idx].item())

            modality_dict[outcome_name] = feature_dict

        output_dict[modality] = modality_dict

    save_path = f"{save_dir}/all_predictor_weights.json"
    with open(save_path, "w") as f:
        json.dump(output_dict, f)

    print(f"Saved full weights to {save_path}")

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
        model = MultiTaskBayesGraphArch(**arch_kwargs)
        kl = {"loss": bnn.BKLLoss(), "weight": 0.1}
    else:
        model = MultiTaskGraphArch(**arch_kwargs)
        kl = None
    if pretrained is not None:
        model.load(pretrained)
        export_all_weights(model, arch_kwargs["outcome_to_idx"], pretrained.parent)

    # loss = MultiTaskLoss(tau_vi=optim_kwargs.get('tau_vi', 0.1))
    task_to_idx = model.task_to_idx
    graph_to_idx = list(model.dim_dict.keys())
    loss = MultiPredEnsembleLoss(task_to_idx, graph_to_idx)

    if on_gpu:
        model = model.to("cuda")
        loss = loss.to("cuda")
    else:
        model = model.to("cpu")
        loss = loss.to("cpu")

    optimizer_kwargs = {k: v for k, v in optim_kwargs.items() if k in ["lr", "weight_decay"]}
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss.parameters()), **optimizer_kwargs)
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, **optim_kwargs)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    loader_dict = {}
    for subset_name, subset in dataset_dict.items():
        _loader_kwargs = copy.deepcopy(loader_kwargs)
        # if not "train" in subset_name: 
        #     if _loader_kwargs["batch_size"] > 4:
        #         _loader_kwargs["batch_size"] = 4
        #     sampling_rate = 1.0
        ds = MultiTaskGraphDataset(
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
                    output = model.train_batch(model, batch_data, on_gpu, loss, optimizer, kl, optim_kwargs['l1_penalty'])
                    ema({"loss": output[0]})
                    pbar.postfix[1]["step"] = step
                    pbar.postfix[1]["EMA"] = ema.tracking_dict["loss"]
                else:
                    output = model.infer_batch(model, batch_data, on_gpu)
                    step_output += split_batch_output(output)
                pbar.update()
            pbar.close()

            logging_dict = {}
            if loader_name == "train":
                for val_name, val in ema.tracking_dict.items():
                    logging_dict[f"train-EMA-{val_name}"] = val
            elif "infer" in loader_name and any(v in loader_name for v in ["train", "valid"]):
                output = list(zip(*step_output))
                logit, y_list, mask_list = output
                logit = np.array(logit).squeeze()
                y_list = list(y_list)
                mask_list = list(mask_list)

                metrics, avg_metrics = evaluate_multitask(logit, y_list, mask_list, task_to_idx)

                # Log per-class metrics
                logging_dict[f"{loader_name}-Cindex"] = avg_metrics["survival"]
                logging_dict[f"{loader_name}-F1"] = avg_metrics["classification"]
                logging_dict[f"{loader_name}-R2"] = avg_metrics["regression"]

                # Optionally save best model based on survival
                if "valid-A" in loader_name and avg_metrics["survival"] > best_score: 
                    best_score = avg_metrics["survival"]
                    model.save(f"{save_dir}/best_model.weights.pth")

                logging_dict[f"{loader_name}-raw-logit"] = logit
                logging_dict[f"{loader_name}-raw-true"] = y_list

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
        model_dir,
        omics_dims,
        omics_pool_ratio,
        arch_opt,
        train_opt,
        task_to_idx
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
    # node_scalers = [joblib.load(scaler_path[k]) for k in omics_modes.keys()] 
    # transform_dict = {k: s.transform for k, s in zip(omics_modes.keys(), node_scalers)}
    
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
        "aggregation": arch_opt['AGGREGATION']['VALUE'],
        "num_groups": arch_opt['AGGREGATION']['NUM_GROUPS'],
        "task_to_idx": task_to_idx
    }

    omics_name = "_".join(arch_opt['OMICS'])
    if arch_opt['BayesGNN']:
        model_dir = model_dir / f"{omics_name}_Bayes_{arch_opt['GNN']}_{arch_opt['AGGREGATION']['VALUE']}"
    else:
        model_dir = model_dir / f"{omics_name}_{arch_opt['GNN']}_{arch_opt['AGGREGATION']['VALUE']}"

    # if multi-omics and multi-scale, higher dimension, so stronger regularization
    if len(omics_dims) > 1 and len(omics_dims['radiomics']) > 1:
        tau_vi = 0.1
    else:
        tau_vi = 0.1
    print(f"Setting the weight of variational loss to {tau_vi}")

    optim_kwargs = {
        "lr": 3e-4,
        "weight_decay": {
            "MEAN": train_opt['WIEGHT_DECAY']['MEAN'],
            "ABMIL": train_opt['WIEGHT_DECAY']['ABMIL'], 
            "SPARRA": train_opt['WIEGHT_DECAY']['SPARRA']
        }[arch_opt['AGGREGATION']['VALUE']],
        "l1_penalty": {
            "MEAN": train_opt['L1_PENALTY']['MEAN'],
            "ABMIL": train_opt['L1_PENALTY']['ABMIL'], 
            "SPARRA": train_opt['L1_PENALTY']['SPARRA']
        }[arch_opt['AGGREGATION']['VALUE']],
        "tau_vi": tau_vi
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
            preproc_func=None,
            BayesGNN=arch_opt['BayesGNN'],
            data_types=arch_opt['OMICS'],
            sampling_rate=train_opt['SAMPLING_RATE']
        )
    return

def inference(
        split_path,
        omics_dims,
        omics_pool_ratio,
        arch_opt,
        infer_opt,
        task_to_idx,
        outcome_to_idx,
        pretrained_model_dir,
):
    """survival prediction
    """
    splits = joblib.load(split_path)
    # node_scalers = [joblib.load(scaler_path[k]) for k in omics_modes.keys()] 
    # transform_dict = {k: s.transform for k, s in zip(omics_modes.keys(), node_scalers)}
    
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
        "aggregation": arch_opt['AGGREGATION']['VALUE'],
        "num_groups": arch_opt['AGGREGATION']['NUM_GROUPS'],
        "task_to_idx": task_to_idx,
        "outcome_to_idx": outcome_to_idx
    }

    omics_name = "_".join(arch_opt['OMICS'])
    if arch_opt['BayesGNN']:
        model_folder = f"{omics_name}_Bayes_{arch_opt['GNN']}_{arch_opt['AGGREGATION']['VALUE']}"
    else:
        # model_folder = f"{omics_name}_{arch_opt['GNN']}_{arch_opt['AGGREGATION']['VALUE']}"
        model_folder = 'radiomics_pathomics_GCNConv_SPARRA_heter_vi0.1ae1e-2_pool0.7_relugate_>20'
    model_dir = pretrained_model_dir / model_folder

    predict_results = {}
    save_results = {}
    for split_idx, split in enumerate(splits):
        if infer_opt['SAVE_OMICS']:
            new_split = {"infer": [v[0] for v in split["test"]]}
        else:
            new_split = {"infer-valid": split["test"]}
        chkpts = model_dir / f"0{split_idx}/epoch=039.weights.pth"
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
            preproc_func=None,
            pretrained=chkpts,
            BayesGNN=arch_opt['BayesGNN'],
            data_types=arch_opt['OMICS'],
            sampling_rate=infer_opt['SAMPLING_RATE']
        )
        outputs = list(zip(*outputs))
        if len(outputs) == 3:
            logits, out1, out2 = outputs
        elif len(outputs) == 2:
            logits, out1 = outputs
        # saving average features
        if infer_opt['SAVE_OMICS']:
            subject_ids = [d[0] for d in new_split["infer"]]
            save_dir = model_dir / arch_opt['AGGREGATION']['VALUE']
            mkdir(save_dir)
            for i, subject_id in enumerate(subject_ids): 
                # save_name = f"{subject_id}.json"
                # save_path = f"{save_dir}/{save_name}"
                # scores = logits[i].squeeze().tolist()
                # scores_dict = {k: scores[v] for k, v in outcome_to_idx.items()}
                # with open(save_path, "w") as f:
                #     json.dump(scores_dict, f, indent=4)
                save_name = f"{subject_id}.npy"
                save_path = f"{save_dir}/{save_name}"
                np.save(save_path, out1[i])
            print(f"Saving subject omics to {save_dir}...")
        else:
            logit = np.array(logit).squeeze()
            y_list = list(out1)
            mask_list = list(out2)

            metrics, avg_metrics = evaluate_multitask(logit, y_list, mask_list, task_to_idx)

            scores_dict = {
                "C-Index": avg_metrics["survival"],
                "F1": avg_metrics["classification"],
                "R2": avg_metrics["regression"]
            }     
            predict_results.update({f"Fold {split_idx}": scores_dict})
            save_results.update({f"Fold {split_idx}": metrics})

    if len(predict_results) > 0:
        print(predict_results)
        for k in scores_dict.keys():
            arr = np.array([v[k] for v in predict_results.values()])
            print(f"CV {k} mean+std", arr.mean(), arr.std())

    if len(save_results) > 0:
        save_path = model_dir / "multitask_learning_results.json"
        with open(save_path, "w") as f:
            json.dump(save_results, f, indent=4)

    return

def test(
    graph_path,
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

    # node_scalers = [joblib.load(scaler_path[k]) for k in arch_opt['OMICS']] 
    # transform_dict = {k: s.transform for k, s in zip(arch_opt['OMICS'], node_scalers)}
    
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
        preproc_func=None,
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
        save_model_dir = save_model_dir / f"{opt['DATASET']}_multitask"
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
    outcomes = []
    for v in opt['OUTCOME'].values(): outcomes += v
    df_outcome = prepare_multitask_outcomes(
        outcome_dir=opt['SAVE_OUTCOME_DIR'],
        subject_ids=subject_ids,
        outcomes=outcomes
    )
    subject_ids = df_outcome['SubjectID'].to_list()
    logger.info(f"Found {len(subject_ids)} subjects with outcomes")
    classes_dict = extract_class_info_from_outcomes(df_outcome)

    omics_paths = [[k, omics_info['omics_paths'][k]] for k in subject_ids]

    SURVIVAL = opt['OUTCOME']['SURVIVAL']
    CLASSIFICATION = opt['OUTCOME']['CLASSIFICATION']
    REGRESSION = opt['OUTCOME']['REGRESSION']

    # Helper to test NaN
    def _safe_val(x):
        if pd.isna(x):
            return None
        return x

    outcomes = []
    # Precompute columns present
    cols = set(df_outcome.columns)

    for _, row in df_outcome.iterrows():
        y_i = {
            "survival": {},
            "classification": {},
            "regression": {}
        }

        # --- Survival: use task_duration and task_event columns ---
        for task in SURVIVAL:
            dur_col = f"{task}_duration"
            ev_col = f"{task}_event"

            duration = _safe_val(row[dur_col]) if dur_col in cols else None
            event = _safe_val(row[ev_col]) if ev_col in cols else None

            # convert to numeric types if present
            if duration is not None:
                # ensure float
                try:
                    duration = float(duration)
                except Exception:
                    duration = None
            if event is not None:
                try:
                    event = int(event)
                except Exception:
                    event = None

            # store a small dict to make survival loss calculation clear
            y_i["survival"][task] = {"duration": duration, "event": event}

        # --- Classification: keep raw label (if present) and factorized class (if present) ---
        for task in CLASSIFICATION:
            label_val = row[task] if task in cols else None
            class_col = f"{task}_class"
            class_val = row[class_col] if class_col in cols else None

            label_val = _safe_val(label_val)
            class_val = _safe_val(class_val)

            # convert factorized to int
            if class_val is not None:
                try:
                    class_val = int(class_val)
                except Exception:
                    class_val = None

            y_i["classification"][task] = {
                "label": label_val,        # original string label if exists
                "class": class_val         # integer class if exists
            }

        # --- Regression: collect all columns that start with f"{task}_" OR exact task column ---
        for task in REGRESSION:
            task_dict = OrderedDict()

            # 1) all prefixed columns, e.g. "GeneProgrames_GP1"
            prefixed_cols = [c for c in df_outcome.columns if c.startswith(f"{task}_") and c != "SubjectID"]
            for pc in prefixed_cols:
                val = _safe_val(row[pc])
                if val is not None:
                    try:
                        val = float(val)
                    except Exception:
                        val = None
                task_dict[pc] = val

            # 2) fallback: maybe there's a single column named exactly as the task
            if (not prefixed_cols) and (task in cols):
                val = _safe_val(row[task])
                if val is not None:
                    try:
                        val = float(val)
                    except Exception:
                        val = None
                task_dict[task] = val

            # store (empty dict means no measurements for this task)
            y_i["regression"][task] = dict(task_dict)

        outcomes.append(y_i)

    # get the mapping from subtasks to idx
    task_to_idx, outcome_to_idx = generate_task_to_idx_grouped(outcomes, classes_dict)
    num_subtasks, num_preds = count_subtasks_and_predictions(task_to_idx)
    opt['ARCH']['DIM_TARGET'] = num_preds
    logger.info(f"Found {num_subtasks} subtasks with {num_preds} outcomes in total to predict")

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

    from analysis.a05_outcome_prediction.m_prepare_omics_info import radiomics_dims
    from analysis.a05_outcome_prediction.m_prepare_omics_info import pathomics_dims
    from analysis.a05_outcome_prediction.m_prepare_omics_info import radiomics_pool_ratio
    from analysis.a05_outcome_prediction.m_prepare_omics_info import pathomics_pool_ratio
    omics_dims = {"radiomics": radiomics_dims[radiomics_mode], "pathomics": pathomics_dims[pathomics_mode]}
    omics_pool_ratio = {"radiomics": radiomics_pool_ratio[radiomics_mode], "pathomics": pathomics_pool_ratio[pathomics_mode]}
    omics_keys = opt['ARCH']['OMICS']
    omics_dims = {k: omics_dims[k] for k in omics_keys}
    omics_pool_ratio = {k: omics_pool_ratio[k] for k in omics_keys}

    if opt['TASKS']['TRAIN']:
        # training
        training(
            split_path=split_path,
            model_dir=save_model_dir,
            omics_dims=omics_dims,
            omics_pool_ratio=omics_pool_ratio,
            arch_opt=opt['ARCH'],
            train_opt=opt['TRAIN'],
            task_to_idx=task_to_idx
        )

    if opt['TASKS']['INFERENCE']:
        # inference
        inference(
            split_path=split_path,
            omics_dims=omics_dims,
            omics_pool_ratio=omics_pool_ratio,
            arch_opt=opt['ARCH'],
            infer_opt=opt['INFERENCE'],
            task_to_idx=task_to_idx,
            outcome_to_idx=outcome_to_idx,
            pretrained_model_dir=save_model_dir
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
