"""Build the combined baseline/r6/r7 pre/post reasoning dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from m_segmentation_model_evaluation import DEFAULT_ALL_ROOT


MODELS = ("baseline", "r6", "r7")
COHORTS = ("training", "internal_test", "external_test")
CLASSES = ("endometrioma", "ovary", "uterus")
MODEL_COLORS = {"baseline": "#4c78a8", "r6": "#f58518", "r7": "#54a24b"}
STAGE_COLORS = {"pre": "#4c78a8", "post": "#e45756"}


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_model_csv(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for model in MODELS:
        frame = pd.read_csv(root / model / "analysis" / filename)
        frame.insert(0, "model", model)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_candidate_survival(root: Path, output: Path, threshold: float) -> None:
    data = load_model_csv(root, "candidate_rejection_results.csv")
    stages = ("Input", "Anatomy", "Inter-class", "Intra-class")
    columns = (None, "stage_1_anatomical_pass", "stage_2_inter_class_pass", "stage_3_intra_class_pass")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(stages)); width = .24
    for axis, domain in zip(axes, ("D1", "D2")):
        for model_index, model in enumerate(MODELS):
            group = data[(data.model == model) & (data.domain == domain)]
            values = [len(group)] + [int(group[column].fillna(False).sum()) for column in columns[1:]]
            bars = axis.bar(x + (model_index - 1) * width, values, width, color=MODEL_COLORS[model], label=model, alpha=.86)
            for bar, value in zip(bars, values):
                axis.annotate(str(value), (bar.get_x()+bar.get_width()/2, value), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        axis.set_xticks(x, stages)
        axis.set_yscale("log")
        axis.set_title(f"{domain} candidate survival")
        axis.set_ylabel("Endometrioma candidates · log scale")
        axis.grid(axis="y", alpha=.22)
    axes[0].legend(frameon=False, ncol=3)
    fig.suptitle(
        f"Candidate survival through the three rejection stages · threshold {threshold:g}",
        fontsize=14,
    )
    save_figure(fig, output)


def plot_dataset_overview(metrics: pd.DataFrame, volumes: pd.DataFrame, output: Path) -> None:
    source = metrics[(metrics.model == "baseline") & (metrics.stage == "pre")]
    detection = volumes[(volumes.model == "baseline") & (volumes.stage == "pre") & (volumes["class"] == "endometrioma")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axis = axes[0]
    cohorts = list(COHORTS); x=np.arange(len(cohorts)); width=.23
    totals = source.groupby("cohort").scan_name.nunique().reindex(cohorts).to_numpy()
    axis.bar(x, totals, .78, color="#d9dce3", label="Total scans")
    for index, class_name in enumerate(CLASSES):
        counts=[]
        for cohort in cohorts:
            subset=source[(source.cohort==cohort)&(source["class"]==class_name)]
            counts.append(int(subset.eligible.sum()))
        bars=axis.bar(x+(index-1)*width, counts, width, color=list(MODEL_COLORS.values())[index], label=f"{class_name} annotated")
        for bar,value in zip(bars,counts): axis.text(bar.get_x()+bar.get_width()/2,value+1,str(value),ha="center",fontsize=8)
    axis.set_xticks(x,[c.replace("_"," ").title() for c in cohorts]); axis.set_title("Segmentation dataset · annotated evaluation scans")
    axis.set_ylabel("Scans"); axis.grid(axis="y",alpha=.2); axis.legend(frameon=False,fontsize=8)
    axis=axes[1]; groups=[]; labels=[]
    for domain in ("D1","D2"):
        for modality in ("T1","T2","T1FS","T2FS"):
            subset=detection[(detection.domain==domain)&(detection.modality==modality)]
            if len(subset): groups.append((len(subset),int(subset.endometrioma_label.sum()))); labels.append(f"{domain}\n{modality}")
    x=np.arange(len(groups)); total=np.array([g[0] for g in groups]); positive=np.array([g[1] for g in groups])
    axis.bar(x,total,color="#d9dce3",label="All scans"); axis.bar(x,positive,color="#e45756",label="Endometrioma-positive label")
    for xx,pp,tt in zip(x,positive,total): axis.text(xx,tt+1,f"{pp}/{tt}",ha="center",fontsize=8)
    axis.set_xticks(x,labels); axis.set_title("Detection dataset · all D1 and D2 scans"); axis.set_ylabel("Scans")
    axis.grid(axis="y",alpha=.2); axis.legend(frameon=False,fontsize=8)
    axis=axes[2]
    cases=(detection.groupby(["domain","case_id"],as_index=False)
           .agg(endometrioma_label=("endometrioma_label","max")))
    domains=["D1","D2"]; x=np.arange(2)
    total=cases.groupby("domain").case_id.nunique().reindex(domains).to_numpy()
    positive=cases.groupby("domain").endometrioma_label.sum().reindex(domains).to_numpy()
    axis.bar(x,total,color="#d9dce3",label="All patients")
    axis.bar(x,positive,color="#e45756",label="Endometrioma-positive")
    for xx,pp,tt in zip(x,positive,total): axis.text(xx,tt+1,f"{int(pp)}/{int(tt)}",ha="center",fontsize=9)
    axis.set_xticks(x,domains); axis.set_title("Detection dataset · all patients")
    axis.set_ylabel("Patients"); axis.grid(axis="y",alpha=.2); axis.legend(frameon=False,fontsize=8)
    fig.suptitle("Task-specific datasets and denominators", fontsize=14)
    save_figure(fig, output)


def plot_probability_correlations(root: Path, output: Path) -> None:
    data = load_model_csv(root, "probability_channel_correlations.csv")
    pairs = tuple(data.pair.drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, domain in zip(axes, ("D1", "D2")):
        positions, values, colors, labels = [], [], [], []
        position = 1
        for pair in pairs:
            for model in MODELS:
                subset = data[(data.domain == domain) & (data.pair == pair) & (data.model == model)].pearson_correlation.dropna()
                positions.append(position); values.append(subset); colors.append(MODEL_COLORS[model]); labels.append(model)
                position += 1
            position += .7
        boxes = axis.boxplot(values, positions=positions, widths=.7, patch_artist=True, showfliers=False)
        for box, color in zip(boxes["boxes"], colors): box.set_facecolor(color); box.set_alpha(.75)
        centers = [np.mean(positions[i*3:i*3+3]) for i in range(len(pairs))]
        axis.set_xticks(centers, [pair.replace(" vs ", "\nvs\n") for pair in pairs])
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_title(domain); axis.grid(axis="y", alpha=.2); axis.set_ylim(-1, 1)
    axes[0].set_ylabel("Voxelwise Pearson correlation")
    handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], linewidth=8, label=m) for m in MODELS]
    axes[0].legend(handles=handles, frameon=False, ncol=3)
    fig.suptitle("Probability-map correlation between anatomical channels", fontsize=14)
    save_figure(fig, output)


def recall_at_threshold(data: pd.DataFrame, threshold: float = .5) -> pd.DataFrame:
    return data[np.isclose(data.threshold, threshold)].copy()


def plot_gt_recall(root: Path, output: Path, threshold: float) -> None:
    data = recall_at_threshold(
        load_model_csv(root, "gt_recall_scan_level.csv"), threshold
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for axis, class_name in zip(axes, CLASSES):
        x = np.arange(len(MODELS)); width = .34
        for offset, domain, color in ((-.17, "D1", "#4c78a8"), (.17, "D2", "#e45756")):
            recalled, total = [], []
            for model in MODELS:
                subset = data[(data.model == model) & (data.domain == domain) & (data["class"] == class_name)]
                total.append(len(subset)); recalled.append(int((subset.gt_recall > 0).sum()))
            bars = axis.bar(x + offset, total, width, color="#d9dce3")
            axis.bar(x + offset, recalled, width, color=color, label=domain)
            for bar, rr, tt in zip(bars, recalled, total):
                axis.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.5, f"{rr}/{tt}", ha="center", fontsize=8)
        axis.set_xticks(x, MODELS); axis.set_title(class_name.title()); axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Annotated scans")
    axes[0].legend(frameon=False, title="Any GT voxel recalled")
    fig.suptitle(
        f"GT recall at probability threshold {threshold:g} · recalled / annotated scans",
        fontsize=14,
    )
    save_figure(fig, output)


def plot_gt_recall_centres(root: Path, output: Path, threshold: float) -> None:
    data = recall_at_threshold(
        load_model_csv(root, "gt_recall_scan_level.csv"), threshold
    )
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharey=True)
    for row, class_name in enumerate(CLASSES):
        for col, domain in enumerate(("D1", "D2")):
            axis = axes[row, col]
            subset = data[(data["class"] == class_name) & (data.domain == domain)]
            modalities = [m for m in ("T1", "T2", "T1FS", "T2FS") if (subset.modality == m).any()]
            x = np.arange(len(modalities)); width = .24
            for index, model in enumerate(MODELS):
                med, low, high = [], [], []
                for modality in modalities:
                    values = subset[(subset.model == model) & (subset.modality == modality)].gt_recall.dropna()
                    med.append(values.median()); low.append(values.quantile(.25)); high.append(values.quantile(.75))
                med=np.asarray(med); low=np.asarray(low); high=np.asarray(high)
                axis.errorbar(x+(index-1)*width, med, yerr=np.vstack([med-low, high-med]), marker="o", capsize=3, color=MODEL_COLORS[model], label=model)
            axis.set_xticks(x, modalities); axis.set_ylim(0, 1.04); axis.grid(axis="y", alpha=.2)
            axis.set_title(f"{class_name.title()} · {domain}")
    for axis in axes[:, 0]: axis.set_ylabel("Median GT voxel recall (IQR)")
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.suptitle(
        f"GT recall across centres and modalities · threshold {threshold:g}",
        fontsize=14,
    )
    save_figure(fig, output)


def plot_physical_volumes(volumes: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    for row, class_name in enumerate(CLASSES):
        for col, cohort in enumerate(("D1", "D2")):
            axis=axes[row,col]; values=[]; positions=[]; colors=[]; labels=[]; pos=1
            for model in MODELS:
                for stage in ("pre", "post"):
                    subset=volumes[(volumes.model==model)&(volumes.stage==stage)&(volumes.domain==cohort)&(volumes["class"]==class_name)].volume_mm3.dropna()
                    values.append(np.log10(subset+1)); positions.append(pos); colors.append(STAGE_COLORS[stage]); labels.append(f"{model}\n{stage}"); pos+=1
                pos+=.5
            boxes=axis.boxplot(values, positions=positions, widths=.7, patch_artist=True, showfliers=False)
            for box,color in zip(boxes["boxes"],colors): box.set_facecolor(color); box.set_alpha(.78)
            axis.set_xticks(positions, labels, fontsize=8); axis.grid(axis="y",alpha=.2)
            axis.set_title(f"{class_name.title()} · {cohort}")
            if col==0: axis.set_ylabel("log10(volume mm³ + 1)")
    handles=[plt.Line2D([0],[0],color=STAGE_COLORS[s],linewidth=8,label=s.title()) for s in ("pre","post")]
    axes[0,0].legend(handles=handles,frameon=False)
    fig.suptitle("Predicted physical volume before and after reasoning", fontsize=14)
    save_figure(fig, output)


def sensitivity_specificity(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,float]:
    positive=scores[scores>0]
    if positive.size:
        thresholds=np.unique(np.r_[0, np.quantile(positive,np.linspace(0,.99,100)), positive.max()+1])
    else: thresholds=np.array([0.,1.])
    sensitivity=[]; specificity=[]
    for threshold in thresholds:
        prediction=scores>threshold
        tp=np.sum(prediction & labels); fn=np.sum(~prediction & labels); tn=np.sum(~prediction & ~labels); fp=np.sum(prediction & ~labels)
        sensitivity.append(tp/(tp+fn) if tp+fn else np.nan); specificity.append(tn/(tn+fp) if tn+fp else np.nan)
    auc=roc_auc_score(labels,scores) if np.unique(labels).size==2 else np.nan
    return thresholds,np.asarray(sensitivity),np.asarray(specificity),auc


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, seed: int = 42) -> tuple[float,float,float]:
    auc=roc_auc_score(labels,scores); rng=np.random.default_rng(seed); values=[]
    for _ in range(1000):
        index=rng.integers(0,len(labels),len(labels)); sampled=labels[index]
        if np.unique(sampled).size==2: values.append(roc_auc_score(sampled,scores[index]))
    low,high=np.quantile(values,[.025,.975])
    return float(auc),float(low),float(high)


def detection_data(
    volumes: pd.DataFrame, level: str, class_name: str = "endometrioma"
) -> pd.DataFrame:
    data=volumes[volumes["class"]==class_name].copy()
    if level=="scan": return data
    if level!="patient": raise ValueError(level)
    return (data.groupby(["model","stage","domain","case_id"],as_index=False)
            .agg(volume_mm3=("volume_mm3","max"),
                 endometrioma_label=("endometrioma_label","max"),
                 gt_endometrioma_case_volume_mm3=("gt_endometrioma_case_volume_mm3","first")))


def plot_sensitivity_specificity(
    volumes: pd.DataFrame, output: Path, level: str,
    class_name: str = "endometrioma",
) -> None:
    data=detection_data(volumes,level,class_name)
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharey=True)
    for row,cohort in enumerate(("D1","D2")):
        for col,stage in enumerate(("pre","post")):
            axis=axes[row,col]
            for model in MODELS:
                subset=data[(data.model==model)&(data.stage==stage)&(data.domain==cohort)].dropna(subset=["volume_mm3","endometrioma_label"])
                labels=subset.endometrioma_label.to_numpy(dtype=bool); scores=subset.volume_mm3.to_numpy(); threshold,sens,spec,_=sensitivity_specificity(labels,scores)
                auc,low,high=bootstrap_auc(labels,scores)
                axis.plot(threshold,sens,color=MODEL_COLORS[model],linewidth=2,label=f"{model} sens · AUC {auc:.2f} [{low:.2f}, {high:.2f}]")
                axis.plot(threshold,spec,color=MODEL_COLORS[model],linewidth=2,linestyle="--",label=f"{model} spec")
            axis.set_xscale("symlog",linthresh=10); axis.set_ylim(-.02,1.02); axis.grid(alpha=.2)
            axis.set_title(f"{cohort} · {stage.title()} · n={len(subset)}")
            axis.set_xlabel(f"Predicted {class_name} volume threshold (mm³)")
            axis.set_ylabel("Sensitivity / specificity")
            axis.legend(frameon=False,fontsize=8,ncol=2)
    score_label="direct endometrioma score" if class_name=="endometrioma" else "ovary-volume surrogate score"
    fig.suptitle(f"{level.title()}-level binary endometrioma detection using {score_label} · solid sensitivity, dashed specificity",fontsize=14)
    save_figure(fig,output)


def plot_zero_operating_point(volumes: pd.DataFrame, output: Path) -> None:
    metrics=("Sensitivity","Specificity","Balanced accuracy","PPV","NPV")
    colors=("#4c78a8","#72b7b2","#f2cf5b","#e45756","#b279a2")
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharey=True)
    for row,level in enumerate(("scan","patient")):
        data=detection_data(volumes,level)
        for col,domain in enumerate(("D1","D2")):
            axis=axes[row,col]; x=np.arange(len(MODELS)); width=.15
            for metric_index,(metric,color) in enumerate(zip(metrics,colors)):
                values=[]
                for model in MODELS:
                    z=data[(data.model==model)&(data.stage=="post")&(data.domain==domain)]
                    truth=z.endometrioma_label.to_numpy(dtype=bool); pred=z.volume_mm3.to_numpy(dtype=float)>0
                    tp=np.sum(pred&truth); fn=np.sum(~pred&truth); tn=np.sum(~pred&~truth); fp=np.sum(pred&~truth)
                    sensitivity=tp/(tp+fn) if tp+fn else np.nan; specificity=tn/(tn+fp) if tn+fp else np.nan
                    values.append({"Sensitivity":sensitivity,"Specificity":specificity,"Balanced accuracy":(sensitivity+specificity)/2,
                                   "PPV":tp/(tp+fp) if tp+fp else np.nan,"NPV":tn/(tn+fn) if tn+fn else np.nan}[metric])
                axis.bar(x+(metric_index-2)*width,values,width,color=color,label=metric)
            axis.set_xticks(x,MODELS); axis.set_ylim(0,1.03); axis.grid(axis="y",alpha=.2)
            axis.set_title(f"{domain} · {level}-level")
    axes[0,0].set_ylabel("Metric value"); axes[1,0].set_ylabel("Metric value")
    axes[0,0].legend(frameon=False,fontsize=8,ncol=2)
    fig.suptitle("Post-reasoning binary operating point: accepted physical volume > 0 mm³",fontsize=14)
    save_figure(fig,output)


def safe_corr(x: pd.Series,y: pd.Series,method: str) -> float:
    if len(x)<3 or x.nunique()<2 or y.nunique()<2: return np.nan
    return float(pearsonr(x,y).statistic if method=="pearson" else spearmanr(x,y).statistic)


def plot_volume_correlations(
    volumes: pd.DataFrame, output: Path, positive_only: bool,
    class_name: str = "endometrioma",
) -> None:
    data=detection_data(volumes,"patient",class_name)
    if positive_only: data=data[data.gt_endometrioma_case_volume_mm3>0]
    fig,axes=plt.subplots(3,2,figsize=(14,14))
    for row,model in enumerate(MODELS):
        for col,cohort in enumerate(("D1","D2")):
            axis=axes[row,col]
            for stage in ("pre","post"):
                subset=data[(data.model==model)&(data.stage==stage)&(data.domain==cohort)].dropna(subset=["volume_mm3","gt_endometrioma_case_volume_mm3"])
                raw_x=subset.gt_endometrioma_case_volume_mm3; raw_y=subset.volume_mm3
                x=np.log10(raw_x+1); y=np.log10(raw_y+1)
                raw_p=safe_corr(raw_x,raw_y,"pearson"); log_p=safe_corr(x,y,"pearson"); s=safe_corr(raw_x,raw_y,"spearman")
                axis.scatter(x,y,s=24,alpha=.7,color=STAGE_COLORS[stage],label=f"{stage}: raw r={raw_p:.2f}, log r={log_p:.2f}, ρ={s:.2f}, n={len(x)}")
            limits=[0,max(axis.get_xlim()[1],axis.get_ylim()[1])]; axis.plot(limits,limits,color="black",linewidth=.8,linestyle=":")
            axis.set_xlim(limits); axis.set_ylim(limits); axis.grid(alpha=.2); axis.legend(frameon=False,fontsize=8)
            axis.set_title(f"{model} · {cohort}")
            axis.set_xlabel("log10(GT endometrioma volume mm³ + 1)"); axis.set_ylabel(f"log10(predicted {class_name} volume mm³ + 1)")
    population="GT-positive patients only (secondary sensitivity analysis)" if positive_only else "all patients, including GT volume = 0 (primary continuous detection)"
    fig.suptitle(f"Patient maximum predicted {class_name} volume versus case GT endometrioma volume\n{population}",fontsize=14)
    save_figure(fig,output)


def plot_correlation_summary(
    volumes: pd.DataFrame, output: Path, class_name: str = "endometrioma"
) -> None:
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharey=True)
    for row,level in enumerate(("scan","patient")):
        data=detection_data(volumes,level,class_name)
        for col,domain in enumerate(("D1","D2")):
            axis=axes[row,col]; x=np.arange(len(MODELS)); width=.18
            for stage_index,stage in enumerate(("pre","post")):
                pearson_values=[]; spearman_values=[]
                for model in MODELS:
                    z=data[(data.model==model)&(data.stage==stage)&(data.domain==domain)]
                    pearson_values.append(safe_corr(z.gt_endometrioma_case_volume_mm3,z.volume_mm3,"pearson"))
                    spearman_values.append(safe_corr(z.gt_endometrioma_case_volume_mm3,z.volume_mm3,"spearman"))
                offset=(-.27 if stage=="pre" else .09)
                axis.bar(x+offset,pearson_values,width,color=STAGE_COLORS[stage],alpha=.9,label=f"{stage} Pearson")
                axis.bar(x+offset+width,spearman_values,width,color=STAGE_COLORS[stage],alpha=.45,hatch="//",label=f"{stage} Spearman")
            axis.axhline(0,color="black",linewidth=.8); axis.set_ylim(-1,1); axis.grid(axis="y",alpha=.2)
            axis.set_xticks(x,MODELS); axis.set_title(f"{domain} · {level}-level · all {'scans' if level=='scan' else 'patients'}")
    axes[0,0].set_ylabel("Correlation with case GT volume"); axes[1,0].set_ylabel("Correlation with case GT volume")
    axes[0,0].legend(frameon=False,fontsize=8,ncol=2)
    fig.suptitle(f"Continuous endometrioma detection using predicted {class_name} volume\nAll zero/nonzero GT burdens; patient score = maximum scan volume",fontsize=14)
    save_figure(fig,output)


def analyze_endometrioma_effect_on_ovary(
    metrics: pd.DataFrame, volumes: pd.DataFrame, output: Path, results_path: Path
) -> pd.DataFrame:
    labels=(volumes[(volumes.model=="baseline")&(volumes.stage=="pre")&(volumes["class"]=="endometrioma")]
            [["scan_name","endometrioma_label"]].drop_duplicates("scan_name"))
    data=(metrics[(metrics["class"]=="ovary")&metrics.eligible.fillna(False)]
          .merge(labels,on="scan_name",how="inner",validate="many_to_one"))
    data["domain"]=np.where(data.cohort.eq("external_test"),"D2","D1")
    patients=(data.groupby(["model","stage","domain","case_id","endometrioma_label"],as_index=False)
              .agg(dice=("dice","mean"),annotated_scans=("scan_name","nunique")))
    rng=np.random.default_rng(42); rows=[]
    for model in MODELS:
        for domain in ("D1","D2"):
            for stage in ("pre","post"):
                subset=patients[(patients.model==model)&(patients.domain==domain)&(patients.stage==stage)]
                negative=subset.loc[subset.endometrioma_label==0,"dice"].to_numpy(float)
                positive=subset.loc[subset.endometrioma_label==1,"dice"].to_numpy(float)
                observed=float(positive.mean()-negative.mean())
                boot=np.empty(5000)
                for index in range(len(boot)):
                    boot[index]=rng.choice(positive,len(positive),replace=True).mean()-rng.choice(negative,len(negative),replace=True).mean()
                combined=np.r_[negative,positive]; n_positive=len(positive); perm=np.empty(10000)
                for index in range(len(perm)):
                    shuffled=rng.permutation(combined); perm[index]=shuffled[-n_positive:].mean()-shuffled[:-n_positive].mean()
                p_value=(1+np.sum(np.abs(perm)>=abs(observed)))/(len(perm)+1)
                rows.append({"model":model,"domain":domain,"stage":stage,"negative_patients":len(negative),"positive_patients":len(positive),
                             "negative_mean_dice":negative.mean(),"positive_mean_dice":positive.mean(),"dice_difference_positive_minus_negative":observed,
                             "difference_ci_2.5%":np.quantile(boot,.025),"difference_ci_97.5%":np.quantile(boot,.975),"permutation_p":p_value})
    results=pd.DataFrame(rows); results.to_csv(results_path,index=False)
    fig,axes=plt.subplots(3,2,figsize=(14,13),sharey=True); colors={0:"#72b7b2",1:"#e45756"}
    rng_points=np.random.default_rng(7)
    for row,model in enumerate(MODELS):
        for col,domain in enumerate(("D1","D2")):
            axis=axes[row,col]; positions=[]; values=[]; box_colors=[]; labels_text=[]; position=1
            for stage in ("pre","post"):
                for label in (0,1):
                    v=patients[(patients.model==model)&(patients.domain==domain)&(patients.stage==stage)&(patients.endometrioma_label==label)].dice.to_numpy(float)
                    positions.append(position); values.append(v); box_colors.append(colors[label]); labels_text.append(f"{stage}\n{'Absent' if label==0 else 'Present'}"); position+=1
                position+=.5
            boxes=axis.boxplot(values,positions=positions,widths=.65,patch_artist=True,showfliers=False,medianprops={"color":"black","linewidth":1.5})
            for box,color in zip(boxes["boxes"],box_colors): box.set_facecolor(color); box.set_alpha(.72)
            for pos,v in zip(positions,values): axis.scatter(pos+rng_points.normal(0,.045,len(v)),v,s=18,color="#26384a",alpha=.5)
            axis.set_xticks(positions,labels_text); axis.set_ylim(-.02,1.02); axis.grid(axis="y",alpha=.2); axis.set_title(f"{model} · {domain}")
    for axis in axes[:,0]: axis.set_ylabel("Patient-mean ovary Dice")
    fig.suptitle("Effect of endometrioma presence on ovary segmentation\nOnly ovary-annotated scans; each point is one patient",fontsize=14)
    save_figure(fig,output)
    return results


def summary_tables(
    metrics: pd.DataFrame, volumes: pd.DataFrame, ovary_effect: pd.DataFrame
) -> tuple[str, str, str]:
    eligible=metrics[(metrics["class"]=="endometrioma")&metrics.eligible.fillna(False)&(metrics.gt_volume_mm3>0)]
    segmentation=(eligible.groupby(["model","cohort","stage"]).dice.mean().unstack(["cohort","stage"])
                  .reindex(index=MODELS,columns=pd.MultiIndex.from_product([COHORTS,("pre","post")])))
    segmentation.columns=[f"{cohort.replace('_',' ').title()} {stage.title()}" for cohort,stage in segmentation.columns]
    segmentation.index.name="Model"; segmentation=segmentation.reset_index()
    segmentation_html=segmentation.to_html(index=False,float_format=lambda x:f"{x:.2f}",classes="summary-table",border=0)

    binary_rows=[]
    for level in ("scan","patient"):
        data=detection_data(volumes,level)
        for model in MODELS:
            for domain in ("D1","D2"):
                z=data[(data.model==model)&(data.stage=="post")&(data.domain==domain)]
                truth=z.endometrioma_label.to_numpy(dtype=bool); score=z.volume_mm3.to_numpy(dtype=float); prediction=score>0
                tp=np.sum(prediction&truth); fn=np.sum(~prediction&truth); tn=np.sum(~prediction&~truth); fp=np.sum(prediction&~truth)
                sensitivity=tp/(tp+fn); specificity=tn/(tn+fp)
                binary_rows.append({"Level":level.title(),"Model":model,"Centre":domain,"n":len(z),"AUROC":roc_auc_score(truth,score),
                                    "Sensitivity":sensitivity,"Specificity":specificity,"Balanced accuracy":(sensitivity+specificity)/2})
    binary=pd.DataFrame(binary_rows)
    metric_columns=("AUROC","Sensitivity","Specificity","Balanced accuracy")
    binary_parts=['<table class="summary-table grouped-table"><thead><tr><th>Level</th><th>Centre</th><th>Model</th><th>n</th>'+
                  ''.join(f'<th>{column}</th>' for column in metric_columns)+'</tr></thead><tbody>']
    for level in ("Scan","Patient"):
        for centre_index,centre in enumerate(("D1","D2")):
            group=binary[(binary.Level==level)&(binary.Centre==centre)].set_index("Model").reindex(MODELS).reset_index()
            best={column:group[column].max() for column in metric_columns}
            for model_index,row in group.iterrows():
                binary_parts.append('<tr class="group-start">' if model_index==0 else '<tr>')
                if centre_index==0 and model_index==0: binary_parts.append(f'<th scope="rowgroup" rowspan="6" class="level-cell">{level}</th>')
                if model_index==0: binary_parts.append(f'<th scope="rowgroup" rowspan="3" class="centre-cell">{centre}</th>')
                binary_parts.extend([f'<td>{row.Model}</td>',f'<td>{int(row.n)}</td>'])
                for column in metric_columns:
                    value=float(row[column]); formatted=f'{value:.2f}'
                    binary_parts.append(f'<td><strong>{formatted}</strong></td>' if np.isclose(value,best[column]) else f'<td>{formatted}</td>')
                binary_parts.append('</tr>')
    binary_parts.append('</tbody></table>'); binary_html=''.join(binary_parts)

    patient=detection_data(volumes,"patient")
    burden_rows=[]
    for model in MODELS:
        for domain in ("D1","D2"):
            z=patient[(patient.model==model)&(patient.stage=="post")&(patient.domain==domain)]
            burden_rows.append({"Model":model,"Centre":domain,"Patients":len(z),"Pearson r":safe_corr(z.gt_endometrioma_case_volume_mm3,z.volume_mm3,"pearson"),
                                "Spearman rho":safe_corr(z.gt_endometrioma_case_volume_mm3,z.volume_mm3,"spearman")})
    burden=pd.DataFrame(burden_rows)
    burden_parts=['<table class="summary-table grouped-table"><thead><tr><th>Level</th><th>Centre</th><th>Model</th><th>Patients</th><th>Pearson r</th><th>Spearman ρ</th></tr></thead><tbody>']
    for centre_index,centre in enumerate(("D1","D2")):
        group=burden[burden.Centre==centre].set_index("Model").reindex(MODELS).reset_index()
        best={column:group[column].max() for column in ("Pearson r","Spearman rho")}
        for model_index,row in group.iterrows():
            burden_parts.append('<tr class="group-start">' if model_index==0 else '<tr>')
            if centre_index==0 and model_index==0: burden_parts.append('<th scope="rowgroup" rowspan="6" class="level-cell">Patient</th>')
            if model_index==0: burden_parts.append(f'<th scope="rowgroup" rowspan="3" class="centre-cell">{centre}</th>')
            burden_parts.extend([f'<td>{row.Model}</td>',f'<td>{int(row.Patients)}</td>'])
            for column in ("Pearson r","Spearman rho"):
                value=float(row[column]); formatted=f'{value:.2f}'
                burden_parts.append(f'<td><strong>{formatted}</strong></td>' if np.isclose(value,best[column]) else f'<td>{formatted}</td>')
            burden_parts.append('</tr>')
    burden_parts.append('</tbody></table>'); burden_html=''.join(burden_parts)
    ovary_binary_rows=[]
    for level in ("scan","patient"):
        ovary=detection_data(volumes,level,"ovary")
        for model in MODELS:
            development=ovary[(ovary.model==model)&(ovary.stage=="post")&(ovary.domain=="D1")]
            dev_truth=development.endometrioma_label.to_numpy(dtype=bool); dev_score=development.volume_mm3.to_numpy(float)
            thresholds,sens,spec,_=sensitivity_specificity(dev_truth,dev_score); selected=float(thresholds[np.nanargmax((sens+spec)/2)])
            for domain in ("D1","D2"):
                z=ovary[(ovary.model==model)&(ovary.stage=="post")&(ovary.domain==domain)]
                truth=z.endometrioma_label.to_numpy(dtype=bool); score=z.volume_mm3.to_numpy(float); prediction=score>selected
                tp=np.sum(prediction&truth); fn=np.sum(~prediction&truth); tn=np.sum(~prediction&~truth); fp=np.sum(prediction&~truth)
                sensitivity=tp/(tp+fn); specificity=tn/(tn+fp)
                ovary_binary_rows.append({"Level":level.title(),"Model":model,"Centre":domain,"n":len(z),"D1 threshold (mm³)":selected,"AUROC":roc_auc_score(truth,score),
                                          "Sensitivity":sensitivity,"Specificity":specificity,"Balanced accuracy":(sensitivity+specificity)/2})
    ovary_binary=pd.DataFrame(ovary_binary_rows); ovary_metric_columns=("AUROC","Sensitivity","Specificity","Balanced accuracy")
    ovary_binary_parts=['<table class="summary-table grouped-table"><thead><tr><th>Level</th><th>Centre</th><th>Model</th><th>n</th><th>D1 threshold (mm³)</th>'+''.join(f'<th>{c}</th>' for c in ovary_metric_columns)+'</tr></thead><tbody>']
    for level in ("Scan","Patient"):
        for centre_index,centre in enumerate(("D1","D2")):
            group=ovary_binary[(ovary_binary.Level==level)&(ovary_binary.Centre==centre)].set_index("Model").reindex(MODELS).reset_index(); best={c:group[c].max() for c in ovary_metric_columns}
            for model_index,row in group.iterrows():
                ovary_binary_parts.append('<tr class="group-start">' if model_index==0 else '<tr>')
                if centre_index==0 and model_index==0: ovary_binary_parts.append(f'<th rowspan="6" class="level-cell">{level}</th>')
                if model_index==0: ovary_binary_parts.append(f'<th rowspan="3" class="centre-cell">{centre}</th>')
                ovary_binary_parts.extend([f'<td>{row.Model}</td>',f'<td>{int(row.n)}</td>',f'<td>{row["D1 threshold (mm³)"]:.0f}</td>'])
                for column in ovary_metric_columns:
                    value=float(row[column]); formatted=f'{value:.2f}'; ovary_binary_parts.append(f'<td><strong>{formatted}</strong></td>' if np.isclose(value,best[column]) else f'<td>{formatted}</td>')
                ovary_binary_parts.append('</tr>')
    ovary_binary_parts.append('</tbody></table>'); ovary_binary_html=''.join(ovary_binary_parts)

    ovary_patient=detection_data(volumes,"patient","ovary"); ovary_burden=[]
    for model in MODELS:
        for domain in ("D1","D2"):
            z=ovary_patient[(ovary_patient.model==model)&(ovary_patient.stage=="post")&(ovary_patient.domain==domain)]
            ovary_burden.append({"Centre":domain,"Model":model,"Patients":len(z),"Pearson r":safe_corr(z.gt_endometrioma_case_volume_mm3,z.volume_mm3,"pearson"),"Spearman ρ":safe_corr(z.gt_endometrioma_case_volume_mm3,z.volume_mm3,"spearman")})
    ovary_burden=pd.DataFrame(ovary_burden); ovary_burden_parts=['<table class="summary-table grouped-table"><thead><tr><th>Level</th><th>Centre</th><th>Model</th><th>Patients</th><th>Pearson r</th><th>Spearman ρ</th></tr></thead><tbody>']
    for centre_index,centre in enumerate(("D1","D2")):
        group=ovary_burden[ovary_burden.Centre==centre].set_index("Model").reindex(MODELS).reset_index(); best={c:group[c].max() for c in ("Pearson r","Spearman ρ")}
        for model_index,row in group.iterrows():
            ovary_burden_parts.append('<tr class="group-start">' if model_index==0 else '<tr>')
            if centre_index==0 and model_index==0: ovary_burden_parts.append('<th rowspan="6" class="level-cell">Patient</th>')
            if model_index==0: ovary_burden_parts.append(f'<th rowspan="3" class="centre-cell">{centre}</th>')
            ovary_burden_parts.extend([f'<td>{row.Model}</td>',f'<td>{int(row.Patients)}</td>'])
            for column in ("Pearson r","Spearman ρ"):
                value=float(row[column]); formatted=f'{value:.2f}'; ovary_burden_parts.append(f'<td><strong>{formatted}</strong></td>' if np.isclose(value,best[column]) else f'<td>{formatted}</td>')
            ovary_burden_parts.append('</tr>')
    ovary_burden_parts.append('</tbody></table>'); ovary_burden_html=''.join(ovary_burden_parts)
    detection_html=(f'<section class="table-panel"><h3>Binary detection using predicted endometrioma volume</h3><p>All positive and negative scans/patients; post-reasoning sensitivity, specificity and balanced accuracy use accepted physical volume &gt; 0 mm³.</p>{binary_html}</section>'
                    f'<section class="table-panel"><h3>Continuous detection using predicted endometrioma volume</h3><p>All patients, including GT-negative patients with zero burden; patient score is maximum scan volume.</p>{burden_html}</section>'
                    f'<section class="table-panel"><h3>Binary detection using predicted ovary volume</h3><p>Ovary volume is a surrogate endometrioma score. Each post-reasoning threshold is selected on D1 by maximum balanced accuracy and then fixed for D2.</p>{ovary_binary_html}</section>'
                    f'<section class="table-panel"><h3>Continuous detection using predicted ovary volume</h3><p>All patients, including zero GT burden; patient score is maximum predicted ovary volume.</p>{ovary_burden_html}</section>')
    ovary_display=ovary_effect.copy()
    ovary_display["Endometrioma absent Dice"]=ovary_display.negative_mean_dice
    ovary_display["Endometrioma present Dice"]=ovary_display.positive_mean_dice
    ovary_display["Difference (present − absent)"]=ovary_display.dice_difference_positive_minus_negative
    ovary_display["95% CI"]=ovary_display.apply(lambda row:f"[{row['difference_ci_2.5%']:.2f}, {row['difference_ci_97.5%']:.2f}]",axis=1)
    ovary_display["Permutation p"]=ovary_display.permutation_p
    ovary_display=ovary_display[["domain","model","stage","negative_patients","positive_patients","Endometrioma absent Dice","Endometrioma present Dice","Difference (present − absent)","95% CI","Permutation p"]]
    ovary_display.columns=["Centre","Model","Stage","Absent n","Present n","Absent Dice","Present Dice","Difference","95% CI","Permutation p"]
    ovary_html=ovary_display.to_html(index=False,float_format=lambda x:f"{x:.3f}",classes="summary-table",border=0)
    return segmentation_html,detection_html,ovary_html


def write_dashboard(
    output: Path,
    parts: list[tuple[str, str, list[tuple[str, str, str]]]],
    threshold: float,
    metrics: pd.DataFrame,
    volumes: pd.DataFrame,
    ovary_effect: pd.DataFrame,
) -> None:
    nav="".join(f'<a href="#part-{i}">Part {i}: {title}</a>' for i,(title,_,_) in enumerate(parts,1))
    rendered=[]
    segmentation_table,detection_tables,ovary_table=summary_tables(metrics,volumes,ovary_effect)
    for part_index,(part_title,part_description,figures) in enumerate(parts,1):
        figures_html="".join(
            f'<section><h3>{title}</h3><p>{description}</p><img src="figures/{filename}" alt="{title}"></section>'
            for title,description,filename in figures
        )
        tables=""
        if part_index==1:
            tables=(f'<section class="table-panel"><h3>Segmentation summary</h3><p>Mean positive-only endometrioma Dice on annotated scans. Post includes fully rejected predictions as Dice 0.</p>{segmentation_table}</section>'
                    f'<section class="table-panel"><h3>Does endometrioma presence degrade ovary segmentation?</h3><p>Patient-level comparison using only ovary-annotated scans. Difference = mean Dice in endometrioma-present patients minus mean Dice in endometrioma-absent patients; negative values indicate degradation. Confidence intervals use patient bootstrap and p-values use patient-label permutation.</p>{ovary_table}</section>')
        elif part_index==2:
            tables=detection_tables
        rendered.append(f'<div class="part" id="part-{part_index}"><div class="part-head"><span>Part {part_index}</span><h2>{part_title}</h2><p>{part_description}</p></div>{tables}{figures_html}</div>')
    source=metrics[(metrics.model=="baseline")&(metrics.stage=="pre")]
    segmentation_counts={c:int(source[source.cohort==c].scan_name.nunique()) for c in COHORTS}
    detection=volumes[(volumes.model=="baseline")&(volumes.stage=="pre")&(volumes["class"]=="endometrioma")]
    scan_counts={d:int(detection[detection.domain==d].scan_name.nunique()) for d in ("D1","D2")}
    patient_counts={d:int(detection[detection.domain==d].case_id.nunique()) for d in ("D1","D2")}
    sections="".join(rendered)
    output.write_text(f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reasoning model comparison</title><style>
    :root{{--ink:#182230;--muted:#596579;--line:#dfe4ec;--paper:#f5f7fa;--accent:#305f9f;--soft:#edf3fb}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}header{{background:#fff;border-bottom:1px solid var(--line);padding:28px max(24px,5vw)}}h1{{margin:0 0 3px;font-size:30px}}h1+strong{{display:block;font-size:19px;margin-bottom:6px}}header p{{margin:0;color:var(--muted)}}nav{{position:sticky;top:0;z-index:3;display:flex;gap:8px;overflow:auto;padding:10px max(24px,5vw);background:rgba(255,255,255,.96);border-bottom:1px solid var(--line)}}nav a{{white-space:nowrap;color:var(--accent);text-decoration:none;padding:6px 9px;border-radius:5px}}nav a:hover{{background:var(--soft)}}main{{max-width:1500px;margin:auto;padding:22px}}.part{{margin-bottom:40px}}.part-head{{background:#17365d;color:#fff;border-radius:12px;padding:22px;margin-bottom:18px}}.part-head span{{text-transform:uppercase;letter-spacing:.08em;font-size:12px;opacity:.8}}.part-head h2{{font-size:26px;margin:2px 0 4px}}.part-head p{{margin:0;color:#dce8f6}}section{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:20px;margin:0 0 22px}}h3{{margin:0 0 4px;font-size:20px}}section p{{margin:0 0 14px;color:var(--muted)}}img{{display:block;width:100%;height:auto;border:1px solid #eef0f4}}.dataset{{background:var(--soft);padding:10px 12px;border-radius:6px;margin-top:12px}}.table-panel{{overflow-x:auto}}.summary-table{{border-collapse:collapse;width:100%;font-size:14px}}.summary-table th{{background:#17365d;color:white;text-align:left}}.summary-table th,.summary-table td{{padding:8px 10px;border:1px solid var(--line)}}.summary-table tbody tr:nth-child(even){{background:#f5f7fa}}.grouped-table .level-cell{{background:#dce8f6;color:#17365d;vertical-align:top;font-size:15px}}.grouped-table .centre-cell{{background:#edf3fb;color:#17365d;vertical-align:top}}.grouped-table .group-start td,.grouped-table .group-start th{{border-top:2px solid #8fa7c2}}.grouped-table strong{{font-weight:800;color:#0b4381}}code{{background:#eef1f5;padding:2px 5px;border-radius:4px}}@media(max-width:700px){{h1{{font-size:24px}}main{{padding:12px}}section{{padding:12px}}}}</style></head><body><header><h1>Endometriosis: Do NOT fully trust FM – Question it</h1><strong>Baseline, r6 and r7 before versus after reasoning</strong><p>Hierarchical anatomical reasoning and statistical rejection · probability threshold {threshold:g}</p><div class="dataset"><strong>Part 1 segmentation:</strong> annotated scans only — training {segmentation_counts['training']}, internal test {segmentation_counts['internal_test']}, external test {segmentation_counts['external_test']}.<br><strong>Part 2 detection:</strong> all scans and patients — D1 {scan_counts['D1']} scans/{patient_counts['D1']} patients; D2 {scan_counts['D2']} scans/{patient_counts['D2']} patients.<br><strong>Part 3 cancer risk:</strong> future endometrioma-positive outcome cohort.</div></header><nav>{nav}</nav><main>{sections}</main></body></html>''',encoding="utf-8")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning-root",type=Path,default=DEFAULT_ALL_ROOT/"reasoning_model_comparison")
    parser.add_argument("--threshold",type=float,default=0.5)
    args=parser.parse_args(); root=args.reasoning_root; output=root/"figures"; output.mkdir(parents=True,exist_ok=True)
    metrics=pd.read_csv(root/"tables"/"pre_post_segmentation_metrics.csv")
    parts=[
        ("Endometrioma segmentation","Localization and delineation are evaluated only where voxel annotations exist.",[
            ("Segmentation dataset and task denominators","The first panel gives the annotated segmentation cohorts; detection denominators are shown separately for context.","02_dataset_overview.png"),
            ("Positive-only segmentation performance","Dice on annotated positive scans; a fully rejected prediction remains Dice 0.","pre_post_positive_dice.png"),
            ("Endometrioma presence versus ovary segmentation","Only ovary-annotated scans; patient-level Dice distributions are separated by endometrioma presence.","03b_endometrioma_effect_on_ovary.png"),
            (f"GT recall at {args.threshold:g}","Annotated scans containing at least one recovered GT voxel.","04_gt_recall_threshold.png"),
            ("GT recall across centres and modalities","Median voxel recall and IQR on class-annotated scans only.","05_gt_recall_centres_modalities.png"),
        ]),
        ("Endometrioma detection","Binary presence/absence and continuous burden are complementary endpoints evaluated on all scans and all patients.",[
            ("Candidate survival through three stages","All scans; survival through anatomy, inter-class and intra-class rejection.","01_candidate_survival.png"),
            ("Anatomical-channel probability correlation","All scans; dependency among endometrioma, ovary and uterus channels before reasoning.","03_probability_channel_correlations.png"),
            ("Predicted physical-volume distributions","All scans; physical volume in mm³ before and after reasoning.","06_pre_post_physical_volume.png"),
            ("Binary detection — scan level","All positive and negative scans. Threshold 0 uses volume > 0 mm³.","07_scan_sensitivity_specificity.png"),
            ("Binary detection — patient level","All positive and negative patients; patient score is maximum scan volume.","08_patient_sensitivity_specificity.png"),
            ("Binary operating point — any accepted candidate","Post-reasoning sensitivity, specificity, balanced accuracy, PPV and NPV at accepted physical volume > 0 mm³.","08b_zero_volume_operating_point.png"),
            ("Ovary-volume surrogate — scan-level binary detection","All scans; predicted ovary physical volume is used as the endometrioma score.","08c_ovary_scan_sensitivity_specificity.png"),
            ("Ovary-volume surrogate — patient-level binary detection","All patients; score is maximum predicted ovary volume across scans.","08d_ovary_patient_sensitivity_specificity.png"),
            ("Continuous detection — all scans and patients","Primary burden analysis includes GT-negative subjects with GT volume 0; raw-volume Pearson and Spearman are shown.","09_all_population_correlation_summary.png"),
            ("Ovary-volume surrogate — continuous detection","All scans and patients; predicted ovary volume is correlated with case GT endometrioma burden.","09b_ovary_all_population_correlation_summary.png"),
            ("Continuous detection — patient scatter","All patients, including zero burden; plot is log-transformed for display while raw and log Pearson plus Spearman are reported.","10_all_patient_volume_correlation.png"),
            ("Ovary-volume surrogate — patient scatter","All patients, including zero burden; maximum predicted ovary volume versus case GT endometrioma volume.","10b_ovary_all_patient_volume_correlation.png"),
            ("Positive-only burden sensitivity analysis","Secondary analysis asks whether burden is ranked after disease is known.","11_positive_patient_volume_correlation.png"),
        ]),
        ("Ovarian cancer risk — future work","Placeholder: among patients with endometrioma, test whether endometrioma burden, ovary volume and radiomics predict ovarian-cancer risk after outcomes and leakage-safe splits are available.",[]),
    ]
    volumes=pd.read_csv(root/"tables"/"all_scan_pre_post_volumes.csv")
    ovary_effect=analyze_endometrioma_effect_on_ovary(metrics,volumes,output/"03b_endometrioma_effect_on_ovary.png",root/"tables"/"endometrioma_effect_on_ovary_segmentation.csv")
    plot_candidate_survival(root,output/"01_candidate_survival.png",args.threshold); plot_dataset_overview(metrics,volumes,output/"02_dataset_overview.png")
    plot_probability_correlations(root,output/"03_probability_channel_correlations.png"); plot_gt_recall(root,output/"04_gt_recall_threshold.png",args.threshold)
    plot_gt_recall_centres(root,output/"05_gt_recall_centres_modalities.png",args.threshold); plot_physical_volumes(volumes,output/"06_pre_post_physical_volume.png")
    plot_sensitivity_specificity(volumes,output/"07_scan_sensitivity_specificity.png","scan")
    plot_sensitivity_specificity(volumes,output/"08_patient_sensitivity_specificity.png","patient")
    plot_zero_operating_point(volumes,output/"08b_zero_volume_operating_point.png")
    plot_sensitivity_specificity(volumes,output/"08c_ovary_scan_sensitivity_specificity.png","scan","ovary")
    plot_sensitivity_specificity(volumes,output/"08d_ovary_patient_sensitivity_specificity.png","patient","ovary")
    plot_correlation_summary(volumes,output/"09_all_population_correlation_summary.png")
    plot_correlation_summary(volumes,output/"09b_ovary_all_population_correlation_summary.png","ovary")
    plot_volume_correlations(volumes,output/"10_all_patient_volume_correlation.png",False)
    plot_volume_correlations(volumes,output/"10b_ovary_all_patient_volume_correlation.png",False,"ovary")
    plot_volume_correlations(volumes,output/"11_positive_patient_volume_correlation.png",True)
    write_dashboard(root/"reasoning_model_comparison_dashboard.html",parts,args.threshold,metrics,volumes,ovary_effect)
    print(root/"reasoning_model_comparison_dashboard.html")


if __name__=="__main__": main()
