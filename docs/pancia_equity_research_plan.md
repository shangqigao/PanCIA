# Who Is Lost Before the Model? Auditing Selection Bias and Fairness Evaluability in Pan-Cancer Multimodal AI

## Background and theoretical approach

Artificial intelligence (AI) integrating radiology (CT/MRI) and digital pathology could improve cancer prognosis and biological characterisation [6]. However, linked cohorts are not neutral samples. In PanCIA, patients may be lost through failed linkage; unavailable modalities; image quality control; scan conversion; segmentation or feature extraction; paired-modality requirements; missing outcomes; or censoring. Losses may vary by cancer type, centre, period, acquisition protocol and demographic group.

Our previous kidney-cancer study found substantial underrepresentation of Asian patients [7]. Pan-cancer analysis adds risk because cancer type is associated with demographics, imaging practice, prognosis and modality availability; pooled comparisons may therefore mistake disease mix for demographic disparity. TCGA–TCIA are retrospective US convenience resources. Patients also contribute unequal numbers of scans and slides, and follow-up may differ between groups.

We treat the pipeline as sequential selection. Conditioning on complete modalities, successful preprocessing and recorded outcomes may introduce complete-case or collider bias when selection relates to both patient characteristics and prognosis. We distinguish three estimands: source-resource representation, selection into analytic cohorts, and performance conditional on inclusion. Good performance in the final cohort cannot establish fairness for patients excluded earlier.

## Goals of the project

The goal is to develop a reproducible patient-level framework for identifying selection bias and determining when fairness can be evaluated credibly.

1. **Map representation and attrition** across clinical linkage, technical processing, modality matching and outcome requirements.
2. **Evaluate models responsibly**, comparing unimodal and multimodal predictions while accounting for cancer type, acquisition effects and censoring.
3. **Assess mitigation and transportability**, including retention of patients with one modality.
4. **Co-design practical guidance** and an equity-audit toolkit with patient and public involvement (PPI).

## Study design and methodological approach

This 12-month mixed-methods study will use existing de-identified cohorts and PanCIA pipelines. TCGA–TCIA will form the core analysis. CPTAC or an approved local cohort will provide optional transportability analysis if access and harmonisation are completed; it is not required for core delivery.

We will create a versioned inclusion ledger with one row per source-cohort patient and indicators for each stage:

**source cohort → clinical linkage → radiology available → radiology QC/conversion → pathology available → pathology QC → segmentation/feature extraction → paired modalities → outcome/follow-up available → final task-specific cohort.**

We will document exclusion reasons, cancer type, centre, modality, acquisition period and scans/slides per patient, revealing technical and measurement-intensity bias. A small prespecified set of adequately supported tasks will include overall survival and selected molecular phenotypes. This evaluates a research pipeline, not a clinical decision system.

## Plans for data collection and securing diverse participation

No prospective clinical data collection is planned. The ledger will link existing imaging, pathology, clinical and processing records. For age, recorded sex, race/ethnicity and available characteristics, we will preserve missingness and inconsistent categories rather than silently excluding them. Composition will be compared with source-dataset and, where defensible, population benchmarks, without presenting TCGA as representative of UK patients.

PPI contributors will be recruited through Cambridge BRC and partner networks, targeting communities underserved in cancer research, including Asian and other minoritised ethnic communities. Accessible information, flexible formats and reimbursement will support participation. Contributors will prioritise questions, interpret “not evaluable” findings and co-design reporting. PPI diversity will not be presented as making the datasets representative.

## Approach to analysis

At every stage, we will report demographic counts, missingness, absolute and relative inclusion differences, and bootstrap confidence intervals. Models will estimate associations with inclusion while adjusting for cancer type, centre, period and available clinical factors. Cancer-specific and partially pooled estimates will be prioritised over unqualified pan-cancer averages; no causal claims will be made.

Unimodal and multimodal models will be compared on the same patients where paired comparison is required; broader unimodal cohorts will quantify the cost of complete-pair restriction. Splitting will be patient-level. Repeated nested cross-validation will be stratified where feasible by cancer type and event; preprocessing, selection and calibration will be fitted within training folds. Centre/acquisition and cancer-type hold-outs will assess transportability. Patient-level aggregation will prevent domination by patients with more images.

Survival analyses will report events, follow-up and censoring by group, using Harrell and censoring-weighted concordance, time-dependent AUC, calibration and Brier score. Classification analyses will report discrimination and calibration together.

Fairness analyses will report group estimates, contrasts and uncertainty within cancer type. Minimum sample and event requirements will be predefined. Imprecise results will be labelled **“fairness not evaluable,”** not “no disparity.” Intersectional analyses will remain descriptive unless supported. Sensitivity analyses will compare complete-case and explicit “missing” category approaches.

Exploratory mitigation will retain unimodal patients and assess inverse-probability weighting using observed selection predictors, reporting positivity, effective sample size and unstable weights. These remain sensitivity analyses because unmeasured selection cannot be removed. We will test whether preprocessing failures cluster by acquisition or demographic group; model reweighting cannot repair biased preprocessing.

## Project risks and mitigation

Small subgroups and few events will be handled through description, partial pooling, uncertainty and suppression of unstable comparisons. Incomplete ethnicity and socioeconomic data will be documented. Cancer type, centre and treatment context will be addressed through stratification and adjustment. If external access is delayed, the public-data audit remains complete; if modality overlap is limited, unimodal and staged-selection analyses remain informative. PPI recruitment will begin early. Only aggregate outputs will be released.

## Outputs and dissemination activities

Outputs will include: (1) a reproducible patient-level inclusion ledger and multimodal equity-audit workflow; (2) a methods paper or preprint on selection bias and fairness evaluability; (3) a co-designed checklist and reporting template for BRC translational teams; and (4) a stakeholder workshop and accessible public summary. Software and aggregate results will be shared where governance and licensing permit.

**Months 1–3:** governance review, PPI initiation, cohort definitions and protocol. **Months 4–6:** inclusion ledger, attrition and missingness analyses. **Months 7–9:** model evaluation, censoring and transportability sensitivity analyses, and co-design workshop. **Months 10–12:** toolkit pilot, manuscript, guidance and dissemination.

## Advancement of healthcare translational research

The project will identify who is lost before model training and when subgroup evidence is too weak for fairness or clinical-readiness claims. It separates representation, analytic selection and conditional performance, and connects technical failures to equity. The tools will support more inclusive multimodal studies, retention of unimodal data and honest uncertainty reporting. The framework will be adaptable to other linked healthcare datasets.
