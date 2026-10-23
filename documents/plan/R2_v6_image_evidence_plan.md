# R2 v6 image-evidence grounding: final plan

Status: plan, 12 Oct 2026; revised the same day to a size-conditioned shape likelihood (elongation, flatness, tubularity, blobby), frozen in ledger/v6/FROZEN_LIKELIHOOD_v61_20261012.txt. Author: Shangqi Gao (Crispin Lab) with Claude.

This plan supersedes the evidence terms in R2_v6_design §3, §3a and §3b for the image-grounding study. It covers the image observation likelihood only; there is no prior and no posterior.

## 1. Goal

For each candidate region (a mask on an image), compute an observation log-likelihood ratio:

> log Λ(x) = log p(x | T) − log p(x | N)

- **T:** the region is (part of) a tumour.
- **N:** the region is normal anatomy that a segmentation agent could plausibly return.
- **x:** a small set of image features.

The output is log Λ and one of three classes:

| Class | Rule |
|---|---|
| Tumour-like | log Λ > log 3 |
| Organ-like | log Λ < −log 3 |
| Uncertain | otherwise; this means review, and is an expected outcome, not a failure |

**Requirement:** it must work both in-domain (the R2Seg and TotalSegmentator expert data it is built from) and out-of-domain (TCGA agent candidates rated by the user). Nothing is tuned on TCGA.

## 2. What the experiments taught us

**Lesson 1. A segmentation agent only returns regions that already differ from their surroundings.** Any evidence that compares the candidate with neighbouring or host tissue measures selection, not tumour. This was shown repeatedly:
- the appearance term with host patches (inverted);
- the selection-matched null;
- the MMD test of candidate vs candidate + host;
- contrast against far host tissue (0.58 overall, 0.20 on E1, 0.21 on the hardest cases);
- boundary support (0.60 overall, 0.42 on E1).

**Lesson 2. Readers decide by anatomical identity.** On the user's rubric, "not a normal structure" (Q1) predicts the call with AUC 0.99. Identity cannot be computed reliably today:
- TotalSegmentator overlap fails (0.39), because the confounders are unlabelled or merged with the tumour.

Only geometry partly tracks identity:
- elongation vs the user's Q1: 0.71.

**Lesson 3. Hard cases are hard for everyone.** On the user's labels:
- the current score reaches 0.82 on candidates Claude rated decisively, but 0.52 on candidates Claude rated unsure;
- AUCs computed only on decisive cases overstate performance.

The method should put borderline cases in the uncertain band rather than force them.

**Lesson 4. Delineation is not existence.** The user labels partial contours of real tumours as tumour. Leak and contour fit penalise them (AUC 0.58 among partial contours). They are evidence for grow/trim, not for whether a tumour exists.

**Lesson 5. Mask geometry transfers across domains better than intensity.** Elongation was stable in every set (E1–E4, the user's labels, CT and MR). Intensity and texture terms vary with modality, sequence and dataset: MR performance is near chance, and results flip between datasets.

## 3. Keep and drop

**Kept in the likelihood:**

Revised 12 Oct 2026 after the evaluation on all 420 rated candidates (`claude/r2_v6_eval_all_rated_2026-10-12`). The core is the four shape terms that are independent of size and rate the same under both raters.

| Feature | Evidence (AUC on user labels; size-adjusted z) | Role |
|---|---|---|
| elong, sqrt(λ3/λ1) | 0.72; z 3.9; CT 0.76 / MR 0.62; < 3 ml 0.81 | core |
| flat (flatness) | 0.66; z 3.4; distant candidates 0.82 | core |
| tube, Hessian line-likeness (lower = tumour) | 0.56; z 3.0 | core |
| blobby, Hessian isotropy at the candidate's scale | 0.58; z 2.7; expert normals 0.85 | core |

Not in the core:
- **Focal core** (0.65; z 1.2) and **heterogeneity** (0.59; z 1.3) are not size-independent on the user's labels. They are reported, not used.

**Size conditioning.** Candidate volume is not an evidence term. The size of non-tumour candidates depends on the agent, and small real tumours must not be penalised for being small. Volume is a condition instead: the shape likelihood is fitted within size bands (< 3 ml, 3–20 ml, > 20 ml), so how much shape counts depends on size. Shape is very informative for small candidates (elongation 0.81 under 3 ml) and weak for 3–20 ml (0.52).

**Kept outside the likelihood, as a separate organ-level observation:**
- **Organ volume.** The anatomy prior gives calibrated false support of 2.7%, but it is informative only when the tumour is at least about 30% of the organ. It stays a separate, optional term and is reported alongside.

**Moved to delineation (grow/trim, level set):**
- **leak, contour fit and obj_in.** They indicate an incomplete or over-inclusive contour.

**Dropped:**

| Item | Reason |
|---|---|
| All host or surrounding comparisons: appearance term, matched null, MMD, c_mask, boundary support | Agent selection (Lesson 1) |
| Heterogeneity | Penalises homogeneous tumours, which were the lowest-scoring true tumours |
| Scale-matched blob contrast (ab_s) | Intensity contrast again: user 0.59, MR unstable |
| TotalSegmentator overlap and structure match (normal_id, j_max) | Fails as identity (0.39 vs user Q1) |
| Size as a stand-alone likelihood term | The non-tumour size distribution depends on the agent, not the image, and it would penalise small true tumours. Size is used as a condition instead (see above) |
| Focal core, heterogeneity, blob contrast, CNR, enclosure | Not independent of size on the user's labels, or intensity-based |

**Optional, tested only after the core is frozen:**
- **3-D tubular continuity:** does the candidate connect to a long tube-like structure beyond its contour? This targets vessels and bowel, the most common named confounders after organ parts. It is the only identity cue that can be computed without new labels.

## 4. Building the likelihood (expert data only)

**Tumour class T.** R2Seg expert tumour masks from 11 public datasets (OV04 excluded), the main component. Agent-like delineation noise is added, because the user counts partial tumour contours as tumour:
- the original mask;
- an eroded or partial version (a random sub-region covering 30–70% of the tumour);
- an over-inclusive version (dilated 3–5 mm).

Each variant has equal weight.

**Normal class N.** Expert normal structures from the TotalSegmentator CT (no-pathology cases) and MR datasets, and from R2Seg organ labels. They are presented the way agents mistake them, using the failure taxonomy categories with equal weight per category (not tuned on TCGA):

| Category | How the candidate is built |
|---|---|
| Vessel | segment along the vessel, 10–60 mm long |
| Bowel / GI wall | segment of bowel, or wall |
| Whole small structure | prostate, gallbladder, adrenal, thyroid, kidney, spleen |
| Organ part | compact crop of liver, kidney or uterus |
| Lumen / fluid | bladder |
| Bone | vertebral body |
| Organ edge | strip along the organ edge |

**Density model (size-conditioned).** Within each size band (< 3, 3–20, > 20 ml):
- A class-balanced logistic regression on the four core terms, with no volume term. The classes have equal total weight in each band, and the normal categories have equal weight within N.
- A light L2 penalty (C = 1 on standardised features).
- log Λ(x | band) = the fitted logit. It is monotone, has few parameters and is robust.
- One model per band, pooled across modalities. Per-modality behaviour is reported, not fitted.
- log Λ is capped at ±log 100. The class thresholds are ±log 3.

**Freeze.** The feature list, sampling and model are written to a FROZEN file before any evaluation.

## 5. Evaluation (no agent runs; the existing data only)

**In-domain: R2Seg and TotalSegmentator expert data.**
- **Leave one dataset out:** for each R2Seg dataset, refit on the others, then score that dataset's tumours (with the three delineation variants) against normal candidates built from its own organ labels (same scans, same domain).
- **Cross-dataset check:** also score tumours against TotalSegmentator normals.
- **Reported:** per dataset and per modality.

**Out-of-domain: TCGA agent candidates.**
- **Primary labels:** the user's — rubric Q5 (137 items) and E1 (60).
- **Secondary labels:** Claude's E2–E4, reported separately.
- **Stratified by:** rater confidence or unsure status, modality, and the user's Q1 category.

**Metrics** (the same in both domains):
- **Coverage:** the share assigned tumour-like or organ-like.
- **Confident-call error:** tumours called organ-like, and normals called tumour-like.
- **Uncertain band:** the share of rater-unsure items that land in it.
- **AUC:** secondary only.

**Pre-specified success criteria:**
- Confident-call error ≤ 15% in both domains.
- Out-of-domain coverage ≥ 40% of user-decisive items.
- Rater-unsure items mostly in the uncertain band.
- The calibration shift between domains is reported: the in-domain vs out-of-domain distribution of log Λ for each class.

## 6. Known limits (stated in the paper)

- **Identity confounders are not solvable by image geometry and texture:** breast parenchyma and vessels, intra-organ vessels, renal sinus and kidney poles, bowel loops. They should mostly fall in the uncertain band. Any that come out tumour-like are the known failure mode.
- **MR evidence is weaker than CT.**
- **The tumour likelihood comes from primary tumours in 11 datasets.** Hosts without R2Seg data (oesophagus, stomach, endometrium, ovary) rely on the architecture-level pooling.
