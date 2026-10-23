# R2 v6 design: normal-model image grounding

Status: draft, 11 Oct 2026; §3b E_lesion added 12 Oct 2026. Author: Shangqi Gao (Crispin Lab) with Claude.
Builds on: r2_priors_and_agent_errors_principle, kb_geometry_priors_v1, geometry_prior_check_tcga, anatomy_prior_check_tcga (project notes, 10–11 Oct 2026).

## 1. Principle

R2 is anomaly detection: **model the normal, and treat the tumour as what the normal cannot explain.**

- **Normal tissue can be learned.** Organ geometry comes from expert anatomy annotations (the TotalSegmentator datasets). Normal appearance comes from the host tissue in the same scan.
- **Abnormality has a long tail and is hard to model.** The tumour model is therefore deliberately weak:
  - a broad, architecture-level geometry prior from expert tumour annotations (R2Seg);
  - a per-candidate appearance model fitted in the scan.
  - Tumour models are never pooled across lesions or organs.

Standing rules:
- Priors come only from expert annotations, never from agent outputs.
- Agent masks are noisy observations, reconciled at inference.
- Pan-cancer rules only; stage is never used for tuning; sex is never used.
- The ledger (host and tumour reasoning) is unchanged. It supplies the location prior π and the feasible domain.

## 2. Hypotheses (per ledger lesion, in its local domain)

The local domain is the host organ, plus the neighbours the knowledge base allows for invasion, plus a margin. Three hypotheses are compared:

- **H0 (normal only).** The candidate region is normal host. The organ must explain the whole region, so the organ estimate is the agent organ mask joined with the candidate.
- **H1 (normal + tumour).** The candidate region (core) is tumour. The organ estimate is the agent organ mask minus the candidate.
- **H1-trim.** As H1, but the tumour is the trimmed core. It is triggered when the candidate is too large relative to its host (volume ratio above the tumour prior q95, or candidate ≥ the median normal organ volume), i.e. the mask probably includes host.

## 3. Evidence terms

Each term is a log Bayes factor, log p(data | H1) − log p(data | H0).

| Term | Source | Definition | Available when |
|---|---|---|---|
| **E_vol** (organ volume) | KB anatomy prior | Log density of the organ-estimate volume under the normal-organ prior: H1 uses organ minus tumour, H0 uses organ ∪ candidate. Density from a log-normal matched to the stored quantiles. | The host has an anatomy prior and the organ is not truncated by the field of view |
| **E_shape** (tumour plausibility) | KB tumour prior | Mainly elongation (AUC 0.80–0.82 in the TCGA check), plus multiplicity. A tube-like or sheet-like candidate is implausible under H1. | Always (candidate ≥ about 2 ml for the irregularity component) |
| **E_app** (appearance) | Within the scan | The R2 composition statistic with the D6 reference (host ≤ 30 mm, trimmed to 2nd–98th percentile), calibrated by the host-patch null and turned into a Bayes factor as now. | Enough host and candidate voxels |
| **E_obs** (agent agreement) | Ledger | Already in π (BP/VT agreement). Not added again. | — |

Combination: log BF = E_vol + E_shape + E_app, and E_lesion (§3b) replaces E_app as the default image term, because the host-comparison E_app failed under agent selection. E_shape's elongation is already inside E_lesion and must not be counted twice. The terms use different information (organ size, candidate shape, intensity), so they are treated as independent. Each term is capped at |log BF| ≤ log 100 so that no single term decides alone.

Posterior odds = π/(1 − π) × BF.

## 3a. E_app in detail (revised 11 Oct 2026)

**Feature.** Smoothed intensity: Gaussian σ = 2 voxels in-plane, none across thick slices. MR is normalised by the median and IQR of body tissue in the scan.

**Candidate sample.** The candidate eroded by one voxel in-plane, up to 6,000 voxels.

**Reference: the host organ only, never other anatomy.** In this order:
1. **Local:** the host-organ interior (mask eroded 2.5 mm to drop the partial-volume rim), within 30 mm of the candidate, excluding every candidate dilated 5 mm.
2. **Whole organ:** the same, without the 30 mm limit.
3. **Counterpart:** the knowledge base's anatomical counterpart, e.g. the contralateral kidney or lung. This level is also used directly when the organ is largely replaced (H1 normal-organ volume below the anatomy prior's q5).
4. **None:** E_app is unavailable (review / geometry and prior only).

A level is accepted when it has ≥ max(500, 3 × candidate sample) voxels. The last available level is accepted with ≥ 500. The reference type is recorded for every lesion.

**No intensity trimming of either sample.** The earlier "2–98% tissue window" was dropped:
- trimming only the reference makes lumen contents in the candidate look abnormal;
- trimming both would clip genuinely hypo- or hyper-intense tumour.

Lumen contents are part of the normal organ, so they stay in both samples.

**Statistic.** The held-out density-ratio gain, unchanged in form:
- 32 quantile bins;
- 15 mm checkerboard cross-fitting;
- candidate histogram shrunk to the host with κ = 500;
- mean held-out log(q / f_H) clipped to ±5 (nats/voxel).

**Null.** 50 compact reference patches of the candidate sample's size (≤ reference/3). Each is removed from the reference and scored identically.

**Bayes factor.** The half-normal form log BF(s − median s₀, sd s₀, τ). τ is re-derived a priori for this statistic, by the same definition as v5: the median s of a lesion 1 normal-tissue SD brighter than its tissue, 1–80 ml, by simulation.

**H1-trim: cross-fitted core.**
1. On fold A, fit the host and candidate densities and compute the per-voxel gain at fold-A candidate voxels.
2. Smooth it spatially (normalised convolution, σ = 3 mm) and take the core as gain > 0.
3. Apply that core to fold B, and score fold-B core voxels with the fold-A densities. Then swap the folds.

Fold-B intensities are used neither to fit densities nor to select the core. Null patches go through the same core search. Trim is reported when the core's log BF beats the whole mask's by ≥ log 3 and the core is < 90% of the mask.

## 3b. E_lesion: candidate-intrinsic lesion-likeness (added 12 Oct 2026)

**Why.** Every host-comparison intensity test (E_app with host patches, the selection-matched null, the MMD test X vs X ∪ host) failed for the same reason: the agents select regions that differ from parenchyma whether or not they are tumour (vessels, fissures, bowel, lumen, organ edges). Comparing with the host therefore measures "selected", not "tumour".

**What.** E_lesion asks whether the candidate itself looks like a lesion, with no host reference. Four terms, each in [0, 1] and invariant to affine intensity changes (no MR normalisation):

| Term | Definition | Rejects |
|---|---|---|
| elong | sqrt(λ3/λ1) of the main component | vessels, strips, fissures, walls |
| blobby | mean \|λ_small\|/\|λ_large\| of the Hessian at σ = clip(r_eq/2, 1.5, 4 mm) over the eroded candidate | tubular and sheet-like texture |
| core_focal | best multiscale LoG extremum (σ 2–11 mm) inside the candidate: (enclosure + min(cnr_tex, 2)/2)/2 | "nothing focal" regions |
| het | IQR(candidate)/noise, as het/(het+5) | homogeneous lumen and parenchyma |

B is the mean of the available terms. Code: `pancia_kb/lesion_likeness.py`; `r2_v6.e_lesion()`.

**Bayes factor, from expert data only.** log BF = −3.272 + 6.331·B, capped at ±log 100. The mapping is a class-balanced logistic fit of R2Seg expert tumour masks (287) against expert normal-structure candidates (712) from the TotalSegmentator datasets and R2Seg organ labels. No TCGA tuning.

**Validation, read-only on TCGA, rated sets.**

| Set | Rater | Status | AUC |
|---|---|---|---|
| E1 | user | development | 0.79–0.83 |
| E2 | Claude | development | 0.79–0.80 |
| E3 | Claude | fresh; Bfix frozen before | 0.85 [0.76, 0.93], p = 1e-6 |
| E4 | Claude | fresh; B5 frozen before | 0.78 [0.65, 0.91], p = 7e-5 |

- Pooled over E1–E4: about 0.80.
- After adjusting for size, B keeps a coefficient of about 1.3 ± 0.25 (z ≈ 5).
- With the expert-calibrated BF, 64–71% of rated tumours get log BF > 0 and 75–80% of non-tumours get log BF < 0, in every set.
- |log BF| > log 3 is rare, so E_lesion is weak evidence by design and never decides alone.

**Limits.**
- On an expert same-domain check (R2Seg tumours against ball-shaped crops of expert organ labels in the same scans), B falls to 0.59. Most of its power against agent errors comes from shape (elongation), which compact crops remove. The texture terms (blobby, core_focal) keep a modest signal (0.62–0.65).
- MR is weaker than CT (about 0.71–0.76 vs 0.83–0.85).
- E2–E4 are Claude's non-expert ratings; E1 is the user's.
- A boundary-support term (is the candidate edge a real image edge?) was tested and dropped: AUC 0.60 pooled, 0.42 on E1, 0.50 on MR.

## 4. Agent-error handling (inference only, agent-agnostic)

Checks on the agent organ mask under the anatomy prior, before evidence is computed:

| Check | Signature | Action |
|---|---|---|
| Notch | Little of the lesion inside the organ, high contact, and organ ∪ lesion more plausible than the organ alone | Organ estimate = organ ∪ connected candidate |
| Malformed organ | ≥ 2 shape descriptors beyond q5/q95 and not explained by the candidate | Drop E_vol (organ evidence unreliable) and record the error |
| Truncated | Mask touches the field-of-view border | Drop E_vol |
| Missing organ | No mask | Drop E_vol; use the anatomical counterpart if the knowledge base defines one |
| Fragmented | Main component < 90% | Keep the main component and the components touching the candidate |

## 5. Decisions

- **Admit:** posterior probability > 0.5 and at least one evidence term available.
- **Reject:** posterior probability ≤ 0.5 with at least one term available.
- **Review:** no evidence term available, or |log BF| < log 3 and the decision would only follow the prior. This is a flag, not a decision.
- **Trim:** if H1-trim beats H1 by a Bayes factor ≥ 3, report the trimmed core.
- **Recovery:** ledger-rejected but recoverable lesions (unchanged RECOVERABLE set and π cap) are admitted only on log BF > 0 with E_vol or E_app available. Shape alone never recovers a lesion.

## 6. Delineation

- **v6.0:** lesion-level decisions as above. Trim and grow keep the v5 3:1 voxel rule, now using the D6 reference.
- **v6.1 (later):** coupled normal/tumour contour evolution (level set). The anatomy prior regularises the organ, the tumour prior the tumour, and E_app is the data term. Trim, recall, split and merge then happen in one process.

## 7. Calibration and checks (no tuning)

1. **False-positive check on normal anatomy (not TCGA).** On TotalSegmentator-dataset normal organs, mark random organ-sized patches as "candidates" and run H0 vs H1. E_vol + E_shape should favour H0 in at least 95% of cases. This tests the prior-based terms without any tumour data.
2. **Size of E_app:** the host-patch null, as in the null study.
3. **TCGA checks, read-only:**
   - the 60 rated lesions and the 162 confident primaries;
   - a fresh visual review of the decisions v6 changes relative to v3 and v5.

## 8. Known gaps

- **Hollow-organ walls:** no wall prior yet. Bladder, colon, stomach and oesophagus rely on E_vol (organ enlargement) and E_app.
- **Long organs** (oesophagus, colon, lung): the volume priors may be biased small because truncated organs were excluded. Fix before using E_vol for them: per-length or truncation-aware measures.
- **Hosts without an anatomy prior** (breast, ovary, uterus, cervix, lymph nodes, bone): E_shape + E_app only. Uterus and cervix could later use expert labels from R2Seg.
- **Small distant lesions** (lung or liver metastases) give negligible E_vol, so their decisions rest on E_shape + E_app.
- **Thick-slice MR:** shape descriptors are measured on a 1.5 mm isotropic grid, but irregularity is unreliable under about 2 ml.

## 9. Implementation plan

1. New module `pancia_kb/r2_v6.py`. `r2_em.py` stays untouched, so the md5 guard stays valid. Selected with `em_model='v6'`.
2. Evidence functions use `pancia_kb/geometry.py` and `geometry_priors.yaml`.
3. Run the false-positive check (§7.1) on TotalSegmentator normal organs.
4. Pilot on the series of the rated lesions and the confident cohort. Compare with v3 and v5; blinded review of the changed decisions.
5. Only then a full-cohort HPC run.
