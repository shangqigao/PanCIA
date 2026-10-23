# PanCIA v7 next steps (updated 18 Oct 2026)

This plan replaces the first version of 18 Oct.

It follows the revised `documents/main.tex` (md5 661a63…; v7 Methods). The project doc `main_tex_v7_revision_2026-10-18` lists the changes.

## Decisions taken by the user (18 Oct)

- **Flagged and leaking outlines** are penalised extent candidates.
- **Other or indeterminate lesions** are reported only.
- **Primary not detected:** the scan is flagged, and the best primary-host lesion is added as a flagged override that is reported separately (confirmed).
- **TCGA is for visual checks only.** It has no tumour annotation: it is used for visual checks, blind ratings and stage-free yardsticks. Expert data (R2Seg + TotalSegmentator) is the only quantitative validation and the only source of learned quantities.

## Constraints (unchanged)

- Public data only; OV04 is never read.
- Priors and likelihoods come from expert annotations only. Agents (BP, VT, TS) are not run on R2Seg.
- Sex is never used. Nothing is tuned on stage. Rules are pan-cancer.
- Agents are pluggable and treated equally.
- Ledger v3, v6s and v6a stay frozen; new code goes into `ledger_v7` and `grounding_g1`.
- Files are never deleted (they go to `_to_delete/`).

---

## A. Housekeeping

1. **Record decisions.** Update the project and plan docs, the rules drafts (`Anatomy_QC_rules_draft.md`, `Image_grounding_rules_draft.md`) and the comments in `main.tex`.
2. **Draft Supplementary Table S1.** List every constant with its source (expert data / knowledge base / design choice with reason).
   - Design constants: π_two = 0.9, π_single = 0.5, ρ = 1/3, w_h = 1 / 0.3 / 0.15, τ = 10 mm, envelope 10 mm, μ ∈ {0, ½, 1}, the decision thresholds 0.75 / 0.25, and the lobe opening of 3 mm.

## B. Expert testbed (R2Seg + TotalSegmentator; agents not run)

3. **Data audit.**
   - Confirm which R2Seg sets have both expert organ and tumour labels: KiTS, Liver/Pancreas (FLARE23 files), NSCLC-Radiomics, Cervix, Prostate158.
   - Check FLARE23 provenance and the tumour-free cases.
   - Pair Prostate158 T2 zones with the ADC tumour masks.
   - Separate the NSCLC primary from involved nodes.
4. **Candidate simulator.**
   - **Tumour cores:** partial 30–70%, and three-slice cuts.
   - **Outlines:** exact; dilated 3–5 mm; grown into host parenchyma; leaking into a neighbour; organ-sized; a false lobe joined by a thin bridge.
   - **Normal candidates:** the existing vessel and bowel tubes, organ parts and edges, and salient structures.
   - **Perturbed organ masks:** leak, merge, missing part, missing organ, tumour included in the organ.
   - **Position classes:** inside / straddling / exophytic, from expert organ and tumour masks.
5. **Testbed harness.** Run the v7 reasoning and the G1 grounding on simulated cases, with held-out splits (leave one R2Seg collection out; two folds of TotalSegmentator) and the metric code.

## C. Quantities estimated on expert data

6. **Existence tables.** Rebuild with the tumour class = expert masks plus partial cores only (no dilated or leaking variants), and with S5 removed. Strata are organ class × band × modality.
7. **Native-resolution tables** for anchors under 3 ml on scans with slices ≤ 2.5 mm. Check the D19 type: small lesions at native resolution vs on the 5 mm grid.
8. **Compatibility weights β** (logistic, Eq. compat), and g_min, fitted on the testbed host cases.
9. **Parsimony term ψ:** the density of log(|expert tumour| / |core|) over the simulated cores and outlines.
10. **Extent evidence** (Eq. extent_evidence).
    - Estimators for f_a and f_h; fit ν for calibration; check that iso-intense extensions are neutral.
    - **Go / no-go:** if extent choice does not beat "anchor only" and "prior only" on the testbed, switch to the fallback (boundary edge evidence + ψ), and tell the user before changing the text.

## D. Implementation (on device; deploy pattern with md5 checks)

11. **`ledger_v7`** (built from v6a; v6a is untouched):
    - mask confidence μ;
    - zones (core / boundary / contested) and organ states, with knowledge-base regions for any unsegmented host;
    - lobes and extensions;
    - compatibility features;
    - lesions (τ = 10 mm, chaining only via two-agent anchors);
    - warning signals and flags, with no hard drops except corrupt output;
    - π with ρ, and w⁰ with flags and ψ;
    - X_a including flagged outlines;
    - candidates outside H_C (for attribution).
12. **`grounding_g1`:**
    - frozen 𝒩_h and existence signs on the anchor;
    - extent evidence on R_a;
    - one-pass enumeration (Eqs. case_posterior–combination);
    - lesion-level host;
    - p_prim;
    - decision policy with overrides and the "other" label;
    - outputs, plus a per-anchor .npz cache for fast panels.
13. **Smoke and unit tests** on synthetic data, covering:
    - a lobe through a bridge;
    - an iso-intense extension;
    - an unsegmented host;
    - an "other" lesion;
    - primary not detected.

## E. Validation on the testbed

14. **Metrics** (held out):
    - **Existence:** calibration (ECE, reliability) and AUROC, overall and for < 3 ml.
    - **Host:** top-1 accuracy by position class and under organ-mask perturbation.
    - **Extent:** closest-extent rate, Dice, volume ratio, growth into host parenchyma.
15. **Sensitivity** of the design constants: support levels, ρ ∈ {1/2, 1/3, 1/5}, and w_h.
16. **Ablations** (main.tex §3.8):
    - flat priors;
    - each evidence component removed;
    - no ψ;
    - flagged outlines pruned vs penalised;
    - touch vs g_h;
    - one-pass vs iterated;
    - ρ.
17. **Baseline comparison:** v6a + G0 v2 on the same testbed.

## F. TCGA visual check (152 pilot series)

18. **Run v7 + G1 on the 152 pilot series** (sharded, with OMP_NUM_THREADS = 1).
19. **Panels from the cache** (2×2: image only plus overlays) for:
    - accepted, uncertain and rejected anchors;
    - extent changes;
    - other lesions;
    - primary-not-detected overrides;
    - the earlier problem cases (E0–E7, D15, D17, D19, and protected-primary bowel and vessel cases).

    The user then reviews the panels.
20. **Fix implementation bugs only.** Any rule change prompted by TCGA is recorded and disclosed (main.tex §3.1 sentence).

## G. Freeze and full cohort

21. **FROZEN file:** code md5s, the knowledge base version, tables, β, ψ, ν, and all constants.
22. **Full run** on 3,291 baseline series, recording QC exclude / degrade counts.
23. **Stage-free yardsticks** (spread, cross, disagree, miss) for BP alone, VT alone, reasoning only (Λ = 1) and the full model. Table 1 per cancer type.
24. **Blind rating set** drawn from v7 outputs (accepted, uncertain, rejected, other), rated by the user.
25. **Stage correlation**, reported only.

## H. Paper

26. **Fill all [TBD]:**
    - data counts;
    - §3.4 testbed results;
    - Table 1;
    - yardsticks;
    - ablations;
    - qualitative figure.
27. **Figures:**
    - Fig. 1 pipeline (wording only);
    - Fig. 2 PGM (kept);
    - the qualitative grounding figure, regenerated;
    - the knowledge-base graph.
28. **Supplementary:** Table S1, testbed construction, sensitivity tables, compatibility features.
29. **Abstract and Introduction** (still placeholders).
30. **Final consistency pass:** notation, cross-references and limitations; sync the rules-draft docs to the text.

## Suggested order and first milestone

A1–A2 → B3–B5 → C6–C10 (the extent-evidence go/no-go is the key risk) → D11–D13 → E14–E17 → F18–F20 → G21–G25 → H26–H30.

The first milestone is B3–B5 plus C9–C10: the testbed plus the extent model. This tells us early whether the fix for the bad orange extents works, before the ledger is rebuilt.
