# PanCIA v8: execution plan, testbed refit to finished paper inputs (19 Oct 2026)

This plan implements `claude/hyperparameter_plan_2026-10-19.md`; `documents/main.tex` (md5 3d98e2) already describes v8. Each step has a gate. When a gate fails, the default action is pre-specified, so the run can continue unattended.

**Standing constraints:**
- Expert data (R2Seg and TotalSegmentator) only, for every estimate.
- Never run BiomedParse, VoxTell or TotalSegmentator on R2Seg.
- OV04 is excluded.
- Never use sex.
- No tuning on stage or TCGA; the TCGA pilot is a bug check.
- No file is deleted; files are moved to `_to_delete/<date>/`.
- v3, v6s, v6a and v7 outputs stay unchanged.

**Design rule for the whole run.** Store everything needed to recompute decisions (the host features φ for each candidate, the test outcome for each host, the QC metrics). Then most sensitivity analyses and ablations become recomputation, not reruns.

---

## Step 1: grounding module g2 (`pancia_kb/grounding_g2.py`, new)

- **Expert tables.** Reuse the g1 outcome tables (`tables_g1_native.json`, counts per stratum).
  - Choose the smoothing (κ, m) by held-out log-likelihood, leaving out one R2Seg collection and one TotalSegmentator fold at a time.
  - Grid: κ ∈ {0, 0.05, 0.1, 0.2}, m ∈ {0, 5, 10, 20}.
- **Test sets.** For each α ∈ {0.01, 0.05, 0.10} and each stratum:
  - rule-out outcomes: p(o|T) ≤ α and Λ < 1;
  - confirm outcomes: p(o|N) ≤ α and Λ > 1.

  Written to `ledger/v8/grounding/test_sets.json`.
- **Functions:**
  - `support_level(kind, signals)`;
  - `decide(level, test)`, the decision table (conflict → uncertain);
  - `credible_set(P, alpha)`;
  - `host_posterior(phi, beta, no_host_score)`, the choice model;
  - `extent_prior(psi, eta, flags, logratio)`.
- **Gate:** unit tests on toy inputs. The decision table must be monotone, and the credible set must hold ≥ 1 − α of the probability and be minimal.

## Step 2: testbed refit (`scripts/tb_fit2.py`, new; held out by collection)

**2a. Host (choice model).**
- Fit a conditional logit (scipy) on the testbed host rows (`ledger/v7/testbed_host2`).
- The no-host score is the α-quantile of the held-out true-host scores.
- Report, overall, by scenario (excl / incl / leak / partial / missing) and by position (inside / straddle / exophytic):
  - top-1 accuracy;
  - credible-set coverage (the true host is in the set);
  - mean credible-set size.

**2b. Existence.**
- For each α, the held-out false rule-out rate of expert tumour cores (partial, sub, cut2d) and the false confirm rate of expert normal candidates.
- Report overall, by modality, by size band (< 3 ml separately) and by stratum.

**2c. Extent.**
- Fit ψ and η_f (flags: leak, organ, prompt mismatch, flat) jointly as piecewise density ratios.
- Report Dice of the chosen extent, best-extent rate, over-15% growth and median ratio against the v7 prior (π̄ · ψ).

**Gates and default actions:**

| Gate | Pass | Default if it fails |
|---|---|---|
| Host coverage at α = 0.05 | ≥ 0.90 overall | Report it; do not recalibrate. Note in the limitations. |
| Existence false rule-out (held out) | ≤ 2α in every modality × size band | That cell gets no rule-out outcomes (neutral only). This rule is pre-specified here, before the results. |
| Existence false confirm (held out) | ≤ 2α likewise | That cell gets no confirm outcomes. |
| Extent | Dice ≥ v7 prior − 0.01 | Keep ψ alone (η_f = 1) and report. |

Outputs: `ledger/v8/testbed/fit_g2.json` and `fit_report_g2.txt`. The v7 fits are kept unchanged.

## Step 3: testbed extension (optional; only if the expert labels exist)

- **Organ labels.** Check whether the R2Seg Bladder, Breast, Colon and Uterus collections carry organ labels for their own host and neighbours.
  - If they do, add them as host cases and repeat step 2a.
  - If they don't, skip. TotalSegmentator cannot be run on R2Seg, so record it as a limitation.
- **Synthetic invasion cases.** Join an expert tumour to a dilated band into a neighbouring organ mask, a pure geometric construction on expert masks. Report how often the credible set contains both organs.
- **Gate:** none. This is reporting only.

## Step 4: ledger v8 and run script

- **`pancia_kb/ledger_v8.py`** (copied from v7, which stays unchanged):
  - organs outside H_C get the same weight as every candidate;
  - drop π̄ from the extents;
  - store the flag set of each extent;
  - store φ for each candidate of each lesion;
  - store the warning signals for each host.
- **`scripts/g2_run.py`** (from g1_run):
  - host probabilities and credible set for each lesion;
  - level, outcome, Λ and test result for each host in the set;
  - decision (all hosts must agree, otherwise uncertain);
  - extent by w0.
- **Override:**
  - lesion choice: credible set contains the primary, highest level, then largest volume;
  - only that lesion's non-rejected anchors are added;
  - the inferred and imposed hosts and their classes are written correctly (fixes the X42 bug).
- **Outputs:** `ledger/v8_pilot/{anchors,lesions,series}_s*.csv`, plus the cache for panels.
- **Smoke test on 3 series:** the X42 OV series, X55 and one KIRC MR series.
- **Gate:** no exceptions, decisions consistent with the table, and the override labels correct. If any fails, fix and retest; do not start step 5 before it passes.

## Step 5: TCGA pilot rerun (152 series, bug check only)

- Same loop as run 2: four shards, a 45 s budget per call, solo reruns for large series.
- **Summary against run 2** (`claude/v8_pilot_run_<date>.md`):
  - decision counts;
  - series with an accepted tumour;
  - credible-set size distribution;
  - omentum in credible sets;
  - overrides;
  - anchors whose decision differs across hosts.
- **Gate (bugs only):**
  - no KB-region box larger than its organ span;
  - no host out of view;
  - no override with the wrong class.

  Nothing is tuned on these outputs.

## Step 6: panels for the user's visual check

- Whole-slice LPS panels (v8_png3, from v7_png3) for the same cases (`png_index_run1.csv` mapping) plus a new group H: about 40 sampled lesions whose credible set holds more than one host.
- Each title shows:
  - the credible set with probabilities;
  - support level, outcome and test result for each host;
  - the decision;
  - the override with its inferred and imposed host.
- **Gate:** none. This is for the user.

## Step 7: sensitivity and ablation

**Recomputed from stored outputs (no rerun):**
- α ∈ {0.01, 0.05, 0.10};
- conflict rule (ℓ = 2 + rule-out → reject);
- KB host weights 1 / 0.3 / 0.15 instead of equal;
- numeric priors (π_two 0.75 / 0.9, ρ 1/3, thresholds 0.75 / 0.25);
- each evidence component removed (requires storing the signs, see step 4);
- no ψ or η_f; binary touch.

**On the testbed:**
- τ and envelope ∈ {5, 10, 15} mm for host top-1 and coverage;
- extent evidence added (the v7 NO-GO result is reused).

**On the pilot (two reruns, 40 min each):** τ and envelope at 5 and 15 mm, reporting only lesion counts and decision counts.

**QC thresholds** (recomputed from stored QC metrics): the share of degraded scans for f ∈ {0.2, 0.25, 0.33} and slice spacing ∈ {6, 8, 10} mm.

Supplement only: one-at-a-time checks of the lobe opening, claim minimum, flatness, leak and overlap thresholds on the testbed.

## Step 8: fill main.tex and the documents

- **Table S1:** the fitted β, no-host quantile, ψ, η_f and the chosen κ / m.
- **§3.4 (expert validation):** host, existence and extent numbers.
- **§3.6:** the ablation and sensitivity numbers.
- **Limitations:** check them against the results.
- **Pilot numbers:** reported in a supplementary note only; not used for any choice.
- **Backup:** `bak/main_before_v8_numbers.tex`.
- **Project docs:**
  - `claude/v8_testbed_results_<date>.md`;
  - `claude/v8_pilot_run_<date>.md`;
  - `claude/v8_sensitivity_<date>.md`.

## Step 9: hand-over to the user

A short summary with the gate outcomes and every default action taken. The panels are ready for the visual check, and the open questions are listed.

---

## Estimated time

| Step | Time |
|---|---|
| 1 | about 1 h |
| 2 | 1–2 h |
| 3 | about 1 h, or skipped |
| 4 | about 1.5 h |
| 5 | about 45 min |
| 6 | about 45 min |
| 7 | 1–2 h |
| 8 | about 1 h |

Total: about 8–10 h of work.

## Decisions that would still need the user (not taken unattended)

- **Changing the framework or the decision table.** If the testbed shows the table is badly unbalanced, for example almost no confirmations anywhere, it is reported, not changed.
- **Adding a new feature to the compatibility score**, for example to handle diffuse organs such as the omentum.
- **Any change suggested by the TCGA panels** other than a bug fix.
