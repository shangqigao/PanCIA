# Ledger v6 plan: host-uncertain reasoning (13 Oct 2026, draft for approval)

Author: Shangqi Gao (Crispin Lab) with Claude. Data: public only (TCGA-TCIA, R2Seg public sets, TotalSegmentator datasets; R2Seg ovary excluded).
Standing rules:
- pan-cancer rules only;
- never use sex;
- no tuning on stage;
- priors only from expert annotations, never from agent outputs;
- agents are pluggable;
- v3 stays frozen as the baseline.

## Principle (user, 13 Oct)
Both the anatomy agent and the tumour agents make segmentation errors. So neither the host nor the tumour can be selected with certainty, and each gets a prior. The pipeline separates reasoning by data type:

- **Host reasoning and tumour reasoning:** set-based only (masks, KB relations). No image.
- **Image grounding:** the only stage that reads intensities. It combines the image signs with the host and tumour priors in one posterior.

## Stage 0: candidate host anchors (unchanged planner)
Per scan, the tumour-hosting anchors are the KB spread set in the field of view, each with its KB prior w_h (primary 1.0, local invasion 0.3, distant site 0.15).

## Stage 1: host reasoning (set-based)
1. **Possible hosts of a claim.** These are only the anchors whose envelope the claim touches (contact within 3 mm). If a claim touches no anchor, its only option is "no host".
2. **Exclude completely infeasible hosts** (hard, set-based, no thresholds on overlap share):
   - anchors outside the spread set;
   - anchors with no organ reference in the scan;
   - side conflicts for paired organs;
   - h is excluded as host when the claim is the organ itself (organ-shaped relative to h: Dice / volume ratio as in the existing rule). The claim is then recorded as an organ claim.
3. **Host prior for each claim.** Over the remaining possible hosts plus "none":

   P(h | c) ∝ w_h · L(g_c,h)

   where g_c,h is set geometry: the share of the claim inside h's envelope, the share of the claim's surface in contact with h, and the distance gap.
4. **The 50% host rule is removed.** A claim lying between organs keeps all of them, each with a probability.

### L: one pan-cancer geometric likelihood
- **Built only from expert annotations.** From R2Seg expert labels (tumour plus organ labels; FLARE also labels neighbouring organs), measure g for the true host and for every labelled non-host organ the tumour touches. Pool across organs and datasets with equal dataset weights: one L for all cancers, with no organ-specific constants.
- **Form.** A density ratio p(g | true host) / p(g | touched non-host), constrained to be monotone in overlap and contact, and fixed before any TCGA result.
- **Agent noise.** g is measured on the dilated envelope, as the ledger does. L has a floor ε, so geometry alone never sets a touched host to 0. This tolerates absorption, notches and an over-segmented neighbour. ε is fixed a priori: the L value of a claim that only touches the organ.
- **Check before use** (expert data only): calibration of P(true host) on held-out R2Seg datasets (leave one dataset out).

## Stage 2: tumour reasoning (set-based)
- **Conditional prior per possible host.** For each claim and each possible host h, π(z = 1 | c, h) comes from the v3 rules evaluated as if h were the host:
  - agreement prior: 0.9 / 0.75 / 0.5 / 0.25;
  - primary rules (largest feasible, multifocal nodule rule, VT-silent);
  - local invasion needs contiguity with a primary candidate;
  - a distant site needs BP ∩ that site's prompted VT;
  - slab and organ-shaped rules, applied relative to h.
- **Rules that depend on other lesions** (largest primary, contiguity) use a two-pass scheme. The first pass computes marginals; the second applies contiguity against candidates whose primary marginal is ≥ 0.5.
- **Marginal prior:** P(z = 1 | c) = Σ_h P(h | c) · π(z = 1 | c, h). A candidate is infeasible only when every possible host makes it infeasible.
- **Recorded per candidate:**
  - host distribution, host entropy, and the number of hosts with P ≥ 0.1;
  - contact area with each organ (multi-organ contact for downstream risk analysis; recorded, not interpreted).

## Stage 3: image grounding (joint posterior)
**Posterior:**

P(z, h | image) ∝ P(h | c) · π(z | c, h) · p(image | z, h)

**Signs.** The current workable signs:
- contrast router (|zh_med| ≥ 1.5 against host h's normal model);
- closedness against h (30 mm);
- patchy heterogeneity;
- elongation;
- fix C.

Host-dependent signs are computed per possible host. Intrinsic signs are shared. A candidate whose possible hosts are only "none" gets the intrinsic signs only.

**Likelihood.** The rule's three outcomes are turned into likelihood ratios from expert in-domain frequencies (R2Seg GT tumours vs TotalSegmentator normals). For example, with the current rule:

| Outcome | GT tumours | Normals | Likelihood ratio |
|---|---|---|---|
| Tumour-like | 76.5% | 4.9% | ≈ 15.6 |
| Organ-like | 6.6% | 68.6% | ≈ 0.10 |
| Uncertain | 16.9% | 26.5% | ≈ 0.64 |

These are frozen before TCGA.

**Image evidence can also move the host.** The candidate's contrast against each possible host's parenchyma enters p(image | z, h).

**Decisions:**
- **admit** if P(z = 1 | ·) > 0.5;
- **report the host** as the full distribution plus the most probable host;
- **flag for review** when the decision follows the prior only, or when the two most probable hosts are within 0.2 of each other.

## Stage 4: evaluation, v6 vs v3 (no tuning on stage)
1. **Degenerate check.** With L set to v3's hard 50% rule and image grounding off, v6 must reproduce v3 statuses exactly.
2. **Cohort scope.** Baseline-study series, stage-free yardsticks:
   - series spread;
   - same-day RECIST disagreement;
   - index disagreement;
   - miss rate;
   - admitted-primary rate where the primary is visible (v3: 65 sampled series with none).
3. **Audit sets.** The gate-audit 22 and the absorb-audit 34. The user labels them first; my reads are not ground truth. Count:
   - lost masses recovered;
   - false admissions of whole normal organs (spleen, bladder) — the main risk of soft hosts.
4. **Rated TCGA sets (yA / yB).** Coverage and confident error, as now.
5. **Stage.** Patient-level ρ with T stage is reported only. Multi-organ contact vs T4 is reported as an exploratory association.

## Implementation
- **New modules:**
  - `pancia_kb/ledger_v6.py` (host and tumour reasoning);
  - `pancia_kb/host_lik.py` (L);
  - image grounding in `r2_v6`.
- **Untouched:** `ledger.py`, `r2_em.py` and the md5 guards.
- **Build script:** `scripts/build_host_likelihood.py` (R2Seg expert labels only). It writes `knowledge_base/host_likelihood.yaml`, which is frozen and logged.
- **Output:** `TCGA_Seg/Ledger_v6`; comparison with `ledger_r2_compare.py`.
- **Order:**
  1. L build and expert check.
  2. Set-based v6 with the degenerate check.
  3. Pilot on the audit series and the 23 pilot series.
  4. Image grounding.
  5. Full cohort on HPC.
  6. Comparison with v3.

## Open choices (proposed defaults)
- **Contact radius** for "possible host": 3 mm (the ledger's contact constant).
- **"None" option:** it gets w = the recall prior 0.25 × L at zero contact.
- **The distant rule keeps the site-prompt requirement.** It is evidence, not host geometry.

## Decisions log
- 13 Oct 2026 (user): possible hosts without an organ mask stay possible at the floor ε and are flagged as uncertain for later review.
- 13 Oct 2026: step 1 done. L is built from the expert 3 mm shell share (`knowledge_base/host_likelihood.yaml`; scripts `host_lik_extract.py`, `host_lik_fit.py`). See the project note `ledger_v6_step1_host_likelihood_2026-10-13`. The steepness of the floor is open for decision.
- 13 Oct 2026 (user): L is kept as built from expert data, with no cap. A candidate is flagged "host uncertain – review" when its primary host sits at the floor ε while a non-primary neighbour dominates (rule only, no new constant).
