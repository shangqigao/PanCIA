# Image grounding on ledger v6: plan (16 Oct 2026; revised overnight to v2, see §10)

Author: Shangqi Gao (Crispin Lab) with Claude.

**Builds on:**
- ledger v6 rules FINAL v3: `documents/plan/Ledger_v6_rules_final_draft.md`; implementation `knowledge_base/pancia_kb/ledger_v6s.py`;
- R2 v6 design §6 (coupled contour evolution);
- the image-evidence studies of 10–12 Oct: geometry, normative, closedness transfer, abduction, v6.2, v6.3, GT rejection check.

**Standing rules:**
- Public data only.
- Priors and calibrations come from expert annotations only (R2Seg, the TotalSegmentator datasets), never from agent outputs or TCGA.
- No tuning on stage; sex is never used.
- Pan-cancer rules.
- Agents are pluggable.
- BP/VT are not run on R2Seg.

## 1. Goal

For every ledger lesion, the image either corrects its delineation or vetoes it; it never invents tumour.
- **Delineation:** move the agents' tumour outlines **and** the anatomy agent's host-organ mask together until they agree with the image (a bi-level set).
- **Existence:** veto-only. Evidence that the evolved tumour is organ-like, or that it vanishes, lowers the ledger prior. Missing tumour evidence never counts against.
- **Host:** a lesion with several possible hosts is evolved under each host. A host under which the tumour is vetoed loses weight.

## 2. What the earlier work established (inputs to this design)

**Effective signs.** These are coded as research scripts; §6 S1 consolidates them.

| Sign | Measures | Evidence | Code |
|---|---|---|---|
| Normative intensity Z | voxel deviation from the host's normal tissue at the same depth below the organ surface | in-domain detection 59–100% of GT, 2–4% false alarms | `norm_feats3.depth_std`, `norm_geo.run_regions` |
| Closedness (30 mm) | does the candidate's own deviation continue inside the host (open = normal structure)? | AUC 0.13–0.15 expert, 0.16–0.21 TCGA CT, 0.18–0.20 MR (low = tumour); robust to resampling | `norm_geo30.closed30` |
| Patchy heterogeneity `rel_tex` (≥ 3 ml) | structured internal variation in the candidate's frame | 0.07 expert, 0.26 TCGA CT, 0.32 MR | `norm_geo.geo` |
| Elongation | tube- or strip-like shape | 0.89 expert, 0.79 TCGA CT, 0.65 MR | `norm_geo.geo`, `pancia_kb/geometry.py` |
| Organ volume (`e_vol`) | organ too large or small for a normal organ, explained by the tumour | false support 2.7%; useful when the tumour is ≥ ~30% of the organ | `pancia_kb/r2_v6.py` |
| Organ overlap (D10 tests a, b) | the lesion *is* or *contains* the organ | ledger v6 | `ledger_v6s.py` |

Frozen thresholds (`ledger/v6/FROZEN_GEO_20261012.txt`, with fix C):
- open > 5.27, closed ≤ 1.42, computed only when |median Z| ≥ 1.5;
- `rel_tex` ≤ 1.047;
- elongation < 0.363.

**Lessons:**
- Comparisons with agent-selected surroundings do not decide existence.
- Identity (normal structure or not) decides, and cannot be fully computed.
- Image evidence is reliable as a veto: only 2.2% of GT tumours are rejected, and 83–93% of organ-like calls are correct.
- Delineation is not existence.

## 3. Units and runs

- **Run unit:** one (lesion, host, agent) triple.
  - **Lesion and host:** from the ledger (D7). An anchor in several hosts' lesions is evolved **independently under each host**, because each host has its own normal-tissue model and its own prior. The prior does not say which host is right.
  - **Agent:** one evolution per agent. The tumour is initialised with **that agent's outlines in the lesion**, merged across its prompts (VT cancer and host prompts together; prompts of one agent are not independent).
- **No fixed anchor.** Over-segmentation (tumour plus normal organ, even where both agents agree) must be correctable, so any part of the initial contour, the anchor included, may vanish when the image says so.
- **Cost:** about 3,200 lesions × ≤ 2 agents in the pilot, each evolution confined to a local domain (below).

## 4. Bi-level set

### 4.1 Two contours
- **Tumour φ_T:** initialised from the agent's outlines in the lesion.
- **Host φ_H:** initialised from the ledger host organ H (TS ∪ VT organ mask; the KB box is used only as a domain limit, not as H).
- **Local domain Ω** (cost and safety):
  - tumour: the initial tumour dilated by **10 mm** (user, 16 Oct; revise if too small);
  - host: the host inside Ω ∪ a 30 mm reference shell around Ω for the normal-tissue statistics.
  - Voxels outside Ω keep their labels.
- Every voxel in Ω is labelled tumour, host or other.

### 4.2 Energy
E = E_data + E_disagree + E_geom + E_vol + E_smooth, minimised by alternating updates of φ_T and φ_H until both stop changing (the balance).

1. **E_data (per voxel).**
   - **Normative term.** Z is computed against the current host region minus the current tumour, with depth measured from the current host surface; it is re-estimated as φ_H moves.
     - Tumour voxels are rewarded for |Z| ≥ 1.5 in **either** direction, so necrotic and enhancing parts both count.
     - Host voxels are rewarded for |Z| < 1.
   - **Local region term** (localised Chan–Vese) at each boundary point: in a ball of about 5–7 mm, the voxel is compared with the local inside and outside means. This handles brightness drifts (MR coil field, contrast phase).
   - **Holes** enclosed by the tumour are filled after each update.
2. **E_disagree (user, 16 Oct).** Tumour and host must disagree as much as possible on the robust signs:
   - **Overlap:** T ∩ H is penalised. A voxel belongs to one of them.
   - **Complete overlap** — the tumour *is* the organ, or *contains* the organ (D10 tests a/b) — is strongly penalised.
   - **Normative separation:** median |Z| inside T versus inside H; a small gap is penalised.
   - **Closedness of T relative to H:** a deviation that stays open, continuing along the host, is penalised for T.
   - Patchy heterogeneity is *not* used as a penalty, because homogeneous tumours exist; it is reported only.
3. **E_geom (tumour geometry).** Elongation below 0.363, or a sheet-like shape, is penalised (the organ-like tumour penalty).
4. **E_vol (host anatomy).** The host volume after removing the tumour is scored under the KB normal-organ volume prior. An implausible organ, too large because it swallowed the tumour or too small because the tumour swallowed it, is penalised. This also corrects anatomy-agent errors such as a TS organ containing the tumour (median 78% in TCGA) or a malformed organ.
5. **E_smooth.** Curvature regularisation for both contours.

The weights are set **only** on perturbed expert data (§5 E1) and frozen before any TCGA run.

### 4.3 Missing host reference
- **Too little normal host tissue** (about < 200 voxels, e.g. a whole-organ claim): Z is undefined. Only the local term, E_geom and E_vol act, and the run is flagged `no_normal_reference`.
- **Hosts without an organ mask** (mask-less) are not evolved.

### 4.4 After convergence
1. **Agreement across agents.** If the BP-initialised and VT-initialised tumours converge (Dice ≥ 0.5; user, 16 Oct: two radiologists may agree only about this much), the alternative outlines are settled. Otherwise both are kept as alternatives and the lesion is flagged `agents_disagree`.
2. **Regrouping.** Per host, the evolved tumours are regrouped with the ledger rule (touch, τ), so lesions can merge, and disconnected pieces become separate lesions (split).
3. **Corrected host.** The evolved host H* is stored; it replaces the agent organ mask for downstream use (graph, volumes).

## 5. Existence and host evidence (veto-only)

Computed on the **evolved** tumour T* and host H*, per (lesion, host).

| Outcome | Evidence |
|---|---|
| T* vanishes, or shrinks below 0.5 ml with no agreement | organ-like |
| T* still organ-like: D10 a/b against H*, or E_disagree at its maximum | organ-like |
| Frozen tree rule says organ-like (open or elongated, neither closed nor heterogeneous; fix C) | organ-like |
| Otherwise (tumour-like or uncertain) | neutral |

- **Existence:** an organ-like outcome multiplies the odds of p_tumour by 1/3 (LR 1/3). Neutral leaves p_tumour unchanged. Admission stays at posterior ≥ 0.5. Soft candidates can only go down.
- **Host:** for a lesion evolved under several hosts, a host with an organ-like outcome has its P(h) multiplied by 1/3 and the hosts are renormalised. This is the first image-informed host evidence. It stays veto-only: no host is promoted by image evidence.
- **Agent disagreement:** reported; it does not change the prior.
- **Outputs per anchor:**
  - posterior p_tumour and status;
  - evolved tumour mask and volume;
  - corrected host mask;
  - flags: `vanished`, `agents_disagree`, `no_normal_reference`, `host_vetoed`;
  - three volumes: anchor, outline, evolved.

## 6. Implementation steps

1. **S1, consolidate the signs.** Create a new module `pancia_kb/grounding.py` with normative Z, closedness, `rel_tex`, elongation, `e_vol`, the D10 tests and the frozen tree rule. Regression check: it must reproduce the stored `geo30` values on the expert sets (Spearman ≥ 0.99 and identical rule calls) before anything new runs.
2. **S2, bi-level set.** Create a new module `pancia_kb/bilevel.py` and smoke-test it on a few R2Seg cases.
3. **S3, freeze.** Write `ledger/v6/FROZEN_BILEVEL_*.txt`: energy terms, weights, step sizes, iteration limits, convergence criterion, thresholds.
4. **S4, in-domain evaluation** (E1, E2 below).
5. **S5, TCGA pilot** on the same 158 series (v6s ledger): E3 and review panels.

`ledger_v6s.py` and `r2_em.py` stay unchanged; grounding reads the ledger outputs.

## 7. Evaluation (frozen before running)

**E1, delineation, in domain (R2Seg, expert data only).** Tumours with same-scan organ labels (kidney, liver, pancreas, lung).
- **Simulated agent errors:**
  - tumour masks eroded or partial (30–70%), or dilated 3–5 mm into the organ;
  - host = expert organ ∪ tumour (TS-like), or organ minus a notch.
- **Measured:** Dice of T* and H* against the expert masks, before and after evolution.
- **Success:**
  - mean Dice gain > 0 on perturbed masks;
  - Dice change ≥ −0.02 on clean masks;
  - expert tumours vanishing ≤ 2%.

**E2, existence, in domain.**
- **Tumour side:** R2Seg tumours, leave-one-dataset-out. Success: veto rate ≤ 5%.
- **Normal side:** TotalSegmentator agent-like normal structures (salient structures, vessel, edge, organ part) used as initial tumour contours. Report the veto rate.

**E3, TCGA (out of domain, no tuning).**
- **Candidates:** the 420 rated candidates mapped to v6s anchors; the user's labels are primary.
- **Success:**
  - veto precision ≥ 80% (vetoed items are non-tumours);
  - user-labelled tumours vetoed ≤ 10%.
- **Also reported:** agent-convergence rate, host-veto cases, volume changes.
- **Review:** panels of T*, H* and the initial outlines.

## 8. Known limits

- Normal sub-structures that resemble tumour (intra-organ vessels, renal sinus, bowel loops) remain the main existence confounder. Many will stay neutral rather than be vetoed.
- MR evidence is weaker than CT. Hosts without an anatomy volume prior (breast, ovary, uterus, cervix) lack E_vol.
- Evolution is local (10 mm). A tumour missed far beyond its initial outline is not recovered.

## 9. Decisions (accepted autonomously on 16 Oct, as proposed; the user delegated decisions overnight)

1. **Local host evolution:** the host evolves only in the lesion's domain plus a 30 mm reference shell, not the whole organ (cost; anatomy errors matter near the tumour).
2. **Veto strength:** LR 1/3 for organ-like (the same 3:1 convention used before).
3. **Host veto:** applied per host, as above, and never promoting a host.

## 10. Revision 16 Oct overnight: grounding v2 (autonomous; the user delegated decisions)

**What happened to the bi-level set.** S1 passed: `pancia_kb/grounding.py` reproduces the stored `geo30` values exactly (16 regions, max |Δ zh_med| 4e-7, identical closedness). S2–S4 failed: five versions of `pancia_kb/bilevel.py` were run on R2Seg (synthetic smoke tests passed from v4 on), and none met E1.

| Version (E1, 11 R2Seg tumours, 4 organs) | Clean Dice change | Perturbed gain (dil_in / erode / agent) |
|---|---|---|
| v4b voxel front, robust-Gaussian evidence | −0.17 | −0.03 / −0.03 / −0.01 |
| v5 3 mm block lattice + voxel refinement | −0.23 | −0.08 / −0.07 / −0.02 |
| v5 blocks, δ = 2, no refinement | −0.07 to −0.09 | −0.00 to −0.02 / ±0.01 / ±0.01 |

**Why (ceiling check, 41 R2Seg tumours).** Even the best possible voxel classifier on the normative Z, given the true tumour ± 5 mm band, agrees with the experts only at Dice 0.72 (kidney), 0.77 (liver), 0.85 (lung) and 0.67 (pancreas). In lung, half of the expert rim voxels (median 46%) look like host tissue. Image-driven redrawing therefore cannot improve on an agent outline that is already at expert level, and it damages outlines on average. This is the same lesson as 12 Oct ("delineation is not existence"), now quantified.

**Decision (v2).**
1. **Delineation:** the agent outline (ledger anchor/outline) is kept; the image does not redraw it. The bi-level code stays in the KB, marked NOT ADOPTED.
2. **Host correction:** H* = H0 \ admitted tumour (`grounding.host_correct`). In E1 this alone was the only robust host gain (host_union +0.03 Dice).
3. **Existence (veto-only, LR 1/3):** veto = V1 or V2, frozen in `ledger/v6/FROZEN_GROUNDING_V2_20261016.txt` before TCGA was evaluated.
   - V1 is the v6.2 fixed-sign shape score being organ-like (expert-calibrated, 12 Oct).
   - V2 is strict closedness: open_ratio > 17.07, the 98th percentile over R2Seg expert tumours.
   - The 30 mm tree rule is not used for vetoes. On the outline it vetoed 7.8% of GT tumours and 22% of TCGA CT tumours.
4. **Per host:** V2 depends on the host frame, so it is applied per host (veto the host's P(h) by 1/3). V1 is host-independent.

**Results.**

| Set | Veto rate / precision | Criterion |
|---|---|---|
| R2Seg GT tumours, V2 | 1.5% (leave-one-dataset-out: kidney 0%, lung 1.7%, pancreas 0%, **liver 8.5%**) | ≤ 5%: met overall, liver fails |
| R2Seg GT tumours, V1 (12 Oct) | 2.2% | met |
| TotalSegmentator normals, V2 | 8.7% (salient 27%, vessel 8%, others ≤ 1.4%) | report |
| TCGA user labels (164 rated, 81 tumours), V1 or V2 | 21 vetoed, precision 86%, tumours vetoed 3.7%, non-tumours vetoed 22% (V1 alone: 12%) | met |
| TCGA merged labels (387, 161 tumours), V1 or V2 | 76 vetoed, precision 90%, tumours vetoed 5.0%, non-tumours vetoed 30% | met |
| TCGA CT, merged | tumours vetoed 10.1% | borderline |

**Known limits.** V2 vetoes some small lung nodules beside vessels (4 merged-label tumours) and is weaker for liver (leave-one-dataset-out 8.5%). MR has few closedness vetoes. Most non-tumours stay neutral (≈ 70–78%). The veto removes part of the false positives; it is not a classifier.

**Pilot (done overnight; details in `claude/image_grounding_overnight_2026-10-16.md`).**
- Of 1,586 v6s anchors with prior ≥ 0.5, 334 are vetoed (21%).
- Without an exemption, 97 of 575 admitted anchors are demoted, and 4 series lose their only admitted primary.
- Proposed: exempt the dominant (largest admitted) primary anchor of each series. With it, 89 anchors are demoted and all 147 series keep their primary.

**Open decisions for the user:**
- adopt v2;
- keep V2 after reviewing the montages;
- adopt the dominant-primary exemption.
