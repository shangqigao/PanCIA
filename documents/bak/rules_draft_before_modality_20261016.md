# Image grounding rules — DRAFT v2: Stage G0 definitions and inference (16 Oct 2026; bracketed proposals G1–G8 confirmed by the user; main.tex updated to match)

This follows the form of the ledger v6 FINAL v3 rules (`Ledger_v6_rules_final_draft.md`). Grounding reads the ledger's outputs (claims, anchors, outlines, lesions, hosts, priors) and the image. It changes no ledger rule.

v1 (same day) is in `_to_delete/2026-10-16/Image_grounding_rules_draft_v1.md`.

**Tags:**
- **[12 Oct]** = established in the 12 Oct studies;
- **[16 Oct]** = a lesson from the bi-level and contour checks;
- **[user 16 Oct]** = decided in today's discussion;
- **[new]** = proposed here;
- **[decide]** = open; listed at the end.

**What changed from v1 (user, 16 Oct):**
1. A grounding case is the tuple c = (R, outlines_R, hosts_R).
2. The goal is the posterior p(T, N | c, I) = p(T | N, c, I) · p(N | c, I): infer the normal model first, then the tumour model given it, alternating.
3. Evidence uses only signs found to transfer. Its likelihoods come from expert data: p(e | N, h) from TotalSegmentator normals, p(e | T) from R2Seg GT tumours.
4. Trim and grow are set-based: a choice among extents built from the agents' outlines by set operations, with weights updated by EM from the ledger priors. No voxel is redrawn.

## Principles
- **G-P1. Only transferable evidence** [user 16 Oct].
  - Tumour and normal evidence are built only from signs that transferred from expert data to TCGA (G5).
  - Signs found not useful are excluded by rule (G5, excluded list).
  - Local contrast selected the candidate, so it only routes.
- **G-P2. Reliable likelihoods** [user 16 Oct].
  - p(e | N, h) comes from TotalSegmentator expert normal anatomy.
  - p(e | T) comes from R2Seg expert tumours.
  - Nothing comes from agent outputs or TCGA, and nothing is tuned on stage. Sex is never used.
- **G-P3. Set-based reasoning, not redrawing** [user 16 Oct; 16 Oct contour check].
  - Grounding reasons over sets the agents produced: anchors, outlines, hosts, and their set combinations.
  - Segmentation agents cannot do this; it is the multi-agent reasoning step.
  - No voxel is added or removed beyond the agents' outlines (as P4: claims are never cut).
- **G-P4. Pan-cancer and agent-symmetric.**
  - No rule names a cancer type or an agent.
  - Host-specific quantities come from that host's own normal tissue in the same scan and from expert normal anatomy of that organ.
- **G-P5. Unknown is not evidence.** A sign that does not apply is unknown and contributes a likelihood ratio of 1.
- **G-P6. Priors are starting weights, not truths.** The ledger priors (D8 host weights, D9 voxel priors) initialise the mixture weights, and inference updates them. Primary protection (P1) is kept: P(primary | R) = 1 when R touches the primary.

## Stage G0. Definitions

### G1. Image grounding
Image grounding is posterior inference over a grounding case:

p(T, N | c, I) = p(T | N, c, I) · p(N | c, I),

where
- **N** is the normal model: which host R belongs to, and that host's normal reference;
- **T** is the tumour model: whether R is tumour, and its extent;
- **I** is the image.

**Outputs, per case:**
- the existence posterior p_tumour;
- the host posterior P(h | c, I);
- the extent posterior over the candidate extents and its MAP extent;
- the evidence behind them.

**Out of scope:**
- finding lesions no agent proposed;
- choosing the primary site (P1).

### G2. Grounding case [user 16 Oct]
c = (R, O_R, H_R), where:
- **R** = an anchor (D4), the located, agreed core;
- **O_R** = R's outlines (D5), the tumour candidates. These are the kept claims overlapping R, with their D10, D10b and slab flags;
- **H_R** = R's host candidates: the hosts R is assigned to (D7), plus "none" for a no-host anchor. Each host carries its prior P(h | R) (D8).

Anchors of one lesion share host weights (G8). Lesion-level output is reporting only.

### G3. Candidate extents — the set-based trim and grow [new]
For case c, the tumour extent is one of a small set 𝒳(c), built only by set operations on the agents' masks:

| Extent | Definition | Role |
|---|---|---|
| X_R | R | maximal trim (agreed core) |
| X_k | each outline o_k ∈ O_R, unflagged or flagged | one agent's reading |
| X_∪ | union of the unflagged outlines | maximal grow |
| X_∪\\N | X_∪ minus every organ other than R's hosts (TS labels and H of non-assigned hosts) | grow without leaks (the D10b set operation) |

- Duplicates (Dice ≥ 0.95) are merged. 𝒳(c) has at most about six members.
- **Initial weight** w0(X) ∝ the D9 summary voxel prior of X (0.9 / 0.75 / 0.5 per voxel). For example, an extent that is mostly agreement voxels starts high.

### G4. Normal model N [12 Oct reference; tightened, new]
For each h ∈ H_R:
- **Normal reference** [revised 16 Oct with main.tex]. N(h) = {v ∈ H(h) : q(v) < π_none = 0.1}, minus a 2 mm margin around the excluded voxels.
  - q(v) is the current voxel tumour probability: q(v) = max_a p(z_a = 1) · Σ_{X ∋ v} p(X | z_a = 1).
  - At the start, q is the voxel prior, so every kept claim is excluded.
  - Voxels return to N(h) only when the posterior makes them confidently normal, for example an anchor rejected with p_tumour < 0.1.
- **Depth and statistics.**
  - Depth is the distance below the surface of H(h) ∪ X.
  - Per depth bin (0, 1.5, 3, 5, 8, 12, 18, 25, 35, 50 mm, beyond): the median and robust SD of the 1 mm-smoothed image over N(h) (`grounding.depth_std`).
- **Valid** if N(h) has ≥ 200 voxels in the 50 mm window. Otherwise every host-frame sign is unknown, flagged `no_normal_reference`.
- **Host-frame Z.** Z = the depth-standardised deviation from N(h).

### G5. Evidence — transferable signs only [user 16 Oct; 12 Oct transfer tests]
For a pair (extent X, host h), the evidence is e(X, h) = (S0, S1, S2, S3, S4, S5). Each sign is three-valued: yes, no or unknown.

| Sign | Definition | Values | Applies when | Transfer evidence |
|---|---|---|---|---|
| S0 router | zh = median Z over X | deviating if \|zh\| ≥ 1.5 | N(h) valid | Router only, not evidence |
| S1 context (closedness) | open_ratio: volume of X's same-sign deviation connected to X within 30 mm, inside H(h) + 2 mm, ÷ vol(X) | closed ≤ 1.42 / open > 5.27 / between | S0 deviating, and ≥ 50% of X inside H(h) + 2 mm [new, decide G6] | AUC 0.15 (GT vs salient), 0.16–0.21 TCGA CT, 0.18–0.20 MR |
| S2 internal structure (patchy) | rel_tex = median local SD (2 mm) of the candidate-frame image over X's core | patchy ≤ 1.047 | vol(X) ≥ 3 ml | AUC 0.07 / 0.26 / 0.32 |
| S3 elongation | √(λmin/λmax) | elongated < 0.363 | always | AUC 0.89 / 0.79 / 0.65 |
| S4 shape | v6.2 fixed-sign score (elong, flat, tube, blobby), size-banded | organ-like / uncertain / tumour-like | always | Organ-like: 2.2% of GT; 83–93% correct on TCGA |
| S5 anatomy (organ overlap) | D10 against H(h): (a) X is the organ; (b) X contains the organ | yes / no | H(h) is a mask | Set-based (v6) |

- **Host dependence.** S0, S1 and S5 depend on h. S2, S3 and S4 are intrinsic.
- **Excluded by rule (found not useful):**
  - local contrast or intensity as evidence (selection effect);
  - Hessian/curvature signs (ridge_tub, smooth_ax, axis_coh, exits, peak);
  - rim–core;
  - voxel appearance densities and voxel EM (R2 EM 5 Oct, PU check, host mixture, bi-level);
  - kNN novelty;
  - abductive explanation signs;
  - MMD / matched-null tests;
  - continuity v6.3.
- **Organ volume e_vol** is included only if a transfer check passes [decide G5].

### G6. Evidence likelihoods from expert data [user 16 Oct]
**Revision (user, 16 Oct, after the step-1 tables):**
- **S4 (v6.2 shape score) is dropped.** On the rated TCGA candidates it added 1 veto beyond the tree on your labels (and that one was a tumour), and 4 on merged labels. Its expert-normal shapes reflect how we sampled the normals, not what agents select (the v6.1 root cause).
- **E = (tree outcome, S5)**: 6 cells. No-host cases use the tree without the context level (3 cells).
- **Organ classes are solid and hollow only** (lung and breast → solid; uterus and cervix → hollow).
- **Uterus fibroids are included** in p(E | T) as a host-frame dataset (wall ∪ cavity as host). The task is tumour segmentation.
- **Likelihoods are frequencies, per organ class and size band** (user, 16 Oct; bands < 3, 3–20, > 20 ml, as v6.2).
  - **Smoothing:** each class's relative frequencies are mixed with 10% of the stratum's pooled (T/N equal-weight) frequencies. A cell empty in both classes is therefore neutral (Λ = 1).
  - **Sparse cells:** the log Λ of a cell supported by n regions (T + N) is multiplied by n / (n + 10).
  - **Small candidates:** below 3 ml, evidence against is capped at Λ ≥ 1/3 (the 3:1 convention, not fitted). Expert small tumours are organ-like in 7%; on agent candidates, small tumours look "open" far more often.
  - **Fallback for strata with < 30 regions:** the same band over all classes, then the class over all bands.
- **Agent-like expert data** (user, 16 Oct):
  - Tumour outline variants added: cut2d (3 slices), leak_host (+ host within 10 mm), leak_nb (+ labelled neighbour within 15 mm).
  - Vessel and bowel normals are tube segments (20–40 mm), replacing ball crops.
**Hierarchical summary.** The hierarchy of the 12 Oct tree is kept as the way the signs are combined. Each (X, h) gets one evidence state:

E = (tree outcome ∈ {tumour-like, organ-like, uncertain}, S4 class, S5 yes/no).

- **Tree outcome** [12 Oct, with fix C]: Level 0 S0 routes; Level 1 S1; Level 2 S2; Level 3 S3.
  - **tumour-like** = (closed or patchy) and not open and not elongated.
  - **organ-like** = (open or elongated) and neither closed nor patchy.
  - **Fix C:** if Level 1 was skipped and elongation is the only organ-like sign, the outcome is uncertain.
  - **uncertain** = everything else.

**Likelihoods.** These are the frequencies of E (at most 18 cells, with a Dirichlet/Laplace prior):
- **p(E | T):** R2Seg GT tumours, main components and their perturbed variants (partial, dilated). The variants are included so the model covers agent-quality outlines.
  - **Host-frame part (user, 16 Oct):** only from datasets with expert organ labels (kidney, liver, pancreas, NSCLC lung, cervix/uterus, prostate zones).
  - **Intrinsic part (S2–S4):** from all R2Seg datasets.
  - **Hosts without an expert-labelled tumour dataset** (for example bladder, colon, breast) use the host-frame table pooled by organ class.
  - TotalSegmentator is not run on R2Seg.
- **p(E | N, h):** TotalSegmentator agent-like normal structures of h's organ class: salient intra-organ structures, vessel, edge, organ part, adjacent, bowel, fluid, and whole-organ claims [12 Oct sets].
  - If h has no TotalSegmentator normals: the organ-class pool (solid organ / hollow viscus / gland), else all organs [decide G7].
  - h = none uses the intrinsic part (S2–S4) only.

**Rules for the likelihood ratio:**
- Λ(E | h) = p(E | T) / p(E | N, h), capped at ±log 100 (as v6.2).
- **Support only inside the host** (user, 16 Oct, after the step-1 expert tables). Λ > 1 counts only when ≥ 50% of the extent lies inside the hole-filled host dilated by 2 mm (the closedness applicability test). Otherwise Λ is capped at 1, so the veto side is unchanged.
  - Held out: false support among normals falls from 18% to 6.2%; host-frame tumours are still supported in 55%.
  - Reason: patchiness gave false support to structures outside the organ (bowel 43%, adjacent organ 42%).
- **Applicability is measured against the organ including a tumour of that organ** (clarified 16 Oct). Expert labels exclude the tumour; organ agents mostly include it.
- Unknown signs are marginalised: their cell is the sum over the unknown sign's values in both tables.
- Held-out estimates (leave one R2Seg dataset out, 2-fold TotalSegmentator) are used for the checks. The final tables use all expert data.

### G7. Inference — normal first, then tumour, alternating [user 16 Oct; new]
**Latent variables per case:**
- h ∈ H_R;
- z ∈ {tumour, normal structure};
- X ∈ 𝒳(c), defined when z = tumour.

**Joint model:**
- P(h, z = 1, X | c, I) ∝ P(h | ·) · π(R, h) · w(X) · p(E(X, h) | T)
- P(h, z = 0 | c, I) ∝ P(h | ·) · (1 − π(R, h)) · p(E(R, h) | N, h)

where:
- π(R, h) is the ledger prior of R under h (D9 summary prior);
- w(X) are the extent weights;
- P(h | ·) are the host weights.

**Algorithm (EM over shared weights):**
1. **Normal step.** For each host h, build N(h) (G4) and compute E(X, h) for every X ∈ 𝒳(c) and for R.
2. **E-step.** For each case, compute responsibilities over (h, z, X) from the joint model above.
3. **M-step.** Update the weights from the responsibilities:
   - **extent weights:** w(X) ← (posterior mass of X + α·w0(X)) / (1 + α), with α = 1, so the prior counts as one pseudo-case;
   - **host weights, shared by the anchors of a lesion:** P(h | lesion) ← (Σ_anchors responsibility of h, volume-weighted + α·P0(h)) / (1 + α);
   - **primary protection:** P(primary) = 1 is held fixed.
4. **Update the normal model.** Remove the MAP tumour extents from N(h) (G4) and recompute the host-frame signs.
5. Repeat steps 2–4 until the weights and MAP extents stop changing, at most 10 iterations. Changes are damped by ½, as in R2 EM, to avoid oscillation.

**Why EM here.** The unknowns are the discrete weights over sets (extents, hosts), not appearance densities. The likelihood tables are fixed from expert data, so the iteration cannot drift into normal tissue (the R2 EM and host-mixture failure). The coupling that needs iteration is real:
- the normal reference depends on the chosen extents;
- anchors of a lesion share a host.

### G8. Decisions from the posterior
- **Existence.** p_tumour(R) = Σ_{h,X} P(h, z = 1, X | c, I).
  - Admitted if ≥ 0.5 (ledger T8 threshold).
  - **Recovery needs evidence** [new, 16 Oct; to confirm]. An anchor that the set rules soft-rejected under every host is admitted only if Σ_h P(h | a) · Λ(X*, h) > 1. A kept prior alone (for example a 0.75 slab soft rejection) does not admit it.
  - Dominant-primary exemption (confirmed G8).
- **Host.** P(h | c, I) = Σ_z,X P(h, z, X | ·).
  - Flag `host_changed` if the MAP host differs from the ledger's.
  - Primary protection holds.
- **Extent (trim/grow).** X* = argmax_X P(X | z = 1, ·). It is reported with its source (anchor, outline id, union, leak-free union). Outline volume and anchor volume are still reported.
- **Flags:** `no_normal_reference`, `context_not_applicable`, `iso` (S0 not deviating), `host_changed`, `extent_changed`, `not_converged`.

### Units used by each grounding question
| Question | Unit |
|---|---|
| Normal reference N(h), Z | host, per scan |
| Evidence state E | (extent, host) |
| Existence, host, extent posteriors | case c = (R, O_R, H_R) |
| Shared host weights | lesion (anchors of one lesion) |
| Reporting | lesion |

### Constants
| Name | Value | Source |
|---|---|---|
| Image smoothing | 1 mm Gaussian, in-plane when slice/pixel ≥ 1.5 | 12 Oct |
| Depth bins | 0, 1.5, 3, 5, 8, 12, 18, 25, 35, 50 mm, beyond | 12 Oct |
| Minimum normal reference | 200 voxels, 50 mm window | 12 Oct |
| Reference exclusion margin | 2 mm around tumour extents | new |
| Router | \|zh\| ≥ 1.5 | 12 Oct |
| Closedness radius / zone | 30 mm / H(h) + 2 mm | 12 Oct |
| Closed / open | ≤ 1.42 / > 5.27 | 12 Oct |
| Closedness applicability | ≥ 50% of X inside H(h) + 2 mm | new [decide G6] |
| Patchy | rel_tex ≤ 1.047, X ≥ 3 ml | 12 Oct |
| Elongated | < 0.363 | 12 Oct |
| v6.2 classes | ±log 3 | 12 Oct |
| Λ cap | ±log 100 | as v6.2 |
| Extent duplicates | Dice ≥ 0.95 merged | new |
| EM prior strength α | 1 (prior = one pseudo-case) | new [decide G4] |
| EM damping / maximum iterations | ½ / 10 | as R2 EM 5 Oct |
| Admission | p_tumour ≥ 0.5 | ledger T8 |

## Open decisions
- **G1. Extent set.** Is 𝒳(c) = {R, each outline, union, leak-free union} the right trim/grow menu? Should pairwise intersections of outlines be added (e.g. the part of two outlines both agents agree on, beyond R)?
- **G2. Evidence state.** Use the tree outcome × v6.2 class × organ overlap (18 cells, proposed), or the raw three-valued signs with naive independence (more cells, a stronger assumption)?
- **G3. Tumour table.** Include perturbed R2Seg variants in p(E | T)? Proposed yes: outlines from agents are not expert-clean.
- **G4. EM prior strength α = 1.** Does the ledger prior count as one pseudo-case? A larger α trusts the ledger more.
- **G5. Organ volume.** Include e_vol as a sign only after a transfer check on TCGA rated candidates? Proposed: reporting only until then.
- **G6. Closedness applicability:** ≥ 50% of X inside H(h) + 2 mm (a priori).
- **G7. Hosts without TotalSegmentator normals** (ovary, uterus, cervix, breast): which pooled normal table? TotalSegmentator has no ovary or breast.
- **G8. Dominant-primary exemption:** the largest admitted primary anchor is never demoted (P1).

## Checks before use (expert data first, then TCGA without tuning)
1. **Expert likelihood tables.** p(E | T) and p(E | N, h), held out (leave one R2Seg dataset out; 2-fold TotalSegmentator). Report:
   - Λ per cell;
   - the share of GT tumours with Λ < 1/3;
   - the share of normals with Λ > 3.
2. **Extent choice, in domain.** R2Seg tumours with simulated agent outline sets: core, dilated into organ, leaking into a neighbouring organ, partial. Does X* match the expert best (Dice of X* vs the best available extent)?
3. **Host choice, in domain.** R2Seg scans with labelled neighbouring organs, using the GT tumour plus the true and adjacent organs as H_R. Is the true host chosen?
4. **Freeze** the tables, constants and decisions (`FROZEN_GROUNDING_G0_*.txt`).
5. **TCGA.** The 420 rated candidates and the v6s pilot. Success criteria:
   - veto precision ≥ 80%;
   - user-labelled tumours vetoed ≤ 10%;
   - confident error ≤ 15% for two-sided calls;
   - primaries kept;
   - review panels of the changed extents and changed hosts.
