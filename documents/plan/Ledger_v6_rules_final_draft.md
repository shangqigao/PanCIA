# Ledger v6 rules — FINAL v3 (definitions and T1–T8 confirmed by the user, 15 Oct 2026; full consistency check and fixes 15 Oct, including an independent review; decisions 1–7 accepted; D10b outline leak and two-volume reporting added after the first v6s pilot)

Implemented in `knowledge_base/pancia_kb/ledger_v6s.py`; pilot: `clinical/PanCIA_outputs/scripts/v6s_check.py`.

Set-based only: masks, the KB and agent agreement. No image is read; image grounding comes afterwards.
- **[v3]** = unchanged from ledger v3.
- **[new]** = changed in v6.
- **[D]** = follows from the Stage 0 definitions.
- **[fix 15 Oct]** = a correction found in the full check (consistency only, no new logic).
- **[decide]** = still open; listed under Open decisions.

Previous versions:
- 13 Oct FINAL: `_to_delete/2026-10-15/Ledger_v6_rules_final_draft_13Oct.md`
- FINAL v2, with the 15 Oct organ-overlap revision: `_to_delete/2026-10-15/Ledger_v6_rules_final_v2_15Oct.md`

## Principles
- **P1. Protect the primary.** The scans are pre-surgical and the cancer is confirmed, so a primary tumour exists unless the primary organ is out of the field of view. Lesions touching the primary organ are decided first, as primary. Only then are the others considered as invasion or distant disease.
- **P2. Agent masks are noisy observations.** Near the primary, shape thresholds never hard-reject. The only drops there are the claim-level ones: the size floors (0.5 ml, crumb groups 0.2 ml), the prompt gate and the slab rule. Away from the primary, the host is a probability. [fix 15 Oct: the old wording, "thresholds never decide alone", overstated this.]
- **P2b (user, 15 Oct). Tolerate agent imperfection.**
  - No organ agent's masks are excluded: H is the union of all organ agents' masks (today TotalSegmentator ∪ VoxTell) everywhere.
  - A potential primary tumour that overlaps the primary organ or a surrounding organ in an organ-like way is **softly rejected**: it is flagged, keeps its prior, and goes to image grounding.
- **P4 (user, 15 Oct). Claims are never cut.**
  - An agent's connected component is the smallest unit. It is kept or dropped whole, and its geometry is never altered.
  - Claims are grouped; they are never split.
  - Anchors are derived references, not claims.
- **P5 (user, 15 Oct). Agents are treated equally.**
  - No agent is the "core" that others attach to.
  - Differences between agents are attributes of a claim (prompt type, flat or not), never structural roles.
  - Agreement counts agents, not prompts.
  - No rule names an agent.
- **P6 (user, 15 Oct). Overlap is segmentation uncertainty.**
  - Different agents' outlines of one tumour may overlap; they are alternative outlines, kept side by side until image grounding.
  - An anchor touching several non-primary hosts is host uncertainty, kept as competing hypotheses.

## Order of computation [fix 15 Oct: stated explicitly]
1. Hosts, organ references and envelopes (D2).
2. Claims, crumb groups and claim-level drops (D3).
3. Agreement set and anchors (D4), over the kept claims.
4. Outlines (D5).
5. Touch (D6) and host probabilities (D8).
6. Organ-shape exclusion away from the primary [decide 5], then renormalise P(h | a).
7. Lesions (D7) and organ-overlap flags (D10).
8. Tumour rules: pass 1 (T1–T4), pass 2 (T6–T7), then T8.

## Stage 0. Definitions [D]

### D1. Agents, prompts and masks
- **BiomedParse (BP):** one cancer-specific tumour prompt, e.g. "kidney tumor" for KIRC.
- **VoxTell (VT):**
  - the same cancer-specific tumour prompt (VT's first prompt);
  - one host prompt per spread-set host in view (VT's second and later prompts);
  - organ prompts. A VT organ mask is the main phrase, or the voxel majority of the main phrase and its aliases when aliases were prompted [decide 4].
- **TotalSegmentator (TS):** organ labels.
- **Pluggable agents.** A new segmentation agent enters as one more tumour agent (cancer prompt, and host prompts if it supports them) or one more organ agent.

### D2. Organ, host and envelope
- **Organ reference.** H(o) = the union of all organ agents' masks for o (today the TS mask ∪ the VT organ mask).
- **KB prior box.** Used as H only for a primary organ with no organ mask. It is flagged and is never used in the organ-overlap test (D10). Its envelope, touch and the rules that depend on them are set by [decide 2].
- **Host.** A spread-set organ in the field of view, with its role and KB weight w_h: primary 1.0, local-invasion neighbour 0.3, cancer-specific distant site 0.15.
  - "Primary out of view" comes from the planner.
  - **Dual role.** An organ that is both a local-invasion neighbour and a distant site of the cancer (KB `also_distant`) has the local role and w = 0.3. T6 sends its non-contiguous lesions to T7, which judges the same lesion under the distant reason. [fix 15 Oct]
- **Mask-less host.** A secondary host with neither mask. It has no envelope, so no anchor can touch it. It is listed for the scan, and any anchor that touches no host is flagged `host_without_mask_possible` when the scan has a mask-less host. Prior π = 0; it is never chosen. [fix 15 Oct: "possible host" was undefined once touch became the rule.]
- **Envelope.** E(h) = H(h) dilated by 10 mm, for every host with a mask.

### D3. Claim — the smallest unit
- **What it is.** A connected component (3-D, 26-connected) of one agent's tumour mask from one prompt, of at least 0.5 ml.
- **Crumbs.** Components under 0.5 ml that touch the primary envelope are grouped per agent and prompt (≤ 5 mm apart). A group of at least 0.2 ml is one claim made of several components; it is the only kind of claim that is not a single component. [C0. fix 15 Oct: made agent-symmetric (v3 applied it to VT only) — decide 6.]
- **Attributes:** agent; prompt type (`cancer`, or `host:h`); flat or not; volume.
- **Never cut.** A claim is dropped whole only by these two rules, applied in this order:
  1. **the prompt gate:** host-prompt claims only; dropped if less than 50% lies inside E(h) of its own prompt host [C2];
  2. **the slab rule:** a claim is *flat* if its through-plane extent (along the acquisition axis) is ≤ 0.25 of its in-plane diagonal and its median per-slice coverage of the **host organ** it overlaps most is ≥ 40%. [audit 15 Oct: "organ" means host organ, as in v3. Testing against any TS organ (a vertebra, a rib, a muscle) dropped 260 more claims in the pilot and lost two primaries.] A flat claim with no kept, non-flat claim of a different agent overlapping it by ≥ 0.5 ml is:
     - **soft-rejected** if it touches E(primary) (primary in view): it is kept as a claim, flagged `slab` like a D10 outline flag, keeps its prior, and goes to image grounding. A lesion whose outlines are all flagged this way is soft-rejected as `primary_slab` (user, 15 Oct, decision 8);
     - **dropped** otherwise.
   The rule as originally written: [C3, agent-symmetric; fix 15 Oct: "2-D" replaced by v3's actual definition].
- [fix 15 Oct: the organ-shape rule away from the primary is no longer a claim drop. It acts on anchor-to-host assignment (step 6, decide 5) and never removes a claim.]
- **Raw output.** Each agent's tumour masks from all prompts, before any drop, at any size. Used only as evidence for *trace* and *silent* (T terms, D9).

### D4. Agreement and anchors — the unit of location
- **Agent mask.** M_k = the union of agent k's kept claims, all prompts.
- **Agreement set.** A = voxels in at least two different agents' M_k. A VT cancer-prompt claim overlapping a VT host-prompt claim is same-agent confirmation, recorded but not agreement.
- **Two-agent anchor.** A connected component of A of at least 0.5 ml.
- **Single-agent anchor.** A connected component of some M_k that does not overlap any two-agent anchor. Crumb-group claims count as connected for this purpose, so one crumb group gives at most one anchor.
  - [fix 15 Oct: defined on M_k, so that two overlapping prompts of the same agent form one anchor, not two overlapping anchors.]
  - ["does not overlap" rather than "contains no": with three or more agents, a two-agent anchor can partly overlap a third agent's component.]
- **Disjointness.**
  - Two-agent anchors never overlap each other, or single-agent anchors.
  - Single-agent anchors of the same agent never overlap.
  - Single-agent anchors of different agents overlap only inside agreement components under 0.5 ml.
- **What anchors are used for.** Anchors are what host reasoning measures (D6, D8) and what lesions group (D7).

### D5. Outlines — the unit of geometry
- **Definition.** Every kept claim is an outline of each anchor it overlaps. A claim overlapping several anchors is an outline of each.
- **Geometry.** Outlines keep the agent's geometry unchanged.
- **Uses.** Outlines are what the shape tests (D10), the per-voxel priors (D9) and image grounding (trim and grow) use.
- **Overlap.** Outlines may overlap (P6).

### D6. Touch — the single spatial relation
- **Definition.** Anchor a touches host h if a ∩ E(h) ≠ ∅, that is, a lies within 10 mm of H(h).
- **One relation, three uses:**
  - lesion membership (D7);
  - the set of possible hosts (D8);
  - primary protection (D8).
- **Where "inside" percentages remain:** the prompt gate and slab coverage (D3), the organ-shape tests (D10), the band share (D8) and the KB-box rule [decide 2].

### D7. Lesion — the unit for grouping and counting
- **Grouping.** For each host h, the anchors assigned to h are grouped. Two anchors join if they lie within τ of each other; chaining applies (if A joins B and B joins C, all three are one lesion).
  - τ = max(5 mm, slice spacing) [decide 3].
- **A lesion** is (h, its group of anchors). Its outlines are all outlines of its anchors.
- **Assignment of an anchor to hosts.**
  - **It touches the primary** (in view): it is assigned to the primary only. Other hosts it touches are *involved organs*, not competing lesions. [fix 15 Oct: D7 and D8 disagreed on this.]
  - **Otherwise:** it is assigned to every masked host it touches, except hosts it is excluded from under [decide 5]. It belongs to a lesion of each. These are competing host hypotheses (P6).
  - **It touches no host:** it is a *no-host lesion* on its own.
- **Why anchors, not outlines.** A large leaky outline cannot bridge separate tumours: b ∩ v1 and b ∩ v2 are separate anchors and join only if they are within τ of each other.

### D8. Host probability (replaces the likelihood table L; user, 15 Oct)
- **Band share.** s_h(a) = the share of anchor a's two-sided 3 mm boundary band inside H(h). The band is 3 mm outside a (other anchors excluded) plus a's 3 mm inner rim.
- **Primary protection.** If the primary is in view and a touches it, P(primary | a) = 1.
  - Paired organ with unresolved side: the side a touches; if both, the side with the larger s; if s is equal (e.g. both 0), the nearer side; if still tied, the organ without a side, flagged `side_unresolved`. The flag is not raised when both sides are KB boxes, which overlap by design (15 Oct).
  - Involved organs are recorded with their s.
- **Otherwise,** over the hosts a is assigned to:
  - P(h | a) ∝ w_h · s_h(a);
  - if s_h = 0 for all of them, P(h | a) ∝ w_h.
- **Consistency.** P(h | a) sums to 1 over the hosts a is assigned to.
- **Lesion level.** A lesion's weight under its host (the volume-weighted mean of its anchors' P(h | a)) is for reporting only; decisions use anchors (T8).
- **No-host lesion.** Soft rejection, flagged `no_host` (plus `host_without_mask_possible` if applicable), with prior π_none [decide 1].
- **Flags.**
  - `multi_host`: the top two hosts both have P ≥ 0.1.
  - `primary_involves_other_organs`: the anchor has involved organs.
  - `host_uncertain_review`: the anchor touches the primary only through the outer band (s_primary < 5%) while one neighbour holds ≥ 30% of its band. [fix 15 Oct: the old thresholds were defined through L.]

### D9. Priors per voxel — replace the support classes
Within each outline, every voxel gets a prior:

- **0.9:** inside a two-agent anchor.
- **0.75 (the silent case, primary only):** all three conditions hold (user, 15 Oct):
  - the voxel is outside every two-agent anchor;
  - it is inside E(primary), in an outline of a primary lesion;
  - *every* other tumour agent is silent, i.e. its raw output from any prompt has no voxel in E(primary).

  [fix 15 Oct: restricted to E(primary), so that a long outline running into a neighbour does not carry 0.75 there.]
  - **Why.** The primary is known to exist (P1). An agent that is silent across the whole primary organ has most likely missed the tumour, so its silence is not evidence against the one claim we have.
  - **Agent-symmetric form of v3's R1 / `BP_vtsilent`.** It applies in both directions. v3 gave VT-only 0.5.
  - **Not used for other hosts.** For local and distant hosts, no tumour is known to exist, so silence carries no information there.
- **0.5:** outside every two-agent anchor, otherwise.

[fix 15 Oct: "claimed by one agent only" was wrong for voxels that two agents claim inside agreement components under 0.5 ml. These now count as 0.5 too.]

- These three values are fixed by rule to preserve the order: agreement > silent-in-primary > single. They are not fitted.
- **Summary prior of an anchor:** the volume-weighted mean voxel prior over the union of the anchor's unflagged outlines, or of all its outlines if every one is flagged. It is used in T8 and for reporting. T1 orders by support class, not by summary prior, so one large leaky outline cannot push a well-agreed lesion down the order.
- **Soft rejections** keep their priors. **Hard rejections** (dropped claims, mask-less hosts) are 0.

### D10. Organ-overlap test (V2, 15 Oct; per outline and per anchor)
- **Scope.** Run on each anchor and each outline of a primary lesion.
  - Anchors are tested against the primary organ and every involved organ.
  - Outlines are tested against every organ they overlap, i.e. every organ with a mask in the scan, in or outside the spread set. A host organ is tested once, on H (the union of organ masks); other organs on their TS label. [audit 15 Oct] [fix 15 Oct: an outline of a kidney tumour that swallows the whole liver, where the anchor is more than 10 mm from the liver, was never tested before. This is the KIRC case.]
  - H = the union of organ masks. The KB box is never tested.
- **Flag `organ_overlap`** if either holds:
  - **(a) is the organ:** all three of
    - volume ratio (its volume / the organ's volume) ≥ 0.6;
    - median per-slice Dice ≥ 0.6, over its acquisition-axis slices that contain the organ;
    - ≥ 80% of it inside H;
  - **(b) contains the organ:** ≥ 80% of H inside it, and the organ makes up ≥ 50% of its volume.
- **Outcome.** A flagged outline or anchor is softly rejected: its prior is kept, and image grounding decides and trims. A leaky outline (e.g. tumour plus whole liver) is caught on its own shape, without affecting the anchor's host.
- **Away from the primary:** [decide 5].

### D10b. Outline leak (user, 15 Oct; after the first v6s pilot)
- **Test.** For each anchor–outline pair, take the part of the outline beyond the anchor that lies inside an organ (any TS label or host H) other than the anchor's assigned hosts. If it is ≥ 50% of the outline's volume, the outline is soft-rejected for that anchor (`leak`).
  - A KB-box host counts as its box region.
  - Involved organs are not exempt.
- **Effect.** Same as a D10-flagged outline: the prior is kept, the outline is excluded from the lesion extent, the summary prior and the admitted outlines, and image grounding decides.
- **A single-agent anchor's own claim is never affected,** because that claim is the anchor. Another agent's claim that overlaps it by less than 0.5 ml is still an outline of it and can leak. [audit 15 Oct: the earlier wording said "never affected".]
- **Why.** Leaky outlines that cover only part of a neighbouring organ pass both organ-shape tests. The KIRC case: a VT liver-prompt claim of 1,471 ml covers 62% of the liver, under the 80% needed for either test.

### D11. Tumour
- **What it is.** A judgement on a lesion under its host: status (admitted / soft rejection / hard rejection), reason and prior. It is not a geometry.
- **Admitted once.** A tumour is admitted once and carries all its outlines with it.
  - Within one host, lesions never share anchors (grouping is a partition).
  - An anchor in several hosts' lesions is admitted at most once, through T8.
  - Alternative outlines are (i) the several outlines of one anchor, and (ii) a claim shared by two lesions, flagged `shared_outline`. Neither creates a new lesion.
- **Final set-based tumours.** The T8-admitted anchors, grouped by their lesion under the host that admitted them. [fix 15 Oct: lesions are judged by T1–T7, anchors by T8; this is the link.]
- **Admitted outlines.** The unflagged outlines of admitted anchors.
- **Further outputs (audit 15 Oct):**
  - primary-tumour outline and anchor volumes;
  - lesion weight (D8);
  - per anchor, any agent whose different prompts overlap it (same-agent confirmation, D4).
  - An admitted anchor reports the admitting host with the largest P.
- **Reporting (user, 15 Oct).** Two volumes are reported:
  - **outline volume:** the union of admitted outlines (unflagged, not leaking). This is the widest set-based reading;
  - **anchor volume:** the union of admitted anchors, i.e. the located, agreed core.

  Image grounding trims between the two.
- **Primary tumour.** The admitted anchors of primary lesions.

### Units used by each rule

| Question | Unit |
|---|---|
| Kept or dropped (size floors, crumbs, prompt gate, slab) | claim |
| Where is it; which host (touch, band share, primary protection, organ-shape exclusion) | anchor |
| Grouping, counting, multifocality, satellites, tumour rules T1–T7 | lesion (group of anchors per host) |
| Contiguity with the admitted primary (T6) | anchors |
| Combining competing hosts, final status (T8) | anchor |
| Organ-overlap test (D10) | anchor and outline |
| Per-voxel priors, size order and nodule test (lesion extent), image grounding | outline (claim) |

### Constants [fix 15 Oct: completed]

| Name | Value | Used in |
|---|---|---|
| Envelope / touch | H + 10 mm | D6, D7, D8, prompt gate, crumbs, silent |
| Contiguity (T6) | lesion anchors within 10 mm of P* | T6 |
| Silent | no raw output of the other agent in E(primary) | D9, T terms |
| Slab | flat (through-plane ≤ 0.25 × in-plane diagonal), median slice coverage ≥ 40% | D3 |
| Grouping τ | max(5 mm, slice spacing) [decide 3] | D7 |
| Band | 3 mm, two-sided | D8 |
| Trace | another agent's raw output within 3 mm | T terms |
| Minimum claim | 0.5 ml | D3 |
| Crumb grouping | ≤ 5 mm apart, group ≥ 0.2 ml | D3 |
| Minimum two-agent anchor | 0.5 ml | D4 |
| Prompt gate | ≥ 50% inside E(own prompt host) | D3 |
| Organ-overlap (a) | volume ratio ≥ 0.6, median slice Dice ≥ 0.6, ≥ 80% inside H | D10 |
| Organ-overlap (b) | ≥ 80% of H inside, organ ≥ 50% of volume | D10 |
| Nodule (T4) | ≥ 10 mm LD, ≥ 3 slices, ≥ 10 mm through-plane | T4 |
| Distant agreement (T7) | ≥ 0.5 ml | T7 |
| KB host weights | 1.0 / 0.3 / 0.15 | D8 |
| Voxel priors | 0.9 / 0.75 / 0.5 | D9 |
| No-host prior π_none | [decide 1] | D8 |
| `multi_host` | top two P ≥ 0.1 | D8 |
| `host_uncertain_review` | s_primary < 5%, neighbour ≥ 30% | D8 |
| Set-based admission | p_admit ≥ 0.5 | T8 |
| Outline leak | ≥ 50% of the outline beyond its anchor inside non-host organs | D10b |

## Stage 1. Claims — now D3
- **C0** crumbs: kept, in D3, per agent.
- **C1** "reference host of a claim": **superseded**; host reasoning is on anchors (D8).
- **C2** prompt gate: kept, in D3 (host-prompt claims only).
- **C3** slab: kept, in D3, agent-symmetric.
- **C4** claim-level organ shape: near the primary, replaced by D10. Away from the primary: an anchor-to-host assignment rule [decide 5], no longer a claim drop.

## Stage 2. Lesions — now D4–D7, D9
- **B1–B3** (VT core, BP within 5 mm, BP inside other organs near the primary, BP leftovers): **superseded** by anchors and grouping. They caused the BP splits (B7; 287 splits in the pilot).
- **B4** support classes: **superseded** by the D9 voxel priors and the T support classes.

## Stage 3. Host of each anchor — now D6–D8
- **V0.** If the primary is out of view, no anchor is primary-protected; every anchor follows the D8 "otherwise" branch.
- **V1 / V3** (primary candidate, certain host, involved organs): now D7 assignment plus D8 primary protection.
- **V2** organ-overlap test: now D10.
- **V4** uncertain host: P(h | a) ∝ w_h · s_h (D8). **The likelihood table L is dropped** (user, 15 Oct); the fitted table stays in `knowledge_base/host_likelihood.yaml` for reference only.
- **V5** review flags: listed in D8 and D10.

## Stage 4. Tumour rules (confirmed by the user, 15 Oct)

### Terms used by T1–T8
- **Primary lesion.** A lesion of the primary host (D7). It exists only when the primary is in view.
- **Flagged anchors and outlines (D10).** A flagged anchor or outline is soft-rejected on its own: it keeps its prior and stays visible to image grounding.
  - The lesion continues with its unflagged anchors and outlines.
  - A lesion is flagged `organ_overlap` only when **all** its anchors are flagged, or all its outlines are flagged or leaking (D10b). [decide 7: the confirmed draft flagged the whole lesion if *any* anchor was flagged.]
- **Lesion extent.** The union of the lesion's unflagged outlines, or of all its outlines if every outline is flagged. Extent is used for the T1 size order and the T4 nodule test. Contiguity uses anchors (T6).
- **Support class of a lesion** (agent-symmetric), computed over its unflagged anchors and checked in this order; the first that applies wins:
  - **two-agent:** it has at least one two-agent anchor;
  - **silent:** a primary lesion whose anchors all come from one agent, and every other tumour agent is silent (D9);
  - **single, confirmed:** no two-agent anchor, and another tumour agent's raw output lies within 3 mm of the lesion (a *trace*). Another prompt of the same agent does not count. This includes a lesion whose single-agent anchors come from different agents but overlap by less than 0.5 ml [fix 15 Oct: that case had no class];
  - **single, unconfirmed:** otherwise.

  [fix 15 Oct: silent and trace both use raw output, as v3 did for trace; v3 measured silence on kept claims. The check order resolves the rare case that satisfies both.]
- **Shared outline.** A claim that overlaps two *different* anchors lying in different lesions is flagged `shared_outline` (possible bridge). Each lesion keeps it as an outline, and reported volumes use unions, so it is not counted twice. One anchor sitting in several hosts' lesions does not trigger the flag. [fix 15 Oct: as written before, the flag fired on every multi-host anchor.]

### Pass 1: primary lesions (only if the primary is in view)
- **T1 [new].** Order of primary lesions: support class (two-agent > silent > single confirmed > single unconfirmed), then extent volume, largest first. Flagged lesions are soft-rejected (`primary_organ_overlap`) and are not eligible for T2.
- **T2 [v3].** The first unflagged primary lesion is admitted as the primary tumour, whatever its class. Reason: `primary_<class>`.
  - If every primary lesion is flagged, no primary is admitted on set rules; the flagged lesions carry their priors into image grounding.
- **T3 [v3].** A later two-agent primary lesion is admitted as multifocal (`primary_multifocal_two_agent`).
- **T4 [new].** A later single-agent primary lesion is admitted as multifocal (`primary_multifocal_single`) if both hold:
  - it is silent, or single confirmed;
  - its extent is a nodule: ≥ 10 mm LD, ≥ 3 slices, ≥ 10 mm through-plane. All three are measured on the acquisition axis. LD = the longest in-plane (Feret) diameter over the acquisition slices. [audit 15 Oct: v3 used the in-plane bounding-box diagonal, which overestimates LD by up to 1.4×, and the S–I axis.]

  Otherwise it is a soft rejection: `primary_extra_unconfirmed` (fails the first test) or `primary_satellite_single` (fails the size test).
- **T5.** Removed (user, option b).

### Pass 2: all other lesions, given the admitted primary tumour
- **Admitted primary tumour.** P* = the union of the unflagged anchors of the primary lesions admitted in pass 1. If none was admitted (primary out of view, or all flagged), P* is empty.
  - [fix 15 Oct: P* and contiguity use anchors, not outline extents. Otherwise a long unflagged outline would make any neighbour lesion it reaches "contiguous", which is the bridging that D7 avoids.]
- **T6 [v3, 10 mm].** A lesion of a local-invasion host is *contiguous* if any of its unflagged anchors has a voxel within 10 mm of P*.
  - Contiguous and two-agent: admitted (`local_two_agent_contiguous`).
  - Contiguous and single-agent: soft rejection (`local_single_agent`).
  - Not contiguous: T7 if that organ is also a distant site of the cancer; otherwise a soft rejection (`local_not_contiguous`).
- **T7 [new].** A lesion of a distant host h is admitted if at least 0.5 ml of its anchors' voxels is claimed both by a host-prompt claim for h and by a claim of a *different* agent. The voxels are pooled over all host-prompt claims for h. Otherwise it is a soft rejection (`distant_needs_site_prompt_agreement`). [fix 15 Oct: stated as a voxel test.]
  - Today that means BiomedParse ∩ VoxTell's site prompt, as before.
  - VoxTell's first prompt ∩ its own site prompt is the same agent and does not count.
  - The lesion touches the site by construction (D7).
- **T8 [new]. Final status of each anchor.** Anchor a is in one lesion per host it is assigned to (D7), with P(h | a) from D8.
  - **Admitted under h** means: a's lesion under h is admitted by T2–T7, **and** a itself is not flagged by D10. A flagged anchor inside an admitted lesion stays a soft rejection (`anchor_organ_overlap`). [fix 15 Oct: before, it inherited admission from its lesion.]
  - **π(a)** = the D9 summary prior of a. Every T1–T7 outcome is admission or soft rejection, so π is never set to 0 here; prior 0 arises only from claim drops (D3) and mask-less hosts (D2).
  - p_admit(a) = Σ_h P(h | a) · 1[admitted under h]; p_tumour(a) = π(a).
  - The set-based status of a is admitted if p_admit(a) ≥ 0.5. For a primary anchor, P = 1, so this is its primary lesion's status, unless a is flagged.
  - **A no-host anchor:** soft rejection, p_admit = 0, p_tumour = π_none.

### What changes in effect versus the earlier v6 logic (confirmed 15 Oct)
- **The order follows the priors:** two-agent > silent > single confirmed > single unconfirmed, whichever agent it is. v3 put VT-only second.
- **Confirmation is symmetric.** A VT-only extra lesion now needs a BP trace within 3 mm, or the silent case. v3 confirmed every VT-only lesion.
- **Leaky outlines are set aside, not the lesion.** A two-agent tumour with one organ-swallowing outline is still admitted on its agreed anchor and its other outlines.

## Zero and low priors
- **Prior 0:**
  - claims dropped by the size floors, the prompt gate or the slab rule (away from the primary only);
  - mask-less hosts;
  - an anchor's assignment to a host it is excluded from under decide 5. The anchor keeps its other hosts; if it has none left, it becomes a no-host lesion with π_none.
- **Low prior π_none = 0.1 (user, 15 Oct), for:**
  - no-host lesions [decide 1];
  - **single-agent anchors away from the primary that are soft-rejected:** local not contiguous, local single-agent, distant without site agreement, organ overlap. Their status is unchanged; only p_tumour drops from 0.5 to 0.1.
  - Two-agent anchors away from the primary keep their D9 prior, as do all primary-lesion candidates.
- No-host lesions: This replaces the earlier prior 0 (user, 15 Oct). For scale: 713 of 2,573 pilot5 lesions, 94% of them BiomedParse-only.
- **Everything else** keeps its D9 prior.

## Open decisions
1. **No-host prior π_none.** Proposed: 0.1, below every support prior, fixed by rule.
2. **KB-box primaries** (e.g. ovary with no mask). The box has no envelope, so nothing could ever touch it, and such scans would have no primary lesion. Proposed:
   - set E(primary) = the box itself (no dilation), for every rule that uses the envelope (crumbs, prompt gate, silent);
   - an anchor touches the box primary only if ≥ 50% of it lies inside the box (v3's rule);
   - the paired-side rule does not apply;
   - D10 never tests the box.
3. **τ** = max(5 mm, slice spacing). Proposed.
4. **H for VT aliases.** The majority vote shrank the cervix in the B7 series (26 vs 43 ml). Check later.
5. **Organ shape away from the primary** (old C4 and the V4 exclusion). Applied at step 6, after touch, before grouping; it never removes a claim. Proposed: run D10 test (a) on every non-primary anchor against each host it touches.
   - A **single-agent** anchor that *is* that organ is not assigned to that host (hard, as in v3: one agent outlining a whole organ is a known VT failure).
   - A **two-agent** anchor that is the organ is flagged and soft-rejected.
   - Outlines failing (a) or (b) are soft-rejected individually, as near the primary.
6. **Crumbs for every agent.** Making C0 symmetric adds BP crumb groups near the primary: BP has a median of 4 small fragments (0.7 ml) per scan, VT a median of 1 (0.02 ml). Proposed: symmetric, as P5 requires.
7. **Flagged anchor handling.** Proposed: set aside only the flagged anchor, and flag the lesion only when all its anchors are flagged. With "any anchor flags the lesion", one small organ-shaped anchor next to a large tumour would soft-reject the whole tumour.

## Implementation audit (15 Oct)
`ledger_v6s.py` was checked rule by rule against this document, with an independent reviewer and 19 invariant checks on the pilot outputs. All invariants pass.

**Fixed in the code:**
- acquisition axis for the slab rule, the D10 slice Dice and T4;
- LD as the Feret diameter;
- D10 host organs tested once, on H;
- KB role weights;
- reported host of admitted anchors;
- no primary envelope when the primary is out of view;
- T7 union;
- unassigned KB boxes in D10b;
- extent from the outlines of unflagged anchors;
- output keys when a scan has no claims;
- missing outputs (primary volumes, lesion weight, same-agent confirmation).

**Code kept, wording to confirm:**
- (i) Paired-side full tie: the code keeps the first side and flags `side_unresolved` (0 cases in the pilot), rather than using the organ without a side.
- (ii) The silent case needs at least two tumour agents present (all pilot series have both).

**Decision 8 (resolved by the user, 15 Oct): slab and nodule axis.** Use the acquisition axis, as written. A flat claim touching E(primary) is soft-rejected, not dropped (D3). The background follows. Using the acquisition axis, as written, rather than v3's S–I axis changes only the non-axial MR series (40 of 89 MR series). In the pilot it drops 32 more flat claims and costs 8 v3-admitted lesions, all on coronal kidney MR (KIRC 7, KIRP 1), including one series' primary.

**Pilot after decision 8 (run 6).**
- 147 of 157 series have an admitted primary.
- 63 lesions are now `primary_slab` soft rejections (31 CT, 32 MR) instead of being dropped.
- The 8 coronal-MR cases: 1 admitted, 5 `primary_slab` soft (prior 0.5–0.71), 2 `primary_satellite_single` soft.
- All invariants still pass.

## Checks before use
- **K1.** v3 mode must still reproduce stored v3 exactly. The new definitions apply to v6 mode only.
- **K2.** Pilot on the same 158 series:
  - gains and losses vs v3 and vs pilot5;
  - counts of alternative outlines, `shared_outline`, multi-host anchors, `organ_overlap` per outline and per anchor, no-host lesions;
  - B7 and the KIRC whole-liver cases checked by name;
  - panels for review (fix the panel organ-name prefix bug first).
