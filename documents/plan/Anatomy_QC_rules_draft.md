# Image quality control, host reasoning and tumour reasoning — rules DRAFT 1 (17 Oct 2026)

Status: draft for user review. Nothing implemented.
- Extends ledger v6s (FINAL v3 definitions) Stage 0 D2 and the tumour rules.
- Evidence: `anatomy_feasibility_audit_2026-10-17` (157 pilot series).

User decisions (17 Oct):
- three quality states;
- outlines renamed (ω ∈ Ω_a) so that o means organ only;
- only the primary gets the KB region when it cannot be segmented;
- host reasoning = R(A_i, A_j) over KB organ–organ relations;
- tumour reasoning = R(u, Ĥ_o) over KB tumour–organ relations.

Principles kept:
- set logic only, no intensities in host or tumour reasoning;
- every threshold is the KB's own (`qc_rules.yaml`, `entities.yaml`) or fixed a priori;
- nothing is tuned on TCGA, stage or agent outputs;
- sex is never used;
- agents are pluggable and treated equally;
- the anatomy stays cut: tumour evidence never reshapes an organ.

## 0. Notation (main.tex)

**Anatomy masks.** A_{k,o} is organ agent k's mask of organ o. Defaults: A_{TS,o} = T_o, A_{VT,o} = V_{K,o}.

**Organ-feasibility state.** For each mask, φ(A_{k,o}) ∈ {feasible, flagged, infeasible}.

**Organ reference.** Ĥ_o = ⋃_{k: φ ≠ infeasible} A_{k,o} \ ⋃_{o'} C_{o,o'}.

**Contested set.** C_{o,o'} = (⋃_k A_{k,o}) ∩ (⋃_k A_{k,o'}) for KB-unrelated organs o, o'. It belongs to both organs as host territory and to neither as normal tissue.

**Tumour units.** u ∈ {claim, anchor a, outline ω}. Outlines of anchor a: Ω_a (was 𝒪_a); unflagged outlines: Ω°_a.

**KB relations.**
- ℛ_AA (organ–organ):
  - has_part / lies_in / invested_by / has_duct: containment, overlap allowed;
  - adjacent_to (contact, direction);
  - spatial_prior (relation, dist_mm);
  - disjoint: the default for organs not related by containment.
- Unary attributes: volume_ml range, expected components, laterality, vertebral span.
- ℛ_TA (tumour–organ): inside / touches (host), extends into (adjacent_to + contiguity with the primary), distant site (cancer-specific list), is / contains the organ, lies in an unrelated organ, lies in a contested region, lies in the region of an unsegmentable primary.

**Definitions.**
- Host reasoning evaluates R(A_{k,o}, A_{k',o'}) ∈ ℛ_AA and the unary checks. It uses anatomy masks only.
- Tumour reasoning evaluates R(u, Ĥ_o) ∈ ℛ_TA.

## 1. Image quality control: state Q ∈ {good, degraded, exclude} (step 0)

A scan is graded once, before reasoning. The grade uses only what the anatomy agents could segment and the acquisition.

**Q1. Acquisition.**
- Through-plane spacing > 8 mm → degraded. Signs computed on the 5 mm grid are unreliable beyond this.
- Image noise in a homogeneous reference tissue (eroded paraspinal muscle, A_{k,autochthon}):
  - CT: σ in HU;
  - MR: SNR = median / σ;
  - σ = robust (MAD) high-pass residual.
- Threshold: the 1st percentile over the expert datasets (TotalSegmentator CT/MR, R2Seg), fixed before any TCGA result. Below it → degraded.
- **Open:** whether noise alone may give exclude. Proposed: no, degraded only.

**Q2. Anatomy-agent failure pattern.**
- 𝓔 = organs the planner expects in view: obligatory presence, span within the vertebral frame, not truncated by the FOV.
- An expected organ *fails* if every organ agent's mask of it is missing or infeasible (H1), or its references conflict with an unrelated organ by > 5% (H2a).
- Failures are split by tumour claims:
  - F_tum: failed organs whose 10 mm envelope (KB region for a missing organ) is touched by a tumour claim of any agent;
  - F_away: all other failed organs.
- F_tum is not a quality signal. It is a tumour signal (T-rules below): a large tumour replacing or swallowing organs.
- With f = |F_away| / |𝓔 \ F_tum| (computed when |𝓔 \ F_tum| ≥ 4):
  - f ≥ 0.5 → exclude;
  - 0.25 ≤ f < 0.5 → degraded;
  - otherwise good.
- Fractions are fixed a priori, not fitted.

**Effect of each state.**
- **good:** the full model.
- **degraded:** host and tumour reasoning run; image grounding is switched off (Λ = 1 for every case); the scan is flagged and reported separately.
- **exclude:** the scan leaves the evaluation cohort and is counted with its reason (Q1 / Q2) in the Data section.

**Review.** Panels of every exclude / degraded scan go to the user before freezing. These are visual checks only; no threshold is changed after the panels.

## 2. Host reasoning: R(A_i, A_j) over ℛ_AA

**H1. Unary feasibility of each mask A_{k,o}.** Evaluated per agent, so one agent's error does not condemn the organ. A mask is *infeasible* if any of these holds:
- **H1a volume:** outside [0.5·min, 2·max] of KB volume_ml (planner rule). Skipped when the mask touches the scan edge (truncated → flag only).
- **H1b components:** more 26-components of ≥ 5% of the mask volume than KB expected_components (solid organ: 1 per side; bowel ≤ 3; vessel: 1 dominant). Components below 5% are dropped (qc component_count). Only the excess makes the mask infeasible.
- **H1c side:** a paired organ's centroid on the wrong side of the mid-sagittal plane (planner handedness-corrected midline).
- **H1d span:** the mask centroid's vertebral level outside the KB span ± (span_var + 1) levels (scan frame from the planner). Skipped when the frame is provisional.

**H2. Pairwise relations R(A_{k,o}, A_{k',o'}).** Over all organs with masks, any agents.
- **H2a disjoint:** organs not related by containment in the KB (2-hop has_part closure) must not overlap. The overlap voxels form C_{o,o'}. When the overlap exceeds 5% of the smaller mask, both masks are marked *conflicting* (used by Q2 and reported). The region rule (H4) applies at any size.
- **H2b part in whole:** |part ∩ whole| / |part| ≥ 0.9. Otherwise the part is flagged and its outside voxels are clipped from the part (qc part_in_whole).
- **H2c contact:** for KB adjacent_to with contact *abuts*, min distance ≤ 10 mm when both organs are in view and neither is truncated. Otherwise both are *flagged*.
- **H2d spatial prior:** for KB spatial_prior with dist_mm, min distance ≤ dist_max + 10 mm. Otherwise *flagged*.
  - The relation respects side_link.
  - `between` is checked as: centroid between the two landmarks along the left–right axis.
- Flags (H2b–d) never remove a mask: tumours displace organs. They count as failures in Q2 only when the organ also fails H1.

**H3. Same organ, different agents** (qc ts_vt_arbitration).
- Infeasible masks are dropped.
- If several agents' masks are feasible, Ĥ_o takes their union, as now. Their disagreement is kept for reporting.

**H4. Organ reference and contested regions.**
- Ĥ_o = (union of feasible masks of o) \ ⋃_{o'} C_{o,o'}.
- **Touch and envelope** use Ĥ_o ∪ ⋃_{o'} C_{o,o'}: a contested region is territory of both organs.
- **Normal tissue 𝒩_h** uses Ĥ_h only: a contested region is normal tissue of neither organ.
- This depends on anatomy masks only; tumour claims are not consulted. The cut is preserved.

**H5. Unsegmentable organ.**
- An expected organ (𝓔) with no feasible mask from any organ agent is *unsegmentable*.
- **Primary:** replaced by the coarse KB region on vertebral levels, flagged, as now (user, 17 Oct). This extends today's rule (primary with no mask) to a primary whose masks are all infeasible.
- **Any other host:** no reference; prior 0, as a mask-less secondary host today.

## 3. Tumour reasoning: R(u, Ĥ_o) over ℛ_TA

**Existing rules, restated as relations.** Constants unchanged.

| Relation | Rule |
|---|---|
| touches(a, h): a ∩ E_h ≠ ∅ | Host candidates 𝓗_a, lesion membership, primary protection |
| boundary share s_h(a) | Host prior P(h \| a) ∝ w_h s_h(a). The contested region counts for both organs. |
| outside(prompt claim, E_h) | Prompt gate (hard) |
| flat_on(claim, Ĥ_o) | Slab rule (hard away from the primary, flag near it) |
| is(u, Ĥ_o) / contains(u, Ĥ_o) | D10 organ-overlap flags (soft) |
| leaks_into(ω, Ĥ_{o'}), o' ∉ 𝓗_a | Leak flag (soft) |
| contiguous(a, P*) + adjacent_to(h, primary) | Local invasion (pass 2) |
| site(h) ∈ distant list + site agreement | Distant lesion (pass 2) |

**New relations.**
- **T-unrelated:** an anchor with ≥ 50% of its volume inside Ĥ_o of a feasible organ o ∉ 𝓗_C.
  - It is that organ, not tumour of the cancer type: soft rejection (π_none), flag `in_non_host_organ`.
  - Primary protection still applies to an anchor touching the primary; the flag is kept for grounding.
  - This is qc atom_not_in_other_organ applied to anchors.
- **T-contested:** an anchor touching C_{h,o'} has both h and o' as host candidates when o' ∈ 𝓗_C. It is flagged `contested` (host uncertainty, resolved by grounding through P(h | a)).
  - Audit: about half of each contested region is covered by tumour claims. There the conflict is usually the tumour itself, and we cannot tell which agent is wrong.
- **T-replaced:** the primary is unsegmentable (H5), or every mask of it is infeasible, and the primary anchor lies in its KB region.
  - Flag `organ_replaced`: a large tumour that swallowed the organ.
  - The anchor keeps primary protection.
  - In grounding, no normal tissue exists for the primary, so only intrinsic evidence is used (as the no-host case).

**Grounding (G0).** Uses Ĥ_h and the contested sets as above; switched off for degraded scans.

## 4. Validation (descriptive)
1. **H1–H2 on expert labels:** TotalSegmentator CT/MR expert labels (single-agent, exclusive maps). Rate of infeasible / flagged masks; expected ≈ 0. Any rule firing on > 2% of expert organs is reported to the user and not adopted as is.
2. **Q1 measures** on expert datasets: set the noise thresholds (1st percentile).
3. **157 pilot series:**
   - Q states with reasons;
   - H1/H2 counts per agent and organ;
   - contested regions (tumour-covered share);
   - unsegmentable organs;
   - T-unrelated / T-contested / T-replaced counts;
   - hosts and lesions changed vs v6s run 6.
4. **Panels:** exclude / degraded scans, T-replaced and T-contested anchors, for user review.

## 5. KB fixes (before validation)
- kidney → liver spatial prior (`inferior_to`, 0–40 mm): add side_link right.
- `between` relations: the centroid-between check above, not a distance.
- Check expected_components for bowel, lung (lobes merged per side) and vessels.

## 6. Implementation and pilot (17 Oct 2026) — changes forced by the data

**Code.**
- `pancia_kb/anatomy_qc.py` (md5 8efd6d…).
- `pancia_kb/ledger_v6a.py` (built from the unchanged `ledger_v6s.py` by `scripts/make_v6a.py`, md5 a05aaf…).
- KB 0.5.8 (`build_kb.py`):
  - kidney→liver spatial prior `side_link: right`;
  - thyroid `expected_components: 2`.
- Noise thresholds: `knowledge_base/qc_thresholds.json` (CT σ p99 = 76.6 HU, n = 104; MR SNR p01 = 1.64, n = 73).

**Changes to the draft** (each found on expert labels or on the pilot before any result was accepted; none tuned on stage):
1. **H1 volume:** too large (> 2·max) → infeasible, except for the cancer's primary organ (flag `large_primary`: a tumour enlarges its organ; CESC cervix 268 ml). Too small (< 0.5·min) → flag `small` only. Removing small MR kidneys / prostates cut primaries off their host (KIRC MR primary 97 ml lost).
2. **H1 components:** flag only. Neither keep_largest_k nor infeasible: a tumour can split its organ (UCEC / KIRC primaries lost their host when the second component was dropped).
3. **H1 side:** only when the planner midline comes from midline structures and the header handedness from anatomy or TS sided pairs with agreement (breast MR with assumed RAS headers failed the side test).
4. **H2 contested:** only between organs with a boundary of their own (organ, hollow organ, gland, bone, muscle, vessel). Lymph stations, fat, fascia, spaces, ducts and nerves are defined around organs.
5. **Organ relations of tumour units use the territory** (reference ∪ contested): touch, boundary share, D10 organ overlap, slab, leak. Testing against the cleaned reference made anchors look bigger than their organ (pilot run 4).
6. **Q2 failure = unsegmentable only.** Conflicts shape the references but do not grade the scan: in the pelvis they were mostly one agent's error (VT uterus / vagina masks in a male scan).
7. **Q2 "touched by tumour"** = touched by the tumour agents' agreement (≥ 2 agents' cancer prompts). Host-prompt answers mark every prompted organ, and raw single-agent claims mark nearly all organs.
8. **Q2 expected organs:** obligatory presence only (surgical: kidney, gallbladder may be absent), with at least half the KB span in the frame.
9. **organ_replaced** only when an organ agent could segment the primary but did not (all masks infeasible, or TS has the class and found nothing), and the anchor is ≥ 0.5 × KB min volume. An OOD primary without any mask is not evidence.
10. **Noise** computed on the 5 mm grid, in-plane block-averaged to about 1.5 mm.

**Expert validation** (`scripts/aqc_expert.py`, `ledger/anat_qc/expert/`). TotalSegmentator CT: 120 no-pathology adults; MR: 150 usb adults, 140 with labels.

| Measure | CT | MR |
|---|---|---|
| Infeasible masks | 7 / 6,384 (0.11%; side only) | 10 / 2,451 (0.41%; 9 too large) |
| Q states | 120 good | 136 good, 4 degraded (spacing > 8 mm) |
| `no_contact` (KB 'abuts') flags | 3.1% of masks | 10.5% of masks |

- CT Q2 fraction: maximum 0.09.
- `no_contact` is reported only: the KB contact attribute is not a guaranteed contact.

**Pilot** (`scripts/v6a_check.py`, `ledger/v6a_ledger/`; 158 v6s pilot series, 152 compared, 6 timeouts).

| Measure | Result |
|---|---|
| Q | 131 good, 20 degraded (19 spacing > 8 mm, 1 anatomy), 1 exclude |
| Primary admitted | unchanged: 142 / 152 in both v6s and v6a |
| Admitted anchors | 557 → 555; anchor volume 8.52 → 8.50 L |

- The exclude is a BRCA dynamic breast MR (fl3d dynaVIEWS): spine, IVC, trachea, oesophagus and stomach are not depicted.
- Status changes on matched anchors: only 4, all `distant_site_agreement → in_non_host_organ` (e.g. a KIRC "liver metastasis" that is the gallbladder).
- 22 / 6 admitted anchors are unmatched by bbox (claims changed through the territory envelope).
- New flags: `in_non_host_organ` 514 (16 among admitted, primary-protected), `contested` 442 (206 admitted), `organ_replaced` 4 (1 admitted).
- Host references: TS 867, VT 184, TS+VT 34, KB box 40, none 109. 711 / 1,234 hosts have a contested region (most are small boundary overlaps).

**Panels.** `ledger/v6a_ledger/png/`: Q (21), R (7), C (16), N (16), X (4); montages `M_*.png`.

**Decision (user, 17 Oct, after the pilot panels):** missing anatomy alone never excludes a scan. In breast imaging (and other dedicated protocols) it is common that no other anatomy is visible. Q2 with f >= 0.25 gives *degraded*; *exclude* requires f >= 0.5 **and** a failing image measure (slice spacing > 8 mm or noise beyond the expert percentile). The pilot BRCA dynamic MR is now degraded; the pilot has no excluded scan. (`anatomy_qc.py` md5 see README.)
