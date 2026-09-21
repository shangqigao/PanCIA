# PanCIA anchor-centric anatomical knowledge base (v0.4.1, draft)

Knowledge that lets a **deterministic planner** turn a TotalSegmentator (TS) result into VoxTell prompts for any
scan, any cancer type: which anchors *should* be in the field of view, where they must be, which surrounding
structures matter for staging, and how to gate and check each prompt. Cancer type is used **only** for the tumour
phrase; everything anatomical is keyed by anchor organ, so an unseen cancer in an external cohort is handled by its
host organ's profile.

```
knowledge_base/
  kb_meta.yaml          version, counts, sources, coverage-tier legend
  entities.yaml         191 structures: VoxTell terms + aliases, TS class mapping, tier, volume/HU, priority;
                        33 are anchors with span / position / presence class / spatial prior
  relations.yaml        364 typed edges = anchor profiles (has_part, invested_by, adjacent_to, supplied_by,
                        drained_by, has_duct, drains_lymph_to, lies_in, wall_of, landmark_for), each with the
                        staging question it supports
  relation_types.yaml   meaning + prompt gate per relation type (written once, applies to all anchors)
  regions.yaml          14 body regions / spaces with landmark definitions and vertebral spans
  landmarks.yaml        10 non-vertebral landmarks with vertebral levels (frame estimation when TS vertebrae fail)
  tumour_prompts.yaml   TCGA cohorts → host + compositional phrases; generic templates for unknown cancers
  prompt_rules.yaml     templates, alias ensembling, gate-as-output-filter rules, single-pass tiers, budget
  qc_rules.yaml         stage-1 reasoning rules: SET THEORY + TOPOLOGY only (containment, disjointness, part⊂whole,
                        component count, tubular continuity, adjacency/contact, laterality, TS–VT arbitration, absence,
                        tumour fusion by connectivity to the agreed core); statistical rules (HU, KS existence gate,
                        two-sided MMD) are listed as deferred and not applied
  evidence_template.csv columns the pilot fills per structure × modality
pancia_kb/              loader + validator (kb.py), planner (planner.py),
                        adapter (adapter.py): TS *_structures.csv / _labels.json / mask (+ optional tumour mask) → TSOutput
scripts/run_planner_batch.py  one plan per TS output under a folder → <img>_plan.json + planner_batch_summary.csv
tests/test_planner.py   synthetic pelvic-MR (CESC) and abdominal-CT (KIRC) plans; example plans as JSON
build_kb.py             authoring source; regenerates knowledge_base/ (edit here, not the YAML)
seed/ts_class_maps.json TotalSegmentator class maps used for cross-checking
```

## Planner (pancia_kb.Planner.plan)
**Single pass.** The plan is compiled once from TS + KB and VoxTell is called **once** per scan with every prompt in
it. Tiers (0 ruler, 1 completion, 2 profile) only order and budget prompts; nothing is re-planned after inference.
Profiles of anchors TS cannot name (ovary, cervix, prostate …) are planned up front under gates; if the anchor turns
out absent, its dependent prompts simply come back empty or fail their gate in the reasoning layer.

1. **Frame** — vertebral span of the FOV from TS vertebrae; else fallback landmarks (`landmarks.yaml`, using the
   top/bottom/centroid of the landmark mask as specified); else a provisional wide span plus tier-0 ruler prompts
   (spine, sacrum, hip bones) whose masks the reasoning layer uses for frame and laterality checks.
2. **Found anchors** — TS classes → entities, with volume-range and laterality plausibility.
3. **Expected anchors** — every anchor whose canonical span overlaps the FOV (≥2 levels). Presence class and sex are
   **never** used to exclude an anchor: sex is optional metadata recorded in the plan for validation only, and the image
   decides existence (an empty return for a sex-dependent, surgical or variable anchor is recorded as *absent*, for an
   obligatory anchor as *not found*). This avoids bias from missing or wrong records and from inferring anatomy from a
   label. **Missing** = expected − found (per side).
4. **Tier 1 (List B, completion)** — missing anchors prompted under a gate compiled from their spatial prior against
   found landmarks. Surgically-removable anchors are recorded as *absent candidates* so an empty return can be
   logged as absence rather than failure.
5. **Tumour prompt** — only when the scan has no initial VoxTell tumour mask (`include_tumour_prompt`): a re-issued
   VoxTell tumour prompt is the same model on the same image and therefore not an independent rater. The two tumour
   raters of the fusion are BiomedParse and the initial VoxTell tumour mask.
6. **Tier 2 (profiles)** — 2-hop profile of the host; 1-hop of TS-unnameable missing and tumour-near anchors.
   An entity already planned in tier 1 is not duplicated; its reason chain and priority are merged.
   Targets TS already found are prompted (List A, TS-bbox gate) only when priority 1; otherwise the TS mask is kept
   (`ts_only` in the log). Host-derived items get a priority boost; OOD structures are demoted.
7. **Budget** — tumour/ruler prompts first, then tier, then priority; capped at `max_prompts_per_scan` (default 60).
   Tier 0/1 and tumour prompts are never dropped (they decide which anchors exist); the cap trims tier 2 only.
   Every prompt carries its gate, priority and a reason chain; the plan serialises to JSON for audit.

**Gates are output filters, not input crops.** VoxTell is a whole-volume model (trained on full scans; its text
conditioning uses global context), so inference always runs on the entire image with every prompt of the plan in one
call. Each gate is applied afterwards: connected components of the returned mask are kept only if ≥80% of their
voxels lie inside the gate region; the rest are rejected as anatomically impossible fragments. An expected structure
that is empty after filtering is recorded as *absent* (sex-dependent, surgical, variable) or *not found* (obligatory).

## Batch planning over TotalSegmentator outputs
```
python scripts/run_planner_batch.py --kb knowledge_base --ts_dir <TS save_dir> --ts_obj organ \
    --out_dir <plans dir> --meta_csv scan_meta.csv --tumour_dir <VT save_dir> --tumour_dir <BP save_dir> \
    --tumour_obj tumor --workers 8
```
Pipeline position: the **initial tumour masks (BiomedParse 2D and VoxTell 3D, already computed for the cohort) and the
TotalSegmentator organs are both inputs to the planner**. `--tumour_dir` is repeatable (VT first, then BP).

**Cancer type (and site) is the primary anchor for the host; tumour masks only supply evidence that is consistent
with it.** BP and VT have zero overlap on ~60 % of series, so a raw centroid is unreliable. The adapter builds the
expected host's envelope (its TS mask dilated 10 mm) and accepts, in order: the agreement `T_BP ∩ T_VT` if ≥50 % of it
lies in the envelope; else the connected components of VT, then BP, that lie ≥50 % inside; else nothing. With
nothing validated the centroid is `None`, the cancer-type host is kept, a bilateral host is prompted on **both** sides
(a side is never guessed) and tumour-near anchors are skipped; where the raters put the tumour instead is recorded
(`raters_elsewhere`, possible metastasis or organ leakage).

*Host without a TS class* (cervix, uterus, ovary, vagina, rectum, breast — the pelvic-MR and breast-MR cohorts): the
envelope is replaced by a **KB prior region**: the host's spatial-prior landmarks, followed transitively (cervix →
uterus → bladder, hips, sacral promontory), plus the members of its body region; those TS did segment define a
bounding box expanded by 30 mm (cervix on MR: hips + sacrum + bladder → the true pelvis). Components are validated
against that box exactly as against the envelope; evidence tags read `prior_region_*`. Only when no TS structure at
all is available is the evidence `unverified_*`. For midline hosts the centroid only affects tumour-near anchors
(the host's own KB profile is prompted regardless); a bilateral host without a TS mask (breast) takes its side from
the tumour's offset to the midline only when that offset is ≥30 mm, otherwise both sides are prompted. The
definitive host localisation comes from the anatomy call itself: the host is always a tier-1 completion prompt, so
the VoxTell host mask (after set/topology checks) is what the fusion step uses, not the prior. Column
`tumour_evidence` in the summary CSV gives the per-scan outcome (`agreement`, `rater_0_validated`,
`rater_1_validated`, `prior_region_*`, `unverified_*`, `none`); `validation_region` and `prior_region_landmarks`
in the plan JSON say what the check was made against. The same two masks
are later the tumour raters of the set/topology fusion.
**Trust order (the design rule).** BP and VT give *candidate* tumours (unreliable); TotalSegmentator gives the
*data-driven* anatomical prior of this scan; where the host is out of TS's vocabulary the KB gives the *data-agnostic*
anatomical prior (spatial relations, body region), which locates the host approximately, validates the candidates,
and feeds the host's profile to VoxTell so the OOD structures get segmented. Reference for fusion = TS host mask if
it exists, else the VoxTell host mask from the anatomy call. Because everything anatomical is keyed by host anchor,
an **unseen cancer type** needs only its host: add a `host` column (KB anchor id, e.g. `cervix`) to the metadata and
the same pipeline applies; `cancer_type` then only selects the tumour phrase (generic templates if unknown).

`scan_meta.csv` needs `img_name, modality, cancer_type` and optionally `host`, `sex`, `site` (img_name as written by
m_tumor_segmentation.py, i.e. the TS file stem without `_<seg_obj>`). Sex only travels into the plan JSON for later
validation; it changes no prompt. Without the file, modality comes from `_labels.json` and the planner falls back to
generic tumour templates. The FOV comes from the TS mask header; the tumour centroid and host organ
(by voxel overlap with the TS mask; the cancer-type host wins unless ≥50 % of the tumour lies in another anchor) come
from the tumour mask when present. Classes under 0.05 ml are dropped as speckle. Each plan JSON records its inputs, KB
version and diagnostics; the summary CSV has per-scan frame method, found/expected/missing counts (and how many
"missing" are TS classes that failed plausibility), prompt counts per tier, budget drops and OOD prompt count — the cohort audit
of step 2 starts from this file. Plans are single-pass: the JSON is the complete prompt list for the one VoxTell call.

## Stage-1 reasoning: sets and topology, no statistics
With four raters' masks per scan — BP tumour, VT tumour, TS organs, VT organs — the reasoning layer uses only set
operations and topology (`qc_rules.yaml`). Anatomy: alias majority vote → components kept if ≥80 % inside the gate →
part ⊂ whole, pairwise disjointness (overlap goes to the entity that stays connected), component count, tubular
continuity, contact with required neighbours, laterality; TS vs VoxTell arbitration by Dice with the consensus
`O_TS ∩ O_VT` as the normal reference. Tumour: `A = T_BP ∩ T_VT` is the agreed core, `D = T_BP △ T_VT` is split into
connected atoms; an atom is kept iff it touches the core (or a kept atom), lies within the dilated host envelope, and is
not inside another organ's consensus; fused = A ∪ kept atoms, holes filled.

**Hosts are a weighted set, not one organ (v0.3.0).** A cancer sits in its primary site with high prior but may
have invaded neighbours or spread. `tumour_prompts.spread` defines the data-agnostic prior: primary (1.0, from cancer
type / explicit `host`), local-invasion neighbours (0.3; the anchors linked to the primary by `adjacent_to` /
`invested_by`, derived from the relation graph so unseen cancers are covered) and cancer-specific distant sites
(0.15; liver, lung, adrenal, spine … per cancer). The data-driven update is set-based: for every candidate present in
the scan, evidence = 2 if the BP∩VT agreement lies in its TS envelope, 1 if one rater's component does, 0 otherwise;
weight = prior × (1 + evidence). The primary is always kept; a secondary host earns prompts only with evidence ≥ 1
(its own extent plus a 1-hop *extent* profile: capsule/fascia, vessels, node stations, neighbours — no sub-parts);
evidence-0 neighbours in the FOV are kept as `contact_check` hosts, tested for tumour–organ contact on TS masks in
the reasoning layer without VoxTell prompts. The coarse KB prior box is never used as evidence for a secondary host.
In fusion every atom is assigned to the kept host whose extent it overlaps most, giving a primary lesion and
secondary lesions per host (KIRC example: kidney primary, ev 1, w 2.0; liver secondary, ev 1, w 0.6; spleen /
pancreas / colon / psoas contact_check). `plan.hosts` and the summary column `hosts` record the set per scan.

**Host reference when TS *does* have the host (non-OOD).** TS and VoxTell disagree on the host for a reason: TS organ
labels usually include masses inside the organ, VoxTell's organ concepts (KiTS/LiTS-style training) tend to exclude
the tumour. So (`host_reference` rule): `H_TS` is the primary, data-driven reference; the **core** `C_H = (H_TS ∩ H_VT)`
eroded, minus all tumour claims, is the normal-tissue reference; the **extent** `H_ext` = components of `H_TS ∪ H_VT`
touching the core is the envelope in which tumour atoms may live (organ ∪ tumour); the feature/graph boundary is
`H_TS` unless it fails plausibility (volume range, component count, truncation) while `H_VT` passes, or Dice < 0.5 with
`H_VT` passing. The disagreement `H_TS △ H_VT` is split: components overlapping a tumour claim are *tumour-related* and
join the tumour candidate pool with provenance `organ_disagreement` — the organ rater that excluded them is an extra
witness that the region is not normal organ; the rest is boundary noise. This gives the `atom_support` rule its third
witness: support = BP claim + VT claim + organ-rater exclusion, which is what decides among single-rater components
on the ~61 % of series where `T_BP ∩ T_VT = ∅`. For an OOD host there is no `H_TS`: `H = H_VT`, core = `H_VT` minus
tumour claims, extent = `H_VT ∪` tumour claims touching it. Two-sided MMD, HU tests and the R²-Seg
existence gate are deferred to stage 2, once the pilot shows where set/topology rules are insufficient.

Run-time components still to implement on top of this: the single VoxTell call per scan
(`voxtell-predict -i img -o out -p <all terms of the plan>`), gate resolution on actual masks, the set/topology
reasoning module implementing `qc_rules.yaml`, and evidence write-back.

## Coverage tiers (from VoxTell's published vocabulary, v0.4.1)
Two fetched sources: the **VoxTell v1.1 label set** behind its Hugging Face text embeddings
(`embeddings/voxtell_v1.1/labels.json`, 14,194 prompt strings = training classes + rewritten synonyms, 190 datasets;
`seed/voxtell_v1.1_labels.json`) gives *presence*; the paper's **Table 10** (1,078 v1.0 labels with # training
volumes) + **Table 7** (test classes), parsed into `seed/voxtell_vocabulary.csv`, gives the *counts*. `build_kb.py`
aligns every entity to the merged vocabulary: the main phrase becomes the
exact vocabulary wording when one exists ('left iliopsoas', 'L1 vertebra', 'liver segment 4', 'left lung upper lobe',
'left iliac vena'), aliases are ordered exact › near › rest, and `coverage_tier` is set from the match — `in_vocab`
(exact label with ≥ 20 training volumes or unknown count), `rare` (exact label, < 20 volumes; expect weak masks), `near` (no exact label;
zero-shot wording sharing the head noun), `ood` (no label with this head noun). `voxtell_evidence` on each entity
records the matched label and its count; `knowledge_base/voxtell_vocab_check.csv` lists every phrase.

What this means for the project: uterus, cervix, rectum, prostate (+ zones), iliopsoas, iliac vessels, liver
segments, lung lobes, individual vertebrae C1–L5 (a vertebral ruler for MR, where TS `total_mr` has no levels) are
real labels. Seminal vesicles (unsided), endometrial cavity, myometrial tissue, omentum and para-aortic lymph nodes exist in v1.1
only. **Ovary, vagina, parametrium, mesorectum, urethra, diaphragm and most abdominal / pelvic node stations are not in
the vocabulary** (mediastinal, hilar, cervical and para-aortic nodes are); prompting them is zero-shot and the planner
budgets them last. Renal vein is labelled on 70
volumes unsided but on 1 volume per side, so the unsided label is used and the side comes from the gate. The three
tissue compartments (subcutaneous fat, visceral fat, skeletal muscle) are no longer anchors: VoxTell has only
whole-body 'fat' / 'muscles' labels and TS `tissue_types` is not run. `scripts/extract_voxtell_vocab.py` re-grades
the KB and can merge the Hugging Face embedding `.npz` keys as a second source.

## Editing
Edit `build_kb.py`, run it (it asserts referential integrity and TS class names), then `python tests/test_planner.py`.
Bump `VERSION` for any content change; per-scan plans record the KB version so prompt sets stay reproducible.

## Sources
TotalSegmentator class maps (wasserth/TotalSegmentator, `map_to_binary.py`); VoxTell (arXiv 2511.11450,
MIC-DKFZ/VoxTell); AJCC Cancer Staging Manual 8th ed. (T/N criteria, regional node lists); FIGO 2018 (cervix),
2023 (endometrium), 2014 (ovary); FMA / UBERON anatomical relations. Note: TS `tissue_types` is a licensed model.
