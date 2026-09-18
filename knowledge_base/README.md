# PanCIA anchor-centric anatomical knowledge base (v0.1.0, draft)

Knowledge that lets a **deterministic planner** turn a TotalSegmentator (TS) result into VoxTell prompts for any
scan, any cancer type: which anchors *should* be in the field of view, where they must be, which surrounding
structures matter for staging, and how to gate and check each prompt. Cancer type is used **only** for the tumour
phrase; everything anatomical is keyed by anchor organ, so an unseen cancer in an external cohort is handled by its
host organ's profile.

```
knowledge_base/
  kb_meta.yaml          version, counts, sources, coverage-tier legend
  entities.yaml         191 structures: VoxTell terms + aliases, TS class mapping, tier, volume/HU, priority;
                        36 are anchors with span / position / presence class / spatial prior
  relations.yaml        364 typed edges = anchor profiles (has_part, invested_by, adjacent_to, supplied_by,
                        drained_by, has_duct, drains_lymph_to, lies_in, wall_of, landmark_for), each with the
                        staging question it supports
  relation_types.yaml   meaning + prompt gate per relation type (written once, applies to all anchors)
  regions.yaml          14 body regions / spaces with landmark definitions and vertebral spans
  landmarks.yaml        10 non-vertebral landmarks with vertebral levels (frame estimation when TS vertebrae fail)
  tumour_prompts.yaml   TCGA cohorts → host + compositional phrases; generic templates for unknown cancers
  prompt_rules.yaml     templates, alias ensembling, ROI per gate kind, waves, budget
  qc_rules.yaml         plausibility / acceptance rules (volume, laterality, containment, adjacency, …)
  evidence_template.csv columns the pilot fills per structure × modality
pancia_kb/              loader + validator (kb.py), planner (planner.py)
tests/test_planner.py   synthetic pelvic-MR (CESC) and abdominal-CT (KIRC) plans; example plans as JSON
build_kb.py             authoring source; regenerates knowledge_base/ (edit here, not the YAML)
seed/ts_class_maps.json TotalSegmentator class maps used for cross-checking
```

## Planner (pancia_kb.Planner.plan)
1. **Frame** — vertebral span of the FOV from TS vertebrae; else fallback landmarks (`landmarks.yaml`, using the
   top/bottom/centroid of the landmark mask as specified); else wave-0 ruler prompts (spine, sacrum, hip bones).
2. **Found anchors** — TS classes → entities, with volume-range and laterality plausibility.
3. **Expected anchors** — anchors whose canonical span overlaps the FOV (≥2 levels), filtered by presence class and sex
   (unknown sex → both branches). **Missing** = expected − found (per side).
4. **Wave 1 (List B, completion)** — missing anchors prompted under a gate compiled from their spatial prior against
   found landmarks. Surgically-removable anchors are recorded as *absent candidates* so an empty return can be
   logged as absence rather than failure.
5. **Tumour prompts** — cohort phrases (or generic templates) on the host; host side from the tumour centroid.
6. **Wave 2 (profiles)** — 2-hop profile of the host; 1-hop of promoted (TS-unnameable) and tumour-near anchors.
   Targets TS already found are prompted (List A, TS-bbox gate) only when priority 1; otherwise the TS mask is kept
   (`ts_only` in the log). Host-derived items get a priority boost; OOD structures are demoted.
7. **Budget** — tumour/ruler prompts first, then by (priority, wave), capped (`max_prompts_per_scan`, default 40).
   Every prompt carries its gate, priority and a reason chain; the plan serialises to JSON for audit.

Run-time components still to implement on top of this: gate resolution on actual masks, VoxTell calls
(`voxtell-predict -i img -o out -p <all terms>`; text is embedded once per prompt set), QC rules, promotion of
wave-1 results into a second `plan()` call, and evidence write-back.

## Coverage tiers (assumptions until the pilot)
VoxTell ships no vocabulary file. Tiers were assigned from the datasets known to be in its training corpus
(TS classes, AMOS/BTCV/FLARE organs, KiPA renal vessels, MSD hepatic vessels, LNQ nodes, prostate-zone and breast
MRI sets). `ood` entries (parametrium, Gerota fascia, perinephric/perivesical fat, mesorectal fascia, neurovascular
bundle, gonadal/uterine vessels, junctional zone) are experimental: strict gates, retire if the pilot fails them.

## Editing
Edit `build_kb.py`, run it (it asserts referential integrity and TS class names), then `python tests/test_planner.py`.
Bump `VERSION` for any content change; per-scan plans record the KB version so prompt sets stay reproducible.

## Sources
TotalSegmentator class maps (wasserth/TotalSegmentator, `map_to_binary.py`); VoxTell (arXiv 2511.11450,
MIC-DKFZ/VoxTell); AJCC Cancer Staging Manual 8th ed. (T/N criteria, regional node lists); FIGO 2018 (cervix),
2023 (endometrium), 2014 (ovary); FMA / UBERON anatomical relations. Note: TS `tissue_types` is a licensed model.
