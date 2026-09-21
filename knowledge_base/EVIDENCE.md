# Evidence behind the PanCIA anchor-centric knowledge base (KB v0.4.1)

This file states, for every kind of content in `knowledge_base/*.yaml`, **where it comes from, how far it has been
verified, and where to check it**. The KB was authored in `build_kb.py` by Shangqi Gao with Claude on 18–20 Sep 2026.
It has **not yet been reviewed fact-by-fact by a radiologist**, and no content has been validated on the cohort — the
pilot (30 CT + 30 MR) is the first validation step. Please treat the verification status column as the honest state.

Verification status used below:

| status | meaning |
|---|---|
| **fetched** | the source file/text was downloaded during authoring and the KB content was checked against it |
| **reference** | encoded from standard, citable references (named below) from the authors' knowledge, *not* re-checked entry by entry against the page |
| **assumption** | a design choice or a guess to be replaced by pilot evidence (`evidence_template.csv`) |

---

## 1. Segmentation-model vocabularies (what TS and VoxTell can name)

| KB content | source | status | check here |
|---|---|---|---|
| `entities[*].ts` — mapping of KB entities to TotalSegmentator classes for tasks `total` (117 CT classes), `total_mr` (50 MR classes), `body`, `tissue_types` | TotalSegmentator `map_to_binary.py` class maps, fetched 2026-09-18 into `seed/ts_class_maps.json`; `build_kb.py` asserts every referenced TS class name exists | **fetched** | https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py |
| TS model description, `--fast` 3 mm, licensing of `tissue_types` | TotalSegmentator repo README; Wasserthal et al., *Radiology: AI* 2023 (CT); D'Antonoli et al. 2024 (MR) | **fetched** (README) / reference (papers) | https://github.com/wasserth/TotalSegmentator · https://doi.org/10.1148/ryai.230024 · https://arxiv.org/abs/2405.19492 |
| `entities[*].voxtell.main / aliases` — VoxTell prompt phrases; `coverage_tier` ∈ {in_vocab 131, near 35, rare 3, ood 22}; `voxtell_evidence` per entity (matched label, source, # training volumes) | Two fetched sources, merged. **(a) VoxTell v1.1 label set** behind the precomputed text embeddings on Hugging Face (`mrokuss/VoxTell`, `embeddings/voxtell_v1.1/labels.json`): 14,194 unique prompt strings = training classes plus their rewritten synonyms, for v1.1's 190 datasets — the operational vocabulary the text encoder was trained on; presence, no counts. Saved as `seed/voxtell_v1.1_labels.json`. **(b) Paper Table 10** (1,078 v1.0 training labels with the number of training volumes each) **+ Table 7** (test classes), parsed into `seed/voxtell_vocabulary.csv`; supplies the counts. `build_kb.py` re-phrases every entity to the exact vocabulary wording when one exists — preferring a Table-10 label with a known count over an HF rewrite ('left iliopsoas', 'L1 vertebra', 'liver segment 4', 'left lung upper lobe', 'left iliac vena', 'seminal vesicles') — and sets the tier: in_vocab = exact label with ≥ 20 training volumes or count unknown (HF-only); rare = exact label with < 20 (left ureter: 1); near = only a label sharing all content tokens or the head noun (zero-shot wording); ood = no label with this head noun. | **fetched** (presence and counts); the *behaviour* on near/ood phrases remains for the pilot | https://huggingface.co/mrokuss/VoxTell/tree/main/embeddings/voxtell_v1.1 · https://arxiv.org/abs/2511.11450 (Tables 7, 10) · `seed/voxtell_v1.1_labels.json`, `seed/voxtell_vocabulary.csv`, `knowledge_base/voxtell_vocab_check.csv` |
| Consequences of the vocabulary check for this project | **Not in VoxTell v1.1's vocabulary (ood, 22 entities)**: ovary (only "ovarian vein"), vagina, parametrium, mesorectum, urethra (only periurethral tissue), diaphragm, pericardium, perinephric / perivesical fat, pectoralis major, coccyx, nipple, junctional zone, and the axillary, supraclavicular, inguinal, pericolic and presacral node stations. **Present**: uterus (583), cervix (949), rectum (1,281), prostate + zones, **seminal vesicles (v1.1 only, unsided)**, endometrial cavity, myometrial tissue, omentum, iliopsoas (4,610), iliac artery/vein (4,514), every vertebra C1–L5 (≈5,500 each — a vertebral ruler for MR), liver segments 1–8, lung lobes, renal artery (171) / renal vein (70, unsided), portal + splenic vein (2,947), celiac trunk (533), SMA (171), hepatic vessels (494), para-aortic lymph nodes (v1.1 only), mediastinal / hilar / cervical nodes. **Near (zero-shot wording)**: the abdominal and pelvic node stations other than para-aortic (obturator, iliac, celiac, hepatic hilar, renal hilar, mesenteric …), pancreas parts, renal pelvis / medulla, Gerota fascia, peritoneum, levator ani, chest wall, pleura. **Rare** (< 20 volumes): ureter, splenic vein, internal iliac vein. | fetched | same |
| BiomedParse (2D rater) | Zhao et al., *Nature Methods* 2024 | reference | https://doi.org/10.1038/s41592-024-02499-w · https://github.com/microsoft/BiomedParse |
| R²-Seg (our prior training-free method that the stage-2 statistics extend) | Gao et al. 2025 | reference | https://arxiv.org/abs/2511.12691 |

The tier of every entity, its matched label and training-volume count are in `entities.yaml` (`voxtell_evidence`) and
`knowledge_base/voxtell_vocab_check.csv` (one row per phrase and side).

## 2. Anatomy: entities, part–whole and neighbour relations

| KB content | source | status | check here |
|---|---|---|---|
| 191 entities: names, categories, laterality; 364 typed relations `has_part`, `invested_by`, `adjacent_to`, `supplied_by`, `drained_by`, `has_duct`, `wall_of`, `lies_in` | Standard human anatomy as formalised in the Foundational Model of Anatomy (FMA) and Uberon (`part_of`, `adjacent_to`, `supplies`, `drains`), with Gray's Anatomy (Standring, 42nd ed., Elsevier 2020) and Netter's Atlas as textbook references. Entity `sources` tags: 142 `TS+FMA`, 29 `TS`, 20 `AJCC8+FMA` (lymph-node stations). | **reference** — relations were written from anatomical knowledge with these ontologies as the intended standard; they were not exported from FMA/Uberon programmatically | FMA on BioPortal: https://bioportal.bioontology.org/ontologies/FMA · Uberon: https://obophenotype.github.io/uberon/ · OLS browser: https://www.ebi.ac.uk/ols4/ontologies/uberon |
| `relation_types.yaml` — meaning and gate per relation type (10 types) | design; the *staging question* each type supports follows the AJCC T/N logic (confined / capsule / adjacent-organ invasion / vascular invasion / nodal) | reference | AJCC 8th (below) |
| `relations[*].staging` tags (283 edges) — which T/N criterion an edge informs, e.g. kidney `invested_by` Gerota fascia → T3/T4; renal vein / IVC → T3a–c; cervix `adjacent_to` parametrium → IIB, pelvic sidewall → IIIB | AJCC Cancer Staging Manual, 8th ed. (Amin MB et al., eds., 2017; ISBN 978-3-319-40617-6) for TNM; FIGO for cervix (2018), endometrium (2023), ovary (2014) | **reference** | AJCC 8th: the book is distributed by the American College of Surgeons (https://www.facs.org/quality-programs/cancer-programs/american-joint-committee-on-cancer/cancer-staging-systems/); its Springer DOI 10.1007/978-3-319-40618-3 is registered in Crossref but Springer has withdrawn the pages, so it does **not** resolve — use the ISBN or the editors' overview article Amin et al., *CA Cancer J Clin* 2017: https://doi.org/10.3322/caac.21388 · FIGO cervix 2018 (Bhatla et al.): https://doi.org/10.1002/ijgo.12749 · FIGO endometrium 2023 (Berek et al.): https://doi.org/10.1002/ijgo.14923 · FIGO ovary 2014 (Prat): https://doi.org/10.1016/j.ijgo.2013.10.001 |
| Regional lymph-node stations per primary site (20 `ln_*` entities, `drains_lymph_to` edges) | AJCC 8th regional-node definitions per chapter (kidney: renal hilar, para-aortic/caval; bladder: perivesical, obturator, internal/external/common iliac, presacral; cervix: parametrial, obturator, iliac, presacral, para-aortic; breast: axillary levels, internal mammary, supraclavicular; lung: hilar, mediastinal, supraclavicular; colon: pericolic, mesenteric; stomach: perigastric, celiac; oesophagus: mediastinal, celiac) | **reference** | AJCC 8th (above) |

## 3. Anchors: vertebral spans, presence class, spatial priors

| KB content | source | status | check here |
|---|---|---|---|
| 36 anchors with canonical vertebral spans, e.g. liver T9–L2, kidney T12–L3, adrenal T11–L1, spleen T10–L1, pancreas T12–L2, aorta T4–L4, heart T5–T9, lung T1–T12, bladder S3–coccyx, prostate S5–coccyx, uterus S2–coccyx, cervix S4–coccyx, ovary S1–S4 (full list in `entities.yaml`) | Textbook surface/vertebral anatomy (Gray's Anatomy; Moore's *Clinically Oriented Anatomy*); spans are deliberately **generous** (they define what to *expect*, not where the organ must lie) | **reference** — approximate by design; the pilot audit (TS `zmin/zmax` per class vs frame) will tighten them | textbook tables of vertebral levels (Gray's Anatomy ch. on abdomen/pelvis; Moore's *Clinically Oriented Anatomy*, surface-anatomy boxes); Radiopaedia articles on individual structures (site blocks automated fetching, so no URL is asserted here) |
| 10 landmarks with vertebral levels: carina T4, xiphoid T9, diaphragm dome T10, renal hilum L1, aortic bifurcation L4, iliac crest L4, sacral promontory S1, bladder dome S3, femoral head / pubic symphysis ≈ coccyx level | classic surface-anatomy landmarks (Gray's; Moore) | **reference** — carina T4/T5, bifurcation L4, iliac crest L4 are textbook; diaphragm dome varies T8–T10 with respiration | Gray's / Moore surface anatomy; verify on Radiopaedia by searching the landmark name |
| Presence classes: obligatory / sex_female / sex_male / surgical / variable | anatomy; **never used as a filter** (image decides) — see standing decision in the pipeline doc | reference | — |
| `spatial_prior` per anchor (e.g. kidney: lateral to psoas 0–20 mm, anterior to quadratus lumborum, inferior to liver; cervix: inferior to uterus, posterior to bladder 0–15 mm, anterior to rectum 0–20 mm; ovary: lateral to uterus 10–50 mm; prostate: inferior to bladder, anterior to rectum, 5–40 mm posterior to symphysis) | anatomical relations from the same references; the **distance windows in mm are the authors' estimates**, tuned only on two synthetic tests | **assumption** (distances) | to be measured on the pilot from TS masks (distance transforms between classes) |
| 14 body regions with spans (thorax T1–T12, retroperitoneum T12–L5, true pelvis S1–coccyx, parametrial space S3–coccyx …) | regional anatomy (Gray's) | reference | — |

## 4. Quantitative priors used by the planner's plausibility checks

| KB content | values | source | status |
|---|---|---|---|
| Organ volume ranges (ml), applied as [0.5×min, 2×max] | sacrum 120–350; hip 150–600; lung 1000–4500 (per side; lowered from 1500 for expiration MR on 2026-09-20); heart 400–1000; liver 900–2500; gallbladder 10–120; spleen 80–600; pancreas 40–150; adrenal 2–15; kidney 90–300; bladder 20–800; prostate 15–150; uterus 30–400; cervix 10–120 | typical adult imaging volumetry (e.g. liver ~1.2–1.8 L, kidney ~130–190 ml, spleen ~100–250 ml, prostate 20–40 ml normal, adrenal 3–6 ml); ranges widened to cover pathology and partial visibility | **reference / assumption** — wide by design; to be replaced by cohort percentiles from the TS audit (`planner_batch_summary.csv` + `_structures.csv`) |
| CT attenuation ranges (HU) | aorta 150–500 (contrast), lung −950–−600, liver 40–80, kidney 30–60, fat −190–−30, muscle −29–150 | standard CT tissue attenuation; fat −190…−30 HU and muscle −29…150 HU are the conventional body-composition windows (Mitsiopoulos et al. 1998; widely reused in sarcopenia CT studies) | reference — **not applied in stage 1** (statistical rules deferred) |
| RECIST-derived thresholds: measurable component ≥ 0.5 ml (≈ 10 mm sphere), longest diameter rules used in `recist_v3` | RECIST 1.1 (Eisenhauer et al., *Eur J Cancer* 2009) | **reference** — https://doi.org/10.1016/j.ejca.2008.10.026 |
| Planner/reasoning thresholds: host envelope 10 mm; KB prior box +30 mm; component "inside" ≥ 50 % (evidence) / ≥ 80 % (gate filter); Dice 0.5 for TS–VT arbitration; alias vote ≥ 2/3; budget 60 prompts; landmark trusted only if untruncated and ≥ 25 % of min volume | design choices | **assumption** — expose in the pilot as sensitivity analysis |

## 5. Tumour prompts and spread priors

| KB content | source | status | check here |
|---|---|---|---|
| Cancer type → primary host (KIRC/KIRP/KICH kidney; LIHC liver; BRCA breast; LUAD/LUSC lung; BLCA bladder; PRAD prostate; CESC cervix; UCEC uterus; OV ovary; STAD stomach; ESCA oesophagus; COAD colon; READ rectum; PAAD pancreas) and tumour phrases | TCGA project definitions (GDC) | **reference** | https://portal.gdc.cancer.gov/projects · https://www.cancer.gov/ccg/research/genome-sequencing/tcga/studied-cancers |
| Spread prior classes and weights: primary 1.0, local invasion 0.3 (derived automatically from `adjacent_to`/`invested_by` edges), distant 0.15 | design | **assumption** — weights are ordering scores, not probabilities |
| Distant metastatic sites per cancer (e.g. RCC → lung, liver, adrenal, bone/spine, pancreas; HCC → lung, adrenal, bone; breast → liver, lung, bone; lung → adrenal, liver, bone; prostate → bone; ovary → peritoneal surfaces/spleen/bowel; gastric → liver, ovary, lung; colorectal → liver, lung) | textbook patterns of metastatic spread (e.g. *DeVita, Hellman & Rosenberg's Cancer*; AJCC chapter M-category notes) restricted to anchors the KB can segment (brain and generic bone are not in the KB, so "bone" is represented by spine/hip/sacrum) | **reference** | AJCC 8th (above); large-cohort autopsy/registry summaries, e.g. Budczies et al. 2015 *Oncotarget* "The landscape of metastatic progression patterns across major human cancers": https://doi.org/10.18632/oncotarget.2677 |

## 6. Empirical evidence gathered so far (this cohort)

| finding | where recorded |
|---|---|
| BP and VT tumour masks have zero overlap on ~61 % of series; VT lesions track T stage (9/13 cohorts), BP lesions rarely do | dashboard v14; `stage_correlation_audit_2026-09-16.md` |
| TCGA-LIHC MR NIfTI headers are x-mirrored relative to RAS+ (TS `kidney_right` at patient-left x; anatomy votes unanimous) → header handedness estimated per scan | `planner_first_real_scans_findings_2026-09-20.md`; `header_handedness` log entry in every plan |
| `total_mr` returns partial hips/sacrum at the FOV edge and 0-ml adrenals; lung volumes 180–650 ml on abdominal MR | same |
| VoxTell tumour masks fragment (up to 28 components on one scan) | same; `rater_0_components` in plan diagnostics |

## 7. What the pilot must fill in (`knowledge_base/evidence_template.csv`)

Per structure × modality: VoxTell return rate and anatomical validity per alias, TS reliability (Dice vs VoxTell,
plausibility-flag rate), observed vertebral span (from TS `zmin/zmax` against the frame), observed volume percentiles,
observed landmark-distance windows, and — for `ood` entities — keep/retire decisions. These replace every row marked
**assumption** above; rows marked **reference** should additionally get a radiologist's sign-off before publication.

## 8. Link check (2026-09-21)

Checked against the Crossref API or by direct fetch during writing: 10.3322/caac.21388 (Amin 2017, AJCC 8th overview) ✔ ·
10.1002/ijgo.12749 (FIGO cervix 2018) ✔ · 10.1002/ijgo.14923 (FIGO endometrium 2023) ✔ · 10.18632/oncotarget.2677
(Budczies 2015) ✔ · 10.1007/978-3-319-40618-3 (AJCC 8th book) ✘ registered but withdrawn by Springer — replaced above.
Not re-checked in this session (network limits), cited from the authors' records: 10.1016/j.ijgo.2013.10.001 (FIGO
ovary 2014), 10.1016/j.ejca.2008.10.026 (RECIST 1.1), 10.1148/ryai.230024 (TotalSegmentator), 10.1038/s41592-024-02499-w
(BiomedParse), arXiv 2405.19492 (TotalSegmentator MRI). GitHub / arXiv / Hugging Face links are the canonical project
pages and were opened during authoring. Please report any link that fails; DOIs are the identifiers to prefer.
