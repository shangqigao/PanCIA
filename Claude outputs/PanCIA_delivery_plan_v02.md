# PanCIA — Project Delivery Plan

*Pan-cancer multi-modal image analysis, built as software that radiologists, pathologists, clinicians and researchers can use.*

| | |
|---|---|
| Owner | Shangqi Gao (Crispin Lab) |
| Version | 0.2 (decisions of 9 Oct recorded, §8) |
| Date | 9 Oct 2026 |
| Status | Research core in progress; product layer not started |

> **Primary principle: easy to use.** A first-time user should get a useful result from one command or one click, without knowing which environment, GPU library or model version is underneath. Everything else in this plan serves that rule.

---

## 1. Progress at a glance

Legend: ✅ done · 🟡 in progress · ⬜ not started · 🧪 prototype exists (research scripts, not yet product-grade)

| # | Workstream | Status | Progress |
|---|---|---|---|
| W1 | Multi-agent segmentation (BP, VT, TS agents) | ✅ outputs for the TCGA/TCIA cohort | `██████████` 100% |
| W2 | Anatomical knowledge base + planner | ✅ KB v0.5.7, planner run on the whole cohort | `█████████░` 90% |
| W3 | Reasoning: ledger (host + logic) + R2 image grounding | 🟡 final run (Ledger_v5) on HPC | `████████░░` 80% |
| W4 | Paper 1: segmentation and reasoning | 🟡 methods and pipeline figure updated; results pending v5 | `██████░░░░` 60% |
| W5 | Anatomical graph (stage 4): the carrier of all features | 🟡 plan agreed (29 Sep), code not started | `██░░░░░░░░` 20% |
| W6 | Agent interface, registry and runner (core of the software) | ⬜ design in this document | `█░░░░░░░░░` 10% |
| W7 | Feature extractors (radiology, pathology, others) as agents | 🧪 `analysis/a03` scripts | `███░░░░░░░` 30% |
| W8 | Integration (aggregators) and prediction | 🧪 `analysis/a04`, `a05` scripts (GNN, survival, signatures) | `███░░░░░░░` 30% |
| W9 | Explanation and reports | 🧪 ledger reasons, R2 evidence, QC figures | `██░░░░░░░░` 20% |
| W10 | User interfaces (local browser app, CLI, viewer plug-ins later) | ⬜ technology chosen | `░░░░░░░░░░` 0% |
| W11 | Packaging, documentation, testing, release | ⬜ (one Docker image for BiomedParse only) | `█░░░░░░░░░` 5% |

**Overall:** the research engine (W1–W4) is close to a first paper. The product layer (W6, W10, W11) is the main gap. It should start now, so that the anatomical graph, feature extraction and prediction are built *as agents* from day one, not ported later.

---

## 2. What users get

Users choose by **what they want to do**, not by which model to run. The four tasks map to the downstream work:

| Task | Typical user | Input | Output they see |
|---|---|---|---|
| **Segment** | radiologist, researcher | CT/MR (DICOM or NIfTI) + cancer type | tumour and organ masks, lesion list (host, class, size), overlay viewer, DICOM-SEG for PACS |
| **Integrate** | researcher, pathologist | segmented scans and/or whole-slide images (WSI), other modalities | per-patient feature table and anatomical/multi-modal graph |
| **Predict** | clinician, researcher | integrated features (or raw data: the earlier steps run automatically) | risk, survival, subtype or signature, with confidence |
| **Explain** | everyone | any of the above | plain-language report: why each lesion was accepted or rejected, which organs are involved, which features drove a prediction, with figures |

Each task works with **defaults and no configuration**. Advanced users can open "Options" to swap any component.

---

## 3. Design principles (easy to use first)

1. **One entry point.**
   - Command line: `pancia segment|integrate|predict|explain`.
   - The same tasks are buttons in the app.
   - No user ever runs an agent script directly.
2. **Zero setup for defaults.**
   - The first run downloads the default agents (container images and weights) and checks the GPU.
   - Users never create conda environments by hand.
3. **Same behaviour everywhere.**
   - One recipe runs unchanged on a laptop, a workstation, an HPC cluster (SLURM + Apptainer) or a hospital server.
   - Only a "where to run" setting changes.
4. **Bring your own model in minutes.** Any model from GitHub, Hugging Face or the user's own code can fill a slot by writing one short manifest file, or by using the wizard.
5. **Plain-language results.** Every output carries a human-readable summary. Errors say what went wrong and how to fix it.
6. **Progressive disclosure.**
   - Defaults are shown first.
   - Options appear on request.
   - Expert parameters live in recipe files, not on the main screen.
7. **Safe by default.**
   - Data stays local, and nothing is uploaded.
   - Every result records which agents, versions and weights produced it (provenance), so it can be reproduced.
8. **Never silent.** Untested or low-evidence results are flagged for review, never dropped or passed off as confident. This follows the "no power → expert review" rule in R2.

---

## 4. Architecture

```
            ┌───────────────────────────────────────────────────────────┐
  Users     │  Local browser app ·   CLI (pancia ...)   ·   Python API    │
            │  3D Slicer / QuPath plug-ins (later)                       │
            └───────────────────────────┬───────────────────────────────┘
                                        │  tasks + recipes
            ┌───────────────────────────▼───────────────────────────────┐
  Core      │  Orchestrator: recipe → steps → run on executor            │
  (light,   │  Case store (standard folder) · Provenance · Reports       │
  no GPU    │  Reasoning: KB + planner + ledger + R2 + anatomical graph  │
  deps)     │  Agent registry (built-in · GitHub · Hugging Face · local) │
            └───────────────────────────┬───────────────────────────────┘
                                        │  standard files in / out
            ┌───────────────────────────▼───────────────────────────────┐
  Agents    │  each in its OWN environment (container image or conda)    │
  (plug-in) │  segmenters: BP · VT · TS · user models                    │
            │  radiology extractors · pathology extractors · others      │
            │  aggregators · predictors · explainers                     │
            └───────────────────────────┬───────────────────────────────┘
                                        │
  Executors │  local (Docker/Podman) · HPC (SLURM + Apptainer) · conda   │
```

### 4.1 Agents: one contract for every model

Every model is an **agent**, declared by a manifest `agent.yaml` that sits next to a thin adapter. The manifest needs no PanCIA internals.

```yaml
name: voxtell
role: segmenter.promptable          # segmenter.tumour | segmenter.anatomy | extractor.radiology |
                                    # extractor.pathology | aggregator | predictor | explainer
version: 1.1
source: hf://<org>/<repo>@<rev>     # or github://user/repo@tag, or local:/path
environment:
  image: ghcr.io/pancia/voxtell:1.1 # preferred; built automatically from `conda:` if missing
  conda: env.yml                    # fallback
  gpu: optional                     # required | optional | none
inputs:  {image: nifti, prompts: text-list}
outputs: {masks: nifti-multi, labels: json}
run: python adapter.py --in {in_dir} --out {out_dir}
license: <licence>                  # shown to the user before first download
test: tests/sample_case             # used by `pancia agent check`
```

- **Isolation:** the core never imports an agent's code. It writes standard inputs to a folder, starts the agent in that agent's environment, and reads standard outputs back. Conflicting dependencies between BP, VT, TS and future extractors therefore cannot clash.
- **Standard I/O types**, a small fixed vocabulary:
  - NIfTI image/mask;
  - DICOM in, DICOM-SEG out;
  - WSI (OpenSlide-readable) plus tile coordinates;
  - feature tables (Parquet/CSV);
  - graphs (`npz`, matching the stage-4 contract);
  - JSON for labels and reports.
- **Adding a model:**
  - `pancia agent add hf://…` or `pancia agent add github://…`;
  - or the app's *Add model* wizard, which asks for role, input and output, then generates the manifest and adapter from a template;
  - then `pancia agent check <name>` runs the agent on a bundled public sample case and reports pass/fail in plain language.

### 4.2 Recipes: what runs, in plain YAML

A recipe lists the steps of a task and which agent fills each slot. Defaults ship with PanCIA. Users copy and edit one only when they want to.

```yaml
recipe: segment-default
task: segment
slots:
  anatomy:   totalsegmentator
  tumour_2d: biomedparse
  tumour_3d: voxtell
  reasoning: pancia-ledger-r2        # built-in, not swappable in v1
outputs: [masks, lesion_table, dicom_seg, report]
```

**Role presets** (one click in the app):

| Preset | Recipe |
|---|---|
| Radiologist | segment + explain |
| Pathologist | WSI features + explain |
| Clinician | predict (runs everything upstream) + explain |
| Researcher | all four, cohort mode |

### 4.3 Case store: one predictable folder per patient

```
<project>/
  cases/<patient>/<study>/<series>/
    input/        original or converted image (never modified)
    agents/<agent>@<version>/   raw agent outputs
    reasoning/    ledger, R2 evidence, final lesions
    graph/        anatomical / multi-modal graph
    features/     per-agent feature tables
    report/       report.html, report.pdf, dicom_seg.dcm
    provenance.json   agents, versions, weights hash, recipe, timestamps
  cohort/         aggregated tables, predictions, dashboards
```

This replaces today's mix of `TCGA_Seg/<Agent>/Radiology/<rel>…` trees, `clinical/PanCIA_outputs` and hard-coded paths. A one-off importer maps the existing TCGA outputs into the new store, so nothing is recomputed.

### 4.4 Executors: the same recipe, anywhere

| Executor | Use | Environment |
|---|---|---|
| `local` | laptop or workstation | Docker/Podman images; conda fallback |
| `slurm` | CSD3 and other HPC | Apptainer images; job arrays; resumable |
| `server` (later) | hospital / lab server with a web front end | containers behind a simple queue |

All runs are **resumable** (finished steps are skipped). This is already how the ledger and VoxTell runners work.

### 4.5 The anatomical graph is the carrier of features

Every downstream step works on one data structure: the **anatomical graph** of a patient.

- **Nodes:**
  - organs and their parts;
  - lesions (from the ledger);
  - later, nodes for other modalities, e.g. a pathology slide node and a clinical node.
- **Edges:** KB relations and observed relations: host, invasion, contact, adjacency, lymphatic and vascular links. Later also temporal edges across studies, via lesion tracking.
- **Features are attached to nodes, never kept as loose tables:**
  - a radiology extractor writes one feature vector per organ or lesion node, using that node's mask;
  - a pathology extractor writes features for slide or region nodes;
  - those nodes are linked to the lesion they sampled (normally the primary tumour node).
- **Aggregators and predictors consume the graph**, so any extractor can be swapped without touching the downstream steps. They are graph models or simpler pooling baselines.
- **Explanations map back onto nodes:** "the prediction relied on the primary lesion and its invasion of organ B".

On disk, a graph file holds the structure, which is fixed once built. Each extractor adds a feature layer keyed by node ID (`features/<agent>@<version>.npz`). Several extractors can therefore coexist on the same graph and be compared.

---

## 5. Workstreams and milestones

Timing is relative (Q1 = the first quarter after this plan is agreed) so it can be adjusted. Each milestone has a single **definition of done (DoD)**.

### M1 — Freeze the research engine (W3, W4) · 🟡 now

- [x] R2 candidate-specific grounding: admission rule, 3:1 trim/grow, recovery cap.
- [x] `em_model='candidate'` made the default; `run_ledger_v4.sh` writes to `Ledger_v5`.
- [x] main.tex methods and pipeline figure updated.
- [ ] Ledger_v5 finished on HPC; compared with v4 by cancer type (admitted / rejected / no-power / trimmed / grown).
- [ ] Decision on the LIHC L2 variant (reference factor / erosion); open cases reviewed (LUSC spine, OV L17, ESCA primary, LIHC L0 liver).
- [ ] Stage analysis rebuilt on v5, as a check only (no tuning on stage).

**DoD:** paper 1 results section written from v5; code tagged `v0.5-research`.

### M2 — Agent contract and runner (W6) · ⬜ Q1

- [ ] Freeze the `agent.yaml` schema and the standard I/O types (§4.1).
- [ ] Wrap the three existing agents (BiomedParse, VoxTell, TotalSegmentator) as manifest + adapter + container image (Docker and Apptainer). Reuse `docker/` and `m_tumor_segmentation.py`.
- [ ] Executors: `local` and `slurm`.
- [ ] `pancia agent list|add|check`.
- [ ] Bundled public sample case (TCIA) for checks and tutorials.

**DoD:** on a clean machine, `pancia agent check voxtell` passes after one install command.

### M3 — `pancia segment` end to end (W1–W3 as a product) · ⬜ Q1

- [ ] Move the planner, VoxTell runner, ledger and R2 behind the orchestrator. The core library is `pancia_kb`, unchanged.
- [ ] Case store and importer for the existing TCGA outputs.
- [ ] Outputs:
  - masks;
  - lesion table (host, class, size, RECIST in the acquisition plane);
  - DICOM-SEG;
  - first plain-language report, including the "no power: expert review" flags.

**DoD:** `pancia segment scan.nii.gz --cancer KIRC` produces the report on a laptop GPU in a reasonable time, and the same command runs on HPC for a cohort list.

### M4 — Anatomical graph and explanation v1 (W5, W9) · ⬜ Q1–Q2

- [ ] Stage-4 Phase A (graph per scan, agreed 29 Sep plan) as a built-in reasoning step.
- [ ] Freeze the **graph contract** (§4.5): node IDs that stay stable across runs, node types, and the feature-layer format that extractors write into. M5 and M6 build on this contract.
- [ ] Phase B loader compatibility for `analysis/a04` / `a05`.
- [ ] Explanation report:
  - lesion → organ relations (primary, invasion, metastasis);
  - why candidates were rejected;
  - evidence figures (the per-candidate posterior figure style).

**DoD:** a radiologist can read one report and see each lesion, its host, its relations and the evidence, without opening any code.

### M5 — Feature extractors as agents, writing onto graph nodes (W7) · ⬜ Q2 (after M4)

- [ ] Radiology extractors write one feature vector per organ or lesion node: handcrafted radiomics, and deep image encoders pooled within each node's mask (Phase C). Port from `analysis/a03`.
- [ ] Pathology extractors:
  - tissue masking, tiling and stain normalisation (port from `a01`);
  - patch encoders as agents, any pathology foundation model from Hugging Face plus user models;
  - slide-level pooling.
- [ ] Slide-level pathology features become a pathology node, linked to the lesion it sampled.
- [ ] Extension point for other modalities (clinical tables, omics) through the same contract: a new node type, or a new feature layer.

**DoD:** the user picks an extractor from a list (or adds one from Hugging Face). Its features appear as a new feature layer on every case's graph.

### M6 — Integration and prediction (W8) · ⬜ Q2–Q3

- [ ] Aggregators as agents that take the graph with its feature layers: graph models (GNN, from `a04`), attention MIL over nodes, and simple pooling baselines.
- [ ] Predictors as agents: survival, phenotype/subtype and signature prediction (from `a05`).
- [ ] Standard evaluation:
  - patient-level splits;
  - baseline-study rule;
  - no leakage across a patient's scans.
- [ ] Explanation of predictions: feature and node attributions mapped back onto the anatomy graph.

**DoD:** `pancia predict --task survival --cohort cohort.csv` trains or applies a model and reports performance with confidence intervals, plus a per-patient explanation.

### M7 — Easy-to-use app (W10) · ⬜ Q2–Q3, starts with M3

**Technology (decided): a local browser app.** The user runs `pancia app`, and a page opens in their browser. Data and computation stay on their machine or their HPC; on HPC the page is reached through an SSH tunnel. There is nothing to install beyond PanCIA itself, and the same app later runs on a lab or hospital server for several users.

**First users: cancer researchers.** So the first version is built around the *Researcher* preset:

- cohort import (a folder or a CSV list);
- batch runs on HPC;
- cohort tables and dashboards;
- per-case drill-down.

The radiologist, pathologist and clinician screens follow, reusing the same pages.

- [ ] The screens follow the four tasks:
  - open data;
  - choose task (or preset);
  - run;
  - review (slice viewer with overlays, lesion table, report).
- [ ] *Options* panel to swap agents per slot; *Add model* wizard.
- [ ] Viewer plug-ins (later): 3D Slicer for radiology, QuPath for pathology, both reading the same case store.
- [ ] Usability rounds:
  - **Round 1:** 3–5 cancer researchers, starting with lab members and collaborators, on public TCGA/TCIA data.
  - **Round 2:** radiologists and pathologists.
  - **Round 3:** clinicians.
  - Each round measures time-to-first-result and error rate, and the top issues are fixed before the next round.

**DoD:**
- **Round 1:** a cancer researcher who is new to PanCIA goes from a cohort folder to a cohort table and per-case reports, without help from us.
- **Later rounds:** a clinician with no command-line experience goes from a DICOM folder to a report in under 15 minutes on first use.

### M8 — Release and community (W11) · ⬜ Q3

- [ ] `pip install pancia`, plus prebuilt images. One-page quick start; one tutorial per task on public TCGA/TCIA data.
- [ ] Documentation site: user guide (task-first), agent author guide, recipe reference.
- [ ] CI:
  - unit tests (KB, planner, ledger, R2 already have tests);
  - an agent contract test per built-in agent;
  - a nightly end-to-end test on the sample case.
- [ ] Versioning: semantic versions for the core; agents pinned by version and weights hash in provenance.
- [ ] Public demo, using public data only.

**DoD:** an external lab installs PanCIA and reproduces a tutorial result without contacting us.

---

## 6. Proposed repository layout (target, reached step by step)

```
PanCIA/
  pancia/                 core package (light dependencies)
    cli/  app/  orchestrator/  executors/  store/  registry/  report/
  pancia_kb/              reasoning library (KB, planner, ledger, R2, graph) — existing, kept as is
  agents/                 one folder per built-in agent: agent.yaml, adapter.py, Dockerfile/env.yml, tests/
    biomedparse/  voxtell/  totalsegmentator/  radiomics/  ...
  recipes/                default recipes and role presets
  analysis/               research scripts (a01–a05), migrated into agents step by step
  documents/              paper and plans
  tests/  docs/
```

**Migration rule:** move code only when a milestone needs it, keep the old entry points working until the replacement passes the same test, and record each move (as done for `clinical/PanCIA_outputs`).

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Agent environments conflict or break | strict isolation (one image per agent); `agent check` on a sample case; pinned versions |
| Model licences / weights cannot be redistributed | the manifest records the licence; weights are downloaded from the original source on first use, after the user accepts |
| GPU memory and runtime vary across sites | per-agent resource hints; CPU fallback where possible; resumable runs; HPC executor |
| Clinicians over-trust outputs | evidence-first reports; no-power and low-evidence flags; "research use only" until validated |
| Scope creep across many tasks | milestones gated by DoD; defaults first, options later |
| Data protection | local processing by default; no telemetry; only public data in demos and docs |
| Single-maintainer bottleneck | agent contract and templates let others add models without touching the core; documented recipes |

---

## 8. Decisions

| # | Question | Decision | Status |
|---|---|---|---|
| 1 | App technology | **Local browser app** (`pancia app`), CLI underneath; Slicer/QuPath plug-ins later | ✅ decided 9 Oct |
| 2 | How agents' environments are packaged | **Containers:** one image per agent, built as Docker images, run with Apptainer on HPC. Conda is a fallback for development only. See 8.1. | 🟡 recommended; confirm when M2 starts |
| 3 | Package name and licence | **`pancia`** (free on PyPI as of 9 Oct 2026). **Apache-2.0** for PanCIA's own code (already the repo's LICENSE). Every agent keeps its own model licence, shown before first download. See 8.2. | 🟡 recommended; confirm before release (M8) |
| 4 | Order after M3 | **Anatomical graph first (M4), then feature extractors (M5).** The graph is the carrier: features attach to its nodes (§4.5). | ✅ decided 9 Oct |
| 5 | First test users | **Cancer researchers**; then radiologists/pathologists; then clinicians | ✅ decided 9 Oct |

### 8.1 Containers in plain terms (decision 2)

**The problem.** BiomedParse, VoxTell, TotalSegmentator and future extractors each need different library versions. Installing them together breaks things, and asking users to manage several conda environments is the opposite of easy.

**A container image is a sealed box** holding one model with exactly the libraries it needs. PanCIA starts the box, gives it the input files and collects the outputs. Users never see the box, and the boxes never interfere with each other.

- **Docker** is the standard way to build these boxes and to run them on laptops and workstations.
- **Apptainer** runs the same boxes on HPC clusters such as CSD3, where Docker is usually not allowed. One Docker image is converted automatically, so we maintain one recipe per agent.
- **Conda** environments stay as a fallback, for development or where containers are unavailable. They are not the default for users.
- **Cost:** a little learning for us when wrapping each agent (a Dockerfile, usually about 20 lines). Users pay nothing and download images on first use.
- **First step (M2):** build the BiomedParse image (`docker/` already has a start), then VoxTell and TotalSegmentator, and check that each runs under Apptainer on CSD3.

### 8.2 Name and licence in plain terms (decision 3)

- **Name:** `pancia` matches the project and the repository and is unclaimed on PyPI. The command is `pancia`; the app is "PanCIA".
- **Licence of our code:** the repository already carries **Apache-2.0**, a permissive licence widely used for medical-imaging software. It allows academic and commercial reuse and includes a patent grant. Keeping it avoids relicensing work.
- **Licences of models are separate.** Each agent's weights and code keep their authors' licence. The manifest records it, and the app shows it before the first download, so PanCIA never redistributes anything it may not.
- **Before the public release:** a short check of third-party code copied into the repo (e.g. the BiomedParse-derived `modeling/` and `pipeline/` folders) for licence notices.

---

## 9. How progress is tracked

- This file is the single plan. The **Progress at a glance** table (§1) and the milestone checklists (§5) are updated at the end of each work session.
- Detailed findings stay in the project notes (`claude/*.md`). Each milestone links to its notes.
- **9 Oct 2026:** decisions 1, 4 and 5 made; 2 and 3 recommended. §4.5 (graph as feature carrier) added; M4, M5 and M7 updated.
- **Next update:** after Ledger_v5 finishes (closes most of M1).
