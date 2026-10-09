# PanCIA — Project Delivery Plan

*Pan-cancer multi-modal image analysis, built as software that radiologists, pathologists, clinicians and researchers can use.*

| | |
|---|---|
| Owner | Shangqi Gao (Crispin Lab) |
| Version | 0.4 (deployment designed: §4.4; decisions 1–2 settled by design) |
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
| W3 | Reasoning: ledger (host + logic) + R2 image grounding | 🟡 final run (Ledger_v5) on HPC; evaluation ready | `████████▌░` 85% |
| W4 | Paper 1: segmentation and reasoning | 🟡 methods, figures and results skeleton done; numbers pending v5 | `███████░░░` 70% |
| W5 | Anatomical graph (stage 4): the carrier of all features | 🟡 plan agreed (29 Sep), code not started | `██░░░░░░░░` 20% |
| W6 | Agent interface, registry and runner (core of the software) | ⬜ design in this document | `█░░░░░░░░░` 10% |
| W7 | Feature extractors (radiology, pathology, others) as agents | 🧪 `analysis/a03` scripts | `███░░░░░░░` 30% |
| W8 | Integration (aggregators) and prediction | 🧪 `analysis/a04`, `a05` scripts (GNN, survival, signatures) | `███░░░░░░░` 30% |
| W9 | Explanation and reports | 🧪 ledger reasons, R2 evidence, QC figures | `██░░░░░░░░` 20% |
| W10 | User interfaces (PanCIA app, CLI, viewer plug-ins later) | ⬜ design settled (§4.4) | `░░░░░░░░░░` 0% |
| W12 | LLM assistant: chat that drives PanCIA through tools | ⬜ idea agreed 9 Oct | `░░░░░░░░░░` 0% |
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
   - Install with one installer or one command.
   - On first use, PanCIA downloads the default agents (their environments and weights) and checks the hardware.
   - Users never create environments, write job scripts or open SSH tunnels.
3. **The app is where you are; the compute is where the data and GPUs are.**
   - The same recipe runs on this computer, a GPU workstation, an HPC cluster or a lab server.
   - The user picks *where to run* once, with a wizard; PanCIA handles the transport (§4.4).
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
  Users     │  Chat assistant (LLM) · Local browser app · CLI · Python API │
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
  Compute   │  this computer · GPU workstation · HPC (SLURM) · lab server │
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

### 4.4 Deployment: one engine, three front doors, any compute

#### Why not a pure web app

A page that runs its model inside the browser tab (as RADAR-web does) is the easiest experience there is, but it only works for one modest model. PanCIA needs:
- several GPU models with conflicting dependencies;
- cohort runs of thousands of scans;
- gigapixel pathology slides;
- data that must stay on site.

The browser cannot carry that load. A hosted web service run by us would mean receiving patient data, which we exclude.

So the design separates **where people work** from **where computation runs**.

```
 FRONT DOORS (same UI code)            ENGINE (Python, one package)          COMPUTE TARGETS
 ───────────────────────────           ─────────────────────────────         ─────────────────────────
 1. PanCIA app  (individual)    ─┐     orchestrator · case store       ┌──▶ this computer (CPU/GPU)
 2. PanCIA server (lab/hospital) ├──▶  reasoning · graph · reports  ───┼──▶ GPU workstation (SSH)
 3. Python / CLI / MCP (power     │     agent registry                  ├──▶ HPC cluster (SSH + SLURM)
    users, scripts, LLM)         ─┘                                     └──▶ lab server queue
```

#### Front doors

| Front door | Who | What the user does | When |
|---|---|---|---|
| **PanCIA app** | individual researchers first, later clinicians | **First release:** one-line installer, then `pancia app`; the UI opens in the browser and is served by a small engine on their own computer. **Later:** a double-click desktop installer (Windows/macOS/Linux) wrapping exactly the same UI. | M7 |
| **PanCIA server** | a lab or hospital | An administrator installs it once on a site server. Everyone else just opens a URL, which is the RADAR-web experience, with the data staying inside the institution. Adds user accounts and a shared job queue. | after M8 |
| **Python / CLI / MCP** | power users, pipelines, LLM assistant (§4.6) | `pancia …` commands, `import pancia`, or `pancia mcp` | from M3 |

All three use **one engine and one UI codebase**, so every feature is built once.

#### Compute targets: "where to run"

The engine sends each step to a compute target chosen in a one-time wizard:

| Target | Setup by user | How PanCIA uses it |
|---|---|---|
| This computer | none | runs light steps (reasoning, graph, reports, viewer) always; runs GPU agents if a suitable GPU is detected |
| GPU workstation | host name and login, entered once | runs agents over SSH; results synced back |
| HPC cluster (e.g. CSD3) | host, login and account, entered once; PanCIA checks the connection and the quota | writes and submits SLURM jobs itself, monitors them, resumes after time-outs, and pulls back the small results (tables, previews, reports) |
| Lab server | URL, entered once | submits to the server's queue |

**Automatic routing.** The app picks the target that can run each step, and explains its choice in plain words. Example: "This Mac has no CUDA GPU; VoxTell will run on CSD3 (estimated 40 min for 120 scans)." The user can override it.

**Data stays where it is.** A cohort already on HPC is analysed on HPC. Only small results and image previews come back, so a researcher can browse cases on a laptop without copying terabytes.

#### Agent environments, chosen per target (replaces the earlier "Docker everywhere" idea)

| Target | How each agent's isolated environment is provided | Why |
|---|---|---|
| This computer, workstation | **Auto-managed environments**: PanCIA creates one locked environment per agent from its manifest, via conda-forge/PyPI with lock files; no admin rights needed | Docker Desktop needs admin rights and a separate install, and is often blocked on hospital and university PCs. |
| HPC | **Apptainer images**, the HPC standard, or the same auto-managed environments where Apptainer is unavailable | Docker is usually not allowed on shared clusters. |
| Lab server | Apptainer or Docker, as the administrator prefers | The server has an administrator. |

The **agent manifest stays the same** (§4.1): it declares an environment *spec*, and the runner turns it into whatever the target supports. Agent authors write one spec; users never see this layer.

#### Viewers inside the UI

- Radiology: a web-based slice viewer for NIfTI/DICOM with mask overlays.
- Pathology: a tiled deep-zoom viewer for slides.
- Both are open-source viewer components, chosen in M7 by licence (permissive only) and by performance on remote previews.
- 3D Slicer and QuPath plug-ins come later, for experts who live in those tools.

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

### 4.6 LLM assistant: talk to PanCIA

Users can type what they want, and an LLM operates PanCIA for them.

> "Segment all KIRC CT scans in this folder, then show me which lesions were rejected and why."

**How it works.** The LLM never does the analysis itself. It only calls PanCIA's **tools**, the same functions the app buttons call, and explains the results back.

```
 user ──chat──▶ LLM ──tool calls──▶ PanCIA tools ──▶ orchestrator / agents / reasoning
      ◀─answer─      ◀──results────               (numbers, tables, figures, report links)
```

- **Tools.** A small set of well-described functions, for example:
  - `import_cohort`, `list_cases`, `list_agents`;
  - `run_task(task, cases, recipe)`, `job_status`;
  - `get_lesions`, `get_graph`, `explain_case`, `compare_runs`, `make_figure`;
  - `add_agent(source)`.
- **Standard connection.** The tools are published through the **Model Context Protocol (MCP)**, an open standard for connecting LLMs to tools. One `pancia mcp` server then works with:
  - the chat panel inside the PanCIA app;
  - Claude (desktop app, Cowork, Claude Code), i.e. what we are doing in this project today, made official;
  - other MCP-capable clients and open-weight models.
- **Choice of LLM.** This is a setting. The table below gives the options.

| LLM option | When to use it |
|---|---|
| Local open-weight model on the user's GPU or HPC (e.g. served with Ollama or vLLM; the repo already has a Qwen MLLM environment) | default whenever data must not leave the site |
| Cloud LLM API | only for public data, or where the institution has approved it; by default only tool results (text and numbers) are sent, never images |

- **Guardrails (easy to use *and* safe):**
  - every number in an answer comes from a tool result, and the answer links the case, run or report it came from;
  - long or irreversible actions show a confirmation card before they run, e.g. starting a 3,000-scan HPC run, installing a new agent, or deleting results;
  - every tool call the assistant makes is logged in the case provenance;
  - the assistant states uncertainty and "no power: expert review" flags, and never gives treatment advice;
  - the assistant is optional: everything it does can also be done with buttons.

---

## 5. Workstreams and milestones

Timing is relative (Q1 = the first quarter after this plan is agreed) so it can be adjusted. Each milestone has a single **definition of done (DoD)**.

### M1 — Freeze the research engine (W3, W4) · 🟡 now

- [x] R2 candidate-specific grounding: admission rule, 3:1 trim/grow, recovery cap.
- [x] `em_model='candidate'` made the default; `run_ledger_v4.sh` writes to `Ledger_v5`.
- [x] main.tex methods and pipeline figure updated.
- [x] Code committed and pushed (9 Oct).
- [x] Evaluation prepared (9 Oct): `ledger_r2_compare.py` and `run_v5_eval.sh`, tested on the pilot. The R2 baseline is **v3**: v5's pre-R2 ledger equals v3, while the local `Ledger_v4` is run 1, whose logic differs.
- [ ] Ledger_v5 finished on HPC and synced; `run_v5_eval.sh` run; compared with v3 by cancer type (admitted / rejected / no-power / trimmed / grown).
- [x] LIHC L2 variant: **not adopted** (9 Oct). The current R2 is kept, and LIHC MR L2 is reported as a known limitation (thick-slice MR, partial volume at the liver dome).
- [ ] Open cases reviewed on v5 (LUSC spine, OV L17, ESCA primary, LIHC L0 liver).
- [ ] Stage analysis rebuilt on v5, as a check only (no tuning on stage).
- [x] Paper: Experiments and Results skeleton with placeholders, the grounding figure, and fixes to the bibliography and the missing figure (9 Oct).
- [ ] Prior ablation (flat π, no recovery cap) on the same series.

**DoD:** paper 1 results section written from v5; code tagged `v0.5-research`.

### M2 — Agent contract and runner (W6) · ⬜ Q1

- [ ] Freeze the `agent.yaml` schema and the standard I/O types (§4.1). The environment is declared as a spec, and the runner materialises it per target (§4.4).
- [ ] Wrap the three existing agents (BiomedParse, VoxTell, TotalSegmentator) as manifest + adapter + container image (Docker and Apptainer). Reuse `docker/` and `m_tumor_segmentation.py`.
- [ ] Compute targets: `this computer` (auto-managed environments) and `HPC` (SSH + SLURM + Apptainer or managed environments). Verify on CSD3 which container runtime is available.
- [ ] `pancia agent list|add|check`.
- [ ] Bundled public sample case (TCIA) for checks and tutorials.

**DoD:** on a clean machine, `pancia agent check voxtell` passes after one install command, both locally (if a GPU is present) and on CSD3 through the HPC target.

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

**Design (§4.4).** The PanCIA app is installed on the user's computer:
- **first release:** one-line installer, then `pancia app`, which opens in the browser;
- **later:** a desktop installer wrapping the same UI.

Heavy steps go to compute targets (workstation, HPC) set up once in a wizard, so there are no SSH tunnels or job scripts for the user. The same UI becomes **PanCIA server** for labs later.

- [ ] **Connect-to-HPC wizard:** host, login and account; connection and quota test; then SLURM submission, monitoring and result sync handled by the engine.
- [ ] **Automatic routing** of each step to a capable target, with a plain-language explanation and an estimated time.

**First users: cancer researchers.** So the first version is built around the *Researcher* preset:

- cohort import (a folder or a CSV list);
- batch runs on HPC;
- cohort tables and dashboards;
- per-case drill-down.

The radiologist, pathologist and clinician screens follow, reusing the same pages.

- [ ] **Lessons from RADAR-web** (a browser app for a generalist CT model, liked for its simplicity):
  - open a page and drop a DICOM zip, folder or NIfTI;
  - the right series is picked automatically, with scouts and short series skipped;
  - models download once and are cached;
  - a "research use only" banner;
  - images never leave the machine.

  PanCIA's heavy agents run on a compute target, not inside the tab, but the user-facing flow should feel the same.
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

### M7b — LLM assistant (W12) · ⬜ starts after M3; grows with each milestone

1. [ ] **Tools first.** Expose the Python API from M2/M3 as about 10 MCP tools, with clear descriptions, typed inputs and outputs, and confirmation flags on heavy actions (`pancia mcp`).
2. [ ] **Quick win, no UI work.** Connect the MCP server to Claude (Cowork / Claude Code) and run the researcher tutorials by chat on public TCGA data.
3. [ ] **Chat panel in the browser app**, using the same tools with a selectable LLM backend (local open-weight model by default). Answers show inline tables and figures and link to the viewer.
4. [ ] **Evaluation with cancer researchers:**
   - fixed scripted tasks;
   - measure task success, wrong-tool calls, unsupported claims (numbers not from a tool) and time saved against using the buttons.
5. [ ] Add tools as new milestones land: graph (M4), features (M5), prediction (M6).

**DoD:** a cancer researcher completes the round-1 tutorial tasks by chat alone, every number in the answers traces to a tool result, and no heavy action runs without confirmation.

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
| Personal computers forbid admin installs (Docker) | managed per-agent environments that need no admin rights; heavy steps routed to HPC |
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
| 1 | App technology | **One engine, three front doors** (§4.4): the PanCIA app (browser UI served locally, then a desktop installer), PanCIA server later, and Python/CLI/MCP. Heavy compute goes to targets chosen once in a wizard. | ✅ designed 9 Oct (delegated to Claude; principle: easy to use) |
| 2 | How agents' environments are packaged | **Per target:** auto-managed locked environments on personal computers; Apptainer (or managed environments) on HPC; Docker optional on servers. One environment spec per agent. See 8.1. | ✅ designed 9 Oct |
| 3 | Package name and licence | **`pancia`** (free on PyPI as of 9 Oct 2026). **Apache-2.0** for PanCIA's own code (already the repo's LICENSE). Every agent keeps its own model licence, shown before first download. See 8.2. | 🟡 recommended; confirm before release (M8) |
| 4 | Order after M3 | **Anatomical graph first (M4), then feature extractors (M5).** The graph is the carrier: features attach to its nodes (§4.5). | ✅ decided 9 Oct |
| 5 | First test users | **Cancer researchers**; then radiologists/pathologists; then clinicians | ✅ decided 9 Oct |

### 8.1 Environments in plain terms (decision 2)

**The problem.** BiomedParse, VoxTell, TotalSegmentator and future extractors each need different library versions. Installing them together breaks things, and asking users to manage several environments is the opposite of easy.

**The answer: each agent gets its own sealed environment, and PanCIA builds it.**

- **On a personal computer:** PanCIA creates a separate, locked Python environment per agent, on first use, inside its own folder. No admin rights or extra software are needed. We chose this over Docker for personal computers because Docker Desktop needs admin rights and a separate install, and is often not allowed on hospital and university machines.
- **On HPC:** Apptainer images are the standard on shared clusters, where Docker is usually not allowed. Where Apptainer is missing, PanCIA builds the same managed environments there.
- **On a lab server:** whatever the administrator prefers (Docker or Apptainer).
- **What we write per agent:** one environment spec in its manifest, plus an optional image recipe. PanCIA turns it into the right form for each target.
- **First step (M2):**
  - BiomedParse as the first agent; `docker/` has a start for the image recipe;
  - then VoxTell and TotalSegmentator;
  - test both forms, managed environment and Apptainer, on CSD3.

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
- **9 Oct 2026 (v0.3):** LLM assistant added (§4.6, M7b), with lessons from RADAR-web added to M7.
- **9 Oct 2026 (v0.4):** deployment designed (§4.4): front doors, compute targets, per-target environments. Decisions 1 and 2 settled by design; the user delegated system design with "easy to use" as the only principle.
- **9 Oct 2026:** M1 work while v5 runs: LIHC decision recorded, evaluation scripts ready and tested, results skeleton in main.tex, code pushed.
- **Next update:** after Ledger_v5 finishes (closes most of M1).
