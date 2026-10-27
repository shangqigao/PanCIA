# Stage 2: voxel-wise Bayesian tumour segmentation — run guide

Code: `knowledge_base/pancia_kb/stage2/` (library) and `knowledge_base/scripts/stage2/` (these scripts).
Stage 1 is frozen at ledger v8.4 and only read. Outputs go to `clinical/PanCIA_outputs/stage2/`.
The fitted values go next to the code, in `knowledge_base/knowledge_base/stage2/stage2_fit.json`.

## Requirements
Python 3.10 with numpy, scipy, pandas, nibabel and matplotlib. Nothing else is needed (no scikit-learn, no GPU).

Data roots are found in this order:
1. the environment variables `PANCIA_PANCIA`, `PANCIA_CLINICAL`, `PANCIA_TCGA_SEG`, `PANCIA_TCGA_NIFTI`, `PANCIA_R2SEG` and `PANCIA_R2SEG_SEG` (TotalSegmentator masks on R2Seg);
2. `~/mnt/<name>`;
3. the Mac paths in `pancia_kb/stage2/paths.py`.

The clinical and PanCIA folders are in OneDrive. Make sure they are downloaded ("Always keep on this device"), because cloud-only files are slow to open.

## Order

| step | command | what it does | review |
|---|---|---|---|
| 4 (first, 1 min) | `python s2_selftest.py` | Phantom tests of the variational-Bayes model: recovery, components, Bayes factor for true and false anchors, coupling | must print ALL PASS |
| 2a | `python s2_r2seg_index.py` | Usability of every R2Seg label file (usable / small / no_tumour / normal_only; TS mask present) | `stage2/expert/index/r2seg_index*.csv` |
| 2 | `python s2_expert_build.py --n-per 100` | Spatial-prior samples: up to 100 random usable cases per spatial collection (10); every tumour ≥ 0.5 ml is a simulated lesion; anatomy from TotalSegmentator only | `stage2/expert/notes*.csv` |
| 3 | `python s2_fit.py` | Fits r, η (noisy-OR over a scan's lesions), coupling and ν (sites count equally); no appearance prior (inferred within each scan) | **review point 1**: `stage2_fit_report.txt` |
| 1 | `python s2_check_inputs.py` | Stage-1 → stage-2 inputs on the TCGA pilot (no model run) | `stage2/check_inputs/summary.txt` |
| 5 | `python s2_validate.py --n-val 20` | Leave-one-site-out validation on R2Seg (9 site folds; the two lung collections form one fold; each fold has its own fit). `run_stage2.sh validate` runs 3 shards; cost varies a lot by case: about 2 min for a kidney case, but a large lung tumour ran 119 VB iterations (~1 s each, not yet converged) for one anchor and one scenario on the 4-core VM. Validation is resumable and `--n-val` counts already-done cases, so `N_VAL=10 ./run_stage2.sh validate` gives a first report and a later run with `N_VAL=20` adds the rest | **review point 2**: `stage2/validation/report.txt` |
| 6 | `python s2_run_tcga.py` | Stage 2 on the 152-series TCGA pilot (accepted and uncertain lesions) | `stage2/tcga_v1/*.csv` |
| 6 | `python s2_png.py --n 60` | Whole-slice 2×3 panels | `stage2/tcga_v1/png/` |

`run_stage2.sh` runs steps 4 → 2a → 2 → 3 → 1 and stops at review point 1. The TCGA step uses `NSHARD_TCGA` (default 2) shards,
because a large cluster can need several GB. Run `./run_stage2.sh validate` and `./run_stage2.sh tcga` after reviewing.

All steps can be resumed: rerunning skips the cases and series that are already done. To start a step again, move its output folder into `_to_delete/`.

Steps 2, 1 and 6 can be split into parallel shards:
`for i in 0 1 2 3; do SHARD=$i/4 python s2_expert_build.py & done; wait`
Each shard writes its own `*_s<i>` files; `run_stage2.sh` writes its logs to `clinical/PanCIA_outputs/stage2/logs/`. Shards need memory: about 3 GB each for R2Seg, and more for large TCGA CTs.

## Design values (disclosed, not fitted)

Model and fit:
- K_max = 4 tumour components, with α₀ = 1/4.
- Appearance: within-scan Normal–Wishart priors only (no cross-scan prior), strength β₀ = 1, ν₀ = D + 2.
- 3 generic background classes.
- L2 penalty of 1e-4 on η.
- Basis knots: signed distance at −10, −5, 0, 5, 10, 20 and 40 mm; depth at 5 and 15 mm.

Image features: smoothed intensity and the local standard deviation at 2 mm (D = 2). MR uses a robust z-score over the body.

Expert simulation:
- Weighting: every tumour site counts equally in every estimate (η, r, coupling, ν); a site's weight is shared equally by its cases (25 Oct, user).
- Anatomy for fitting: TotalSegmentator only (25 Oct; before: equal weight for the expert-organ scenarios excl, incl, lost and ts).
- 2000 tumour and 6000 other samples per anchor and scenario.
- Sampling radius of 60 mm.

Run settings:
- At most 5 region growths.
- 300 iterations.
- Convergence: max |Δp| < 1e-3, or {p ≥ ½} unchanged for 10 iterations with mean |Δp| < 1e-4.

Reporting thresholds: the refined tumour is p ≥ ½, the flag is log BF < 0, and k_eff counts components with weight ≥ 0.05 (display only).

## Changes from the main.tex stage-2 text (23 Oct), found while building and testing
1. **Image features.** Intensity plus the local SD at 2 mm, with no local means and no 6 mm scale. On the phantom, the larger scales blurred boundaries and Dice fell from 0.99 to 0.92.
2. **Territory feature.** It has three levels (in host h / in another segmented organ / in none) plus a flag for a knowledge-base region. Credible-host uncertainty enters only through the mixture over hosts.
3. **Background mixing weights.** They are learned per scan per voxel type (Dir(1)) and are not scaled by P(h|L).
4. **The 'exact' outline is left out of the simulation.** No agent draws the expert mask.
5. **q_v is not always 1 on the anchors.** Some candidate extents (U − non-host, and some outlines) do not contain the whole anchor.
6. **Bayes factor.** It is computed as a refit on N_L = anchors ⊕ r within the region, with **matched flexibility**: BOTH models get K extra free background classes with the tumour prior, and the tumour model adds K tumour classes placed by the spatial prior π.
   - The extra classes take the within-scan tumour prior (25 Oct; before: the expert tumour prior).
   - Without extra classes, the model with more components always won. A false anchor gave log BF > 0 on the phantom.
   - With extra classes in the background model only, a true kidney anchor gave log BF < 0 on R2Seg (−0.098 per voxel), because the free classes also explained normal-tissue heterogeneity far from the anchor.
   - With both (current): kidney true anchor +0.058 (ts) / +0.225 (excl) per voxel, false anchor −0.034; phantom T4 +4553, T3 −741.
   - N_L must include normal tissue around the claim. log BF is reported in total and per voxel; it is never exponentiated.
7. **The Potts term is used unnormalised** (pairwise penalty). The ELBO for the Bayes factor is that of the model without coupling.
8. **MR regions are not limited by a body mask.** An Otsu body dropped hypo-intense T2 tumours on R2Seg cervix.

## Changes of 24 Oct
- **Appearance priors are keyed by sequence family:** CT, MR T2W, MR T1post and MR ADC.
  - **Sources (all R2Seg tumour-labelled collections; expert labels only):**
    - T2W: bladder, cervix, prostate T2, and uterus (normal tissue only);
    - T1post: breast DCE, first post-contrast phase;
    - ADC: prostate.
  - **Not used as tumour:** UMD (Uterus_Tumor_00) is benign uterine myoma. Its uterus labels (1, 2) give normal-tissue statistics only; myomas and nabothian cysts are left out of every class.
  - **TCGA MR series:** the family comes from fixed rules on the dcm2niix sidecar (`pancia_kb/stage2/sequence.py`). Families without expert data (T1 with unknown contrast, T1pre, DWI, STIR, SSFP, subtraction) get the broad prior.
  - **Pilot coverage:** 57 of 89 MR series (64%) have an expert prior.
- **Self-review fixes:**
  - The fit subsample is uniform, and rows keep their sampling weights.
  - The large voxel × class arrays are single precision.
  - Voxel types are the distinct organ-membership rows, with no 62-organ limit.
  - Windows are padded by max(30 mm, r).
- **Option A: TotalSegmentator territories (R2Seg_Seg).** TS masks on R2Seg are used only as anatomy; tumour truth stays expert.
  - TS labels are grouped by knowledge-base entity and side (`kb.ts_to_entity`); lung lobes merge into one lung per side.
  - The true host is the collection's site entity (`expert.SITE_ENTITY`). If TS has no class for it, or none of its voxels are in the crop, the host is unsegmented and its territory is the knowledge-base host region (kb flag). This is the case for Breast and Cervix.
  - The spatial build and the validation now cover all 10 spatial collections. Collections with expert organ labels keep the 'excl/incl/lost' scenarios and add 'ts'; the go/no-go uses 'ts', which matches TCGA (TS anatomy).
  - Label files stored with a scale factor (Prostate) are read as integers (`expert._read_labels`).
- **Bayes factor with matched flexibility on both models** (see design value 6).
- **Smoke (24 Oct, one case each, `partial` core):** kidney ts Dice 0.978, log BF/voxel +0.058, false anchor −0.034; bladder ts +0.064, false anchor +0.008 (not flagged); breast (KB host) Dice 0.953, +0.076. The false-anchor flag rate over folds is part of review point 2.
- **Self-review of option A (24 Oct, evening):**
  - Simulated outlines never equal the expert mask. Breast (KB host) had 'host10' and 'organ' outlines identical to the expert mask, so they are now only built when the host is segmented. 'part80' kept every voxel of thin tumours because many voxels share the minimum depth; it is now the deepest 80% by rank, and any outline identical to the mask is dropped.
  - TotalSegmentator masks are checked against the image grid (shape and affine); on a mismatch the 'ts' scenario is skipped for that case with a warning.
  - Validation folds are tumour sites, so Lung_Tumor_00 and Lung_Tumor_01 are held out together. Folds run in parallel shards and `--summary` merges them.

## Changes of 25 Oct (user decisions after review point 1, first attempt)
- **Bug fixed:** the spatial build read the appearance pass's resume list (`done_app*.txt`) as its own, and skipped 406 of 930 cases. It now reads only `done(_s<i>).txt`.
- **Case selection:** up to 100 cases per collection drawn at random from the **usable** cases (largest expert tumour ≥ 0.5 ml; TotalSegmentator mask present), via a usability index of every label file (`s2_r2seg_index.py`). Before, unusable files counted against the 100, so Liver, Pancreas and Prostate had about 50 each. The appearance pass uses the same rule per collection and sequence (Prostate T2 and ADC separately).
  - Index on 25 Oct: 4206 files; usable 3404 (3316 for the spatial prior); no tumour label 418 (Liver 83, Pancreas 219, Prostate 113); Uterus benign 300; small < 0.5 ml 73. Minimum size 0.5 ml kept (user).
- **Fitting uses TotalSegmentator anatomy only** (territories and the organs the simulated outlines are drawn around), as on TCGA. Expert organ labels are used only in validation ('excl'). The 'incl' and 'lost' scenarios are dropped.
  - Reason: expert label maps are exclusive, so the tumour is never inside the expert organ, while the TotalSegmentator host contains 34–100% of it (18 cases checked). Fitting on expert organs taught the prior an anatomy convention TCGA does not have.
  - The expert host is no longer required, so the 11 Lung_Tumor_01 cases without one are usable.
- **Tumour-free scans as negatives (evaluation only):** in each held-out site with such scans (Liver, Pancreas, Prostate T2, Lung_Tumor_01), up to `--n-val` scans get a false anchor in the TotalSegmentator host, its volume drawn from the fold's partial cores. The report gives the flag rate per site.
- **Positive-case false anchors** are placed in the TotalSegmentator host when it is segmented, otherwise in the expert host.
- **Sites count equally (user: pan-cancer, no site bias).** Each tumour site has weight 1/(number of sites), shared equally by its contributing cases: case weights in the η regression, a weighted 95th percentile for r, weighted moments for (a_ω, b_ω), a weighted median for ν, and weighted means and between-case covariances in every appearance pool. The two lung collections form one site. The fit report lists the cases per site.
- **Self-review for site bias (25 Oct):**
  - The η regression's column standardisation uses the fit weights, so the L2 penalty's scale no longer depends on which site has more rows.
  - The validation report is site-weighted throughout: summary means (per-site means in `summary_per_site.csv`), the go/no-go (a site-weighted paired Dice difference, with the per-site values and the number of sites improved), Bayes-factor flag rates (per site and site-weighted, for true anchors, false anchors in tumour scans, and tumour-free scans), and calibration (per site, then averaged).
  - TCGA panels are sampled with an equal share per cancer type.
  - The fit report marks tumour appearance priors that come from a single site.

## Changes of 25 Oct (evening): appearance is inferred within each scan only (user decision)
- **Reason:** appearance does not transfer across scans. CT varies with contrast phase, MR has no absolute scale and varies with sequence and scanner, and the expert priors came mostly from single sites (cancer-specific). The transferable part is the spatial (shape and location) prior, as in BayeSeg; appearance is inferred per scan.
- **Model (`vb.class_priors`, used by `run.py` and the phantom test):** every class gets a weak Normal–Wishart prior (β₀ = 1, ν₀ = D + 2) from its own voxels in this scan:
  - tumour components: voxels weighted by the stage-1 marginal q;
  - each organ class: its territory, weighted by 1 − q;
  - generic classes: all voxels, weighted by 1 − q.

  The Bayes factor uses the same priors on N_L for both models, including the extra free classes. Tumour and background are told apart by the spatial prior, the territories, the start from q and the coupling.
- **Removed:**
  - `s2_expert_appearance.py` and `s2_mr_sequences.py` (moved to `_to_delete`);
  - the appearance priors in the fit (`fit_appearance`, `_pool`, `run.Priors`).

  The MR sequence family is still classified per series (`sequence.py`), for reporting and stratification only. The solid/hollow class is a validation stratum only.
- **ν (Student-t tail shape):** fitted on the tumour-intensity samples of the spatial build (`nu_samples*.jsonl`), standardised within each case and site-weighted.
- **Phantom with within-scan priors:**
  - T1 Dice 0.918 (0.991 with the old stand-in priors), recall beyond the extent 0.996. The difference is a one-voxel outward shell at the tumour–organ partial-volume boundary: edge voxels with high local SD fit neither the organ class nor the generic classes and go to a spare tumour component.
  - The result is the same for a shared region prior, per-class priors, ν₀ = D, generic classes started from all voxels, or region-wide generic priors. The old 0.991 relied on an arbitrary, extremely broad generic stand-in class.
  - Giving the background K spare classes made it worse (0.88–0.91).
  - The expert validation will show whether a boundary bias exists on real tumours (Dice and the over-segmented fraction against the stage-1 extent).
  - T2 now checks composition on the rim + necrosis tumour (rim-like and necrosis-like components, each weight ≥ 0.05) and sparsity on a homogeneous tumour (one component emptied). Before, it asked for 2–3 components on the rim + necrosis tumour; with within-scan priors, the partial-volume boundaries take components of their own.
  - T3 −404, T4 +4048, T5 unchanged in direction.

## Changes of 25 Oct (night): simulation with several tumours and under-segmenting agents (user decision, review point 1)
- **Why:** the first fit gave a prior of about 0.001 everywhere beyond the agents' outlines. In the simulation, only the largest tumour got a lesion, and 5 of its 6 outlines contained it entirely. The only tumour beyond every outline was other tumours, which were treated as missed by all agents. R2Seg has several tumours (≥ 0.5 ml) in 16% of cases (site-weighted): liver 68%, NSCLC lung 52%, bladder, kidney and prostate 8%, colon 4%.
- **(a) Every tumour ≥ 0.5 ml is a simulated lesion** (`expert.simulate_lesions`), with its own anchor per core type and its own claims. A scan's lesions form one cluster, q = 1 − Π(1 − q_L), as `run.py` combines a stage-1 cluster. Components are counted with 26-connectivity, so diagonal fragments are one tumour; specks under 0.5 ml stay tumour truth without an anchor.
- **(b) Agents can under-segment.** New outlines per anchor:
  - **sub:** a random compact 30–70% of the tumour, joined with the anchor;
  - **trunc:** the tumour cut at a random slice beyond the anchor at one end of the acquisition stack;
  - **core4:** the anchor grown by 4 mm.

  These join part80 (under) and dil4, host10, organ, nb15, lobe (over/leak). The claims of an anchor are a random subset of its outline types, every subset equally likely (`expert.agent_claims`). No rate of agent errors is set. Real v8.4 anchors carry 0–9 outline extents after merging (a sanity check only).
- **(c) A tumour with no proposal:** every simulated tumour is proposed, so no rate of missed tumours is assumed. Far-away tumours found by the image are reported as new candidates.
- **(d) Validation** uses the same simulation and reports, per lesion, recall beyond the union of its candidate extents, by distance band (0–5, 5–10, 10–20, > 20 mm), site-weighted.
- **Fit (`fit.fit_eta_noisy_or`):** η is fitted for π_v = 1 − Π_L(1 − σ(η·ψ_v(L))) over a scan's lesions, as at inference. With one lesion it is the logistic regression (checked: identical coefficients; a two-lesion simulation recovers the true η). The samples store per-lesion q_L and sd_L (`qL`, `sdL`).
- **Prior basis:** q hinges at 0.25, 0.5 and 0.75 (mid-range calibration).
- **Usability index:** 26-connectivity and the number of tumours ≥ 0.5 ml (`n_tumours`).
- **Smoke checks** (scratch root; numbers not meaningful):
  - The build works on 9 sites.
  - Liver case: 3 lesions per core type; lung case: 2.
  - Share of tumour volume beyond every candidate extent: 0–4.5% (cervix 13.7%).
  - Fit: 21.5% of voxel rows have more than one lesion.
  - Validation: per-lesion rows, the 3-lesion cluster inferred jointly.
- **Claims design (user, 25 Oct):** uniform random subsets of the outline types (kept), rather than two agents each over- or under-segmenting with equal probability.
- **Self-review fixes (25 Oct, before the rerun):**
  - The usability index writes fixed columns. Rows for tumour-free and benign files lacked `n_tumours`, which shifted columns and could make the index unreadable.
  - A tumour too small or thin for a core type (cores need ≥ 20 voxels) uses its partial core, or the whole tumour, as that anchor. Every tumour is then proposed in every core group, and small tumours no longer reappear as unproposed tumour beyond every outline at a voxel-size-dependent rate.
  - Checked on the device: index with mixed statuses; build of an 18-tumour liver scan (18 lesions in every core group); noisy-OR fit with up to 18 lesions per voxel.

## 25 Oct (late): review point 1 (multi-lesion build) and ν option A (on hold)
- **Review point 1 passed:** 911 cases, no skips; r = 36.6 mm; coupling Gamma(1.81, 0.69); in-sample calibration good in every bin. Prior beyond every candidate extent ≈ 0.003–0.004 (was 0.001): with uniform claim subsets, 0–1% of tumour lies beyond every extent (breast 3.9%). On those voxels, observed vs fitted: 0–5 mm 0.0068 vs 0.0036; 5–10 mm 0.0010 vs 0.0031.
- **ν option A implemented** (`s2_nu.py`, `fit.fit_tumour_components`, `fit.pooled_nu`, `vb.nu_update`, `learn_nu`):
  - The number of components (1–4) is chosen by the lower bound, because a fixed K = 4 from k-means splits one tissue into copies that are never merged back.
  - Synthetic check: right K, and ν 3 → 3.0, 4 → 4.0, 30 → 29, Gaussian → 86–100.
  - A convergence fix for learned ν: the "segmentation unchanged" rule stopped runs at iteration 11.
- **On hold:** on real tumours, all voxels (smoothed features, so correlated) make the lower bound pick K = 4 almost always, and ν → 100 (tails absorbed by components). With voxels thinned to a 4 mm grid (≈ independent), K = 1–2 and ν mostly large, with one bladder component at ν ≈ 11; but there are only 65–444 voxels per tumour. The same voxel correlation inflates stage 2's own component count. `s2_nu.py` is commented out in `run_stage2.sh`, and the fit falls back to the whole-tumour ν (8.61) until the user decides.

## Tail study (review only, 26 Oct): s2_tail_study.py

Evidence for the intra-tumoral tail prior (step 1) and its model form (step 2). Nothing here enters `stage2_fit.json`.
All usable R2Seg tumour scans of the spatial collections except Breast (no host organ), plus up to 50 tumour-free scans per
collection. Per scan: tumour interior, peritumoral ring (equal voxel count) and far tissue of the same organ; statistics I, D1, D2,
G1 and LoG at sigma 1/2/4 mm; Gaussian vs Student-t at equal n; 1..4-component Student-t (K by lower bound); shared-scale vs
order-specific t on (I, D1, LoG_s1); covariates (noise, spacing, thickness, n). Library: `pancia_kb/stage2/tails.py`.

    cd <PanCIA>/knowledge_base/scripts/stage2
    ./run_tail_study.sh                      # 2 single-threaded low-priority shards, resting half of each scan's time
    NSHARD=4 PAUSE=0 ./run_tail_study.sh     # faster, louder
    ./run_tail_study.sh summary              # tables and report only

Outputs: `clinical/PanCIA_outputs/stage2/tail_study/` (records, fits.csv in the tails2_fits.csv format, components.csv,
form.csv, stats_*.csv, report.txt, samples/). Resumable: rerun the same command after an interruption.

## Tail-weight prior (26 Oct): s2_nu_prior.py, run_nu_prior.sh

The Student-t tail weight nu gets a prior per cell of modality (CT, MR) x the host's tissue class from the knowledge base
(`tissue_class`, KB v0.5.9: parenchymal / hollow / aerated; other classes such as bone, nodes, fat, brain take the pan-cancer
fallback). Data: the tail study's joint-feature components (tumour interiors for the tumour prior, rings for the normal prior;
breast tumour interiors via `s2_tail_study.py --tumour-only`), regions with >= 100 grid voxels, sites equal. Hierarchy:
components within sites within cells within the pan-cancer mean (moment estimates); leave-one-site-out validation against the
single pan-cancer prior; split-half check of how much a scan's own estimate depends on the prior.

    ./run_nu_prior.sh            # breast (200 random scans, 2 low-priority shards), tail-study summary, split-half, fit

Outputs: `knowledge_base/knowledge_base/stage2/nu_prior.json`; `clinical/PanCIA_outputs/stage2/tail_study/nu_prior/`.

## Changes of 26 Oct (night): features of order 0-2 and the tail-weight prior in the model
- **Features (D = 4):** I (normalised intensity), D1a, D1b (in-plane forward differences), LoG1 (in-plane Laplacian of Gaussian,
  1 mm), exactly the tail study's statistics (`features.image_features`, checked identical to `tails.filters`). One Student-t scale
  is shared by the four. Before: smoothed intensity and local SD at 2 mm (D = 2). The within-scan prior strength nu0 = D + 2 = 6.
- **Tail weights:** `s2_fit.py` stores `nu_prior` (`nuprior.fit_prior`: modality x tissue class from the tail study, the excluded
  collections left out in validation folds). `run.py` gives every class a prior on log(1/nu): tumour components the mixture over
  the cluster's credible hosts (P(h|L)), organ classes their own organ's cell, generic classes and the Bayes-factor extra classes
  the normal fallback; `vb.Model(nu_prior=...)` updates every nu by MAP (`vb.nu_map`, once per iteration) and adds the log prior
  to the ELBO. Convergence now also needs nu to settle (relative change < 1 %, `vb.NU_TOL`).
- **Outputs:** per-component nu (`model.nu_tumour`), background nu, and an atypicality map 1/E[u_v] (`atyp`; per lesion its median
  and 90th percentile inside the refined tumour).
- **Self-test:** T2 now checks that the interior of a homogeneous tumour is one component by the lower bound (with derivative
  features a whole homogeneous tumour uses components for its edges at fixed nu = 5); new T6 (nu prior on the phantoms: Dice
  0.999 on both, an emptied component on the homogeneous tumour, Bayes-factor signs kept) and T7 (MAP nu: data win at n = 5000,
  prior wins at n = 50). ALL PASS.
- **Cost:** nu settling adds iterations (R2Seg pancreas smoke: 129 vs 22 iterations, 65 s vs 11 s per fit; same Dice 0.976 vs the
  chosen extent's 0.889). An ECME step (`vb.nu_map_marginal`) converged faster but lost the tumour's boundary layer on the phantom
  (Dice 0.95), so it is not used.
- **No rebuild needed:** the spatial-prior samples do not use the image features; rerun `s2_selftest.py`, `s2_fit.py`, then validation.
- **Shards** of `run_stage2.sh` are single-threaded and low priority.
