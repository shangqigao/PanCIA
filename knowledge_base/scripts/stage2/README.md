# Stage 2: voxel-wise Bayesian tumour segmentation — run guide

Code: `knowledge_base/pancia_kb/stage2/` (library) and `knowledge_base/scripts/stage2/` (these scripts).
Stage 1 is frozen at ledger v8.4 and only read. Outputs go to `clinical/PanCIA_outputs/stage2/`.
The fitted values go next to the code, in `knowledge_base/knowledge_base/stage2/stage2_fit.json`.

## Requirements
Python 3.10 with numpy, scipy, pandas, nibabel and matplotlib. Nothing else is needed (no scikit-learn, no GPU).

Data roots are found in this order:
1. the environment variables `PANCIA_PANCIA`, `PANCIA_CLINICAL`, `PANCIA_TCGA_SEG`, `PANCIA_TCGA_NIFTI` and `PANCIA_R2SEG`;
2. `~/mnt/<name>`;
3. the Mac paths in `pancia_kb/stage2/paths.py`.

The clinical and PanCIA folders are in OneDrive. Make sure they are downloaded ("Always keep on this device"), because cloud-only files are slow to open.

## Order

| step | command | what it does | review |
|---|---|---|---|
| 4 (first, 1 min) | `python s2_selftest.py` | Phantom tests of the variational-Bayes model: recovery, components, Bayes factor for true and false anchors, coupling | must print ALL PASS |
| 2 | `python s2_expert_build.py --n-per 100` | Expert samples from R2Seg (5 collections, never Ovary_Tumor_00) | `stage2/expert/notes*.csv` |
| 3 | `python s2_fit.py` | Fits r, η, coupling, ν and the appearance priors | **review point 1**: `stage2_fit_report.txt` |
| 1 | `python s2_check_inputs.py` | Stage-1 → stage-2 inputs on the TCGA pilot (no model run) | `stage2/check_inputs/summary.txt` |
| 5 | `python s2_validate.py --n-val 20` | Leave-one-collection-out validation on R2Seg (includes 5 fold fits) | **review point 2**: `stage2/validation/report.txt` |
| 6 | `python s2_run_tcga.py` | Stage 2 on the 152-series TCGA pilot (accepted and uncertain lesions) | `stage2/tcga_v1/*.csv` |
| 6 | `python s2_png.py --n 60` | Whole-slice 2×3 panels | `stage2/tcga_v1/png/` |

`run_stage2.sh` runs steps 4 → 2 → 3 → 1 and stops at review point 1. Run `./run_stage2.sh validate` and `./run_stage2.sh tcga` after reviewing.

All steps can be resumed: rerunning skips the cases and series that are already done. To start a step again, move its output folder into `_to_delete/`.

Steps 2, 1 and 6 can be split into parallel shards:
`for i in 0 1 2 3; do SHARD=$i/4 python s2_expert_build.py & done; wait`
Each shard writes its own `*_s<i>` files. Shards need memory: about 3 GB each for R2Seg, and more for large TCGA CTs.

## Design values (disclosed, not fitted)

Model and fit:
- K_max = 4 tumour components, with α₀ = 1/4.
- Normal–Wishart strength: β₀ = 1, ν₀ = D + 2.
- 3 generic background classes.
- L2 penalty of 1e-4 on η.
- Basis knots: signed distance at −10, −5, 0, 5, 10, 20 and 40 mm; depth at 5 and 15 mm.

Image features: smoothed intensity and the local standard deviation at 2 mm (D = 2). MR uses a robust z-score over the body.

Expert simulation:
- Equal weight for the three anatomy scenarios (excl, incl, lost).
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
6. **Bayes factor.** It is computed as a refit on N_L = anchors ⊕ r with **matched flexibility**: the background-only model gets K extra classes with the tumour prior, placed freely.
   - Without matched flexibility, the model with more components always won. A false anchor gave log BF > 0 on the phantom.
   - N_L must include normal tissue around the claim.
7. **The Potts term is used unnormalised** (pairwise penalty). The ELBO for the Bayes factor is that of the model without coupling.
8. **MR regions are not limited by a body mask.** An Otsu body dropped hypo-intense T2 tumours on R2Seg cervix.
