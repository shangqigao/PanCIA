"""Stage 2: voxel-wise Bayesian tumour segmentation (main.tex Sec. 'Stage 2', 23 Oct 2026).

Modules
  paths     data roots (Mac defaults, ~/mnt in the Cowork VM, or PANCIA_<NAME> environment variables)
  features  image features y_v (smoothed intensity, local mean / sd at two scales), body mask, MR normalisation
  prior     spatial tumour prior pi_v = sum_h P(h|L) sigma(eta . psi_v(h)) (basis, fitted weights)
  vb        mean-field variational Bayes: Student-t tumour components (sparse Dirichlet) + background classes,
            latent-weight Potts coupling on the tumour indicator, ELBO and the approximate Bayes factor
  expert    expert-data simulation on R2Seg (agent-like anchors and extents around real expert tumours)
  fit       fitting of r, eta, coupling hyperparameters, appearance priors and nu on expert data only
  interface stage-1 (ledger v8.4) -> per-lesion stage-2 inputs (q_v, chosen extent, credible hosts, regions, clusters)
  run       one cluster of lesions end to end (region growth, refined tumour, new candidates, Bayes factor)

Stage 1 (ledger v8.4) is frozen and only read. Priors come from expert data (R2Seg, never Ovary_Tumor_00); TCGA is used for
visual and bug checks only and nothing is tuned on it."""
VERSION = 's2.0'
