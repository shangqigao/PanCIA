# Stage-1 refits on real agent outlines (v8.5 items brought forward)

Stage 1 itself (ledger v8.4, g3_run_v84.py) is frozen; these scripts refit stage-1 parameters on R2Seg BiomedParse + VoxTell outputs
(agent-neutral rules: every outline is a candidate on equal terms, no agent identity is used; sites count equally).

## Extent prior psi (29 Oct)
psi(log |xi| / |a|) scores every candidate extent of every anchor (anchor, each outline after lobe splitting, U, U - non-host organs;
merged at Dice >= 0.95); w0 ~ psi, normalised over the anchor's candidates; q_a = sum w0 1[v in xi]; xi_a = argmax w0.
v8.4 psi was fitted on simulated cores (30-70 % of the tumour), so it prefers extents 1.5-2.2x the anchor and discards large correct
outlines (real-agent validation, kidney: chosen extent Dice 0.36 vs union 0.82).

    python s1_extent_rows.py [--n-per 100]   # candidate rows per true anchor + voxel patterns (exact q >= 1/2 Dice); sharded
    python s1_fit_psi.py                     # M1 density ratio / M2 conditional logit, leave one site out, gate vs v8.4 psi

or `cd ../stage2 && ./run_stage2.sh psi_fit`. Outputs: clinical/PanCIA_outputs/ledger/v85/extent/ (extent_rows*, patterns*,
fit_psi.json, fit_psi_report.txt). The stage-2 real-agent steps (agent_fit, validate_agents_v2) use fit_psi.json when it exists
(validation: the psi fold that left the held-out site out). eta_f stays {} (the v8.4 flags are computed inside the TCGA ledger and
are not reproduced on R2Seg; disclosed). The TCGA stage-1 run keeps v8.4 psi until v8.5.
