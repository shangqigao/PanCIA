TCGA stage 2 on the HPC (24 Nov 2026; step H): accepted lesions only, --no-bf, VB on the A100 (CuPy), fast EDT.

1. Code: git pull PanCIA (needs s2_run_tcga.py --fit/--accepted-only, vb.py GPU path, interface.py keep=, and
   knowledge_base/knowledge_base/stage2/stage2_fit_agent_beyond.json of 23 Nov).
2. Copy from the Mac (~/Datasets/PanCIA_outputs) to $PANCIA_OUTPUTS on the HPC:
     ledger/v86_tcga_all_bp2/{anchors_s*.csv,series_s*.csv}   ledger/v85/extent/fit_psi.json   ledger/v8/testbed/fit_host6.json   ledger/v8/testbed/relational_fit.json
     ledger/v6/grounding/g0/h5/tables_g1.json   ledger/v8/grounding/test_sets.json
   TCGA_Seg on the HPC must hold the SAME inputs stage 1 used on the Mac: BiomedParse = bp2, VoxTell initial + the bp2 plan call
   (plan masks, *_plan_manifest.json), Planner plans of the bp2 rerun, TotalSegmentator. Copy any the HPC lacks from the Mac.
3. Environment (small): conda create -n pancia_s2gpu python=3.10; pip install cupy-cuda12x scipy pandas nibabel pyyaml edt
   (cupy-cuda13x if the GPU nodes have a CUDA 13 driver). Edit env_tcga.sh. Run: bash check_tcga_s2.sh
   Smoke on one GPU node (interactive or a short job), checks that stage 1 is rebuilt identically (checks_s0.csv: P(h|L) / S match):
     PANCIA_GPU=1 PANCIA_FAST_EDT=1 python s2_run_tcga.py --stage1 v86_tcga_all_bp2 --accepted-only --no-bf --tag tcga_bp2_smoke --only TCGA-BLCA
   (stop it after a few series; look at stage2/tcga_bp2_smoke/{lesions,checks,errors}*.csv; move the folder to _to_delete afterwards)
4. Submit: sbatch --array=0-7 run_tcga_s2.slurm  (8 jobs x 1 A100 x K=8 shard processes = 64 shards; RAM ~3 GB per shard).
   Time-out: resubmit the same command; finished series are skipped. Same NTASK x K must be kept for a resubmission.
5. Progress: cat $PANCIA_OUTPUTS/stage2/tcga_bp2/done_s*.txt | wc -l   (of the series with an accepted lesion); errors_s*.csv.
6. Copy $PANCIA_OUTPUTS/stage2/tcga_bp2/ (without maps/ first, it is large) back to the Mac.
