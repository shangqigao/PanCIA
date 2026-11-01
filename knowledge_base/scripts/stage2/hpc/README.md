# validate_agents_v3 on a server (no SLURM)

Same run as `./run_stage2.sh validate_agents_v3` on the Mac (same 423 R2Seg scans, true-lesion prior with "in an outline beyond xi",
per-site held-out psi), split into NSHARD single-thread processes. Outputs land in
`$PANCIA_CLINICAL/PanCIA_outputs/stage2/validation_agents_v3/` (rows_s<i>.csv, calibration_s<i>.csv, report.txt).

1. Code (Mac): commit and push, including the new files (git status shows them as untracked):
   knowledge_base/pancia_kb/stage2/agents.py, knowledge_base/scripts/stage1/, knowledge_base/scripts/stage2/{s2_agent_build.py,
   s2_prior_calibration.py, s2_validate_agents.py, hpc/}. Server: git pull.
2. Fitted inputs (Mac -> server): clinical/hpc_bundle_v3.tgz (51 KB: fit_host6.json, fit_psi.json, the usability index, the 9 fold fits
   of folds_agent_beyond). On the server: cd $PANCIA_CLINICAL && tar xzf hpc_bundle_v3.tgz
3. Data on the server: R2Seg/<collection>/{images,labels} and R2Seg_Seg/{BiomedParse,VoxTell,TotalSegmentator}/<collection>.
   TotalSegmentator (1.7 GB) may only be on the Mac: rsync -a R2Seg_Seg/TotalSegmentator <server>:<R2Seg_Seg>/   (Uterus_Tumor_00 is not used)
4. Edit env.sh (four paths, conda env).
5. bash check_hpc.sh       (paths, files, packages, code version, agent masks per site; computes nothing)
6. NSHARD=32 nohup bash run_server.sh > logs/run_server.out 2>&1 &
   (survives logout; NSHARD defaults to min(cores - 2, free RAM GB / 3); ~2-3 GB per shard). v2 was ~89 CPU-h, single scans up to
   ~1 h: 32 shards ~3-5 h, 16 shards ~6-8 h.
7. bash progress.sh        (finished scans of 423, errors, running shards); logs/v3_s<i>.txt per shard
8. bash stop_server.sh     (stops all shards). Rerun step 6 to resume (any NSHARD; finished scans are skipped).
9. Copy validation_agents_v3/ back to the Mac's clinical/PanCIA_outputs/stage2/.
