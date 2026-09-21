"""Batch planning over TotalSegmentator outputs (single-pass: one plan = one VoxTell call per scan).

For every ``*_structures.csv`` under --ts_dir, build a TSOutput (pancia_kb.adapter), run the KB planner and write
``<out_dir>/<img_name>_plan.json`` plus one summary row. Metadata (modality, cancer type, sex) comes from
--meta_csv when given (columns: img_name, modality, cancer_type, sex; extra columns ignored); otherwise modality is
read from the TS ``_labels.json`` and cancer_type / sex are left empty (planner then uses generic tumour templates; sex is
metadata only and never filters anchors). Tumour masks are looked up as ``<tumour_dir>/<img_name>_<tumour_obj>.nii.gz`` when --tumour_dir
is given; --tumour_dir may be repeated (VT first, then BP): the agreement of the raters drives the centroid when
non-empty, else the first non-empty rater. Provides the tumour centroid and a host-organ guess by TS overlap.

Example
  python scripts/run_planner_batch.py --kb knowledge_base --ts_dir /path/TS_out --out_dir /path/plans \
      --meta_csv scan_meta.csv --tumour_dir /path/VT_out --tumour_obj tumor --workers 8
Resumable: existing plan files are skipped unless --overwrite.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, glob, time, traceback
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pancia_kb import KnowledgeBase, Planner
from pancia_kb.adapter import load_ts_output

KB = None
ARGS = None


def _init(kb_path, args):
    global KB, ARGS
    KB = KnowledgeBase.load(kb_path)
    ARGS = args


def img_name_from_csv(p: str, ts_obj: str, ts_dir: str) -> str:
    """img_name = path of the TS table relative to --ts_dir, minus `_<ts_obj>_structures.csv`; nested layouts
    (Radiology/<CT|MR>/<Project>/<series_uid>/<desc>) are preserved so the same relative path resolves the tumour masks."""
    rel = os.path.relpath(p, ts_dir)[:-len('_structures.csv')]
    return rel[:-len(ts_obj) - 1] if ts_obj and rel.endswith('_' + ts_obj) else rel


def meta_lookup(meta: dict, img_name: str) -> dict:
    """Match metadata by full img_name, else by any path component (e.g. the series_uid folder)."""
    if img_name in meta:
        return meta[img_name]
    for part in reversed(img_name.split(os.sep)):
        if part in meta:
            return meta[part]
    return {}


def expected_host_for(kb, meta):
    """Host anchor: explicit `host` column (unseen cancer types / external cohorts) beats the TCGA cancer-type table.
    Cancer type is used only for the tumour phrase; the anatomy is keyed by host, so any anchor in the KB works."""
    h = (meta.get('host') or '').strip()
    if h:
        if h not in kb.entities or not kb.entities[h].get('is_anchor'):
            raise ValueError(f"meta host {h!r} is not a KB anchor; anchors: {sorted(e['id'] for e in kb.anchors())}")
        return h
    return (kb.tumour_prompts['cancer_types'].get(meta.get('cancer_type') or '') or {}).get('host')


def plan_one(job):
    csv_path, meta = job
    t0 = time.time()
    img_name = meta['img_name']
    out_path = os.path.join(ARGS.out_dir, f'{img_name}_plan.json')
    row = dict(img_name=img_name, status='ok', modality=meta.get('modality'), cancer_type=meta.get('cancer_type'), meta_host=meta.get('host'), sex=meta.get('sex'))
    if os.path.exists(out_path) and not ARGS.overwrite:
        row['status'] = 'skipped'
        return row
    try:
        tumour_mask = [os.path.join(d, f'{img_name}_{ARGS.tumour_obj}.nii.gz') for d in (ARGS.tumour_dir or [])]
        tumour_mask = [c for c in tumour_mask if os.path.exists(c)] or None
        ts, diag = load_ts_output(csv_path, tumour_mask=tumour_mask, kb=KB, min_volume_ml=ARGS.min_volume_ml,
                                  expected_host=expected_host_for(KB, meta), cancer_type=meta.get('cancer_type'))
        modality = meta.get('modality') or diag.get('modality') or ('MR' if ts.task == 'total_mr' else 'CT')
        has_vt = bool(tumour_mask) and ARGS.tumour_dir and os.path.dirname(tumour_mask[0]) == ARGS.tumour_dir[0]   # first --tumour_dir = VT
        plan = Planner(KB).plan(ts, modality=modality, cancer_type=meta.get('cancer_type') or None, sex=meta.get('sex') or None,
                                include_tumour_prompt=ARGS.tumour_prompt == 'always' or (ARGS.tumour_prompt == 'if_no_vt' and not has_vt))
        plan.log.insert(0, dict(step='tumour_evidence', evidence=diag.get('tumour_evidence'), voxels=diag.get('tumour_voxels'),
                                host=ts.tumour_host_guess, override=diag.get('host_override'), conflict_ignored=diag.get('host_conflict_ignored'),
                                site=meta.get('site')))
        d = json.loads(plan.to_json())
        d['input'] = dict(img_name=img_name, structures_csv=csv_path, tumour_mask=tumour_mask, modality=modality,
                          cancer_type=meta.get('cancer_type'), sex=meta.get('sex'), kb_version=KB.meta.get('version'),
                          ts_task=ts.task, fov_z_mm=list(ts.fov_z_mm), n_structures=len(ts.structures), diagnostics=diag)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(d, f, indent=1)
        tiers = [p.tier for p in plan.prompts]
        row.update(task=ts.task, frame_method=plan.frame.get('method'), frame_span='-'.join(plan.frame.get('span', []) or []),
                   fov_source=diag.get('fov_source'), n_ts_structures=len(ts.structures), n_dropped_small=len(diag['dropped_small']),
                   n_found=len(plan.anchors_found), n_expected=len(plan.anchors_expected), n_missing=len(plan.anchors_missing),
                   missing=';'.join(f"{m['entity']}{'/' + m['side'] if m.get('side') else ''}" for m in plan.anchors_missing),
                   n_missing_implausible=sum(1 for m in plan.anchors_missing
                                             if any(f['entity'] == m['entity'] and f.get('side') == m.get('side') and not f['plausible'] for f in plan.anchors_found)),
                   n_prompts=len(plan.prompts), n_tier0_ruler=tiers.count(0), n_tier1_completion=tiers.count(1), n_tier2_profile=tiers.count(2),
                   n_budget_dropped=sum(len(l['dropped']) for l in plan.log if l.get('step') == 'budget'),
                   n_ood=sum(1 for p in plan.prompts if KB.entities[p.entity].get('coverage_tier') == 'ood'),
                   host=ts.tumour_host_guess, hosts=';'.join(f"{h['entity']}:{h['cls'][:3]}:ev{h.get('evidence')}:w{h.get('weight')}" for h in plan.hosts), n_secondary_hosts=sum(h['role'] == 'secondary' for h in plan.hosts), tumour_evidence=diag.get('tumour_evidence'), tumour_voxels=diag.get('tumour_voxels'),
                   overlap_host=diag.get('overlap_host'), overlap_host_frac=diag.get('overlap_host_frac'), host_override=diag.get('host_override'),
                   n_plausibility_flags=sum(1 for l in plan.log if l.get('step') == 'found_plausibility'),
                   seconds=round(time.time() - t0, 2))
    except Exception as e:  # keep the batch going; record the failure
        row.update(status='error', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-800:])
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kb', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'knowledge_base'))
    ap.add_argument('--ts_dir', required=True, help='root containing *_structures.csv (searched recursively)')
    ap.add_argument('--ts_obj', default='organ', help='seg_obj suffix used when TS was run (strip it to get img_name)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--meta_csv', default=None, help='key column img_name OR series_uid; then modality, cancer_type [, host, sex, site]; host = KB anchor id, required for cancer types not in tumour_prompts.yaml')
    ap.add_argument('--tumour_dir', action='append', default=None, help='repeatable; order = rater priority (VT, then BP)')
    ap.add_argument('--tumour_obj', default='tumor')
    ap.add_argument('--tumour_prompt', choices=['never', 'if_no_vt', 'always'], default='if_no_vt',
                    help='add the tumour phrase to the anatomy call; default only when the scan has no initial VT tumour mask')
    ap.add_argument('--min_volume_ml', type=float, default=0.05)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    meta = {}
    if args.meta_csv:
        with open(args.meta_csv, newline='') as f:
            for r in csv.DictReader(f):
                meta[r.get('img_name') or r['series_uid']] = r
    # enumerate TS tables: from the metadata (site/series_uid → one folder each; fast on network mounts) when possible,
    # else a recursive glob over --ts_dir
    csvs, missing_dirs = [], 0
    if meta and all(('site' in r or ('modality' in r and 'cancer_type' in r)) and 'series_uid' in r for r in meta.values()):
        for r in meta.values():
            site = r.get('site') or f"{r['modality']}/{r['cancer_type']}"
            site = site.split('Radiology/')[-1]                     # site is stored as Radiology/<CT|MR>/<project>
            d = os.path.join(args.ts_dir, site, r['series_uid'])
            if not os.path.isdir(d):
                missing_dirs += 1; continue
            csvs += sorted(glob.glob(os.path.join(d, f'*_{args.ts_obj}_structures.csv')))
        print(f'{len(csvs)} TS tables located from metadata; {missing_dirs} series without a TS folder', flush=True)
    else:
        csvs = sorted(glob.glob(os.path.join(args.ts_dir, '**', '*_structures.csv'), recursive=True))
    if args.limit:
        csvs = csvs[:args.limit]
    jobs = []
    for p in csvs:
        name = img_name_from_csv(p, args.ts_obj, args.ts_dir)
        m = dict(meta_lookup(meta, name)); m['img_name'] = name
        jobs.append((p, m))
    print(f'{len(jobs)} TS outputs; {sum(1 for _, m in jobs if len(m) > 1)} with metadata', flush=True)

    kb = KnowledgeBase.load(args.kb)
    errs = kb.validate()
    if errs:
        sys.exit('KB invalid:\n' + '\n'.join(errs))

    rows = []
    with Pool(args.workers, initializer=_init, initargs=(args.kb, args)) as pool:
        for i, row in enumerate(pool.imap_unordered(plan_one, jobs, chunksize=4), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(jobs):
                n_err = sum(r['status'] == 'error' for r in rows)
                print(f'  {i}/{len(jobs)} done, {n_err} errors', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != 'img_name', k))
    summ = os.path.join(args.out_dir, 'planner_batch_summary.csv')
    with open(summ, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r['status'] == 'ok']
    if ok:
        import statistics as st
        print(f"summary -> {summ}\n  frames: " + ', '.join(f"{m}={sum(1 for r in ok if r['frame_method'] == m)}" for m in sorted({r['frame_method'] for r in ok}))
              + f"\n  prompts/scan median {st.median(r['n_prompts'] for r in ok)}, missing anchors/scan median {st.median(r['n_missing'] for r in ok)}")
    print(f"  errors: {sum(r['status'] == 'error' for r in rows)}, skipped: {sum(r['status'] == 'skipped' for r in rows)}")


if __name__ == '__main__':
    main()
