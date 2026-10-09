"""Stage 3 batch: lesion ledger + admission gate for every scan with a VoxTell plan manifest.

  python scripts/run_lesion_ledger.py --seg_root <.../TCGA_Seg> --workers 8 [--overwrite] [--rel_list list.txt]
                                     [--out_name Ledger] [--params '{"r1": false, "r3": false}']
                                     [--img_root <.../TCGA_NIFTI>] [--shard i --nshard n]

v4 (R2 = calibrated EM, pancia_kb/r2_em.py) needs the images: --img_root (default <seg_root>/../TCGA_NIFTI). --shard/--nshard
split the scan list for a SLURM array (each shard writes ledger_summary_<i>of<n>.csv); the run is resumable (existing outputs
are skipped unless --overwrite).

Writes <seg_root>/<out_name>/Radiology/<rel>_lesions.json + _lesions.npz and <seg_root>/<out_name>/Radiology/ledger_summary.csv.
--out_name (default Ledger) lets a new rule version be written beside the current ledger for comparison; --params overrides
ledger parameters (e.g. '{"r1": false, "r3": false}' reproduces v2 rules)."""
import argparse, csv, glob, json, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pancia_kb import KnowledgeBase
from pancia_kb.ledger import build_ledger, save_ledger

KB = None


def init(kb_dir):
    global KB
    KB = KnowledgeBase.load(kb_dir)


def one(args):
    rel, seg_root, overwrite, out_name, params = args
    out = os.path.join(seg_root, out_name, 'Radiology', rel)
    row = dict(rel=rel, project=rel.split('/')[1], modality=rel.split('/')[0], status='ok')
    try:
        if os.path.exists(out + '_lesions.json') and not overwrite:
            led = json.load(open(out + '_lesions.json')); row['status'] = 'skipped'
        else:
            t = time.time()
            led, crops, aff, shp = build_ledger(rel, seg_root, KB, params)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            save_ledger(led, crops, aff, shp, out)
            row['seconds'] = round(time.time() - t, 1)
        L = led['lesions']
        adm = [l for l in L if l['status'] == 'admitted']
        r2 = led.get('r2') or {}
        row.update(r2_converged=r2.get('converged'), r2_iterations=r2.get('iterations'), r2_feasible=r2.get('n_feasible'),
                   r2_accepted=r2.get('n_accepted'), r2_rejected=r2.get('n_rejected'), r2_recovered_lesions=r2.get('n_recovered_lesions'),
                   r2_rejected_lesions=r2.get('n_rejected_lesions'), r2_recalled_ml=r2.get('recalled_ml'),
                   r2_note=r2.get('note') or r2.get('error'))
        row.update(primary_visible=led['primary_visible'], n_claims=led['n_claims'], n_dropped=len(led['dropped_claims']),
                   n_lesions=len(L), n_admitted=len(adm),
                   n_primary=sum(l['cls'] == 'primary' for l in adm), n_local=sum(l['cls'] == 'local_invasion' for l in adm),
                   n_distant=sum(l['cls'] == 'distant' for l in adm),
                   primary_ml=round(sum(l['ml'] for l in adm if l['cls'] == 'primary'), 1),
                   primary_reason=';'.join(sorted({l['reason'] for l in adm if l['cls'] == 'primary'})),
                   version=led.get('version', 'v1'),
                   primary_support=';'.join(sorted({str(l.get('support')) for l in adm if l['cls'] == 'primary'})),
                   n_primary_unconfirmed=sum(l['cls'] == 'primary' and not l.get('confirmed', True) for l in adm),
                   n_primary_vtsilent=sum(l['cls'] == 'primary' and l.get('support') == 'BP_vtsilent' for l in adm),
                   n_attached=sum(l.get('assign') == 'attached' for l in adm),
                   secondary_hosts=';'.join(sorted({f"{l['host']}:{l['cls'][:3]}" for l in adm if l['cls'] != 'primary'})),
                   extends_into=';'.join(sorted({k for l in adm for k in (l.get('extends_into') or {})})),
                   n_organ_like=sum(l['organ_like'] for l in adm),
                   drop_reasons=';'.join(f"{k}={v}" for k, v in sorted(_count(d['reason'] for d in led['dropped_claims']).items())),
                   reject_reasons=';'.join(f"{k}={v}" for k, v in sorted(_count(l['reason'] for l in L if l['status'] != 'admitted').items())))
    except Exception as e:
        row.update(status='error', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-800:])
    return row


def _count(xs):
    c = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg_root', required=True)
    ap.add_argument('--kb', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'knowledge_base'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--rel_list', default=None, help='optional text file with one <Mod>/<project>/<uid>/<series> per line')
    ap.add_argument('--out_name', default='Ledger', help='output folder under seg_root (default Ledger)')
    ap.add_argument('--params', default=None, help='JSON dict of ledger parameter overrides')
    ap.add_argument('--img_root', default=None, help='image NIfTI root for R2 (default <seg_root>/../TCGA_NIFTI)')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    a = ap.parse_args()
    if a.rel_list:
        rels = [l.strip() for l in open(a.rel_list) if l.strip()]
    else:
        base = os.path.join(a.seg_root, 'VoxTell', 'Radiology') + '/'
        rels = sorted(p[len(base):-len('_plan_manifest.json')] for p in glob.glob(base + '*/*/*/*_plan_manifest.json'))
    rels = rels[a.shard::a.nshard]
    print(f'{len(rels)} scans (shard {a.shard}/{a.nshard})', flush=True)
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.kb,)) as ex:
        prm = json.loads(a.params) if a.params else {}
        if a.img_root:
            prm['img_root'] = a.img_root
        for i, r in enumerate(ex.map(one, [(rel, a.seg_root, a.overwrite, a.out_name, prm) for rel in rels], chunksize=4)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f'{i + 1}/{len(rels)}', flush=True)
    out = os.path.join(a.seg_root, a.out_name, 'Radiology',
                       'ledger_summary.csv' if a.nshard == 1 else f'ledger_summary_{a.shard}of{a.nshard}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('rel', 'project', 'modality', 'status'), k))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    err = [r for r in rows if r['status'] == 'error']
    print(f'done: {len(rows)} scans, errors {len(err)} -> {out}')
    for r in err[:5]:
        print(r['rel'], r['error'])
