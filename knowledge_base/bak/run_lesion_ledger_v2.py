"""Stage 3 batch: lesion ledger + admission gate for every scan with a VoxTell plan manifest.

  python scripts/run_lesion_ledger.py --seg_root <.../TCGA_Seg> --workers 8 [--overwrite] [--rel_list list.txt]

Writes <seg_root>/Ledger/Radiology/<rel>_lesions.json + _lesions.npz and <seg_root>/Ledger/Radiology/ledger_summary.csv."""
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
    rel, seg_root, overwrite = args
    out = os.path.join(seg_root, 'Ledger', 'Radiology', rel)
    row = dict(rel=rel, project=rel.split('/')[1], modality=rel.split('/')[0], status='ok')
    try:
        if os.path.exists(out + '_lesions.json') and not overwrite:
            led = json.load(open(out + '_lesions.json')); row['status'] = 'skipped'
        else:
            t = time.time()
            led, crops, aff, shp = build_ledger(rel, seg_root, KB)
            save_ledger(led, crops, aff, shp, out)
            row['seconds'] = round(time.time() - t, 1)
        L = led['lesions']
        adm = [l for l in L if l['status'] == 'admitted']
        row.update(primary_visible=led['primary_visible'], n_claims=led['n_claims'], n_dropped=len(led['dropped_claims']),
                   n_lesions=len(L), n_admitted=len(adm),
                   n_primary=sum(l['cls'] == 'primary' for l in adm), n_local=sum(l['cls'] == 'local_invasion' for l in adm),
                   n_distant=sum(l['cls'] == 'distant' for l in adm),
                   primary_ml=round(sum(l['ml'] for l in adm if l['cls'] == 'primary'), 1),
                   primary_reason=';'.join(sorted({l['reason'] for l in adm if l['cls'] == 'primary'})),
                   version=led.get('version', 'v1'),
                   primary_support=';'.join(sorted({str(l.get('support')) for l in adm if l['cls'] == 'primary'})),
                   n_primary_unconfirmed=sum(l['cls'] == 'primary' and not l.get('confirmed', True) for l in adm),
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
    a = ap.parse_args()
    if a.rel_list:
        rels = [l.strip() for l in open(a.rel_list) if l.strip()]
    else:
        base = os.path.join(a.seg_root, 'VoxTell', 'Radiology') + '/'
        rels = sorted(p[len(base):-len('_plan_manifest.json')] for p in glob.glob(base + '*/*/*/*_plan_manifest.json'))
    print(f'{len(rels)} scans')
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.kb,)) as ex:
        for i, r in enumerate(ex.map(one, [(rel, a.seg_root, a.overwrite) for rel in rels], chunksize=4)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f'{i + 1}/{len(rels)}', flush=True)
    out = os.path.join(a.seg_root, 'Ledger', 'Radiology', 'ledger_summary.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('rel', 'project', 'modality', 'status'), k))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    err = [r for r in rows if r['status'] == 'error']
    print(f'done: {len(rows)} scans, errors {len(err)} -> {out}')
    for r in err[:5]:
        print(r['rel'], r['error'])
