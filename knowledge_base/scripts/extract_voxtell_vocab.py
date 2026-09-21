"""Extract VoxTell's training vocabulary and grade the KB's VoxTell phrases against it.

The VoxTell paper (arXiv 2511.11450) lists the training vocabulary in Table 10 ("Summary of anatomical structures",
1078 labels with the number of training volumes per label) and the held-out test classes in Table 7; VoxTell v1.1 was
trained on both (train + test sets), so their union is the vocabulary. `seed/voxtell_vocabulary.csv` is that union
parsed from the PDF (names with nested parentheses — brain arteries, teeth, small veins — were not captured).
The precomputed text-embedding .npz shipped via Hugging Face can be added as a second source. This script (run in the VoxTell environment) does:

  1. find the .npz under the Hugging Face cache (or --npz), dump its keys → seed/voxtell_vocabulary.txt
  2. optionally merge a text file of class names copied from the paper's appendix (--paper_txt, one name per line)
  3. grade every KB entity's voxtell.main and aliases:
        exact      — phrase (case/whitespace/side-normalised) is a vocabulary entry trained on ≥ --min_volumes volumes
        rare       — exact vocabulary entry but trained on fewer volumes (Table 10 counts; e.g. 'left renal vein (1)')
        near       — token-level match: every content token of the phrase appears in one vocabulary entry
                     (e.g. 'left renal vein' vs 'renal vein left'), or the phrase is a vocabulary entry + a side word
        ood        — no vocabulary entry shares its head noun
     → knowledge_base/voxtell_vocab_check.csv  and a summary of tier changes to apply in build_kb.py

  python scripts/extract_voxtell_vocab.py [--npz PATH] [--paper_txt seed/voxtell_paper_classes.txt] [--kb knowledge_base]
"""
from __future__ import annotations
import argparse, csv, glob, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

SIDE = re.compile(r'\b(left|right|bilateral)\b')
STOP = {'the', 'of', 'a', 'an', 'in', 'and', 'or', 'with', 'to'}


SYN = {'vena': 'vein', 'hipbone': 'hip bone', 'oesophagus': 'esophagus', 'tumour': 'tumor', 'transitional': 'transition', 'vertebrae': 'vertebra'}
ROMAN = {'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5', 'vi': '6', 'vii': '7', 'viii': '8'}
GENERIC = {'muscle', 'gland', 'bone'}


def norm(s: str) -> str:
    s = s.lower().replace('_', ' ').replace('-', ' ')
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    w = [SYN.get(t, t) for t in re.sub(r'\s+', ' ', s).strip().split()]
    if len(w) >= 2 and w[-2] == 'segment' and w[-1] in ROMAN:
        w[-1] = ROMAN[w[-1]]
    return ' '.join(w)


def tokens(s: str):
    return [t for t in norm(s).split() if t not in STOP]


def find_npz(explicit):
    if explicit:
        return [explicit]
    roots = [os.environ.get('HF_HOME'), os.path.expanduser('~/.cache/huggingface'), os.environ.get('HF_HUB_CACHE')]
    hits = []
    for r in roots:
        if r and os.path.isdir(r):
            hits += glob.glob(os.path.join(r, '**', '*.npz'), recursive=True)
    return sorted(set(hits))


def load_vocab_from_npz(paths):
    import numpy as np
    vocab = {}
    for p in paths:
        try:
            z = np.load(p, allow_pickle=True)
        except Exception as e:
            print(f'  skip {p}: {e}'); continue
        keys = list(z.files)
        # either one array per class name, or an array of names + an array of embeddings
        names = []
        for k in keys:
            arr = z[k]
            if arr.dtype.kind in ('U', 'S', 'O') and arr.ndim >= 1 and arr.size > 10:
                names += [str(x) for x in arr.ravel().tolist()]
        if not names and len(keys) > 10:
            names = keys
        print(f'  {p}: {len(keys)} arrays, {len(names)} candidate class names')
        for n in names:
            vocab.setdefault(norm(n), n)
    return vocab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', default=None)
    ap.add_argument('--paper_txt', default=None, help='class names copied from the paper appendix, one per line')
    ap.add_argument('--paper_csv', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'seed', 'voxtell_vocabulary.csv'),
                    help='name,train_volumes,source parsed from arXiv 2511.11450 Table 10 (training) + Table 7 (test); default seed file')
    ap.add_argument('--hf_json', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'seed', 'voxtell_v1.1_labels.json'),
                    help='labels.json from huggingface.co/mrokuss/VoxTell embeddings/voxtell_v1.1 (14,194 prompt strings incl. rewrites); default seed file')
    ap.add_argument('--min_volumes', type=int, default=20, help='exact matches trained on fewer volumes than this are graded near (rare), not in_vocab')
    ap.add_argument('--kb', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'knowledge_base'))
    ap.add_argument('--out_vocab', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'seed', 'voxtell_vocabulary.txt'))
    a = ap.parse_args()

    paths = find_npz(a.npz) if a.npz else []
    print(f'{len(paths)} .npz candidates')
    vocab = load_vocab_from_npz(paths) if paths else {}
    counts = {}
    if a.paper_csv and os.path.exists(a.paper_csv):
        with open(a.paper_csv, newline='') as f:
            for r in csv.DictReader(f):
                vocab.setdefault(norm(r['name']), r['name'])
                counts[norm(r['name'])] = int(r['train_volumes']) if r['train_volumes'] else 10**6   # test-set classes: seen at test, count unknown
        print(f'paper vocabulary: {len(counts)} classes from {a.paper_csv}')
    if a.hf_json and os.path.exists(a.hf_json):
        import json
        hf = json.load(open(a.hf_json))
        for lab in hf:
            vocab.setdefault(norm(lab), lab)
            counts.setdefault(norm(lab), 10**6)          # presence confirmed by the v1.1 embedding labels; count unknown
        print(f'HF v1.1 labels: {len(hf)} from {a.hf_json}')
    if a.paper_txt and os.path.exists(a.paper_txt):
        for line in open(a.paper_txt):
            line = line.strip()
            if line:
                vocab.setdefault(norm(line), line)
    if not vocab:
        sys.exit('no vocabulary found: pass --npz <file> (look under the HF cache of the VoxTell env) or --paper_txt')
    os.makedirs(os.path.dirname(a.out_vocab), exist_ok=True)
    with open(a.out_vocab, 'w') as f:
        f.write('\n'.join(sorted(vocab.values())) + '\n')
    print(f'{len(vocab)} unique class names -> {a.out_vocab}')

    from pancia_kb import KnowledgeBase
    kb = KnowledgeBase.load(a.kb)
    vnorm = set(vocab)
    vtoks = [(set(tokens(v)), v) for v in vocab.values()]
    rows, changes = [], {}
    for e in kb.entities.values():
        phrases = [e['voxtell']['main']] + list(e['voxtell'].get('aliases', []))
        grades = []
        for ph in phrases:
            for side in (['left', 'right'] if '{side}' in ph else [None]):
                p = ph.replace('{side}', side or '').strip()
                pn = norm(p); pt = set(tokens(p)); pt_noside = pt - {'left', 'right'}
                pg = ' '.join(t for t in pn.split() if t not in GENERIC)
                cands = []
                for q in (pn, pg, norm(SIDE.sub('', pn)), norm(SIDE.sub('', pg))):
                    w = q.split()
                    if not w: continue
                    cands += [q, ' '.join(w[:-1] + [w[-1][:-1]]) if w[-1].endswith('s') else ' '.join(w[:-1] + [w[-1] + 's'])]
                key = next((k for k in cands if k in vnorm), None)
                if key is not None:
                    g = 'exact' if counts.get(key, 10**6) >= a.min_volumes else 'rare'
                elif any(pt_noside and pt_noside <= vt for vt, _ in vtoks):
                    g = 'near'
                else:
                    head = tokens(p)[-1] if tokens(p) else ''
                    g = 'near' if any(head in vt for vt, _ in vtoks) else 'ood'
                match = (vocab.get(key) if key else None) or next((v for vt, v in sorted(vtoks, key=lambda x: -counts.get(norm(x[1]), 0)) if pt_noside and pt_noside <= vt), '')
                grades.append(g)
                rows.append(dict(entity=e['id'], phrase=p, grade=g, vocab_match=match, train_volumes=counts.get(norm(match), '') if match else '', current_tier=e.get('coverage_tier')))
        # the near/ood split is decided by build_kb.py (head-noun heuristic); here only label presence is compared
        best = 'in_vocab' if 'exact' in grades else ('rare' if 'rare' in grades else 'near_or_ood')
        cur = e.get('coverage_tier'); cur = cur if cur in ('in_vocab', 'rare') else 'near_or_ood'
        if best != cur:
            changes[e['id']] = (e.get('coverage_tier'), best)
    out = os.path.join(a.kb, 'voxtell_vocab_check.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['entity', 'phrase', 'grade', 'vocab_match', 'train_volumes', 'current_tier']); w.writeheader(); w.writerows(rows)
    print(f'{len(rows)} phrases graded -> {out}')
    from collections import Counter
    print('grades:', Counter(r['grade'] for r in rows))
    print(f'{len(changes)} entities whose label presence disagrees with the KB tier (current -> from vocabulary); 0 expected after build_kb.py:')
    for k, (old, new) in sorted(changes.items()):
        print(f'  {k:32s} {old} -> {new}')


if __name__ == '__main__':
    main()
