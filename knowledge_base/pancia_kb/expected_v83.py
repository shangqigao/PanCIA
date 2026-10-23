"""Expected host list v8.3 (21 Oct 2026; user decisions after panel U4).
(1) kb.spread_hosts (kept unchanged) drops an invasion route whose destination is a PART that no agent segments (not a KB anchor),
    e.g. bladder / ovary / uterus -> colon_sigmoid, oesophagus -> main_bronchus. Here such a destination is mapped to its nearest
    anchor WHOLE along has_part (colon_sigmoid -> colon), keeping the relation's class and side_link ('mapped_from' records it).
    The same mapping applies to the cancer's distant-site list. Same conditions as kb.spread_hosts otherwise.
(2) parts_of(kb, e): the has_part descendants of e (downward chain). Used by ledger_v83: an organ outside the expected list is
    barred from entry only if it is a PART of an expected host (renal cortex of an expected kidney); a whole that contains an expected
    host (colon contains rectum) or an organ linked by lies_in / invested_by / has_duct can enter like any other organ."""
from functools import lru_cache

EXCL = {'peritoneal_spread', 'nodal', 'location', 'confined'}


def _anchor_whole(kb, e, _seen=None):
    """nearest KB anchor (tumour host) that has e as a part, walking has_part upwards; None if there is none"""
    d = kb.entities.get(e, {})
    if d.get('is_anchor') and d.get('tumour_host', True):
        return e
    frontier = [e]; seen = {e}
    while frontier:
        nxt = []
        for x in frontier:
            for r in kb.in_edges.get(x, []):
                if r['type'] == 'has_part' and r['src'] not in seen:
                    w = kb.entities.get(r['src'], {})
                    if w.get('is_anchor') and w.get('tumour_host', True):
                        return r['src']
                    seen.add(r['src']); nxt.append(r['src'])
        frontier = nxt
    return None


def static_hosts(kb, primary, cancer_type=None):
    sp = kb.tumour_prompts.get('spread', {})
    prior = sp.get('prior', dict(primary=1.0, local_invasion=0.3, distant=0.15))
    out = [dict(entity=primary, cls='primary', prior=prior['primary'])]
    seen = {primary}
    distant_raw = (sp.get('distant_by_cancer') or {}).get(cancer_type or '', sp.get('distant_default', []))
    distant = {(_anchor_whole(kb, e) or e) for e in distant_raw}
    for r in kb.out_edges.get(primary, []):
        if not (r['type'] in ('adjacent_to', 'invested_by') or (r['type'] == 'has_part' and 'adjacent_invasion' in (r.get('staging') or []))):
            continue
        if not r.get('staging') or not (set(r['staging']) - EXCL):
            continue
        d = kb.entities.get(r['dst'])
        if not d or not d.get('tumour_host', True):
            continue
        dst = _anchor_whole(kb, r['dst'])                                  # (1) part -> nearest segmented whole
        if dst is None or dst in seen:
            continue
        seen.add(dst)
        out.append(dict(entity=dst, cls='local_invasion', prior=prior['local_invasion'], via=r['type'], side_link=r.get('side_link', 'any'),
                        also_distant=dst in distant, mapped_from=r['dst'] if dst != r['dst'] else None))
    for e in distant_raw:
        w = _anchor_whole(kb, e)
        if w is None or w in seen:
            continue
        seen.add(w); out.append(dict(entity=w, cls='distant', prior=prior['distant'], mapped_from=e if w != e else None))
    return out


def parts_of(kb, e):
    return _parts(id(kb), kb, e)


_PC = {}


def _parts(k, kb, e):
    if (k, e) not in _PC:
        out, fr = set(), {e}
        while fr:
            nx = set()
            for x in fr:
                nx |= {r['dst'] for r in kb.out_edges.get(x, []) if r['type'] == 'has_part'}
            fr = nx - out; out |= nx
        _PC[(k, e)] = out
    return _PC[(k, e)]
