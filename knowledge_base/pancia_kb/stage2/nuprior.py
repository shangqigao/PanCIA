"""Tail-weight prior of the stage-2 Student-t classes (main.tex Eq. nu_prior; Appendix C), keyed by modality (CT | MR) x the host's
knowledge-base tissue class (parenchymal | hollow | aerated; KB v0.5.9). Fitting (from the tail study's joint-feature components) and
use at inference (ClassPriors). x = log(1/nu), nu in [2, 100].

Fitting: components within sites within cells within a pan-cancer mean (moment estimates; sites equal). A component of a new scan
in cell k has prior N(mean_k, sd_k^2); any other host (empty cell, tissue classes without expert tumours) N(fallback).
Inference: tumour components take the mixture over the lesion's credible hosts sum_h P(h|L) N(cell(mod, class(h))); an organ
background class takes its own organ's cell (normal prior); generic background classes (and the extra classes of the Bayes-factor
models) take the normal fallback."""
import os
import numpy as np
import pandas as pd
from . import paths, expert as EX

NU_RANGE = (2.0, 100.0)
CELL_CLASSES = ('parenchymal', 'hollow', 'aerated')
MODALITIES = ('CT', 'MR')
FALLBACK_CLASSES = 'neural, osseous, muscle, lymphoid, diffuse, vascular, space, none'


def tissue_class_of():
    """collection -> tissue class of its host entity, read from the knowledge base (never from data)"""
    import yaml
    ents = yaml.safe_load(open(os.path.join(paths.kb_dir(), 'knowledge_base', 'entities.yaml')))
    tc = {e['id']: e.get('tissue_class') for e in ents['entities']}
    out = {ds: tc.get(ent) for ds, ent in EX.SITE_ENTITY.items()}
    if any(v is None for v in out.values()):
        raise RuntimeError(f'knowledge base without tissue_class (needs v0.5.9): {out}')
    return out, ents.get('version')


def load_components(O, min_n):
    c = pd.read_csv(os.path.join(O, 'components.csv'))
    c = c[c.kind.isin(['tumour_scan', 'tumour_only_scan']) & (c.feature == 'joint') & c.region.isin(['tumour', 'ring'])
          & (c.n_fit >= min_n) & (c.rho >= 0.05) & (c.n >= 100)].copy()
    c['x'] = np.log(1.0 / np.clip(c.nu, *NU_RANGE))
    c['site'] = c.ds.map(EX.SITE_OF)
    c['modality'] = np.where(c['mod'].astype(str).str.upper().str.startswith('CT'), 'CT', 'MR')
    tc, ver = tissue_class_of(); c['tissue'] = c.ds.map(tc)
    c['cell'] = c.modality + ' / ' + c.tissue
    # weights: components equal within a scan, scans equal within a site (sites are made equal in fit())
    c['w'] = 1.0 / c.groupby(['ds', 'file', 'region']).x.transform('size')
    c['w'] = c.w / c.groupby(['site', 'region']).w.transform('sum')
    return c, ver


def fit(c):
    """moment estimates of the hierarchy on one region's components (tumour or ring). Returns dict."""
    sites = []
    for s, g in c.groupby('site'):
        m = float((g.w * g.x).sum() / g.w.sum()); v = float((g.w * (g.x - m) ** 2).sum() / g.w.sum())
        sites.append(dict(site=s, cell=g.cell.iloc[0], mean=m, var=v, n_scans=g.file.nunique(), n_comp=len(g)))
    S = pd.DataFrame(sites)
    sw2 = float(S['var'].mean())                                                      # sites equal
    cells = []
    for k, g in S.groupby('cell'):
        cells.append(dict(cell=k, raw=float(g['mean'].mean()), n_sites=len(g), sites=', '.join(g.site), n_scans=int(g.n_scans.sum()),
                          n_comp=int(g.n_comp.sum()), ss=float(((g['mean'] - g['mean'].mean()) ** 2).sum())))
    C = pd.DataFrame(cells)
    df = float((C.n_sites - 1).clip(lower=0).sum())
    # pooled within-cell between-site variance, minus the sampling variance of a site mean (within-site variance / scans), >= 0
    multi = S[S.cell.isin(C.cell[C.n_sites > 1])]
    samp = float((multi['var'] / multi.n_scans).mean()) if len(multi) else 0.0
    ts2_raw = float(C.ss.sum() / df) if df > 0 else float(S['mean'].var())
    ts2 = max(ts2_raw - samp, 1e-6)
    mu0 = float(C.raw.mean())
    tc2 = max(float(C.raw.var(ddof=1)) - float((ts2 / C.n_sites).mean()), 0.0) if len(C) > 1 else 0.0
    prec = C.n_sites / ts2 + (1.0 / tc2 if tc2 > 0 else np.inf)
    C['mean'] = np.where(tc2 > 0, (C.n_sites * C.raw / ts2 + mu0 / max(tc2, 1e-12)) / prec, mu0)
    C['v_mean'] = np.where(tc2 > 0, 1.0 / prec, 0.0)
    C['sd'] = np.sqrt(C.v_mean + ts2 + sw2)
    v0 = (tc2 + ts2 / C.n_sites.mean()) / len(C)
    fall = dict(mean=mu0, sd=float(np.sqrt(v0 + tc2 + ts2 + sw2)))
    return dict(sites=S, cells=C, sw2=sw2, ts2=ts2, ts2_df=df, tc2=tc2, mu0=mu0, fallback=fall)


def logpdf(x, m, sd):
    return -0.5 * np.log(2 * np.pi * sd ** 2) - 0.5 * ((x - m) / sd) ** 2


def loso(c):
    rows = []
    for s in sorted(c.site.unique()):
        tr, te = c[c.site != s], c[c.site == s]
        if tr.site.nunique() < 3:
            continue
        F = fit(tr); cell = te.cell.iloc[0]; C = F['cells'].set_index('cell')
        pc = (float(C.loc[cell, 'mean']), float(C.loc[cell, 'sd'])) if cell in C.index else (F['fallback']['mean'], F['fallback']['sd'])
        pf = (F['fallback']['mean'], F['fallback']['sd'])
        w = te.w / te.w.sum()
        rows.append(dict(site=s, cell=cell, cell_in_training=cell in C.index, site_mean_x=float((w * te.x).sum()),
                         pred_cell=pc[0], pred_pancancer=pf[0], lp_cell=float((w * logpdf(te.x, *pc)).sum()),
                         lp_pancancer=float((w * logpdf(te.x, *pf)).sum())))
    return pd.DataFrame(rows)




def prior_tables(c):
    """fit on loaded components (load_components): {'tumour': table, 'normal': table} as stored in nu_prior.json"""
    out = {}
    for reg, name in (('tumour', 'tumour'), ('ring', 'normal')):
        F = fit(c[c.region == reg]); C = F['cells']
        out[name] = dict(cells={r.cell: dict(mean_x=round(float(r['mean']), 4), sd_x=round(float(r.sd), 4),
                                             nu_at_mean=round(float(np.exp(-r['mean'])), 2), n_sites=int(r.n_sites), sites=r.sites,
                                             n_scans=int(r.n_scans), n_components=int(r.n_comp)) for _, r in C.iterrows()},
                         fallback=dict(mean_x=round(F['fallback']['mean'], 4), sd_x=round(F['fallback']['sd'], 4),
                                       nu_at_mean=round(float(np.exp(-F['fallback']['mean'])), 2)),
                         variance=dict(within_site=round(F['sw2'], 4), between_site=round(F['ts2'], 4), between_site_df=F['ts2_df'],
                                       between_cell=round(F['tc2'], 4)))
    return out


def fit_prior(O=None, min_n=100, exclude=()):
    """the prior from the tail study (O: tail_study folder), leaving out collections `exclude` (validation folds). None without
    components.csv."""
    O = O or paths.out_dir('tail_study')
    if not os.path.exists(os.path.join(O, 'components.csv')):
        return None
    c, ver = load_components(O, min_n)
    c = c[~c.ds.isin(list(exclude))]
    return dict(version='nu_prior_v1', kb_version=ver, min_n=min_n, excluded=list(exclude), variable='x = log(1/nu), nu in [2, 100]',
                key='modality (CT | MR) x host tissue_class (parenchymal | hollow | aerated); other -> fallback',
                **prior_tables(c))


# ---------------------------------------------------------------------------------------------------------------- inference
_TC = None


def kb_tissue():
    """entity id -> tissue_class from the knowledge base (cached)"""
    global _TC
    if _TC is None:
        import yaml
        ents = yaml.safe_load(open(os.path.join(paths.kb_dir(), 'knowledge_base', 'entities.yaml')))
        _TC = {e['id']: e.get('tissue_class') for e in ents['entities']}
    return _TC


def entity_of(key):
    """KB entity id of an organ / host key: ('host', ent), (ent, side), ('other_side:ent', side), ('ts:name', None), 'ent>part',
    or an expert organ name; None when unknown"""
    tc = kb_tissue()
    e = (key[1] if key[0] == 'host' else key[0]) if isinstance(key, tuple) else key
    e = str(e).split(':')[-1].split('>')[0]
    for cand in (e, e.rsplit('_', 1)[0], e.replace('_left', '').replace('_right', '')):
        if cand in tc:
            return cand
    return None


def tissue_of(key):
    e = entity_of(key)
    return kb_tissue().get(e) if e else None


class ClassPriors:
    """per-class (mixture) priors on x = log(1/nu) for one scan: lists of (weight, mean, sd)"""

    def __init__(self, spec, mod):
        self.spec = spec; self.mod = 'CT' if str(mod).upper().startswith('CT') else 'MR'

    def _cell(self, table, tissue):
        t = self.spec[table]; c = t['cells'].get(f'{self.mod} / {tissue}') if tissue else None
        c = c or t['fallback']
        return float(c['mean_x']), float(c['sd_x'])

    def tumour(self, PhL):
        """mixture over credible hosts (PhL: {host key: P(h|L)}; several lesions: pass the average)"""
        if not PhL:
            return [(1.0,) + self._cell('tumour', None)]
        tot = sum(PhL.values()) or 1.0
        return [(float(w) / tot,) + self._cell('tumour', tissue_of(h)) for h, w in PhL.items()]

    def organ(self, key):
        return [(1.0,) + self._cell('normal', tissue_of(key))]

    def generic(self):
        return [(1.0,) + self._cell('normal', None)]
