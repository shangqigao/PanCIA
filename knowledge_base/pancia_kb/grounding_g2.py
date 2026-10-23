"""Image grounding G2 and the v8 reasoning primitives (19 Oct 2026). main.tex v8 (md5 3d98e2): Sec. 2.1 (Eqs. conditional_joint,
normal_model, likelihood_ratios), Sec. 2.3 (compat, support level, extent_prior, existence test, decision policy, Table 'decision').
Plan: claude/hyperparameter_plan_2026-10-19.md, claude/v8_execution_plan_2026-10-19.md.

  normal model  P(h | L) = softmax_h( beta . phi_h(L) ) over the lesion's candidates (equal weights; choice model fitted on expert data);
                no-host candidate with the fixed score s_0 = alpha-quantile of the true-host scores of expert tumours;
                host credible set S = smallest set of candidates with probability >= 1 - alpha
  tumour model  existence TEST at level alpha (ledger/v8/grounding/test_sets.json): an outcome rules out if p(o|T) <= alpha < p(o|N),
                confirms if p(o|N) <= alpha < p(o|T) (host frame: only for an anchor inside its host), else neutral
                support level l = 2 (two agents) / 1 (single agent or lobe), minus one per warning signal (>= 0)
                decision = TABLE[l][test]; conflict -> uncertain; must agree under every host of S, else uncertain
  extent        w0(xi) ~ psi(log |xi|/|a|) * prod_{f in flags(xi)} eta_f   (fitted on the expert testbed)
Image evidence (signs, tree, normal tissue) is unchanged from G1 (grounding_g1.existence_evidence)."""
from __future__ import annotations
import json
import numpy as np
from . import grounding_g1 as G1

VERSION = 'g2.0'
TREE = G1.TREE
FEATS = G1.FEATS
compat_features = G1.compat_features
existence_evidence = G1.existence_evidence
split_lobes = G1.split_lobes
band = G1.band
organ_class = G1.organ_class
DECISION = {2: {'confirm': 'accepted', 'neutral': 'accepted', 'rule_out': 'uncertain'},
            1: {'confirm': 'accepted', 'neutral': 'uncertain', 'rule_out': 'rejected'},
            0: {'confirm': 'uncertain', 'neutral': 'rejected', 'rule_out': 'rejected'}}
FLAGS = ['leak', 'organ', 'prompt_mismatch', 'flat']
APPL_MIN = 0.5


class TestSets:
    """rule-out / confirm outcomes per stratum and alpha (scripts/g2_tables.py)"""
    def __init__(self, path):
        self.t = json.load(open(path)) if isinstance(path, str) else path

    def entry(self, cls, vol, mod):
        mod = mod if mod in ('CT', 'MR') else 'CT'; b = band(vol)
        return self.t['intrinsic'][f'{b}|{mod}'] if cls is None else self.t['host_frame'][f'{cls}|{b}|{mod}']

    def test(self, cls, vol, mod, tree, appl, alpha):
        """cls None -> intrinsic table (no host frame). Returns 'rule_out', 'confirm' or 'neutral'."""
        if tree not in TREE:
            return 'neutral'
        s = self.entry(cls, vol, mod)['sets'][str(alpha)]
        if tree in s['rule_out']:
            return 'rule_out'
        if tree in s['confirm']:
            if cls is not None and (appl is None or appl < self.t.get('appl_min', APPL_MIN)):
                return 'neutral'
            return 'confirm'
        return 'neutral'

    def lam(self, cls, vol, mod, tree):
        """reported likelihood ratio of the outcome (shrunk; not used by the decision)"""
        if tree not in TREE:
            return 1.0
        return float(self.entry(cls, vol, mod)['lam'][TREE.index(tree)])


def support_level(kind, signals):
    """2 for a two-agent anchor, 1 for a single-agent or lobe anchor; minus one per warning signal; never below 0
    (the no-host candidate carries no extra penalty: main.tex v8 attaches none; it is tested on the intrinsic table)"""
    lv = (2 if kind == 'two' else 1) - len(signals or [])
    return int(max(0, lv))


def decide(level, test):
    return DECISION[int(level)][test]


def decide_set(per_host):
    """per_host: dict h -> decision under h (hosts of the credible set). Same everywhere -> that decision; else uncertain."""
    ds = set(per_host.values())
    return ds.pop() if len(ds) == 1 else 'uncertain'


def score(phi, beta):
    return float(sum(beta.get(k, 0.0) * phi[k] for k in FEATS))


def host_posterior(cands, beta, s0):
    """cands: dict h -> phi (None for the no-host candidate). Choice model: P(h) ~ exp(beta . phi_h); no-host score s0."""
    hs = list(cands); z = np.array([score(cands[h], beta) if cands[h] is not None else s0 for h in hs], float)
    z = np.exp(z - z.max()); z = z / z.sum()
    return {h: float(v) for h, v in zip(hs, z)}


def credible_set(P, alpha):
    """smallest set of candidates (by decreasing probability) whose probability is at least 1 - alpha"""
    items = sorted(P.items(), key=lambda kv: (-kv[1], str(kv[0])))
    out, s = [], 0.0
    for h, p in items:
        out.append(h); s += p
        if s >= 1.0 - alpha - 1e-12:
            break
    return out


class Psi(G1.Psi):
    pass


def extent_prior(cands, psi, eta):
    """cands: list of dict(logratio, flags=list of flag names). w0 ~ psi(lr) * prod eta_f; normalised."""
    w = np.array([psi(c['logratio']) * float(np.prod([eta.get(f, 1.0) for f in c.get('flags', [])])) for c in cands], float)
    return w / w.sum() if w.sum() > 0 else np.full(len(cands), 1.0 / len(cands))


def self_test():
    # decision table is monotone in level and in evidence
    order = {'rejected': 0, 'uncertain': 1, 'accepted': 2}; tests = ['rule_out', 'neutral', 'confirm']
    for lv in (0, 1, 2):
        v = [order[decide(lv, t)] for t in tests]; assert v == sorted(v), (lv, v)
    for t in tests:
        v = [order[decide(lv, t)] for lv in (0, 1, 2)]; assert v == sorted(v), (t, v)
    assert decide(2, 'rule_out') == 'uncertain' and decide(0, 'confirm') == 'uncertain'
    # credible set: minimal and >= 1 - alpha
    P = {'a': 0.6, 'b': 0.3, 'c': 0.07, 'd': 0.03}
    assert credible_set(P, 0.05) == ['a', 'b', 'c'] and credible_set(P, 0.10) == ['a', 'b'] and credible_set(P, 0.5) == ['a']
    assert credible_set({'a': 1.0}, 0.05) == ['a']
    # support level
    assert support_level('two', []) == 2 and support_level('lobe', []) == 1 and support_level('two', ['flat']) == 1
    assert support_level('single', ['a', 'b']) == 0
    # host posterior sums to 1, no-host enters with its score
    hp = host_posterior({'x': {k: 1.0 for k in FEATS}, None: None}, {k: 0.0 for k in FEATS}, 0.0); assert abs(sum(hp.values()) - 1) < 1e-9
    assert abs(hp['x'] - 0.5) < 1e-9
    assert decide_set({'a': 'accepted', 'b': 'accepted'}) == 'accepted' and decide_set({'a': 'accepted', 'b': 'rejected'}) == 'uncertain'
    w = extent_prior([dict(logratio=0.0, flags=[]), dict(logratio=0.0, flags=['leak'])], Psi(), {'leak': 0.5}); assert abs(w[0] - 2 / 3) < 1e-9
    return True
