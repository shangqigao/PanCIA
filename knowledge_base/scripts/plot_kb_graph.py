"""Circular hierarchical edge-bundling plot of the pan-cancer anatomical KB (no cancer-type spread sets).

Ring  : all KB entities, grouped sector -> anchor group -> entity (has_part defines anchor membership).
Inside: non-hierarchical relations bundled along the hierarchy (Holten 2006), coloured by family.
    python plot_kb_graph.py --kb ../knowledge_base --out "../../documents/figs/kb_graph"
"""
import argparse, collections as C, math, yaml, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Wedge
from scipy.interpolate import BSpline

SECTORS = [  # (name, anchors) in cranio-caudal / systemic order around the ring
    ('Head & neck', ['brain', 'oral_cavity', 'pharynx', 'larynx', 'parotid_gland', 'thyroid']),
    ('Thorax', ['trachea', 'lung', 'heart', 'esophagus', 'breast', '_thorax_other']),
    ('Abdomen', ['liver', 'gallbladder', 'stomach', 'spleen', 'duodenum', 'pancreas', 'adrenal', 'kidney', 'small_bowel', 'colon', 'omentum']),
    ('Pelvis', ['rectum', 'urinary_bladder', 'prostate', 'seminal_vesicle', 'uterus', 'cervix', 'vagina', 'ovary']),
    ('Whole body & fluid', ['_whole', '_fluid']),
    ('Bone & muscle', ['spine', 'sacrum', 'hip', '_bones', 'iliopsoas', 'psoas', '_muscles']),
    ('Vessels', ['aorta', 'inferior_vena_cava', 'iliac_artery', 'iliac_vein', '_arteries', '_veins']),
    ('Lymph-node stations', ['_ln_thorax', '_ln_abdomen', '_ln_pelvis']),
]
PSEUDO = {'_bones': 'other bones', '_muscles': 'muscles & body wall', '_whole': 'skin & fat', '_fluid': 'fluid / cyst',
          '_thorax_other': 'other thoracic', '_arteries': 'other arteries', '_veins': 'other veins',
          '_ln_thorax': 'thoracic', '_ln_abdomen': 'abdominal', '_ln_pelvis': 'pelvic'}
LN_GROUP = {'_ln_thorax': ['ln_cervical', 'ln_supraclavicular', 'ln_axillary', 'ln_internal_mammary', 'ln_mediastinal', 'ln_hilar'],
            '_ln_abdomen': ['ln_celiac', 'ln_hepatic_hilar', 'ln_perigastric', 'ln_renal_hilar', 'ln_paraaortic',
                            'ln_mesenteric', 'ln_pericolic', 'ln_abdominal'],
            '_ln_pelvis': ['ln_common_iliac', 'ln_external_iliac', 'ln_internal_iliac', 'ln_obturator', 'ln_presacral',
                           'ln_pelvic', 'ln_inguinal']}
# relation families: (types, colour, linestyle, label). Colours: validated reference slots 1-3 + neutral grey.
FAMILIES = [
    (('adjacent_to',), '#b9b8b2', '-', 'adjacent to (contiguity)'),
    (('invested_by', 'has_duct'), '#1baf7a', '-', 'invested by / has duct'),
    (('supplied_by', 'drained_by'), '#eb6834', '-', 'supplied / drained by (vascular)'),
    (('drains_lymph_to',), '#2a78d6', '-', 'drains lymph to'),
]
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8984'
R_LEAF, R_GROUP, R_SECTOR = 1.0, 0.74, 0.46


def load(kb):
    E = yaml.safe_load(open(f'{kb}/entities.yaml'))['entities']
    R = yaml.safe_load(open(f'{kb}/relations.yaml'))['relations']
    return E, R


def assign(E, R):
    """entity id -> group id (an anchor or a pseudo-group)."""
    ent = {e['id']: e for e in E}
    anchors = {e['id'] for e in E if e.get('is_anchor')}
    grp = {a: a for a in anchors}
    for g, ids in LN_GROUP.items():
        for i in ids: grp[i] = g
    parent = C.defaultdict(list)
    for r in R:
        if r['type'] == 'has_part': parent[r['dst']].append(r['src'])
    def top(i, seen=()):
        if i in anchors: return i
        for p in parent.get(i, []):
            if p not in seen:
                t = top(p, seen + (i,))
                if t: return t
    for i in ent:
        if i not in grp:
            t = top(i)
            if t: grp[i] = t
    # unparented: attach to the anchor that invests it / owns its duct, else by category
    for typ in ('invested_by', 'has_duct'):
        for r in R:
            if r['type'] == typ and r['dst'] not in grp and r['src'] in anchors: grp[r['dst']] = r['src']
    for i, e in ent.items():
        if i in grp: continue
        c = e['category']
        grp[i] = '_arteries' if c == 'vessel_artery' else '_veins' if c == 'vessel_vein' else None
    # remaining (no has_part parent, not invested by / ducted from an anchor): group by what the structure is
    by_cat = {'bone': '_bones', 'muscle': '_muscles', 'fat': '_whole', 'fascia': '_whole', 'region': '_whole',
              'space': '_fluid', 'gland': '_thorax_other'}
    adj = C.defaultdict(C.Counter)  # organs/ducts with no part-of link (e.g. anal canal) go to the anchor they abut
    for r in R:
        if r['type'] == 'adjacent_to':
            for u, v in ((r['src'], r['dst']), (r['dst'], r['src'])):
                if v in anchors: adj[u][v] += 1
    FIG_GROUP = {'skull': 'brain', 'eye': 'brain', 'nasal_cavity': 'pharynx', 'submandibular_gland': 'oral_cavity'}   # figure-only placement of a bone that belongs with its region rather than with the skeleton
    for i, g in FIG_GROUP.items():
        if i in ent and grp.get(i) is None: grp[i] = g
    for i, e in ent.items():
        if grp.get(i) is None:
            c = e['category']
            grp[i] = by_cat[c] if c in by_cat else adj[i].most_common(1)[0][0] if adj[i] else '_whole'
    return grp


def layout(E, R, grp):
    order_in_yaml = {e['id']: k for k, e in enumerate(E)}
    leaves, gid_of, sec_of = [], {}, {}
    members = C.defaultdict(list)
    for i, g in grp.items(): members[g].append(i)
    for s, (sname, gs) in enumerate(SECTORS):
        for g in gs:
            m = members.get(g, [])
            if g in LN_GROUP: m = [i for i in LN_GROUP[g] if i in grp]
            else: m = sorted(m, key=lambda i: (i != g, order_in_yaml[i]))
            for i in m:
                leaves.append(i); gid_of[i] = g; sec_of[i] = s
    missing = set(grp) - set(leaves)
    assert not missing, missing
    n, gap = len(leaves), 1.6  # gap (in leaf slots) between sectors
    pos, t = {}, 0.0
    anc = {e['id'] for e in E if e.get('is_anchor')}
    for k, i in enumerate(leaves):
        if k and sec_of[i] != sec_of[leaves[k - 1]]: t += gap
        w = 1.8 if i in anc else 1.0
        pos[i] = t + (w - 1) / 2; t += w
    total = t + gap
    ang = {i: math.pi / 2 - 2 * math.pi * (pos[i] + 0.5 * gap) / total for i in leaves}  # clockwise from 12 o'clock
    return leaves, ang, gid_of, sec_of, total


def circ_mean(a):
    return math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--kb', default='../knowledge_base'); ap.add_argument('--out', required=True)
    ap.add_argument('--beta', type=float, default=0.82); a = ap.parse_args()
    E, R = load(a.kb); ent = {e['id']: e for e in E}
    grp = assign(E, R)
    leaves, ang, gid_of, sec_of, total = layout(E, R, grp)
    g_ang = {g: circ_mean([ang[i] for i in leaves if gid_of[i] == g]) for g in set(gid_of.values())}
    s_ang = {s: circ_mean([ang[i] for i in leaves if sec_of[i] == s]) for s in set(sec_of.values())}
    P = lambda r, t: np.array([r * math.cos(t), r * math.sin(t)])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(7.2, 7.5)); ax = fig.add_axes([0, 0.065, 1, 0.935]); ax.set_aspect('equal'); ax.axis('off')
    L = 1.6; ax.set_xlim(-L * 1.05, L * 1.05); ax.set_ylim(-L, L)

    # hierarchy (has_part / membership): faint tree root -> sector -> group -> leaf
    for s in s_ang:
        ax.plot(*zip(P(0, 0), P(R_SECTOR, s_ang[s])), color='#dddcd6', lw=0.6, zorder=1)
    for g, t in g_ang.items():
        s = sec_of[next(i for i in leaves if gid_of[i] == g)]
        ax.plot(*zip(P(R_SECTOR, s_ang[s]), P(R_GROUP, t)), color='#dddcd6', lw=0.6, zorder=1)
    for i in leaves:
        ax.plot(*zip(P(R_GROUP, g_ang[gid_of[i]]), P(R_LEAF - 0.015, ang[i])), color='#e9e8e3', lw=0.4, zorder=1)

    # bundled relations
    def ctrl(u, v):
        gu, gv, su, sv = gid_of[u], gid_of[v], sec_of[u], sec_of[v]
        pts = [P(R_LEAF, ang[u]), P(R_GROUP, g_ang[gu])]
        if gu != gv:
            if su != sv: pts += [P(R_SECTOR, s_ang[su]), P(0, 0), P(R_SECTOR, s_ang[sv])]
            else: pts += [P(R_SECTOR, s_ang[su])]
            pts += [P(R_GROUP, g_ang[gv])]
        else:
            pts += [P(R_GROUP * 0.9, g_ang[gu])]
        pts += [P(R_LEAF, ang[v])]
        pts = np.array(pts)
        n = len(pts)  # Holten straightening
        lin = pts[0] + (pts[-1] - pts[0]) * np.linspace(0, 1, n)[:, None]
        return a.beta * pts + (1 - a.beta) * lin

    def curve(pts):
        k = min(3, len(pts) - 1)
        tk = np.r_[[0] * k, np.linspace(0, 1, len(pts) - k + 1), [1] * k]
        return BSpline(tk, pts, k)(np.linspace(0, 1, 80))

    counts = C.Counter()
    for fam, (types, col, ls, lab) in enumerate(FAMILIES):
        for r in R:
            if r['type'] not in types or r['src'] not in ang or r['dst'] not in ang or r['src'] == r['dst']: continue
            c = curve(ctrl(r['src'], r['dst'])); counts[fam] += 1
            ax.plot(c[:, 0], c[:, 1], color=col, lw=0.55 if fam == 0 else 0.8, alpha=0.55 if fam == 0 else 0.8,
                    ls=ls, solid_capstyle='round', zorder=2 + fam)

    # leaves and labels
    for i in leaves:
        e, t = ent[i], ang[i]
        anchor = bool(e.get('is_anchor')); host = anchor and bool(e['anchor'].get('tumour_host', e.get('tumour_host')))
        x, y = P(R_LEAF, t)
        if anchor:
            ax.scatter([x], [y], s=16, color=INK if host else '#ffffff', edgecolor=INK, lw=0.8, zorder=10)
        else:
            ax.scatter([x], [y], s=4, color=MUTED, lw=0, zorder=10)
        deg = math.degrees(t); flip = 90 < (deg % 360) < 270
        tx, ty = P(R_LEAF + 0.035, t)
        name = e['name']
        ax.text(tx, ty, name, rotation=deg + (180 if flip else 0), rotation_mode='anchor',
                ha='right' if flip else 'left', va='center', fontsize=5.2 if anchor else 4.4,
                fontweight='bold' if anchor else 'normal', color=INK if anchor else INK2, zorder=11)
    # group pseudo-labels (no entity of their own) on the group node
    for g, t in ([] if True else g_ang.items()):
        if g in PSEUDO:
            x, y = P(R_GROUP - 0.03, t)
            ax.text(x, y, PSEUDO[g], fontsize=4.6, style='italic', color=INK2, ha='center', va='center',
                    rotation=math.degrees(t) - 90 if math.sin(t) >= 0 else math.degrees(t) + 90, zorder=12,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.85))
    # sector arcs + names
    for s, (sname, _) in enumerate(SECTORS):
        ts = [ang[i] for i in leaves if sec_of[i] == s]
        t0, t1 = math.degrees(min(ts)) - 0.6, math.degrees(max(ts)) + 0.6
        ax.add_patch(Wedge((0, 0), R_LEAF - 0.025, t0, t1, width=0.012, color=INK2, lw=0, zorder=9))
        tm = s_ang[s]; x, y = P(0.79, tm)
        ax.text(x, y, sname, fontsize=6.2, fontweight='bold', color=INK, ha='center', va='center', zorder=12,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#c9c8c2', lw=0.4, alpha=0.95))

    # legend
    lx = fig.add_axes([0.04, 0.0, 0.92, 0.07]); lx.axis('off'); lx.set_xlim(0, 1); lx.set_ylim(0, 1)
    items = [(col, f'{lab}  ({counts[k]})') for k, (_, col, _, lab) in enumerate(FAMILIES)]
    for k, (col, lab) in enumerate(items):
        x0 = 0.02 + (k % 2) * 0.5; y0 = 0.72 - (k // 2) * 0.32
        lx.plot([x0, x0 + 0.04], [y0, y0], color=col, lw=2.0); lx.text(x0 + 0.05, y0, lab, va='center', fontsize=6.4, color=INK)
    y0 = 0.08
    lx.scatter([0.03], [y0], s=16, color=INK, edgecolor=INK, lw=0.8); lx.text(0.05, y0, 'tumour-hosting anchor', va='center', fontsize=6.4)
    lx.scatter([0.30], [y0], s=16, color='white', edgecolor=INK, lw=0.8); lx.text(0.32, y0, 'other anchor', va='center', fontsize=6.4)
    lx.scatter([0.50], [y0], s=4, color=MUTED, lw=0); lx.text(0.52, y0, 'part / associated structure', va='center', fontsize=6.4)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(f'{a.out}.{ext}', dpi=300 if ext == 'png' else None)
    n_anchor = sum(1 for e in E if e.get('is_anchor'))
    print(f'entities {len(leaves)}, anchors {n_anchor}, bundled {sum(counts.values())}, by family {dict(counts)}')
    for g in sorted(set(gid_of.values()), key=str):
        print(g, [i for i in leaves if gid_of[i] == g])


if __name__ == '__main__':
    main()
