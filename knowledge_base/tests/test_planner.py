"""Synthetic tests: (1) pelvic MR cervix case where TS finds only bladder/sacrum/hips/iliac vessels/colon,
(2) abdominal CT kidney case with vertebrae present. Run: python -m pytest tests/ -q  (or python tests/test_planner.py)"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pancia_kb import KnowledgeBase, Planner, TSOutput, TSStructure

KB = KnowledgeBase.load(os.path.join(os.path.dirname(__file__), '..', 'knowledge_base'))


def test_kb_valid():
    errs = KB.validate()
    assert not errs, '\n'.join(errs)


def test_pelvic_mr_cesc_completion():
    ts = TSOutput(task='total_mr', fov_z_mm=(-120, 80), tumour_centroid_mm=(0, -20, -40), tumour_host_guess=None, structures=[
        TSStructure('urinary_bladder', 180, (0, 30, -30), zmin_mm=-55, zmax_mm=-5), TSStructure('sacrum', 150, (0, -60, -10), zmin_mm=-60, zmax_mm=40),
        TSStructure('hip_left', 300, (80, 0, -20), zmin_mm=-95, zmax_mm=60), TSStructure('hip_right', 300, (-80, 0, -20), zmin_mm=-95, zmax_mm=60),
        TSStructure('iliac_artery_left', 8, (40, -20, 40)), TSStructure('iliac_artery_right', 8, (-40, -20, 40)),
        TSStructure('colon', 250, (10, -30, 20)), TSStructure('femur_left', 200, (90, 0, -90), zmin_mm=-120, zmax_mm=-60), TSStructure('femur_right', 200, (-90, 0, -90), zmin_mm=-120, zmax_mm=-60),
        TSStructure('iliopsoas_left', 120, (50, -10, 0)), TSStructure('iliopsoas_right', 120, (-50, -10, 0))])
    plan = Planner(KB).plan(ts, modality='MR', cancer_type='TCGA-CESC', sex='female', include_tumour_prompt=True)
    missing = {m['entity'] for m in plan.anchors_missing}
    # completion must propose the female pelvic organs TS cannot name
    for e in ['uterus', 'cervix', 'vagina', 'rectum', 'ovary']:
        assert e in missing, f'{e} not proposed; missing={missing}'
    # sex is metadata only: male-presence anchors are still proposed (image decides; empty → recorded absent)
    assert 'prostate' in missing and 'seminal_vesicle' in missing
    assert any(l.get('step') == 'expected' and l.get('sex_used_for_filtering') is False for l in plan.log)
    assert 'prostate' in plan.anchors_absent_candidates or any(m['entity'] == 'prostate' for m in plan.anchors_missing)
    ent = {(p.entity, p.side) for p in plan.prompts}
    # cervix profile → parametrium, pelvic node stations
    assert ('parametrium', 'left') in ent or ('parametrium', 'right') in ent
    assert any(p.entity.startswith('ln_') for p in plan.prompts)
    assert plan.frame['method'] in ('fallback_landmarks', 'ts_vertebrae')
    # every prompt has a gate and a reason chain
    for p in plan.prompts:
        assert p.gate and p.reason and p.terms
    return plan


def test_abdominal_ct_kirc():
    verts = [TSStructure(f'vertebrae_{l}', 30, (0, -80, z)) for l, z in [('T11', 150), ('T12', 120), ('L1', 90), ('L2', 60), ('L3', 30), ('L4', 0)]]
    ts = TSOutput(task='total', fov_z_mm=(-20, 170), tumour_centroid_mm=(70, -60, 80), tumour_host_guess='kidney', structures=verts + [
        TSStructure('kidney_left', 160, (70, -60, 80)), TSStructure('kidney_right', 150, (-70, -60, 90)), TSStructure('liver', 1500, (-60, 20, 120)),
        TSStructure('spleen', 200, (90, -40, 130)), TSStructure('aorta', 60, (5, -70, 80)), TSStructure('inferior_vena_cava', 40, (-20, -60, 80)),
        TSStructure('pancreas', 70, (0, -20, 100)), TSStructure('adrenal_gland_left', 5, (60, -70, 120)), TSStructure('colon', 400, (0, 40, 40)),
        TSStructure('iliopsoas_left', 200, (35, -70, 40)), TSStructure('iliopsoas_right', 200, (-35, -70, 40)), TSStructure('urinary_bladder', 5, (0, 60, -10))])
    plan = Planner(KB).plan(ts, modality='CT', cancer_type='TCGA-KIRC', sex='male', include_tumour_prompt=True)
    assert plan.frame['method'] == 'ts_vertebrae'
    missing = {(m['entity'], m['side']) for m in plan.anchors_missing}
    assert ('adrenal', 'right') in missing          # right adrenal not found → completion
    ent = {(p.entity, p.side) for p in plan.prompts}
    for e in [('renal_vein', 'left'), ('perinephric_fat', 'left'), ('ureter', 'left'), ('ln_paraaortic', None)]:
        assert e in ent, f'{e} missing from KIRC plan: {sorted(ent)[:40]}'
    # bladder failed volume plausibility (5 ml) → flagged
    assert any(l.get('step') == 'found_plausibility' and l['entity'] == 'urinary_bladder' for l in plan.log)
    return plan


if __name__ == '__main__':
    test_kb_valid(); print('KB valid')
    p1 = test_pelvic_mr_cesc_completion(); print('CESC MR: frame', p1.frame, '| missing', [(m['entity'], m['side']) for m in p1.anchors_missing])
    print('  prompts:', len(p1.prompts)); [print('  ', p.tier, p.list, p.priority, p.entity, p.side, '|', p.terms[0], '|', p.gate.get('kind'), '|', p.reason[-1]) for p in p1.prompts]
    p2 = test_abdominal_ct_kirc(); print('\nKIRC CT: frame', p2.frame, '| missing', [(m['entity'], m['side']) for m in p2.anchors_missing])
    print('  prompts:', len(p2.prompts)); [print('  ', p.tier, p.list, p.priority, p.entity, p.side, '|', p.terms[0], '|', p.gate.get('kind'), '|', p.reason[-1]) for p in p2.prompts]
    open(os.path.join(os.path.dirname(__file__), 'example_plan_cesc_mr.json'), 'w').write(p1.to_json())
    open(os.path.join(os.path.dirname(__file__), 'example_plan_kirc_ct.json'), 'w').write(p2.to_json())


def test_secondary_host_evidence_topology():
    """v0.4.2: a single rater's isolated blob in a neighbouring organ is a lesion hypothesis, not host evidence;
    only a component growing out of the validated primary tumour into the neighbour counts (evidence 1);
    agreement inside the neighbour counts as evidence 2."""
    import numpy as np
    from pancia_kb.adapter import host_candidates_evidence
    kb = KB
    task = 'total'
    names_ts = {1: 'liver', 2: 'stomach', 3: 'kidney_left'}
    md = np.zeros((60, 60, 30), np.int16)
    md[5:30, 10:50, :] = 1            # liver
    md[32:50, 10:30, :] = 2           # stomach (2 mm gap from liver → inside each other's 10 mm envelope)
    md[32:50, 34:55, :] = 3           # kidney
    sp = (1.0, 1.0, 3.0)
    vt = np.zeros(md.shape, bool); vt[15:22, 25:35, 10:18] = True                      # primary tumour, in liver
    bp = vt.copy()
    bp[40:46, 40:50, 10:18] = True                                                     # isolated BP blob in kidney (1.4 ml)
    cands = kb.spread_hosts('liver', 'TCGA-LIHC')
    hosts = host_candidates_evidence(kb, cands, [vt, bp], ['rater_0', 'rater_1'], md, names_ts, task, sp, primary_tumour=vt & bp)
    h = {x['entity']: x for x in hosts}
    assert h['liver']['evidence'] == 2
    assert h['kidney']['evidence'] == 0 and h['kidney'].get('isolated_lesions') == [('rater_1', 1.4)]
    assert h['stomach']['evidence'] == 0 and not h['stomach'].get('isolated_lesions')
    # BP component that grows out of the agreed tumour across the boundary into the stomach → local invasion evidence 1
    bp2 = vt.copy(); bp2[15:45, 25:30, 10:18] = True
    hosts = host_candidates_evidence(kb, cands, [vt, bp2], ['rater_0', 'rater_1'], md, names_ts, task, sp, primary_tumour=vt & bp2)
    h = {x['entity']: x for x in hosts}
    assert h['stomach']['evidence'] == 1, h['stomach']
    # without a validated primary tumour the same component is only a hypothesis
    hosts = host_candidates_evidence(kb, cands, [vt, bp2], ['rater_0', 'rater_1'], md, names_ts, task, sp, primary_tumour=None)
    h = {x['entity']: x for x in hosts}
    assert h['stomach']['evidence'] == 0            # no validated primary → nothing to grow out of; no single-rater secondary evidence


def test_tier_t_tumour_prompts():
    """v0.5: one 'X tumor' prompt per spread-set host in the FOV, unconditioned; alias ensemble on the primary only when the
    initial VoxTell mask is empty; tier-T prompts survive the budget."""
    ts = TSOutput(task='total', structures=[
        TSStructure(ts_name='kidney_left', volume_ml=160, centroid_mm=(60, 0, 0), truncated=False),
        TSStructure(ts_name='kidney_right', volume_ml=150, centroid_mm=(-60, 0, 0), truncated=False),
        TSStructure(ts_name='liver', volume_ml=1500, centroid_mm=(-80, 20, 60), truncated=False),
        TSStructure(ts_name='spleen', volume_ml=180, centroid_mm=(90, 30, 60), truncated=False),
        TSStructure(ts_name='aorta', volume_ml=120, centroid_mm=(0, -20, 30), truncated=False),
        TSStructure(ts_name='vertebrae_T11', volume_ml=30, centroid_mm=(0, -60, 90), truncated=False),
        TSStructure(ts_name='vertebrae_L3', volume_ml=40, centroid_mm=(0, -60, -30), truncated=False)],
        fov_z_mm=(-120, 130), tumour_centroid_mm=(60, 5, 0), tumour_host_guess='kidney',
        hosts=[dict(entity='kidney', cls='primary', prior=1.0, evidence=2, weight=3.0, region='ts_envelope', role='primary'),
               dict(entity='liver', cls='local_invasion', prior=0.3, evidence=0, weight=0.3, region='ts_envelope', role='contact_check'),
               dict(entity='spleen', cls='local_invasion', prior=0.3, evidence=0, weight=0.3, region='ts_envelope', role='contact_check'),
               dict(entity='lung', cls='distant', prior=0.15, evidence=None, weight=0.15, region=None, role='distant_watch'),
               dict(entity='pancreas', cls='distant', prior=0.15, evidence=0, weight=0.15, region='ts_envelope', role='distant_watch')])
    ts.vt_initial_largest_ml = 12.0
    plan = Planner(KB).plan(ts, modality='CT', cancer_type='TCGA-KIRC')
    T = [p for p in plan.prompts if p.list == 'T']
    ents = sorted((p.entity, p.side) for p in T)
    assert ('kidney', 'left') in ents and ('liver', None) in ents and ('spleen', None) in ents, ents
    assert ('lung', None) not in ents and ('lung', 'left') not in ents          # not in FOV → no prompt
    assert ('pancreas', None) in ents                                           # distant site in FOV, no evidence → tumour prompt, no profile
    assert not any(p.list != 'T' and p.from_anchor == 'pancreas' for p in plan.prompts)
    assert all(p.tier == 3 and p.gate['kind'] == 'tumour_host' for p in T)
    assert not any(p.gate.get('ensemble') for p in T)                          # VT initial present → no alias ensemble
    kid = [p for p in T if p.entity == 'kidney'][0]
    assert kid.terms[0] == 'left kidney tumor', kid.terms
    assert ('kidney', 'right') in ents and [p for p in T if p.entity == 'kidney' and p.side == 'right'][0].gate['host_class'] == 'distant'
    # empty initial VT mask → ensemble added, still one prompt per host
    ts.vt_initial_largest_ml = 0.0
    plan2 = Planner(KB).plan(ts, modality='CT', cancer_type='TCGA-KIRC')
    ens = [p for p in plan2.prompts if p.list == 'T' and p.gate.get('ensemble')]
    assert len(ens) == 1 and ens[0].entity == 'kidney'
    assert len(plan2.prompts) <= KB.prompt_rules['budget']['max_prompts_per_scan']


def test_spread_sides_anatomy():
    """v0.5.3: side-specific invasion routes. Left kidney tumour: no liver/duodenum invasion prompt (right-only routes) but
    liver kept as a KIRC distant site; spleen yes. Liver tumour: only the right kidney/adrenal. Lung tumour: liver is not
    reached through the diaphragm (no invasion staging) — distant only."""
    from pancia_kb.planner import Planner as P
    pl = P(KB)
    sh = {h['entity']: h for h in KB.spread_hosts('kidney', 'TCGA-KIRC')}
    assert pl.spread_sides(sh['spleen'], 'kidney', 'left')[0] == [None]
    assert pl.spread_sides(sh['duodenum'], 'kidney', 'left')[0] is None
    s, cls, _ = pl.spread_sides(sh['liver'], 'kidney', 'left'); assert cls == 'distant'
    assert pl.spread_sides(sh['spleen'], 'kidney', 'right')[0] is None
    assert pl.spread_sides(sh['adrenal'], 'kidney', 'left')[0] == ['left']
    assert pl.spread_sides(sh['liver'], 'kidney', None)[0] == [None]            # side unknown → no restriction
    sl = {h['entity']: h for h in KB.spread_hosts('liver', 'TCGA-LIHC')}
    assert pl.spread_sides(sl['kidney'], 'liver', None)[0] == ['right']
    assert pl.spread_sides(sl['adrenal'], 'liver', None)[0] == ['right']
    assert sorted(pl.spread_targets(sl['adrenal'], 'liver', None)) == [('left', 'distant', 0.15), ('right', 'local_invasion', 0.3)]
    assert sorted(pl.spread_targets(sh['adrenal'], 'kidney', 'left')) == [('left', 'local_invasion', 0.3), ('right', 'distant', 0.15)]
    lu = {h['entity']: h for h in KB.spread_hosts('lung', 'TCGA-LUAD')}
    assert lu['liver']['cls'] == 'distant'



def test_reach_distance_test():
    """v0.5.4: a neighbour is a local-invasion host only where the validated primary tumour lies within reach_mm of its TS
    mask. Sigmoid tumour: bladder local; liver (a COAD distant site) distant; spleen (not a distant site) not a host."""
    import numpy as np
    from pancia_kb.adapter import host_candidates_evidence
    from pancia_kb.planner import Planner as P
    names_ts = {1: 'colon', 2: 'urinary_bladder', 3: 'liver', 4: 'spleen'}
    md = np.zeros((80, 80, 40), np.int16)
    md[10:70, 60:66, :] = 1           # colon: long tube
    md[30:50, 67:79, :] = 2           # bladder, 1 mm from the colon
    md[0:30, 0:40, :] = 3             # liver, far from the colon
    md[50:80, 0:30, :] = 4            # spleen, far
    sp = (1.0, 1.0, 3.0)
    t = np.zeros(md.shape, bool); t[35:45, 60:66, 15:25] = True        # tumour in the colon segment next to the bladder
    hosts = host_candidates_evidence(KB, KB.spread_hosts('colon', 'TCGA-COAD'), [t, t.copy()], ['rater_0', 'rater_1'], md, names_ts,
                                     'total', sp, primary_tumour=t, reach_tumour=t, reach_mm=10.0)
    h = {x['entity']: x for x in hosts}
    assert h['urinary_bladder']['reach'] == {'none': True} and h['urinary_bladder']['reach_mm']['none'] <= 2
    assert h['liver']['reach'] == {'none': False} and h['spleen']['reach'] == {'none': False}
    pl = P(KB)
    assert pl.spread_targets(h['urinary_bladder'], 'colon', None) == [(None, 'local_invasion', 0.3)]
    assert pl.spread_targets(h['liver'], 'colon', None) == [(None, 'distant', 0.15)]
    assert pl.spread_targets(h['spleen'], 'colon', None) == []
    # no TS mask for a neighbour → no reach entry → side_link prior as before
    assert 'reach' not in h.get('stomach', {})


def test_paired_primary_sides():
    """v0.5.4: tumour validated in one kidney → that kidney primary, the other a distant-class host; tumour in both kidneys →
    both primary, no contralateral distant prompt."""
    import numpy as np, nibabel as nib, tempfile, os
    from pancia_kb.adapter import tumour_summary
    names_ts = {1: 'kidney_left', 2: 'kidney_right', 3: 'liver'}
    md = np.zeros((90, 40, 30), np.int16)
    md[5:25, 10:30, 5:25] = 1; md[65:85, 10:30, 5:25] = 2; md[60:90, 0:10, :] = 3
    aff = np.diag([1.0, 1.0, 3.0, 1.0])
    d = tempfile.mkdtemp()
    def run(tumour):
        paths = []
        for i in range(2):
            q = os.path.join(d, f't{i}.nii.gz'); nib.save(nib.Nifti1Image(tumour.astype(np.uint8), aff), q); paths.append(q)
        q = os.path.join(d, 'ts.nii.gz'); nib.save(nib.Nifti1Image(md, aff), q)
        return tumour_summary(paths, q, KB, 'total', names_ts, expected_host='kidney', cancer_type='TCGA-KIRC')[2]
    t = np.zeros(md.shape, bool); t[8:18, 12:22, 8:16] = True
    assert run(t)['primary_sides'] == ['left']
    t2 = t.copy(); t2[70:80, 12:22, 8:16] = True
    assert run(t2)['primary_sides'] == ['left', 'right']
    ts = TSOutput(task='total', structures=[
        TSStructure(ts_name='kidney_left', volume_ml=160, centroid_mm=(15, 20, 45), truncated=False),
        TSStructure(ts_name='kidney_right', volume_ml=150, centroid_mm=(75, 20, 45), truncated=False),
        TSStructure(ts_name='vertebrae_T11', volume_ml=30, centroid_mm=(45, -60, 90), truncated=False),
        TSStructure(ts_name='vertebrae_L3', volume_ml=40, centroid_mm=(45, -60, -30), truncated=False)],
        fov_z_mm=(-120, 130), tumour_centroid_mm=(45, 20, 45), tumour_host_guess='kidney',
        hosts=[dict(entity='kidney', cls='primary', prior=1.0, evidence=2, weight=3.0, region='ts_envelope', role='primary')])
    ts.vt_initial_largest_ml = 5.0
    ts.primary_sides = ['left', 'right']
    T = [(p.entity, p.side, p.gate['host_class']) for p in Planner(KB).plan(ts, modality='CT', cancer_type='TCGA-KIRC').prompts if p.list == 'T']
    assert ('kidney', 'left', 'primary') in T and ('kidney', 'right', 'primary') in T and not any(c == 'distant' and e == 'kidney' for e, s, c in T)
    ts.primary_sides = ['right']
    T = [(p.entity, p.side, p.gate['host_class']) for p in Planner(KB).plan(ts, modality='CT', cancer_type='TCGA-KIRC').prompts if p.list == 'T']
    assert ('kidney', 'right', 'primary') in T and ('kidney', 'left', 'distant') in T, T


def test_new_invasion_neighbours():
    """v0.5.4: UCEC → cervix (T2), vagina (T3b); OV → colon, rectum, bladder (FIGO IIB pelvic extension)."""
    u = {h['entity']: h['cls'] for h in KB.spread_hosts('uterus', 'TCGA-UCEC')}
    assert u.get('cervix') == 'local_invasion' and u.get('vagina') == 'local_invasion'
    o = {h['entity']: h for h in KB.spread_hosts('ovary', 'TCGA-OV')}
    assert all(o[e]['cls'] == 'local_invasion' for e in ('colon', 'rectum', 'urinary_bladder'))
    assert o['colon']['also_distant']


def test_v055_fixes():
    """v0.5.5: spine is a host found from vertebrae; colon does not walk rectal relations; sex-neutral / non-primary wording;
    contralateral ovary is primary-class; primary out of FOV → no primary prompt, local neighbours → distant or dropped;
    missing anchors get no profile unless they are spread-set hosts."""
    from pancia_kb.adapter import _host_ts_labels
    names = {1: 'vertebrae_L1', 2: 'vertebrae_L2', 3: 'liver'}
    assert sorted(_host_ts_labels(KB, 'total', names, 'spine')) == [1, 2]
    hop2 = {r['dst'] for r in KB.profile('colon', hops=2) if r['hop'] == 2}
    assert not hop2 & {'prostate', 'vagina', 'levator_ani', 'ln_inguinal', 'sacrum'}
    assert 'testicular' not in KB.entities['gonadal_vein']['voxtell']['main']
    assert KB.entities['vagina']['tumour_phrase']['main'] == 'vagina tumor'
    assert KB.entities['ovary'].get('contralateral_class') == 'primary'
    ov = {h['entity']: h['cls'] for h in KB.spread_hosts('ovary', 'TCGA-OV')}
    assert ov.get('omentum') == 'distant'
    ts = TSOutput(task='total', structures=[
        TSStructure(ts_name='vertebrae_T11', volume_ml=30, centroid_mm=(0, -60, 90), truncated=False),
        TSStructure(ts_name='vertebrae_L3', volume_ml=40, centroid_mm=(0, -60, -30), truncated=False),
        TSStructure(ts_name='colon', volume_ml=500, centroid_mm=(40, 20, -20), truncated=False),
        TSStructure(ts_name='liver', volume_ml=1500, centroid_mm=(-80, 20, 60), truncated=False)],
        fov_z_mm=(-120, 130), tumour_host_guess='ovary',
        hosts=[dict(entity='ovary', cls='primary', prior=1.0, evidence=None, weight=1.0, region=None, role='primary'),
               dict(entity='colon', cls='local_invasion', prior=0.3, evidence=0, weight=0.3, region='ts_envelope', role='contact_check', also_distant=True, side_link='any'),
               dict(entity='small_bowel', cls='local_invasion', prior=0.3, evidence=0, weight=0.3, region='ts_envelope', role='contact_check', also_distant=False, side_link='any'),
               dict(entity='liver', cls='distant', prior=0.15, evidence=0, weight=0.15, region='ts_envelope', role='distant_watch')])
    ts.vt_initial_largest_ml = 3.0; ts.primary_in_fov = False
    plan = Planner(KB).plan(ts, modality='CT', cancer_type='TCGA-OV')
    T = {(p.entity, p.gate['host_class']) for p in plan.prompts if p.list == 'T'}
    assert ('colon', 'distant') in T and ('liver', 'distant') in T and not any(e == 'ovary' for e, c in T)
    assert not any(e == 'small_bowel' for e, c in T)
    colon = [p for p in plan.prompts if p.list == 'T' and p.entity == 'colon'][0]
    assert all('primary' not in t for t in colon.terms), colon.terms
    assert not any(p.reason and p.reason[0].startswith('host anchor ovary') for p in plan.prompts)
    # contralateral ovary: primary class when the side is resolved
    ts.primary_in_fov = True; ts.primary_sides = ['left']
    ts.hosts[0] = dict(entity='ovary', cls='primary', prior=1.0, evidence=1, weight=2.0, region='kb_prior_region', role='primary')
    T = {(p.entity, p.side, p.gate['host_class']) for p in Planner(KB).plan(ts, modality='CT', cancer_type='TCGA-OV').prompts if p.list == 'T'}
    assert ('ovary', 'left', 'primary') in T and ('ovary', 'right', 'primary') in T, T
