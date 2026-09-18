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
    plan = Planner(KB).plan(ts, modality='MR', cancer_type='TCGA-CESC', sex='female')
    missing = {m['entity'] for m in plan.anchors_missing}
    # completion must propose the female pelvic organs TS cannot name
    for e in ['uterus', 'cervix', 'vagina', 'rectum', 'ovary']:
        assert e in missing, f'{e} not proposed; missing={missing}'
    assert 'prostate' not in missing and 'seminal_vesicle' not in missing
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
        TSStructure('spleen', 200, (90, -40, 130)), TSStructure('aorta', 60, (-5, -70, 80)), TSStructure('inferior_vena_cava', 40, (20, -60, 80)),
        TSStructure('pancreas', 70, (0, -20, 100)), TSStructure('adrenal_gland_left', 5, (60, -70, 120)), TSStructure('colon', 400, (0, 40, 40)),
        TSStructure('iliopsoas_left', 200, (35, -70, 40)), TSStructure('iliopsoas_right', 200, (-35, -70, 40)), TSStructure('urinary_bladder', 5, (0, 60, -10))])
    plan = Planner(KB).plan(ts, modality='CT', cancer_type='TCGA-KIRC', sex='male')
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
    print('  prompts:', len(p1.prompts)); [print('  ', p.wave, p.list, p.priority, p.entity, p.side, '|', p.terms[0], '|', p.gate.get('kind'), '|', p.reason[-1]) for p in p1.prompts]
    p2 = test_abdominal_ct_kirc(); print('\nKIRC CT: frame', p2.frame, '| missing', [(m['entity'], m['side']) for m in p2.anchors_missing])
    print('  prompts:', len(p2.prompts)); [print('  ', p.wave, p.list, p.priority, p.entity, p.side, '|', p.terms[0], '|', p.gate.get('kind'), '|', p.reason[-1]) for p in p2.prompts]
    open(os.path.join(os.path.dirname(__file__), 'example_plan_cesc_mr.json'), 'w').write(p1.to_json())
    open(os.path.join(os.path.dirname(__file__), 'example_plan_kirc_ct.json'), 'w').write(p2.to_json())
