"""Builds the anchor-centric pan-cancer anatomical knowledge base (v0.1) as YAML.

Run:  python build_kb.py  → writes ./knowledge_base/*.yaml
Content is authored in Python for consistency (shared helpers, TS class-map cross-check),
then dumped to YAML which is the artefact of record.
"""
import json, yaml, os, datetime
OUT = os.path.join(os.path.dirname(__file__), 'knowledge_base')
os.makedirs(OUT, exist_ok=True)
TS = json.load(open(os.path.join(os.path.dirname(__file__), 'seed', 'ts_class_maps.json')))
TS_ALL = {task: set(v.values()) for task, v in TS.items()}
VERSION = '0.1.0'
TODAY = str(datetime.date.today())

# ------------------------------------------------------------------ helpers
E = []   # entities
R = []   # relations

def ent(id, name, category, *, lat='none', vt=None, aliases=(), ts=None, modality=('CT', 'MR'),
        tier='assumed_in_vocab', vol=None, hu=None, mr=None, prio=2, anchor=None, notes=None, sources=('TS', 'FMA')):
    """Register an entity. ts = {task: name_or_{side:name}}. vt = VoxTell main term (may contain {side})."""
    if ts:
        for task, names in ts.items():
            vals = list(names.values()) if isinstance(names, dict) else [names]
            for n in vals:
                assert n in TS_ALL[task], f'{id}: {n} not in TS {task}'
    e = dict(id=id, name=name, category=category, laterality=lat,
             voxtell=dict(main=vt or name.lower(), aliases=list(aliases)),
             ts=ts or {}, modality=list(modality), coverage_tier=tier, priority=prio)
    if vol: e['volume_ml'] = vol
    if hu: e['ct_hu'] = list(hu)
    if mr: e['mr_note'] = mr
    if anchor: e['is_anchor'] = True; e['anchor'] = anchor
    else: e['is_anchor'] = False
    if notes: e['notes'] = notes
    e['sources'] = list(sources)
    E.append(e); return id

def A(span, lr, ap, presence, prior=(), span_var=1):
    """Anchor block: canonical vertebral span, transverse position, presence class, spatial prior."""
    return dict(span=list(span), span_var=span_var, position=dict(lr=lr, ap=ap), presence=presence,
                spatial_prior=[dict(landmark=l, relation=r, dist_mm=list(d)) for l, r, d in prior])

def rel(src, type, dst, *, side='same', direction=None, contact=None, staging=(), tier=None, prio=None, note=None):
    d = dict(src=src, type=type, dst=dst, side_link=side)
    if direction: d['direction'] = direction
    if contact: d['contact'] = contact
    if staging: d['staging'] = list(staging)
    if tier: d['coverage_tier'] = tier
    if prio: d['priority'] = prio
    if note: d['note'] = note
    R.append(d)

# ------------------------------------------------------------------ relation types
RELATION_TYPES = [
 dict(id='has_part', meaning='sub-structure of the anchor; supports confined-to-organ and intra-organ location',
      gate=dict(kind='inside', dilate_mm=3), default_priority=2),
 dict(id='invested_by', meaning='capsule, fascia or fat compartment surrounding the anchor; supports beyond-capsule / compartment extension (T3-type criteria)',
      gate=dict(kind='shell', inner_mm=0, outer_mm=40), default_priority=1),
 dict(id='adjacent_to', meaning='physical neighbour; supports adjacent-organ invasion (T4-type criteria). Read in reverse it is the spatial prior for completion',
      gate=dict(kind='neighbour_bbox_or_directional_shell', shell_mm=30), default_priority=2),
 dict(id='supplied_by', meaning='arterial supply down to nameable branches; supports vascular encasement',
      gate=dict(kind='corridor', from_='anchor_hilum', to='parent_trunk', radius_mm=15), default_priority=2),
 dict(id='drained_by', meaning='venous drainage down to nameable branches; supports venous invasion / tumour thrombus (e.g. renal vein, portal vein)',
      gate=dict(kind='corridor', from_='anchor_hilum', to='parent_trunk', radius_mm=15), default_priority=1),
 dict(id='has_duct', meaning='excretory duct or airway; supports obstruction criteria (hydronephrosis, biliary dilatation, atelectasis)',
      gate=dict(kind='corridor', from_='anchor', to='target_organ', radius_mm=12), default_priority=2),
 dict(id='drains_lymph_to', meaning='regional lymph-node station per AJCC 8th primary-site definition; supports N stage',
      gate=dict(kind='region', defined_by='vessels_and_vertebral_levels'), default_priority=2),
 dict(id='lies_in', meaning='body region / compartment membership; supports FOV logic and peritoneal reasoning',
      gate=dict(kind='none'), default_priority=3),
 dict(id='wall_of', meaning='wall or floor the anchor attaches to; supports wall invasion (chest wall, pelvic sidewall, diaphragm)',
      gate=dict(kind='directional_shell', shell_mm=25), default_priority=2),
 dict(id='landmark_for', meaning='non-vertebral landmark with a known vertebral level; used for frame estimation when TS vertebrae are missing',
      gate=dict(kind='none'), default_priority=3),
]

# ------------------------------------------------------------------ regions
REGIONS = [
 dict(id='thorax', span=['T1', 'T12'], defined_by=['ribs', 'sternum', 'lungs', 'heart'], peritoneal=None),
 dict(id='mediastinum', span=['T1', 'T10'], defined_by=['between lungs; contains heart, aorta, oesophagus, trachea'], peritoneal=None),
 dict(id='upper_abdomen', span=['T10', 'L2'], defined_by=['diaphragm dome to renal hilum'], peritoneal='mixed'),
 dict(id='retroperitoneum', span=['T12', 'L5'], defined_by=['posterior to peritoneum: kidneys, adrenals, aorta, IVC, psoas, pancreas, duodenum'], peritoneal='retro'),
 dict(id='peritoneal_cavity', span=['T10', 'S1'], defined_by=['liver, spleen, stomach, small bowel, transverse/sigmoid colon, omentum'], peritoneal='intra'),
 dict(id='pelvis_true', span=['S1', 'coccyx'], defined_by=['below pelvic brim (promontory / iliopectineal line): bladder, uterus/prostate, rectum'], peritoneal='mixed'),
 dict(id='pelvis_false', span=['L5', 'S1'], defined_by=['iliac fossae above pelvic brim'], peritoneal='intra'),
 dict(id='porta_hepatis', span=['T12', 'L1'], defined_by=['between liver hilum and portal vein / common bile duct / hepatic artery'], peritoneal='intra'),
 dict(id='renal_hilum', span=['L1', 'L2'], defined_by=['medial kidney between renal pelvis and aorta/IVC'], peritoneal='retro'),
 dict(id='perivesical_space', span=['S3', 'coccyx'], defined_by=['fat around bladder, bounded by pubis and peritoneal reflection'], peritoneal='extra'),
 dict(id='presacral_space', span=['S1', 'coccyx'], defined_by=['between rectum/mesorectum and sacrum'], peritoneal='extra'),
 dict(id='parametrial_space', span=['S3', 'coccyx'], defined_by=['lateral to cervix, medial to obturator internus, between bladder and rectum'], peritoneal='extra'),
 dict(id='axilla', span=['T2', 'T5'], defined_by=['lateral to pectoralis minor, medial to humerus'], peritoneal=None),
 dict(id='neck', span=['C1', 'T1'], defined_by=['skull base to clavicles'], peritoneal=None),
]

# ------------------------------------------------------------------ landmarks (non-vertebral, with vertebral level)
LANDMARKS = [
 dict(id='carina', entity='trachea', level='T4', tolerance=1, use='bottom', how='caudal end of trachea mask'),
 dict(id='diaphragm_dome', entity='liver', level='T10', tolerance=1, use='top', how='most cranial liver voxel (right dome)'),
 dict(id='xiphoid', entity='sternum', level='T9', tolerance=1, use='bottom', how='caudal end of sternum mask'),
 dict(id='renal_hilum_level', entity='kidney', level='L1', tolerance=1, use='centroid', how='centroid of kidney masks (L1–L2)'),
 dict(id='aortic_bifurcation', entity='aorta', level='L4', tolerance=1, use='bottom', how='caudal end of aorta mask / start of iliac arteries'),
 dict(id='iliac_crest', entity='hip', level='L4', tolerance=1, use='top', how='most cranial hip-bone voxel'),
 dict(id='sacral_promontory', entity='sacrum', level='S1', tolerance=0, use='top', how='most cranial-anterior sacrum voxel'),
 dict(id='femoral_head', entity='femur', level='coccyx', tolerance=1, use='top', how='centroid of femoral head'),
 dict(id='pubic_symphysis', entity='hip', level='coccyx', tolerance=1, use='bottom', how='most anterior-inferior midline hip voxel'),
 dict(id='bladder_dome', entity='urinary_bladder', level='S3', tolerance=2, use='top', how='most cranial bladder voxel (fill-dependent)'),
]
VERTEBRAL_ORDER = [f'C{i}' for i in range(1, 8)] + [f'T{i}' for i in range(1, 13)] + [f'L{i}' for i in range(1, 6)] + [f'S{i}' for i in range(1, 6)] + ['coccyx']

# ================================================================== ENTITIES
# ---- skeleton / ruler
for lvl in VERTEBRAL_ORDER[:-1]:
    ent(f'vertebra_{lvl}', f'{lvl} vertebra', 'bone', vt=f'{lvl} vertebra', aliases=(f'vertebra {lvl}',),
        ts={'total': f'vertebrae_{lvl}'} if f'vertebrae_{lvl}' in TS_ALL['total'] else None, tier='in_vocab', prio=3, sources=('TS',))
ent('spine', 'Spine', 'bone', vt='spine', aliases=('vertebral column', 'lumbar spine', 'thoracic spine'), ts={'total_mr': 'vertebrae'},
    tier='in_vocab', prio=1, anchor=A(['C1', 'coccyx'], 'midline', 'posterior', 'obligatory'), notes='systemic anchor: cranio-caudal ruler; TS total_mr gives one merged class, use VoxTell per level if needed')
ent('sacrum', 'Sacrum', 'bone', vt='sacrum', ts={'total': 'sacrum', 'total_mr': 'sacrum'}, tier='in_vocab', prio=1,
    anchor=A(['S1', 'S5'], 'midline', 'posterior', 'obligatory'))
ent('coccyx', 'Coccyx', 'bone', vt='coccyx', tier='near', prio=3)
ent('hip', 'Hip bone (ilium, ischium, pubis)', 'bone', lat='bilateral', vt='{side} hip bone', aliases=('{side} pelvis bone', '{side} iliac bone'),
    ts={'total': {'left': 'hip_left', 'right': 'hip_right'}, 'total_mr': {'left': 'hip_left', 'right': 'hip_right'}}, tier='in_vocab', prio=1,
    anchor=A(['L5', 'coccyx'], 'bilateral', 'mid', 'obligatory'))
ent('femur', 'Femur (proximal)', 'bone', lat='bilateral', vt='{side} femur', ts={'total': {'left': 'femur_left', 'right': 'femur_right'}, 'total_mr': {'left': 'femur_left', 'right': 'femur_right'}}, tier='in_vocab', prio=3)
ent('sternum', 'Sternum', 'bone', vt='sternum', ts={'total': 'sternum'}, tier='in_vocab', prio=3)
ent('rib_cage', 'Rib cage', 'bone', lat='bilateral', vt='{side} rib cage', aliases=('{side} ribs',), tier='in_vocab', prio=2,
    ts={'total': {'left': 'rib_left_1', 'right': 'rib_right_1'}}, notes='TS gives 24 individual ribs; merge per side. ts entry lists first rib only as representative')
ent('scapula', 'Scapula', 'bone', lat='bilateral', vt='{side} scapula', ts={'total': {'left': 'scapula_left', 'right': 'scapula_right'}, 'total_mr': {'left': 'scapula_left', 'right': 'scapula_right'}}, tier='in_vocab', prio=3)
ent('clavicle', 'Clavicle', 'bone', lat='bilateral', vt='{side} clavicle', ts={'total': {'left': 'clavicula_left', 'right': 'clavicula_right'}, 'total_mr': {'left': 'clavicula_left', 'right': 'clavicula_right'}}, tier='in_vocab', prio=3)
ent('humerus', 'Humerus', 'bone', lat='bilateral', vt='{side} humerus', ts={'total': {'left': 'humerus_left', 'right': 'humerus_right'}, 'total_mr': {'left': 'humerus_left', 'right': 'humerus_right'}}, tier='in_vocab', prio=3)
ent('spinal_cord', 'Spinal cord', 'nerve', vt='spinal cord', ts={'total': 'spinal_cord', 'total_mr': 'spinal_cord'}, tier='in_vocab', prio=3)

# ---- great vessels (systemic anchors)
ent('aorta', 'Aorta', 'vessel_artery', vt='aorta', aliases=('thoracic aorta', 'abdominal aorta'), ts={'total': 'aorta', 'total_mr': 'aorta'}, tier='in_vocab', prio=1, hu=(150, 500),
    anchor=A(['T4', 'L4'], 'midline', 'posterior', 'obligatory'), notes='systemic anchor; bifurcation = L4 landmark')
ent('inferior_vena_cava', 'Inferior vena cava', 'vessel_vein', vt='inferior vena cava', aliases=('IVC',), ts={'total': 'inferior_vena_cava', 'total_mr': 'inferior_vena_cava'}, tier='in_vocab', prio=1,
    anchor=A(['T8', 'L5'], 'right', 'posterior', 'obligatory', prior=[('aorta', 'right_of', (10, 40))]))
ent('superior_vena_cava', 'Superior vena cava', 'vessel_vein', vt='superior vena cava', ts={'total': 'superior_vena_cava'}, tier='in_vocab', prio=3)
ent('iliac_artery', 'Common/external iliac artery', 'vessel_artery', lat='bilateral', vt='{side} iliac artery', aliases=('{side} common iliac artery', '{side} external iliac artery'),
    ts={'total': {'left': 'iliac_artery_left', 'right': 'iliac_artery_right'}, 'total_mr': {'left': 'iliac_artery_left', 'right': 'iliac_artery_right'}}, tier='in_vocab', prio=2,
    anchor=A(['L4', 'S3'], 'bilateral', 'posterior', 'obligatory', prior=[('aorta', 'inferior_to', (0, 60))]))
ent('iliac_vein', 'Common/external iliac vein', 'vessel_vein', lat='bilateral', vt='{side} iliac vein',
    ts={'total': {'left': 'iliac_vena_left', 'right': 'iliac_vena_right'}, 'total_mr': {'left': 'iliac_vena_left', 'right': 'iliac_vena_right'}}, tier='in_vocab', prio=2,
    anchor=A(['L5', 'S3'], 'bilateral', 'posterior', 'obligatory', prior=[('iliac_artery', 'posterior_medial_to', (0, 15))]))
ent('internal_iliac_artery', 'Internal iliac artery', 'vessel_artery', lat='bilateral', vt='{side} internal iliac artery', aliases=('{side} hypogastric artery',), tier='near', prio=2)
ent('internal_iliac_vein', 'Internal iliac vein', 'vessel_vein', lat='bilateral', vt='{side} internal iliac vein', tier='near', prio=3)
ent('portal_vein', 'Portal vein', 'vessel_vein', vt='portal vein', aliases=('main portal vein',), ts={'total': 'portal_vein_and_splenic_vein', 'total_mr': 'portal_vein_and_splenic_vein'}, tier='in_vocab', prio=1,
    notes='TS merges portal + splenic vein; VoxTell separates')
ent('splenic_vein', 'Splenic vein', 'vessel_vein', vt='splenic vein', ts={'total': 'portal_vein_and_splenic_vein', 'total_mr': 'portal_vein_and_splenic_vein'}, tier='near', prio=2)
ent('hepatic_veins', 'Hepatic veins', 'vessel_vein', vt='hepatic veins', aliases=('hepatic vein',), tier='in_vocab', prio=1, notes='MSD Task08 in VoxTell training')
ent('hepatic_artery', 'Hepatic artery', 'vessel_artery', vt='hepatic artery', aliases=('common hepatic artery', 'proper hepatic artery'), tier='near', prio=2)
ent('celiac_trunk', 'Coeliac trunk', 'vessel_artery', vt='celiac trunk', aliases=('coeliac artery',), tier='near', prio=3)
ent('superior_mesenteric_artery', 'Superior mesenteric artery', 'vessel_artery', vt='superior mesenteric artery', aliases=('SMA',), tier='near', prio=2)
ent('superior_mesenteric_vein', 'Superior mesenteric vein', 'vessel_vein', vt='superior mesenteric vein', aliases=('SMV',), tier='near', prio=2)
ent('renal_artery', 'Renal artery', 'vessel_artery', lat='bilateral', vt='{side} renal artery', tier='in_vocab', prio=2, notes='KiPA dataset in VoxTell training')
ent('renal_vein', 'Renal vein', 'vessel_vein', lat='bilateral', vt='{side} renal vein', tier='in_vocab', prio=1, notes='KiPA dataset; key for RCC T3a')
ent('gonadal_vein', 'Gonadal vein', 'vessel_vein', lat='bilateral', vt='{side} gonadal vein', aliases=('{side} testicular vein', '{side} ovarian vein'), tier='ood', prio=3)
ent('pulmonary_artery', 'Pulmonary artery', 'vessel_artery', vt='pulmonary artery', aliases=('main pulmonary artery',), tier='in_vocab', prio=2)
ent('pulmonary_vein', 'Pulmonary veins', 'vessel_vein', vt='pulmonary veins', ts={'total': 'pulmonary_vein'}, tier='in_vocab', prio=3)
ent('brachiocephalic_trunk', 'Brachiocephalic trunk', 'vessel_artery', vt='brachiocephalic trunk', ts={'total': 'brachiocephalic_trunk'}, tier='in_vocab', prio=3)
ent('subclavian_artery', 'Subclavian artery', 'vessel_artery', lat='bilateral', vt='{side} subclavian artery', ts={'total': {'left': 'subclavian_artery_left', 'right': 'subclavian_artery_right'}}, tier='in_vocab', prio=3)
ent('common_carotid_artery', 'Common carotid artery', 'vessel_artery', lat='bilateral', vt='{side} common carotid artery', ts={'total': {'left': 'common_carotid_artery_left', 'right': 'common_carotid_artery_right'}}, tier='in_vocab', prio=3)
ent('brachiocephalic_vein', 'Brachiocephalic vein', 'vessel_vein', lat='bilateral', vt='{side} brachiocephalic vein', ts={'total': {'left': 'brachiocephalic_vein_left', 'right': 'brachiocephalic_vein_right'}}, tier='in_vocab', prio=3)
ent('uterine_artery', 'Uterine artery', 'vessel_artery', lat='bilateral', vt='{side} uterine artery', tier='ood', prio=3)
ent('axillary_vessels', 'Axillary artery and vein', 'vessel_artery', lat='bilateral', vt='{side} axillary vessels', aliases=('{side} axillary artery', '{side} axillary vein'), tier='near', prio=3)

# ---- thorax
ent('lung', 'Lung', 'organ', lat='bilateral', vt='{side} lung', ts={'total': {'left': 'lung_upper_lobe_left', 'right': 'lung_upper_lobe_right'}, 'total_mr': {'left': 'lung_left', 'right': 'lung_right'}},
    tier='in_vocab', prio=1, hu=(-950, -600), vol={'min': 1500, 'max': 4500},
    anchor=A(['T1', 'T12'], 'bilateral', 'mid', 'obligatory', prior=[('heart', 'lateral_to', (0, 30))]), notes='TS total: merge lobes per side; lobes are has_part entities')
for side_lobes in [('left', ['upper', 'lower']), ('right', ['upper', 'middle', 'lower'])]:
    for lobe in side_lobes[1]:
        ent(f'lung_{side_lobes[0]}_{lobe}_lobe', f'{side_lobes[0].title()} {lobe} lobe', 'organ', lat=side_lobes[0], vt=f'{side_lobes[0]} {lobe} lobe of lung',
            ts={'total': f'lung_{lobe}_lobe_{side_lobes[0]}'}, tier='in_vocab', prio=2)
ent('trachea', 'Trachea', 'duct', vt='trachea', ts={'total': 'trachea'}, tier='in_vocab', prio=2, anchor=A(['C6', 'T4'], 'midline', 'mid', 'obligatory'))
ent('main_bronchus', 'Main bronchus', 'duct', lat='bilateral', vt='{side} main bronchus', aliases=('{side} mainstem bronchus',), tier='in_vocab', prio=2)
ent('bronchial_tree', 'Bronchial tree / airways', 'duct', vt='airways', aliases=('bronchi', 'airway tree'), tier='in_vocab', prio=3)
ent('pleura', 'Pleura', 'fascia', lat='bilateral', vt='{side} pleura', aliases=('{side} pleural surface',), tier='near', prio=2)
ent('pleural_effusion', 'Pleural effusion', 'space', lat='bilateral', vt='{side} pleural effusion', tier='in_vocab', prio=3)
ent('heart', 'Heart', 'organ', vt='heart', ts={'total': 'heart', 'total_mr': 'heart'}, tier='in_vocab', prio=1, vol={'min': 400, 'max': 1000},
    anchor=A(['T5', 'T9'], 'midline', 'anterior', 'obligatory'))
ent('pericardium', 'Pericardium', 'fascia', vt='pericardium', tier='near', prio=2)
ent('esophagus', 'Oesophagus', 'hollow_organ', vt='esophagus', aliases=('oesophagus', 'gullet'), ts={'total': 'esophagus', 'total_mr': 'esophagus'}, tier='in_vocab', prio=1,
    anchor=A(['C6', 'T11'], 'midline', 'posterior', 'obligatory', prior=[('trachea', 'posterior_to', (0, 15)), ('aorta', 'anterior_right_of', (0, 20))]))
ent('thyroid', 'Thyroid gland', 'gland', vt='thyroid gland', ts={'total': 'thyroid_gland'}, tier='in_vocab', prio=3, anchor=A(['C5', 'T1'], 'midline', 'anterior', 'surgical'))
ent('thymus', 'Thymus / anterior mediastinal fat', 'gland', vt='thymus', tier='near', prio=3)
ent('chest_wall', 'Chest wall', 'muscle', lat='bilateral', vt='{side} chest wall', aliases=('{side} thoracic wall',), tier='near', prio=2)
ent('diaphragm', 'Diaphragm', 'muscle', vt='diaphragm', tier='near', prio=2)
ent('breast', 'Breast', 'organ', lat='bilateral', vt='{side} breast', aliases=('{side} mammary gland',), tier='in_vocab', prio=1, modality=('MR', 'CT'),
    anchor=A(['T2', 'T7'], 'bilateral', 'anterior', 'surgical', prior=[('pectoralis_major', 'anterior_to', (0, 30))]), notes='Duke breast MRI in VoxTell training (breast + FGT)')
ent('fibroglandular_tissue', 'Fibroglandular tissue', 'organ', lat='bilateral', vt='{side} fibroglandular tissue', aliases=('{side} breast parenchyma',), tier='in_vocab', prio=2, modality=('MR',))
ent('pectoralis_major', 'Pectoralis major', 'muscle', lat='bilateral', vt='{side} pectoralis major muscle', ts={'abdominal_muscles': {'left': 'pectoralis_major_left', 'right': 'pectoralis_major_right'}}, tier='in_vocab', prio=2)
ent('skin', 'Skin', 'fascia', vt='skin', tier='near', prio=3)
ent('nipple', 'Nipple', 'landmark', lat='bilateral', vt='{side} nipple', tier='ood', prio=3)

# ---- upper abdomen
ent('liver', 'Liver', 'organ', vt='liver', aliases=('hepatic parenchyma',), ts={'total': 'liver', 'total_mr': 'liver'}, tier='in_vocab', prio=1, vol={'min': 900, 'max': 2500}, hu=(40, 80),
    anchor=A(['T9', 'L2'], 'right', 'anterior', 'obligatory', prior=[('kidney', 'superior_to', (0, 40)), ('diaphragm', 'inferior_to', (0, 10))]))
for seg in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']:
    ent(f'liver_segment_{seg}', f'Liver segment {seg}', 'organ', vt=f'liver segment {seg}', aliases=(f'Couinaud segment {seg}',), tier='near', prio=3)
ent('liver_lobe', 'Liver lobe', 'organ', lat='bilateral', vt='{side} lobe of the liver', aliases=('{side} hepatic lobe',), tier='in_vocab', prio=2)
ent('gallbladder', 'Gallbladder', 'hollow_organ', vt='gallbladder', ts={'total': 'gallbladder', 'total_mr': 'gallbladder'}, tier='in_vocab', prio=2, vol={'min': 10, 'max': 120},
    anchor=A(['L1', 'L2'], 'right', 'anterior', 'surgical', prior=[('liver', 'inferior_surface_of', (0, 10))]))
ent('bile_duct', 'Common bile duct', 'duct', vt='common bile duct', aliases=('bile duct', 'CBD'), tier='near', prio=2)
ent('intrahepatic_bile_ducts', 'Intrahepatic bile ducts', 'duct', vt='intrahepatic bile ducts', aliases=('biliary tree',), tier='near', prio=3)
ent('spleen', 'Spleen', 'organ', vt='spleen', ts={'total': 'spleen', 'total_mr': 'spleen'}, tier='in_vocab', prio=1, vol={'min': 80, 'max': 600},
    anchor=A(['T10', 'L1'], 'left', 'posterior', 'surgical', prior=[('kidney', 'superior_lateral_to', (0, 30)), ('stomach', 'posterior_lateral_to', (0, 20))]))
ent('stomach', 'Stomach', 'hollow_organ', vt='stomach', aliases=('gastric organ',), ts={'total': 'stomach', 'total_mr': 'stomach'}, tier='in_vocab', prio=1,
    anchor=A(['T10', 'L2'], 'left', 'anterior', 'obligatory', prior=[('liver', 'left_of', (0, 20)), ('spleen', 'medial_to', (0, 30))]))
ent('gastric_wall', 'Gastric wall', 'hollow_organ', vt='gastric wall', aliases=('stomach wall',), tier='near', prio=2)
ent('gastroesophageal_junction', 'Gastro-oesophageal junction', 'landmark', vt='gastroesophageal junction', tier='near', prio=3)
ent('duodenum', 'Duodenum', 'hollow_organ', vt='duodenum', ts={'total': 'duodenum', 'total_mr': 'duodenum'}, tier='in_vocab', prio=2,
    anchor=A(['L1', 'L3'], 'right', 'mid', 'obligatory', prior=[('pancreas', 'lateral_to_head_of', (0, 10))]))
ent('pancreas', 'Pancreas', 'organ', vt='pancreas', ts={'total': 'pancreas', 'total_mr': 'pancreas'}, tier='in_vocab', prio=1, vol={'min': 40, 'max': 150},
    anchor=A(['T12', 'L2'], 'midline', 'posterior', 'surgical', prior=[('splenic_vein', 'anterior_to', (0, 10)), ('stomach', 'posterior_to', (0, 20))]))
for part in ['head', 'body', 'tail', 'uncinate process']:
    ent(f'pancreas_{part.split()[0]}', f'Pancreatic {part}', 'organ', vt=f'pancreatic {part}', aliases=(f'pancreas {part}',), tier='near', prio=3)
ent('pancreatic_duct', 'Main pancreatic duct', 'duct', vt='pancreatic duct', tier='near', prio=3)
ent('adrenal', 'Adrenal gland', 'gland', lat='bilateral', vt='{side} adrenal gland', aliases=('{side} suprarenal gland',), ts={'total': {'left': 'adrenal_gland_left', 'right': 'adrenal_gland_right'}, 'total_mr': {'left': 'adrenal_gland_left', 'right': 'adrenal_gland_right'}},
    tier='in_vocab', prio=1, vol={'min': 2, 'max': 15}, anchor=A(['T11', 'L1'], 'bilateral', 'posterior', 'surgical', prior=[('kidney', 'superior_medial_to', (0, 20))]))
ent('kidney', 'Kidney', 'organ', lat='bilateral', vt='{side} kidney', aliases=('{side} renal organ',), ts={'total': {'left': 'kidney_left', 'right': 'kidney_right'}, 'total_mr': {'left': 'kidney_left', 'right': 'kidney_right'}},
    tier='in_vocab', prio=1, vol={'min': 90, 'max': 300}, hu=(30, 60),
    anchor=A(['T12', 'L3'], 'bilateral', 'posterior', 'surgical', prior=[('psoas', 'lateral_to', (0, 20)), ('quadratus_lumborum', 'anterior_to', (0, 15)), ('liver', 'inferior_to', (0, 40))]))
ent('renal_cortex', 'Renal cortex', 'organ', lat='bilateral', vt='{side} renal cortex', tier='near', prio=3)
ent('renal_medulla', 'Renal medulla', 'organ', lat='bilateral', vt='{side} renal medulla', tier='near', prio=3)
ent('renal_pelvis', 'Renal pelvis / collecting system', 'duct', lat='bilateral', vt='{side} renal pelvis', aliases=('{side} renal collecting system',), tier='near', prio=2)
ent('renal_sinus_fat', 'Renal sinus fat', 'fat', lat='bilateral', vt='{side} renal sinus fat', tier='ood', prio=3)
ent('perinephric_fat', 'Perinephric fat', 'fat', lat='bilateral', vt='{side} perinephric fat', aliases=('{side} perirenal fat',), tier='ood', prio=2, notes='experimental; RCC T3a criterion')
ent('gerota_fascia', "Gerota's fascia", 'fascia', lat='bilateral', vt="{side} Gerota's fascia", aliases=('{side} renal fascia',), tier='ood', prio=3, notes='experimental; RCC T4 criterion')
ent('kidney_cyst', 'Renal cyst', 'space', lat='bilateral', vt='{side} kidney cyst', ts={'total': {'left': 'kidney_cyst_left', 'right': 'kidney_cyst_right'}}, tier='in_vocab', prio=3)
ent('ureter', 'Ureter', 'duct', lat='bilateral', vt='{side} ureter', tier='near', prio=2, notes='corridor renal pelvis→bladder trigone; hydronephrosis = FIGO IIIB')
ent('small_bowel', 'Small bowel', 'hollow_organ', vt='small bowel', aliases=('small intestine', 'jejunum and ileum'), ts={'total': 'small_bowel', 'total_mr': 'small_bowel'}, tier='in_vocab', prio=2,
    anchor=A(['L2', 'S1'], 'midline', 'anterior', 'obligatory'))
ent('colon', 'Colon', 'hollow_organ', vt='colon', aliases=('large bowel', 'large intestine'), ts={'total': 'colon', 'total_mr': 'colon'}, tier='in_vocab', prio=1,
    anchor=A(['T12', 'S3'], 'bilateral', 'anterior', 'obligatory'), notes='TS merges all colon incl. rectum in MR')
for seg, lat in [('ascending', 'right'), ('transverse', 'midline'), ('descending', 'left'), ('sigmoid', 'left'), ('cecum', 'right')]:
    ent(f'colon_{seg}', f'{seg.title()} colon' if seg != 'cecum' else 'Caecum', 'hollow_organ', lat=lat, vt=f'{seg} colon' if seg != 'cecum' else 'cecum', tier='near', prio=3)
ent('rectum', 'Rectum', 'hollow_organ', vt='rectum', ts=None, tier='in_vocab', prio=1,
    anchor=A(['S3', 'coccyx'], 'midline', 'posterior', 'obligatory', prior=[('sacrum', 'anterior_to', (0, 25)), ('urinary_bladder', 'posterior_to', (0, 40))]),
    notes='TS: part of colon class; completion target on pelvic MR')
ent('mesorectum', 'Mesorectum / mesorectal fascia', 'fascia', vt='mesorectal fascia', aliases=('mesorectum',), tier='ood', prio=3, notes='experimental')
ent('anal_canal', 'Anal canal', 'hollow_organ', vt='anal canal', aliases=('anus',), tier='near', prio=3)
ent('mesentery', 'Mesentery', 'fat', vt='mesentery', aliases=('mesenteric fat',), tier='near', prio=3)
ent('omentum', 'Greater omentum', 'fat', vt='omentum', aliases=('greater omentum', 'omental fat'), tier='near', prio=3)
ent('peritoneum', 'Peritoneum', 'fascia', vt='peritoneum', aliases=('peritoneal lining',), tier='ood', prio=3)
ent('ascites', 'Ascites', 'space', vt='ascites', aliases=('peritoneal fluid',), tier='in_vocab', prio=2)

# ---- pelvis
ent('urinary_bladder', 'Urinary bladder', 'hollow_organ', vt='urinary bladder', aliases=('bladder',), ts={'total': 'urinary_bladder', 'total_mr': 'urinary_bladder'}, tier='in_vocab', prio=1, vol={'min': 20, 'max': 800},
    anchor=A(['S3', 'coccyx'], 'midline', 'anterior', 'obligatory', prior=[('hip', 'between', (0, 0)), ('sacrum', 'anterior_to', (30, 120))]))
ent('bladder_wall', 'Bladder wall', 'hollow_organ', vt='bladder wall', aliases=('urinary bladder wall',), tier='near', prio=2)
ent('perivesical_fat', 'Perivesical fat', 'fat', vt='perivesical fat', tier='ood', prio=2, notes='experimental; BLCA T3')
ent('urethra', 'Urethra', 'duct', vt='urethra', tier='near', prio=3)
ent('prostate', 'Prostate', 'gland', vt='prostate', aliases=('prostate gland',), ts={'total': 'prostate', 'total_mr': 'prostate'}, tier='in_vocab', prio=1, vol={'min': 15, 'max': 150},
    anchor=A(['S5', 'coccyx'], 'midline', 'anterior', 'sex_male', prior=[('urinary_bladder', 'inferior_to', (0, 20)), ('rectum', 'anterior_to', (0, 15)), ('pubic_symphysis', 'posterior_to', (5, 40))]))
ent('prostate_peripheral_zone', 'Prostate peripheral zone', 'gland', vt='peripheral zone of the prostate', tier='in_vocab', prio=2, modality=('MR',))
ent('prostate_transition_zone', 'Prostate transition zone', 'gland', vt='transition zone of the prostate', aliases=('central gland',), tier='in_vocab', prio=2, modality=('MR',))
ent('seminal_vesicle', 'Seminal vesicle', 'gland', lat='bilateral', vt='{side} seminal vesicle', tier='near', prio=1,
    anchor=A(['S4', 'coccyx'], 'bilateral', 'posterior', 'sex_male', prior=[('prostate', 'posterior_superior_to', (0, 20)), ('urinary_bladder', 'posterior_inferior_to', (0, 15)), ('rectum', 'anterior_to', (0, 15))]), notes='PRAD T3b')
ent('neurovascular_bundle', 'Neurovascular bundle', 'nerve', lat='bilateral', vt='{side} neurovascular bundle', tier='ood', prio=3, notes='experimental')
ent('uterus', 'Uterus', 'organ', vt='uterus', aliases=('uterine body', 'womb'), ts=None, tier='near', prio=1, vol={'min': 30, 'max': 400},
    anchor=A(['S2', 'coccyx'], 'midline', 'mid', 'sex_female', prior=[('urinary_bladder', 'posterior_superior_to', (0, 20)), ('rectum', 'anterior_to', (0, 30)), ('hip', 'between', (0, 0)), ('sacral_promontory', 'inferior_to', (0, 80))]),
    notes='completion target: TS total_mr has no uterus')
ent('endometrium', 'Endometrium / uterine cavity', 'organ', vt='endometrium', aliases=('endometrial cavity',), tier='near', prio=2, modality=('MR',))
ent('myometrium', 'Myometrium', 'organ', vt='myometrium', tier='near', prio=2, modality=('MR',), notes='UCEC T1a/b = <50% / ≥50% myometrial invasion')
ent('junctional_zone', 'Junctional zone', 'organ', vt='junctional zone of the uterus', tier='ood', prio=3, modality=('MR',))
ent('cervix', 'Uterine cervix', 'organ', vt='cervix', aliases=('uterine cervix', 'cervix uteri'), tier='near', prio=1, vol={'min': 10, 'max': 120},
    anchor=A(['S4', 'coccyx'], 'midline', 'mid', 'sex_female', prior=[('uterus', 'inferior_to', (0, 10)), ('vagina', 'superior_to', (0, 10)), ('urinary_bladder', 'posterior_to', (0, 15)), ('rectum', 'anterior_to', (0, 20))]),
    notes='completion target on pelvic MR')
ent('cervical_stroma', 'Cervical stroma', 'organ', vt='cervical stroma', tier='ood', prio=3, modality=('MR',))
ent('parametrium', 'Parametrium', 'fascia', lat='bilateral', vt='{side} parametrium', aliases=('{side} parametrial tissue',), tier='ood', prio=2, notes='experimental; FIGO IIB')
ent('vagina', 'Vagina', 'hollow_organ', vt='vagina', tier='near', prio=2,
    anchor=A(['S5', 'coccyx'], 'midline', 'mid', 'sex_female', prior=[('cervix', 'inferior_to', (0, 10)), ('urethra', 'posterior_to', (0, 10)), ('rectum', 'anterior_to', (0, 10))]))
ent('vagina_upper_two_thirds', 'Upper two-thirds of vagina', 'hollow_organ', vt='upper vagina', tier='ood', prio=3, notes='FIGO IIA vs IIIA boundary')
ent('ovary', 'Ovary', 'gland', lat='bilateral', vt='{side} ovary', tier='near', prio=2,
    anchor=A(['S1', 'S4'], 'bilateral', 'mid', 'sex_female', prior=[('uterus', 'lateral_to', (10, 50)), ('iliac_artery', 'anterior_medial_to', (0, 25))]))
ent('adnexa', 'Adnexa (ovary + tube)', 'gland', lat='bilateral', vt='{side} adnexa', aliases=('{side} ovary and fallopian tube',), tier='near', prio=3)
ent('pelvic_sidewall', 'Pelvic sidewall', 'muscle', lat='bilateral', vt='{side} pelvic sidewall', aliases=('{side} obturator internus muscle',), tier='near', prio=2)
ent('obturator_internus', 'Obturator internus', 'muscle', lat='bilateral', vt='{side} obturator internus muscle', tier='near', prio=3)
ent('levator_ani', 'Levator ani / pelvic floor', 'muscle', vt='levator ani muscle', aliases=('pelvic floor muscles',), tier='near', prio=2)
ent('external_urethral_sphincter', 'External urethral sphincter', 'muscle', vt='external urethral sphincter', tier='ood', prio=3)
ent('iliopsoas', 'Iliopsoas', 'muscle', lat='bilateral', vt='{side} iliopsoas muscle', ts={'total': {'left': 'iliopsoas_left', 'right': 'iliopsoas_right'}, 'total_mr': {'left': 'iliopsoas_left', 'right': 'iliopsoas_right'}}, tier='in_vocab', prio=2,
    anchor=A(['L1', 'coccyx'], 'bilateral', 'posterior', 'obligatory'))
ent('psoas', 'Psoas major', 'muscle', lat='bilateral', vt='{side} psoas muscle', ts={'abdominal_muscles': {'left': 'psoas_major_left', 'right': 'psoas_major_right'}, 'total': {'left': 'iliopsoas_left', 'right': 'iliopsoas_right'}}, tier='in_vocab', prio=2,
    anchor=A(['T12', 'L5'], 'bilateral', 'posterior', 'obligatory', prior=[('spine', 'lateral_to', (0, 15))]), notes='TS total merges psoas into iliopsoas')
ent('quadratus_lumborum', 'Quadratus lumborum', 'muscle', lat='bilateral', vt='{side} quadratus lumborum muscle', ts={'abdominal_muscles': {'left': 'quadratus_lumborum_left', 'right': 'quadratus_lumborum_right'}}, tier='near', prio=3)
ent('gluteus', 'Gluteal muscles', 'muscle', lat='bilateral', vt='{side} gluteal muscles', ts={'total': {'left': 'gluteus_maximus_left', 'right': 'gluteus_maximus_right'}, 'total_mr': {'left': 'gluteus_maximus_left', 'right': 'gluteus_maximus_right'}}, tier='in_vocab', prio=3, notes='merge maximus/medius/minimus')
ent('autochthon', 'Autochthonous back muscles (erector spinae)', 'muscle', lat='bilateral', vt='{side} erector spinae muscle', ts={'total': {'left': 'autochthon_left', 'right': 'autochthon_right'}, 'total_mr': {'left': 'autochthon_left', 'right': 'autochthon_right'}}, tier='in_vocab', prio=3)
ent('rectus_abdominis', 'Rectus abdominis', 'muscle', lat='bilateral', vt='{side} rectus abdominis muscle', ts={'abdominal_muscles': {'left': 'rectus_abdominis_left', 'right': 'rectus_abdominis_right'}}, tier='near', prio=3)
ent('abdominal_wall', 'Anterior abdominal wall', 'muscle', vt='anterior abdominal wall', aliases=('abdominal wall muscles',), tier='near', prio=3)

# ---- body composition (systemic)
ent('subcutaneous_fat', 'Subcutaneous adipose tissue', 'fat', vt='subcutaneous fat', aliases=('subcutaneous adipose tissue', 'SAT'), ts={'tissue_types': 'subcutaneous_fat'}, tier='in_vocab', prio=3, hu=(-190, -30),
    anchor=A(['C1', 'coccyx'], 'midline', 'mid', 'obligatory'), notes='TS tissue_types is a licensed model; VoxTell fallback')
ent('visceral_fat', 'Visceral adipose tissue', 'fat', vt='visceral fat', aliases=('intra-abdominal fat', 'torso fat', 'VAT'), ts={'tissue_types': 'torso_fat'}, tier='in_vocab', prio=3, hu=(-190, -30),
    anchor=A(['T10', 'S1'], 'midline', 'anterior', 'obligatory'))
ent('skeletal_muscle', 'Skeletal muscle (whole)', 'muscle', vt='skeletal muscle', ts={'tissue_types': 'skeletal_muscle'}, tier='in_vocab', prio=3, hu=(-29, 150),
    anchor=A(['C1', 'coccyx'], 'midline', 'mid', 'obligatory'), notes='L3-level cross-section = sarcopenia index')
ent('body_trunk', 'Body (trunk)', 'region', vt='body', ts={'body': 'body_trunc'}, tier='in_vocab', prio=3)

# ---- lymph node stations (AJCC regional nodes by primary site)
LN = [
 ('ln_mediastinal', 'Mediastinal lymph nodes', 'mediastinal lymph nodes', ('mediastinal nodes', 'thoracic lymph nodes'), 'in_vocab', 'between lungs, T3–T8, around trachea/carina/oesophagus', ['T3', 'T8']),
 ('ln_hilar', 'Hilar lymph nodes', '{side} hilar lymph nodes', ('{side} pulmonary hilar nodes',), 'near', 'around main bronchus and pulmonary vessels', ['T5', 'T7']),
 ('ln_supraclavicular', 'Supraclavicular lymph nodes', '{side} supraclavicular lymph nodes', (), 'near', 'above clavicle lateral to carotid', ['C7', 'T1']),
 ('ln_axillary', 'Axillary lymph nodes', '{side} axillary lymph nodes', ('{side} axillary nodes',), 'near', 'axilla lateral to pectoralis minor', ['T2', 'T5']),
 ('ln_internal_mammary', 'Internal mammary lymph nodes', '{side} internal mammary lymph nodes', (), 'ood', 'parasternal along internal thoracic vessels', ['T1', 'T6']),
 ('ln_celiac', 'Coeliac lymph nodes', 'celiac lymph nodes', ('coeliac nodes',), 'near', 'around coeliac trunk', ['T12', 'L1']),
 ('ln_hepatic_hilar', 'Hepatic hilar / hepatoduodenal nodes', 'hepatic hilar lymph nodes', ('portal lymph nodes', 'hepatoduodenal ligament nodes'), 'near', 'porta hepatis around portal vein', ['T12', 'L1']),
 ('ln_perigastric', 'Perigastric lymph nodes', 'perigastric lymph nodes', ('gastric lymph nodes',), 'near', 'along lesser and greater curvature', ['T10', 'L2']),
 ('ln_renal_hilar', 'Renal hilar lymph nodes', '{side} renal hilar lymph nodes', (), 'near', 'around renal vessels', ['L1', 'L2']),
 ('ln_paraaortic', 'Para-aortic / retroperitoneal lymph nodes', 'para-aortic lymph nodes', ('retroperitoneal lymph nodes', 'paracaval lymph nodes', 'aortocaval lymph nodes'), 'in_vocab', 'around aorta and IVC, T12–L4', ['T12', 'L4']),
 ('ln_mesenteric', 'Mesenteric lymph nodes', 'mesenteric lymph nodes', (), 'near', 'along SMA/SMV in mesentery', ['L1', 'L4']),
 ('ln_pericolic', 'Pericolic lymph nodes', 'pericolic lymph nodes', ('mesocolic lymph nodes',), 'ood', 'along colonic vessels', ['T12', 'S1']),
 ('ln_common_iliac', 'Common iliac lymph nodes', '{side} common iliac lymph nodes', (), 'near', 'along common iliac vessels L4–S1', ['L4', 'S1']),
 ('ln_external_iliac', 'External iliac lymph nodes', '{side} external iliac lymph nodes', (), 'near', 'along external iliac vessels', ['S1', 'S4']),
 ('ln_internal_iliac', 'Internal iliac lymph nodes', '{side} internal iliac lymph nodes', ('{side} hypogastric lymph nodes',), 'near', 'along internal iliac vessels', ['S1', 'S4']),
 ('ln_obturator', 'Obturator lymph nodes', '{side} obturator lymph nodes', (), 'near', 'medial to obturator internus, below external iliac vein', ['S2', 'S5']),
 ('ln_presacral', 'Presacral lymph nodes', 'presacral lymph nodes', (), 'ood', 'anterior to sacrum', ['S1', 'S4']),
 ('ln_inguinal', 'Inguinal lymph nodes', '{side} inguinal lymph nodes', (), 'in_vocab', 'below inguinal ligament along femoral vessels', ['S5', 'coccyx']),
 ('ln_pelvic', 'Pelvic lymph nodes (aggregate)', 'pelvic lymph nodes', (), 'in_vocab', 'aggregate of iliac/obturator/presacral', ['L5', 'coccyx']),
 ('ln_abdominal', 'Abdominal lymph nodes (aggregate)', 'abdominal lymph nodes', ('enlarged abdominal lymph nodes',), 'in_vocab', 'aggregate', ['T10', 'L5']),
]
for id, name, vt, al, tier, region, span in LN:
    ent(id, name, 'lymph_station', lat='bilateral' if '{side}' in vt else 'none', vt=vt, aliases=al, tier=tier, prio=2, notes=f'gate region: {region}; span {span[0]}–{span[1]}', sources=('AJCC8', 'FMA'))

# ================================================================== RELATIONS (anchor profiles)
S = 'confined'; C = 'beyond_capsule'; ADJ = 'adjacent_invasion'; V = 'vascular_invasion'; O = 'obstruction'; N = 'nodal'; W = 'wall_invasion'; L = 'location'

# -- kidney (full depth)
for p in ['renal_cortex', 'renal_medulla', 'renal_pelvis', 'renal_sinus_fat']: rel('kidney', 'has_part', p, staging=[S, L])
rel('kidney', 'invested_by', 'perinephric_fat', staging=[C], note='RCC T3a: perinephric / sinus fat invasion')
rel('kidney', 'invested_by', 'gerota_fascia', staging=[C], note='RCC T4: beyond Gerota fascia')
rel('kidney', 'adjacent_to', 'adrenal', direction='superior_medial', contact='abuts', staging=[ADJ], note='T4 if contiguous adrenal invasion')
rel('kidney', 'adjacent_to', 'liver', side='right', direction='superior', contact='near', staging=[ADJ])
rel('kidney', 'adjacent_to', 'spleen', side='left', direction='superior_lateral', contact='near', staging=[ADJ])
rel('kidney', 'adjacent_to', 'pancreas', direction='anterior', contact='near', staging=[ADJ], note='head on right, tail on left')
rel('kidney', 'adjacent_to', 'duodenum', side='right', direction='anterior_medial', contact='abuts', staging=[ADJ])
rel('kidney', 'adjacent_to', 'colon', direction='anterior_lateral', contact='abuts', staging=[ADJ], note='ascending right, descending left')
rel('kidney', 'adjacent_to', 'psoas', direction='medial', contact='abuts', staging=[ADJ, W])
rel('kidney', 'adjacent_to', 'quadratus_lumborum', direction='posterior', contact='abuts', staging=[W])
rel('kidney', 'adjacent_to', 'diaphragm', direction='superior_posterior', contact='near', staging=[ADJ])
rel('kidney', 'supplied_by', 'renal_artery', staging=[V]); rel('renal_artery', 'supplied_by', 'aorta', side='any')
rel('kidney', 'drained_by', 'renal_vein', staging=[V], note='RCC T3a: renal vein / segmental branch thrombus')
rel('renal_vein', 'drained_by', 'inferior_vena_cava', side='any', staging=[V], note='T3b infradiaphragmatic IVC, T3c supradiaphragmatic')
rel('kidney', 'drained_by', 'gonadal_vein', side='left', staging=[V], note='left gonadal vein drains to left renal vein')
rel('kidney', 'has_duct', 'ureter', staging=[O], note='corridor renal pelvis → bladder trigone')
rel('kidney', 'drains_lymph_to', 'ln_renal_hilar', staging=[N]); rel('kidney', 'drains_lymph_to', 'ln_paraaortic', side='any', staging=[N])
rel('kidney', 'lies_in', 'retroperitoneum', side='any')

# -- liver (full depth)
for seg in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']: rel('liver', 'has_part', f'liver_segment_{seg}', side='any', staging=[L])
rel('liver', 'has_part', 'liver_lobe', side='any', staging=[L]); rel('liver', 'has_part', 'intrahepatic_bile_ducts', side='any', staging=[O])
rel('liver', 'invested_by', 'peritoneum', side='any', staging=[C], note='HCC T4: perforation of visceral peritoneum')
rel('liver', 'adjacent_to', 'diaphragm', side='any', direction='superior', contact='abuts', staging=[ADJ], note='HCC T4 direct invasion')
rel('liver', 'adjacent_to', 'gallbladder', side='any', direction='inferior', contact='abuts', staging=[ADJ, 'HCC T4'])
rel('liver', 'adjacent_to', 'stomach', side='any', direction='left_inferior', contact='near', staging=[ADJ])
rel('liver', 'adjacent_to', 'duodenum', side='any', direction='inferior', contact='abuts', staging=[ADJ])
rel('liver', 'adjacent_to', 'colon', side='any', direction='inferior', contact='near', staging=[ADJ], note='hepatic flexure')
rel('liver', 'adjacent_to', 'kidney', side='right', direction='inferior_posterior', contact='abuts', staging=[ADJ])
rel('liver', 'adjacent_to', 'adrenal', side='right', direction='posterior_medial', contact='abuts', staging=[ADJ])
rel('liver', 'adjacent_to', 'heart', side='any', direction='superior', contact='separated_by_diaphragm', staging=[])
rel('liver', 'adjacent_to', 'chest_wall', side='right', direction='lateral', contact='separated_by_diaphragm', staging=[W])
rel('liver', 'supplied_by', 'hepatic_artery', side='any', staging=[V]); rel('hepatic_artery', 'supplied_by', 'celiac_trunk', side='any'); rel('celiac_trunk', 'supplied_by', 'aorta', side='any')
rel('liver', 'supplied_by', 'portal_vein', side='any', staging=[V], note='HCC T4 / BCLC: portal vein tumour thrombus (main or branch)')
rel('portal_vein', 'drained_by', 'superior_mesenteric_vein', side='any'); rel('portal_vein', 'drained_by', 'splenic_vein', side='any')
rel('liver', 'drained_by', 'hepatic_veins', side='any', staging=[V], note='HCC T4: hepatic vein invasion'); rel('hepatic_veins', 'drained_by', 'inferior_vena_cava', side='any', staging=[V])
rel('liver', 'has_duct', 'bile_duct', side='any', staging=[O], note='corridor porta hepatis → duodenum')
rel('liver', 'drains_lymph_to', 'ln_hepatic_hilar', side='any', staging=[N]); rel('liver', 'drains_lymph_to', 'ln_celiac', side='any', staging=[N]); rel('liver', 'drains_lymph_to', 'ln_paraaortic', side='any', staging=[N])
rel('liver', 'lies_in', 'peritoneal_cavity', side='any'); rel('liver', 'lies_in', 'upper_abdomen', side='any')

# -- uterus / cervix / vagina / ovary (full depth, completion targets)
for p in ['endometrium', 'myometrium', 'junctional_zone', 'cervix']: rel('uterus', 'has_part', p, side='any', staging=[S, L], note='UCEC T1a/T1b = depth of myometrial invasion' if p == 'myometrium' else None)
rel('uterus', 'invested_by', 'peritoneum', side='any', staging=[C], note='uterine serosa; UCEC T3a')
rel('uterus', 'adjacent_to', 'urinary_bladder', side='any', direction='anterior_inferior', contact='abuts', staging=[ADJ], note='UCEC/CESC T4: bladder mucosa')
rel('uterus', 'adjacent_to', 'rectum', side='any', direction='posterior', contact='near', staging=[ADJ], note='T4: rectal mucosa')
rel('uterus', 'adjacent_to', 'sigmoid colon' if False else 'colon_sigmoid', side='any', direction='posterior_superior', contact='near', staging=[ADJ])
rel('uterus', 'adjacent_to', 'small_bowel', side='any', direction='superior', contact='near', staging=[ADJ])
rel('uterus', 'adjacent_to', 'ovary', side='any', direction='lateral', contact='near', staging=[ADJ], note='UCEC T3a: adnexal involvement')
rel('uterus', 'adjacent_to', 'pelvic_sidewall', side='any', direction='lateral', contact='separated_by_parametrium', staging=[W], note='FIGO IIIB: pelvic wall')
rel('uterus', 'supplied_by', 'uterine_artery', side='any', staging=[]); rel('uterine_artery', 'supplied_by', 'internal_iliac_artery', side='any')
rel('uterus', 'drains_lymph_to', 'ln_obturator', side='any', staging=[N]); rel('uterus', 'drains_lymph_to', 'ln_internal_iliac', side='any', staging=[N]); rel('uterus', 'drains_lymph_to', 'ln_external_iliac', side='any', staging=[N])
rel('uterus', 'drains_lymph_to', 'ln_common_iliac', side='any', staging=[N]); rel('uterus', 'drains_lymph_to', 'ln_paraaortic', side='any', staging=[N], note='FIGO IIIC2')
rel('uterus', 'lies_in', 'pelvis_true', side='any')
rel('cervix', 'has_part', 'cervical_stroma', side='any', staging=[S], note='FIGO IB by stromal invasion depth / size')
rel('cervix', 'invested_by', 'parametrium', side='any', staging=[C], note='FIGO IIB: parametrial invasion')
rel('cervix', 'adjacent_to', 'vagina', side='any', direction='inferior', contact='continuous', staging=[ADJ], note='IIA upper 2/3, IIIA lower 1/3')
rel('cervix', 'adjacent_to', 'urinary_bladder', side='any', direction='anterior', contact='abuts', staging=[ADJ], note='IVA bladder mucosa')
rel('cervix', 'adjacent_to', 'rectum', side='any', direction='posterior', contact='near', staging=[ADJ], note='IVA rectal mucosa')
rel('cervix', 'adjacent_to', 'pelvic_sidewall', side='any', direction='lateral', contact='separated_by_parametrium', staging=[W], note='IIIB pelvic wall')
rel('cervix', 'adjacent_to', 'ureter', side='any', direction='lateral', contact='near', staging=[O], note='IIIB hydronephrosis / non-functioning kidney')
rel('cervix', 'drains_lymph_to', 'ln_obturator', side='any', staging=[N]); rel('cervix', 'drains_lymph_to', 'ln_internal_iliac', side='any', staging=[N]); rel('cervix', 'drains_lymph_to', 'ln_external_iliac', side='any', staging=[N])
rel('cervix', 'drains_lymph_to', 'ln_presacral', side='any', staging=[N]); rel('cervix', 'drains_lymph_to', 'ln_common_iliac', side='any', staging=[N]); rel('cervix', 'drains_lymph_to', 'ln_paraaortic', side='any', staging=[N], note='IIIC2')
rel('cervix', 'lies_in', 'pelvis_true', side='any')
rel('vagina', 'has_part', 'vagina_upper_two_thirds', side='any', staging=[L]); rel('vagina', 'adjacent_to', 'urethra', side='any', direction='anterior', contact='abuts', staging=[ADJ])
rel('vagina', 'adjacent_to', 'rectum', side='any', direction='posterior', contact='abuts', staging=[ADJ]); rel('vagina', 'adjacent_to', 'levator_ani', side='any', direction='lateral', contact='abuts', staging=[W])
rel('ovary', 'has_part', 'adnexa', staging=[L]); rel('ovary', 'adjacent_to', 'uterus', direction='medial', contact='near', staging=[ADJ])
rel('ovary', 'adjacent_to', 'iliac_artery', direction='posterior_lateral', contact='near', staging=[W], note='ovarian fossa')
rel('ovary', 'adjacent_to', 'colon_sigmoid', side='left', direction='posterior', contact='near', staging=[ADJ]); rel('ovary', 'adjacent_to', 'small_bowel', side='any', direction='superior', contact='near', staging=[ADJ])
rel('ovary', 'invested_by', 'peritoneum', side='any', staging=[C], note='FIGO IC capsule rupture / surface')
rel('ovary', 'adjacent_to', 'omentum', side='any', direction='superior', contact='near', staging=['peritoneal_spread'], note='FIGO III omental disease')
rel('ovary', 'drained_by', 'gonadal_vein', staging=[]); rel('ovary', 'drains_lymph_to', 'ln_paraaortic', side='any', staging=[N]); rel('ovary', 'drains_lymph_to', 'ln_external_iliac', side='any', staging=[N]); rel('ovary', 'drains_lymph_to', 'ln_obturator', side='any', staging=[N])
rel('ovary', 'lies_in', 'pelvis_true', side='any')

# -- prostate (full depth)
for p in ['prostate_peripheral_zone', 'prostate_transition_zone']: rel('prostate', 'has_part', p, side='any', staging=[S, L])
rel('prostate', 'invested_by', 'neurovascular_bundle', side='any', staging=[C], note='T3a extraprostatic extension marker')
rel('prostate', 'adjacent_to', 'seminal_vesicle', side='any', direction='posterior_superior', contact='abuts', staging=[ADJ], note='T3b seminal vesicle invasion')
rel('prostate', 'adjacent_to', 'urinary_bladder', side='any', direction='superior', contact='abuts', staging=[ADJ], note='T4 bladder neck')
rel('prostate', 'adjacent_to', 'rectum', side='any', direction='posterior', contact='separated_by_fascia', staging=[ADJ], note='T4 rectum')
rel('prostate', 'adjacent_to', 'levator_ani', side='any', direction='lateral_inferior', contact='near', staging=[W], note='T4 levator / pelvic wall')
rel('prostate', 'adjacent_to', 'external_urethral_sphincter', side='any', direction='inferior', contact='continuous', staging=[ADJ], note='T4 external sphincter')
rel('prostate', 'adjacent_to', 'pelvic_sidewall', side='any', direction='lateral', contact='near', staging=[W])
rel('prostate', 'has_duct', 'urethra', side='any', staging=[O])
rel('prostate', 'supplied_by', 'internal_iliac_artery', side='any', staging=[])
rel('prostate', 'drains_lymph_to', 'ln_obturator', side='any', staging=[N]); rel('prostate', 'drains_lymph_to', 'ln_internal_iliac', side='any', staging=[N]); rel('prostate', 'drains_lymph_to', 'ln_external_iliac', side='any', staging=[N])
rel('prostate', 'drains_lymph_to', 'ln_presacral', side='any', staging=[N]); rel('prostate', 'drains_lymph_to', 'ln_common_iliac', side='any', staging=['M1a'])
rel('prostate', 'lies_in', 'pelvis_true', side='any')

# -- bladder
rel('urinary_bladder', 'has_part', 'bladder_wall', side='any', staging=[S], note='T2 muscularis propria')
rel('urinary_bladder', 'invested_by', 'perivesical_fat', side='any', staging=[C], note='T3 perivesical tissue')
rel('urinary_bladder', 'adjacent_to', 'prostate', side='any', direction='inferior', contact='abuts', staging=[ADJ], note='T4a prostatic stroma')
rel('urinary_bladder', 'adjacent_to', 'seminal_vesicle', side='any', direction='posterior_inferior', contact='abuts', staging=[ADJ], note='T4a')
rel('urinary_bladder', 'adjacent_to', 'uterus', side='any', direction='posterior_superior', contact='abuts', staging=[ADJ], note='T4a uterus')
rel('urinary_bladder', 'adjacent_to', 'vagina', side='any', direction='posterior', contact='abuts', staging=[ADJ], note='T4a vagina')
rel('urinary_bladder', 'adjacent_to', 'rectum', side='any', direction='posterior', contact='near', staging=[ADJ])
rel('urinary_bladder', 'adjacent_to', 'pelvic_sidewall', side='any', direction='lateral', contact='near', staging=[W], note='T4b pelvic wall')
rel('urinary_bladder', 'adjacent_to', 'abdominal_wall', side='any', direction='anterior_superior', contact='near', staging=[W], note='T4b abdominal wall')
rel('urinary_bladder', 'adjacent_to', 'small_bowel', side='any', direction='superior', contact='near', staging=[ADJ]); rel('urinary_bladder', 'adjacent_to', 'colon_sigmoid', side='any', direction='superior_posterior', contact='near', staging=[ADJ])
rel('urinary_bladder', 'has_duct', 'urethra', side='any', staging=[O]); rel('urinary_bladder', 'has_duct', 'ureter', side='any', staging=[O], note='ureteric orifice; hydronephrosis')
rel('urinary_bladder', 'drains_lymph_to', 'ln_obturator', side='any', staging=[N]); rel('urinary_bladder', 'drains_lymph_to', 'ln_internal_iliac', side='any', staging=[N]); rel('urinary_bladder', 'drains_lymph_to', 'ln_external_iliac', side='any', staging=[N])
rel('urinary_bladder', 'drains_lymph_to', 'ln_presacral', side='any', staging=[N]); rel('urinary_bladder', 'drains_lymph_to', 'ln_common_iliac', side='any', staging=['N3'])
rel('urinary_bladder', 'lies_in', 'pelvis_true', side='any'); rel('urinary_bladder', 'lies_in', 'perivesical_space', side='any')

# -- rectum
rel('rectum', 'invested_by', 'mesorectum', side='any', staging=[C], note='T3 into mesorectal fat; CRM = mesorectal fascia')
rel('rectum', 'adjacent_to', 'sacrum', side='any', direction='posterior', contact='separated_by_presacral_fat', staging=[W])
rel('rectum', 'adjacent_to', 'prostate', side='any', direction='anterior', contact='separated_by_fascia', staging=[ADJ]); rel('rectum', 'adjacent_to', 'seminal_vesicle', side='any', direction='anterior', contact='near', staging=[ADJ])
rel('rectum', 'adjacent_to', 'vagina', side='any', direction='anterior', contact='abuts', staging=[ADJ]); rel('rectum', 'adjacent_to', 'uterus', side='any', direction='anterior', contact='near', staging=[ADJ])
rel('rectum', 'adjacent_to', 'urinary_bladder', side='any', direction='anterior', contact='near', staging=[ADJ]); rel('rectum', 'adjacent_to', 'levator_ani', side='any', direction='inferior_lateral', contact='abuts', staging=[W])
rel('rectum', 'adjacent_to', 'anal_canal', side='any', direction='inferior', contact='continuous', staging=[L])
rel('rectum', 'drains_lymph_to', 'ln_mesenteric', side='any', staging=[N], note='mesorectal / superior rectal nodes'); rel('rectum', 'drains_lymph_to', 'ln_internal_iliac', side='any', staging=[N]); rel('rectum', 'drains_lymph_to', 'ln_presacral', side='any', staging=[N])
rel('rectum', 'drains_lymph_to', 'ln_inguinal', side='any', staging=['N (low rectal/anal)'])
rel('rectum', 'lies_in', 'pelvis_true', side='any'); rel('rectum', 'lies_in', 'presacral_space', side='any')

# -- colon
for seg in ['cecum', 'ascending', 'transverse', 'descending', 'sigmoid']: rel('colon', 'has_part', f'colon_{seg}', side='any', staging=[L])
rel('colon', 'has_part', 'rectum', side='any', staging=[L], note='TS colon class includes rectum')
rel('colon', 'invested_by', 'peritoneum', side='any', staging=[C], note='T4a visceral peritoneum'); rel('colon', 'invested_by', 'mesentery', side='any', staging=[C], note='T3 pericolic fat')
rel('colon', 'adjacent_to', 'liver', side='any', direction='superior', contact='near', staging=[ADJ], note='T4b'); rel('colon', 'adjacent_to', 'stomach', side='any', direction='superior', contact='near', staging=[ADJ])
rel('colon', 'adjacent_to', 'spleen', side='any', direction='superior', contact='near', staging=[ADJ]); rel('colon', 'adjacent_to', 'kidney', side='any', direction='posterior', contact='near', staging=[ADJ])
rel('colon', 'adjacent_to', 'duodenum', side='any', direction='posterior', contact='abuts', staging=[ADJ]); rel('colon', 'adjacent_to', 'pancreas', side='any', direction='posterior', contact='near', staging=[ADJ])
rel('colon', 'adjacent_to', 'small_bowel', side='any', direction='medial', contact='abuts', staging=[ADJ]); rel('colon', 'adjacent_to', 'urinary_bladder', side='any', direction='inferior', contact='near', staging=[ADJ])
rel('colon', 'adjacent_to', 'uterus', side='any', direction='inferior', contact='near', staging=[ADJ]); rel('colon', 'adjacent_to', 'abdominal_wall', side='any', direction='anterior', contact='near', staging=[W])
rel('colon', 'adjacent_to', 'iliopsoas', side='any', direction='posterior', contact='near', staging=[W])
rel('colon', 'supplied_by', 'superior_mesenteric_artery', side='any', staging=[V]); rel('colon', 'drained_by', 'superior_mesenteric_vein', side='any', staging=[V])
rel('colon', 'drains_lymph_to', 'ln_pericolic', side='any', staging=[N]); rel('colon', 'drains_lymph_to', 'ln_mesenteric', side='any', staging=[N]); rel('colon', 'drains_lymph_to', 'ln_paraaortic', side='any', staging=['M1'])
rel('colon', 'lies_in', 'peritoneal_cavity', side='any')

# -- stomach
rel('stomach', 'has_part', 'gastric_wall', side='any', staging=[S], note='T1–T3 wall layers'); rel('stomach', 'has_part', 'gastroesophageal_junction', side='any', staging=[L])
rel('stomach', 'invested_by', 'peritoneum', side='any', staging=[C], note='T4a serosa'); rel('stomach', 'invested_by', 'omentum', side='any', staging=[C], note='T3 into omentum without serosal breach')
rel('stomach', 'adjacent_to', 'liver', side='any', direction='right_superior', contact='abuts', staging=[ADJ], note='T4b'); rel('stomach', 'adjacent_to', 'spleen', side='any', direction='left_posterior', contact='near', staging=[ADJ])
rel('stomach', 'adjacent_to', 'pancreas', side='any', direction='posterior', contact='near', staging=[ADJ]); rel('stomach', 'adjacent_to', 'colon', side='any', direction='inferior', contact='near', staging=[ADJ], note='transverse colon')
rel('stomach', 'adjacent_to', 'diaphragm', side='any', direction='superior', contact='abuts', staging=[ADJ]); rel('stomach', 'adjacent_to', 'adrenal', side='left', direction='posterior', contact='near', staging=[ADJ])
rel('stomach', 'adjacent_to', 'kidney', side='left', direction='posterior', contact='near', staging=[ADJ]); rel('stomach', 'adjacent_to', 'abdominal_wall', side='any', direction='anterior', contact='near', staging=[W])
rel('stomach', 'adjacent_to', 'esophagus', side='any', direction='superior', contact='continuous', staging=[L]); rel('stomach', 'adjacent_to', 'duodenum', side='any', direction='right', contact='continuous', staging=[L])
rel('stomach', 'supplied_by', 'celiac_trunk', side='any', staging=[V]); rel('stomach', 'drained_by', 'portal_vein', side='any', staging=[V]); rel('stomach', 'drained_by', 'splenic_vein', side='any', staging=[V])
rel('stomach', 'drains_lymph_to', 'ln_perigastric', side='any', staging=[N]); rel('stomach', 'drains_lymph_to', 'ln_celiac', side='any', staging=[N]); rel('stomach', 'drains_lymph_to', 'ln_hepatic_hilar', side='any', staging=[N]); rel('stomach', 'drains_lymph_to', 'ln_paraaortic', side='any', staging=['M1'])
rel('stomach', 'lies_in', 'peritoneal_cavity', side='any'); rel('stomach', 'lies_in', 'upper_abdomen', side='any')

# -- oesophagus
rel('esophagus', 'adjacent_to', 'trachea', side='any', direction='anterior', contact='abuts', staging=[ADJ], note='T4b trachea'); rel('esophagus', 'adjacent_to', 'main_bronchus', side='left', direction='anterior', contact='abuts', staging=[ADJ])
rel('esophagus', 'adjacent_to', 'aorta', side='any', direction='left_posterior', contact='abuts', staging=[ADJ], note='T4b aorta'); rel('esophagus', 'adjacent_to', 'heart', side='any', direction='anterior', contact='separated_by_pericardium', staging=[ADJ])
rel('esophagus', 'adjacent_to', 'pericardium', side='any', direction='anterior', contact='abuts', staging=[ADJ], note='T4a pericardium'); rel('esophagus', 'adjacent_to', 'pleura', side='any', direction='lateral', contact='abuts', staging=[ADJ], note='T4a pleura')
rel('esophagus', 'adjacent_to', 'diaphragm', side='any', direction='inferior', contact='abuts', staging=[ADJ], note='T4a diaphragm'); rel('esophagus', 'adjacent_to', 'spine', side='any', direction='posterior', contact='near', staging=[ADJ], note='T4b vertebral body')
rel('esophagus', 'adjacent_to', 'stomach', side='any', direction='inferior', contact='continuous', staging=[L]); rel('esophagus', 'adjacent_to', 'thyroid', side='any', direction='anterior', contact='near', staging=[ADJ])
rel('esophagus', 'adjacent_to', 'lung', side='any', direction='lateral', contact='separated_by_pleura', staging=[ADJ])
rel('esophagus', 'drains_lymph_to', 'ln_mediastinal', side='any', staging=[N]); rel('esophagus', 'drains_lymph_to', 'ln_supraclavicular', side='any', staging=[N]); rel('esophagus', 'drains_lymph_to', 'ln_perigastric', side='any', staging=[N]); rel('esophagus', 'drains_lymph_to', 'ln_celiac', side='any', staging=[N])
rel('esophagus', 'lies_in', 'mediastinum', side='any')

# -- pancreas
for part in ['head', 'body', 'tail', 'uncinate']: rel('pancreas', 'has_part', f'pancreas_{part}', side='any', staging=[L])
rel('pancreas', 'has_duct', 'pancreatic_duct', side='any', staging=[O]); rel('pancreas', 'adjacent_to', 'bile_duct', side='any', direction='posterior_head', contact='abuts', staging=[O, ADJ], note='biliary obstruction')
rel('pancreas', 'adjacent_to', 'duodenum', side='any', direction='right', contact='abuts', staging=[ADJ]); rel('pancreas', 'adjacent_to', 'stomach', side='any', direction='anterior', contact='near', staging=[ADJ])
rel('pancreas', 'adjacent_to', 'spleen', side='any', direction='left', contact='abuts', staging=[ADJ]); rel('pancreas', 'adjacent_to', 'kidney', side='left', direction='posterior', contact='near', staging=[ADJ])
rel('pancreas', 'adjacent_to', 'adrenal', side='left', direction='posterior', contact='near', staging=[ADJ]); rel('pancreas', 'adjacent_to', 'colon', side='any', direction='anterior_inferior', contact='near', staging=[ADJ])
rel('pancreas', 'adjacent_to', 'inferior_vena_cava', side='any', direction='posterior', contact='abuts', staging=[V]); rel('pancreas', 'adjacent_to', 'aorta', side='any', direction='posterior', contact='near', staging=[V])
rel('pancreas', 'supplied_by', 'celiac_trunk', side='any', staging=[V], note='T4 coeliac axis'); rel('pancreas', 'supplied_by', 'superior_mesenteric_artery', side='any', staging=[V], note='T4 SMA')
rel('pancreas', 'supplied_by', 'hepatic_artery', side='any', staging=[V], note='T4 common hepatic artery'); rel('pancreas', 'drained_by', 'splenic_vein', side='any', staging=[V]); rel('pancreas', 'drained_by', 'superior_mesenteric_vein', side='any', staging=[V]); rel('pancreas', 'drained_by', 'portal_vein', side='any', staging=[V])
rel('pancreas', 'drains_lymph_to', 'ln_celiac', side='any', staging=[N]); rel('pancreas', 'drains_lymph_to', 'ln_hepatic_hilar', side='any', staging=[N]); rel('pancreas', 'drains_lymph_to', 'ln_mesenteric', side='any', staging=[N]); rel('pancreas', 'drains_lymph_to', 'ln_paraaortic', side='any', staging=[N])
rel('pancreas', 'lies_in', 'retroperitoneum', side='any')

# -- lung (per side)
rel('lung', 'has_part', 'lung_left_upper_lobe', side='left', staging=[L]); rel('lung', 'has_part', 'lung_left_lower_lobe', side='left', staging=[L])
for lobe in ['upper', 'middle', 'lower']: rel('lung', 'has_part', f'lung_right_{lobe}_lobe', side='right', staging=[L])
rel('lung', 'has_part', 'main_bronchus', staging=[L], note='T2 main bronchus without carina'); rel('lung', 'has_part', 'bronchial_tree', side='any', staging=[O], note='atelectasis / obstructive pneumonitis T2')
rel('lung', 'invested_by', 'pleura', staging=[C], note='T2 visceral pleura; T3 parietal pleura')
rel('lung', 'adjacent_to', 'chest_wall', direction='lateral', contact='separated_by_pleura', staging=[W], note='T3 chest wall')
rel('lung', 'adjacent_to', 'diaphragm', direction='inferior', contact='separated_by_pleura', staging=[ADJ], note='T4 diaphragm')
rel('lung', 'adjacent_to', 'heart', side='any', direction='medial', contact='separated_by_pericardium', staging=[ADJ], note='T4 heart; T3 parietal pericardium')
rel('lung', 'adjacent_to', 'pericardium', side='any', direction='medial', contact='abuts', staging=[ADJ]); rel('lung', 'adjacent_to', 'aorta', side='left', direction='medial', contact='abuts', staging=[ADJ], note='T4 great vessels')
rel('lung', 'adjacent_to', 'superior_vena_cava', side='right', direction='medial', contact='abuts', staging=[ADJ]); rel('lung', 'adjacent_to', 'esophagus', side='any', direction='medial_posterior', contact='near', staging=[ADJ], note='T4 oesophagus')
rel('lung', 'adjacent_to', 'trachea', side='any', direction='medial_superior', contact='near', staging=[ADJ], note='T4 carina / trachea'); rel('lung', 'adjacent_to', 'spine', side='any', direction='posterior_medial', contact='near', staging=[ADJ], note='T4 vertebral body')
rel('lung', 'adjacent_to', 'rib_cage', direction='lateral', contact='separated_by_pleura', staging=[W]); rel('lung', 'adjacent_to', 'liver', side='right', direction='inferior', contact='separated_by_diaphragm', staging=[])
rel('lung', 'supplied_by', 'pulmonary_artery', side='any', staging=[V]); rel('lung', 'drained_by', 'pulmonary_vein', side='any', staging=[V])
rel('lung', 'has_duct', 'main_bronchus', staging=[O]); rel('main_bronchus', 'has_duct', 'trachea', side='any')
rel('lung', 'drains_lymph_to', 'ln_hilar', staging=['N1']); rel('lung', 'drains_lymph_to', 'ln_mediastinal', side='any', staging=['N2/N3']); rel('lung', 'drains_lymph_to', 'ln_supraclavicular', side='any', staging=['N3'])
rel('lung', 'lies_in', 'thorax', side='any')

# -- breast
rel('breast', 'has_part', 'fibroglandular_tissue', staging=[L]); rel('breast', 'has_part', 'nipple', staging=[L])
rel('breast', 'adjacent_to', 'pectoralis_major', direction='posterior', contact='abuts', staging=[ADJ], note='pectoralis involvement (not T4 unless chest wall)')
rel('breast', 'adjacent_to', 'chest_wall', direction='posterior', contact='separated_by_pectoralis', staging=[W], note='T4a chest wall (ribs, intercostals, serratus)')
rel('breast', 'adjacent_to', 'skin', direction='anterior', contact='abuts', staging=[ADJ], note='T4b skin')
rel('breast', 'adjacent_to', 'rib_cage', direction='posterior', contact='separated_by_muscle', staging=[W])
rel('breast', 'supplied_by', 'axillary_vessels', staging=[]); rel('breast', 'drains_lymph_to', 'ln_axillary', staging=['N1-N2a']); rel('breast', 'drains_lymph_to', 'ln_internal_mammary', staging=['N2b/N3b']); rel('breast', 'drains_lymph_to', 'ln_supraclavicular', staging=['N3c'])
rel('breast', 'lies_in', 'thorax', side='any')

# -- adrenal, spleen, gallbladder, bladder-region misc, oesophagus done; adrenal short profile
rel('adrenal', 'adjacent_to', 'kidney', direction='inferior_lateral', contact='abuts', staging=[ADJ]); rel('adrenal', 'adjacent_to', 'liver', side='right', direction='lateral', contact='abuts', staging=[ADJ])
rel('adrenal', 'adjacent_to', 'inferior_vena_cava', side='right', direction='medial', contact='abuts', staging=[V]); rel('adrenal', 'adjacent_to', 'aorta', side='left', direction='medial', contact='near', staging=[V])
rel('adrenal', 'adjacent_to', 'pancreas', side='left', direction='anterior', contact='near', staging=[ADJ]); rel('adrenal', 'adjacent_to', 'spleen', side='left', direction='lateral', contact='near', staging=[ADJ])
rel('adrenal', 'adjacent_to', 'diaphragm', direction='posterior', contact='abuts', staging=[ADJ]); rel('adrenal', 'drains_lymph_to', 'ln_paraaortic', side='any', staging=[N]); rel('adrenal', 'lies_in', 'retroperitoneum', side='any')
rel('spleen', 'adjacent_to', 'stomach', side='any', direction='anterior_medial', contact='near', staging=[ADJ]); rel('spleen', 'adjacent_to', 'kidney', side='left', direction='inferior_medial', contact='near', staging=[ADJ])
rel('spleen', 'adjacent_to', 'pancreas', side='any', direction='medial', contact='abuts', staging=[ADJ], note='tail'); rel('spleen', 'adjacent_to', 'colon', side='any', direction='inferior', contact='near', staging=[ADJ], note='splenic flexure')
rel('spleen', 'adjacent_to', 'diaphragm', side='any', direction='superior_posterior', contact='abuts', staging=[]); rel('spleen', 'drained_by', 'splenic_vein', side='any', staging=[V]); rel('spleen', 'lies_in', 'peritoneal_cavity', side='any')
rel('gallbladder', 'adjacent_to', 'liver', side='any', direction='superior', contact='abuts', staging=[ADJ]); rel('gallbladder', 'adjacent_to', 'duodenum', side='any', direction='inferior_medial', contact='abuts', staging=[ADJ])
rel('gallbladder', 'adjacent_to', 'colon', side='any', direction='inferior', contact='near', staging=[ADJ]); rel('gallbladder', 'has_duct', 'bile_duct', side='any', staging=[O]); rel('gallbladder', 'drains_lymph_to', 'ln_hepatic_hilar', side='any', staging=[N])
rel('duodenum', 'adjacent_to', 'pancreas', side='any', direction='medial', contact='abuts', staging=[ADJ]); rel('duodenum', 'adjacent_to', 'kidney', side='right', direction='posterior', contact='near', staging=[ADJ]); rel('duodenum', 'adjacent_to', 'inferior_vena_cava', side='any', direction='posterior', contact='abuts', staging=[V])
rel('small_bowel', 'invested_by', 'mesentery', side='any', staging=[C]); rel('small_bowel', 'supplied_by', 'superior_mesenteric_artery', side='any', staging=[V]); rel('small_bowel', 'drains_lymph_to', 'ln_mesenteric', side='any', staging=[N]); rel('small_bowel', 'lies_in', 'peritoneal_cavity', side='any')
# -- great vessels / systemic
rel('aorta', 'has_part', 'celiac_trunk', side='any', staging=[L]); rel('aorta', 'has_part', 'superior_mesenteric_artery', side='any'); rel('aorta', 'has_part', 'renal_artery', side='any'); rel('aorta', 'has_part', 'iliac_artery', side='any')
rel('iliac_artery', 'has_part', 'internal_iliac_artery', staging=[L]); rel('iliac_vein', 'has_part', 'internal_iliac_vein'); rel('inferior_vena_cava', 'has_part', 'iliac_vein', side='any'); rel('inferior_vena_cava', 'has_part', 'renal_vein', side='any'); rel('inferior_vena_cava', 'has_part', 'hepatic_veins', side='any')
rel('heart', 'invested_by', 'pericardium', side='any', staging=[]); rel('heart', 'adjacent_to', 'lung', side='any', direction='lateral', contact='separated_by_pericardium'); rel('heart', 'adjacent_to', 'esophagus', side='any', direction='posterior', contact='near'); rel('heart', 'adjacent_to', 'diaphragm', side='any', direction='inferior', contact='abuts')
rel('heart', 'lies_in', 'mediastinum', side='any'); rel('trachea', 'has_part', 'main_bronchus', side='any', staging=[L]); rel('trachea', 'adjacent_to', 'esophagus', side='any', direction='posterior', contact='abuts'); rel('trachea', 'adjacent_to', 'thyroid', side='any', direction='anterior', contact='abuts'); rel('trachea', 'lies_in', 'mediastinum', side='any')
rel('spine', 'adjacent_to', 'psoas', side='any', direction='lateral', contact='abuts'); rel('spine', 'adjacent_to', 'aorta', side='any', direction='anterior', contact='abuts'); rel('spine', 'adjacent_to', 'esophagus', side='any', direction='anterior', contact='near'); rel('spine', 'has_part', 'spinal_cord', side='any'); rel('spine', 'has_part', 'sacrum', side='any')
rel('sacrum', 'adjacent_to', 'rectum', side='any', direction='anterior', contact='separated_by_presacral_fat'); rel('sacrum', 'adjacent_to', 'hip', side='any', direction='lateral', contact='abuts', note='sacroiliac joint'); rel('sacrum', 'adjacent_to', 'iliac_artery', side='any', direction='anterior_lateral', contact='near')
rel('hip', 'adjacent_to', 'urinary_bladder', direction='medial', contact='near'); rel('hip', 'adjacent_to', 'obturator_internus', direction='medial', contact='abuts'); rel('hip', 'adjacent_to', 'iliopsoas', direction='medial', contact='abuts'); rel('hip', 'adjacent_to', 'gluteus', direction='posterior_lateral', contact='abuts'); rel('hip', 'adjacent_to', 'femur', direction='inferior_lateral', contact='abuts', note='hip joint')
rel('pelvic_sidewall', 'has_part', 'obturator_internus', staging=[L]); rel('pelvic_sidewall', 'adjacent_to', 'iliac_artery', direction='superior', contact='near'); rel('pelvic_sidewall', 'adjacent_to', 'ln_obturator', direction='medial', contact='near')
rel('skeletal_muscle', 'has_part', 'psoas', side='any'); rel('skeletal_muscle', 'has_part', 'autochthon', side='any'); rel('skeletal_muscle', 'has_part', 'rectus_abdominis', side='any'); rel('skeletal_muscle', 'has_part', 'iliopsoas', side='any'); rel('skeletal_muscle', 'has_part', 'gluteus', side='any'); rel('skeletal_muscle', 'has_part', 'pectoralis_major', side='any')
# landmarks
for lm in LANDMARKS: rel(lm['entity'], 'landmark_for', 'spine', side='any', note=f"{lm['id']} ≈ {lm['level']}")

# ================================================================== TUMOUR PROMPTS
TUMOUR = dict(
 version=VERSION,
 generic_templates=['tumor in the {host}', '{host} tumor', '{host} mass', '{host} lesion', 'cancer of the {host}', 'malignant neoplasm in the {host}'],
 host_resolution=['max_overlap_with_anchor_masks', 'cancer_type_default_host', 'planner_flag_unknown'],
 cancer_types={
  'TCGA-KIRC': dict(host='kidney', phrases=['clear cell renal cell carcinoma in the {side} kidney', 'renal cell carcinoma in the {side} kidney', '{side} kidney tumor', 'renal tumor']),
  'TCGA-KIRP': dict(host='kidney', phrases=['papillary renal cell carcinoma in the {side} kidney', 'renal cell carcinoma in the {side} kidney', '{side} kidney tumor']),
  'TCGA-KICH': dict(host='kidney', phrases=['chromophobe renal cell carcinoma in the {side} kidney', 'renal cell carcinoma in the {side} kidney', '{side} kidney tumor']),
  'TCGA-LIHC': dict(host='liver', phrases=['hepatocellular carcinoma', 'liver tumor', 'hepatic tumor', 'liver cancer']),
  'TCGA-BRCA': dict(host='breast', phrases=['breast cancer in the {side} breast', '{side} breast tumor', 'invasive breast carcinoma', 'breast lesion']),
  'TCGA-LUAD': dict(host='lung', phrases=['lung adenocarcinoma in the {side} lung', '{side} lung tumor', 'lung cancer', 'pulmonary nodule']),
  'TCGA-LUSC': dict(host='lung', phrases=['squamous cell carcinoma in the {side} lung', '{side} lung tumor', 'lung cancer', 'bronchial carcinoma']),
  'TCGA-BLCA': dict(host='urinary_bladder', phrases=['bladder cancer', 'urothelial carcinoma of the bladder', 'bladder tumor', 'bladder wall thickening']),
  'TCGA-PRAD': dict(host='prostate', phrases=['prostate cancer', 'prostate tumor', 'prostatic adenocarcinoma', 'prostate lesion']),
  'TCGA-CESC': dict(host='cervix', phrases=['cervical cancer', 'cervix tumor', 'cervical carcinoma', 'tumor in the uterine cervix']),
  'TCGA-UCEC': dict(host='uterus', phrases=['endometrial cancer', 'endometrial carcinoma', 'uterine tumor', 'tumor in the uterine cavity']),
  'TCGA-OV': dict(host='ovary', phrases=['ovarian cancer', 'ovarian tumor', 'adnexal mass', 'ovarian carcinoma', 'peritoneal carcinomatosis']),
  'TCGA-STAD': dict(host='stomach', phrases=['gastric cancer', 'stomach tumor', 'gastric adenocarcinoma', 'gastric wall thickening']),
  'TCGA-ESCA': dict(host='esophagus', phrases=['esophageal cancer', 'esophageal tumor', 'esophageal carcinoma', 'esophageal wall thickening']),
  'TCGA-COAD': dict(host='colon', phrases=['colon cancer', 'colorectal cancer', 'colon tumor', 'colonic adenocarcinoma']),
  'TCGA-READ': dict(host='rectum', phrases=['rectal cancer', 'rectal tumor', 'rectal adenocarcinoma']),
  'TCGA-PAAD': dict(host='pancreas', phrases=['pancreatic cancer', 'pancreatic adenocarcinoma', 'pancreatic tumor', 'pancreatic mass']),
  'CPTAC-*': dict(host=None, phrases=[], note='map CPTAC cohorts to the TCGA host above by organ; CPTAC-CCRCC→kidney, CPTAC-PDA→pancreas, CPTAC-UCEC→uterus, CPTAC-LUAD/LSCC→lung, CPTAC-HNSCC→(head & neck: not covered in v0.1)'),
 },
 side_resolution='from tumour-mask laterality vs anchor masks; if unknown prompt both sides and keep the side whose mask overlaps an anchor',
)

# ================================================================== PROMPT / QC RULES
PROMPT_RULES = dict(version=VERSION,
 templates=dict(lateral='{side} {term}', in_region='{term} in the {region}', tumour_in_host='{tumour_phrase}'),
 alias_ensemble=dict(n_aliases=3, fusion='majority_vote', min_agreement=2, note='main term + first 2 aliases; fuse binary masks by voxel majority'),
 roi_by_gate=dict(inside='anchor mask dilated 3 mm → bbox', shell='anchor bbox dilated by outer_mm', neighbour_bbox_or_directional_shell='TS bbox of neighbour if present else directional half-shell of anchor', corridor='cylinder between endpoints', region='region box from landmarks', directional_shell='half-shell on stated side', none='whole volume'),
 waves=dict(wave0='ruler prompts if TS vertebrae missing: spine, sacrum, hip bone', wave1='missing anchors from completion under spatial-prior gates', wave2='profile items (1–2 hop) of host / promoted / tumour-near anchors', max_waves=2),
 budget=dict(max_prompts_per_scan=40, order='tumour prompts, then by (priority, wave); host-derived items get a one-step priority boost', drop_order='lowest priority first'),
 modality=dict(CT='prefer TS mask when TS reliability ≥ 0.8; VoxTell refine host + vessels', MR='TS reliability from pilot; completion always on for pelvic anchors'),
 encoder_reuse='VoxTell embeds text once; image encoder per volume; pass all prompts of a scan in one predict call (voxtell-predict -p ...)',
)
QC_RULES = dict(version=VERSION, rules=[
 dict(id='volume_range', applies='all', check='volume_ml within entity.volume_ml (if defined) ×[0.5, 2.0]', on_fail='flag'),
 dict(id='laterality', applies='bilateral', check='centroid x-sign matches side in RAS frame from affine', on_fail='swap_or_flag'),
 dict(id='containment', applies='listB', check='≥80% of mask inside gate', on_fail='reject'),
 dict(id='adjacency', applies='listB', check='min surface distance to each required neighbour ≤ d_mm from spatial prior', on_fail='reject'),
 dict(id='symmetry', applies='bilateral', check='volume ratio L/R within [0.5, 2.0] unless surgical absence flagged', on_fail='flag'),
 dict(id='tubular_continuity', applies='vessel_artery,vessel_vein,duct', check='single dominant component ≥70% volume; elongation ≥3', on_fail='flag'),
 dict(id='intensity', applies='CT', check='median HU within entity.ct_hu ± 30', on_fail='flag'),
 dict(id='truncation', applies='all', check='fraction of mask on volume boundary; >5% → truncated flag', on_fail='flag'),
 dict(id='existence_gate', applies='listB', check='R²-Seg L1: max prob, positive ratio, KS vs empty distribution', on_fail='absent_if_expected_else_reject'),
 dict(id='ts_vt_agreement', applies='listA', check='Dice(TS, VoxTell) ≥ 0.5 → use VoxTell boundary; else TS + flag', on_fail='use_ts'),
 dict(id='single_rater_threshold', applies='listB', check='all listB checks pass AND alias agreement ≥ 2/3', on_fail='reject'),
])

# ================================================================== VALIDATE + DUMP
ids = {e['id'] for e in E}
assert len(ids) == len(E), 'duplicate entity id'
region_ids = {r['id'] for r in REGIONS}
for r in R:
    assert r['src'] in ids, f"relation src missing: {r['src']}"
    assert r['dst'] in ids or (r['type'] == 'lies_in' and r['dst'] in region_ids), f"relation dst missing: {r['dst']} ({r['src']} {r['type']})"
    assert r['type'] in {t['id'] for t in RELATION_TYPES}
for e in E:
    if e.get('anchor'):
        for p in e['anchor']['spatial_prior']:
            assert p['landmark'] in ids or p['landmark'] in {l['id'] for l in LANDMARKS}, f"prior landmark missing: {p['landmark']} in {e['id']}"
for lm in LANDMARKS: assert lm['entity'] in ids
meta = dict(name='PanCIA anchor-centric anatomical knowledge base', version=VERSION, date=TODAY, author='Shangqi Gao (Crispin Lab) with Claude',
            status='draft v0.1 — not yet pilot-verified', vertebral_order=VERTEBRAL_ORDER,
            counts=dict(entities=len(E), anchors=sum(e['is_anchor'] for e in E), relations=len(R), regions=len(REGIONS), landmarks=len(LANDMARKS)),
            sources=dict(TS='TotalSegmentator class maps (map_to_binary.py, fetched 2026-09-18)', VoxTell='arXiv 2511.11450; repo MIC-DKFZ/VoxTell (no shipped vocabulary; coverage tiers are assumptions until pilot)',
                         AJCC8='AJCC Cancer Staging Manual 8th ed. T/N definitions and regional node lists', FIGO='FIGO 2018 cervix / 2023 endometrium / 2014 ovary', FMA='Foundational Model of Anatomy relations (part_of, adjacent_to)'),
            coverage_tier_legend=dict(in_vocab='structure or close synonym present in known VoxTell training datasets', near='related concept present; expect degraded but usable', ood='not expected in training; experimental, strict gates', assumed_in_vocab='default for TS classes'))
def dump(name, obj):
    with open(os.path.join(OUT, name), 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True, width=140)
dump('kb_meta.yaml', meta); dump('relation_types.yaml', dict(version=VERSION, relation_types=RELATION_TYPES))
dump('regions.yaml', dict(version=VERSION, regions=REGIONS)); dump('landmarks.yaml', dict(version=VERSION, landmarks=LANDMARKS, vertebral_order=VERTEBRAL_ORDER))
dump('entities.yaml', dict(version=VERSION, entities=E)); dump('relations.yaml', dict(version=VERSION, relations=R))
dump('tumour_prompts.yaml', TUMOUR); dump('prompt_rules.yaml', PROMPT_RULES); dump('qc_rules.yaml', QC_RULES)
# evidence template
import csv
with open(os.path.join(OUT, 'evidence_template.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['entity_id', 'side', 'modality', 'n_scans', 'ts_found_rate', 'ts_vt_dice_median', 'vt_plausibility_pass_rate', 'vt_empty_when_expected_absent_rate', 'vt_seconds_per_prompt', 'decision', 'reviewer', 'date'])
print(json.dumps(meta['counts']))
