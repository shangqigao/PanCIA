import ast
import json
import pathlib
import pandas as pd

from utilities.constants import PANCIA_PROJECT_SITE, PANCIA_PROMPT_TEMPLETES

def prepare_MAMAMIA_info(img_dir, lab_dir=None, lab_mode=None, img_format='nifti', phase='pre-contrast', site='breast', meta_info=None):
    if img_format == 'dicom':
        img_paths = pathlib.Path(img_dir).glob('*')
        img_paths = [p for p in img_paths if p.is_dir()]
        patient_ids = [p.name for p in img_paths]
    else:
        if phase == "pre-contrast":
            img_paths = sorted(pathlib.Path(img_dir).rglob('*_0000.nii.gz'))
        elif phase == "1st-contrast":
            img_paths = sorted(pathlib.Path(img_dir).rglob('*_0001.nii.gz'))
        elif phase == "2nd-contrast":
            img_paths = sorted(pathlib.Path(img_dir).rglob('*_0002.nii.gz'))
        else:
            case_paths = sorted(pathlib.Path(img_dir).glob('*'))
            case_paths = [p for p in case_paths if p.is_dir()]
            img_paths = []
            for path in case_paths:
                nii_paths = path.glob("*.nii.gz")
                multiphase_keys = ["_0000.nii.gz", "_0001.nii.gz", "_0002.nii.gz"]
                nii_paths = [p for p in nii_paths if any(k in p.name for k in multiphase_keys)]
                img_paths.append(sorted(nii_paths))
        if lab_dir is not None:
            lab_paths = sorted(pathlib.Path(f"{lab_dir}/{lab_mode}").glob('*.nii.gz'))
        else:
            lab_paths = [None]*len(img_paths)
    text_prompts = [f'tumor located within the {site}, adjacent to the chest wall on MRI']*len(img_paths)
    # text_prompts = [f'{site} {target}']*len(img_paths)

    # read clinical and imaging info
    if meta_info is not None:
        df_meta = pd.read_excel(meta_info, sheet_name='dataset_info')
        df_meta['pixel_spacing'] = df_meta['pixel_spacing'].apply(ast.literal_eval)
        parent_path = pathlib.Path(meta_info).parent
        meta_list = []
        for patient_id in patient_ids:
            field_strength = df_meta.loc[df_meta["patient_id"] == patient_id, 'field_strength'].values[0]
            bilateral_mri = df_meta.loc[df_meta["patient_id"] == patient_id, 'bilateral_mri'].values[0]
            lateral = 'bilateral' if bilateral_mri == 1 else 'unilateral'
            manufacturer = df_meta.loc[df_meta["patient_id"] == patient_id, 'manufacturer'].values[0]
            meta_data = {
                'field_strength': field_strength, 
                'bilateral': lateral, 
                'scanner_manufacturer': manufacturer
            }
            patient_info_path = f'{parent_path}/patient_info_files/{patient_id}.json'
            with open(patient_info_path, 'r') as file: 
                patient_info = json.load(file)
            breast_coords = patient_info["primary_lesion"]["breast_coordinates"]
            meta_data.update({"breast_coordinates": breast_coords})
            meta_list.append(meta_data)
    else:
        meta_list = None
    
    dataset_info = {
        'name': 'MAMA-MIA',
        'img_paths': img_paths,
        'lab_paths': lab_paths,
        'text_prompts': text_prompts,
        'modality': ['MRI']*len(img_paths),
        'site': ['breast']*len(img_paths),
        'meta_list': meta_list,
        'img_format': ['nifti']*len(img_paths)
    }
    return dataset_info

def _plan_path_for_image(img_path, planner_dir):
    """Planner output for one NIfTI: <planner_dir>/<Modality>/<project>/<series_uid>/<series>_plan.json. Accepts the Planner
    root (…/Planner) or its Radiology subfolder (…/Planner/Radiology)."""
    parts = pathlib.Path(str(img_path)).parts
    rel = pathlib.Path(*parts[-4:-1]) / (parts[-1].replace('.nii.gz', '') + '_plan.json')
    root = pathlib.Path(planner_dir)
    for cand in (root / rel, root / 'Radiology' / rel):
        if cand.is_file():
            return cand
    return None


def load_plan_prompts(plan_path, plan_lists=None):
    """Read a planner *_plan.json into VoxTell prompt jobs.

    Returns dict(plan=<path>, kb_version=…, jobs=[…]); every job keeps the planner fields (entity, side, list, tier, terms,
    gate, priority, relation, from_anchor, reason) and gets the VoxTell output name '<entity>[_<side>]_<seg_obj>' with
    seg_obj = 'tumor' for tier-T prompts and 'organ' for anatomy prompts. plan_lists (e.g. ['T'] or ['T', 'A', 'B'])
    restricts the planner lists; None = all."""
    with open(plan_path, 'r') as f:
        plan = json.load(f)
    keep = None if not plan_lists else {str(x) for x in plan_lists}
    jobs = []
    for i, p in enumerate(plan.get('prompts', [])):
        if keep is not None and str(p.get('list')) not in keep:
            continue
        seg_obj = 'tumor' if p.get('list') == 'T' else 'organ'
        name = p['entity'] + (f"_{p['side']}" if p.get('side') else '') + f'_{seg_obj}'
        jobs.append(dict(p, plan_index=i, seg_obj=seg_obj, name=name))
    return dict(plan=str(plan_path), kb_version=(plan.get('input') or {}).get('kb_version'), jobs=jobs)


def prepare_TCGA_radiology_info(img_json, lab_dir=None, lab_mode=None, img_format='nifti', seg_obj='tumor',
                                planner_dir=None, plan_lists=None):
    """planner_dir: if given, prompts come from each image's planner '*_plan.json' (text_prompts[i] = dict(plan, kb_version,
    jobs)); images without a plan keep the single cancer-type prompt."""
    assert pathlib.Path(img_json).suffix == '.json', 'only support loading info from json file'
    with open(img_json, 'r') as f:
        data = json.load(f)
    included_subjects = data['included subjects']
    img_paths = []
    for k, v in included_subjects.items(): img_paths += v['radiology']
    images, site, modality, prompts = [], [], [], []
    for img_path in img_paths:
        folds = str(img_path).split('/')
        project = folds[-3]
        img_mod = {'MR': 'MRI', 'CT': 'CT'}[folds[-4]]
        if PANCIA_PROJECT_SITE.get(project, False):
            images.append(img_path)
            img_site = PANCIA_PROJECT_SITE[project]
            site.append(img_site)
            modality.append(img_mod)
            if seg_obj == 'tumor':
                target = PANCIA_PROMPT_TEMPLETES[img_site]
                prompts.append(f"{target} on {img_mod}")
            else:
                prompts.append(f"{img_site} on {img_mod}")
    
    plan_paths = [None] * len(images)
    if planner_dir is not None:
        for i, img_path in enumerate(images):
            pp = _plan_path_for_image(img_path, planner_dir)
            if pp is not None:
                prompts[i] = load_plan_prompts(pp, plan_lists)
                plan_paths[i] = str(pp)
        print(f'planner prompts: {sum(p is not None for p in plan_paths)}/{len(images)} images have a plan')

    if lab_dir is not None:
        lab_paths = [str(p).split('/TCGA_NIFTI/')[-1] for p in images]
        lab_paths = [f"{lab_dir}/{lab_mode}/{p}".replace('.nii.gz', f'_{seg_obj}.nii.gz') for p in lab_paths]
    else:
        lab_paths = [None]*len(img_paths)

    dataset_info = {
        'name': 'TCGA-Radiology',
        'img_paths': images,
        'lab_paths': lab_paths,
        'text_prompts': prompts,
        'plan_paths': plan_paths,
        'modality': modality,
        'site': site,
        'meta_list': None,
        'img_format': [img_format] * len(images)
    }

    return dataset_info

def prepare_CPTAC_radiology_info(img_json, lab_dir=None, lab_mode=None, img_format='nifti', seg_obj='tumor',
                                 planner_dir=None, plan_lists=None):
    """planner_dir: if given, prompts come from each image's planner '*_plan.json' (text_prompts[i] = dict(plan, kb_version,
    jobs)); images without a plan keep the single cancer-type prompt."""
    assert pathlib.Path(img_json).suffix == '.json', 'only support loading info from json file'
    with open(img_json, 'r') as f:
        data = json.load(f)
    included_subjects = data['included subjects']
    img_paths = []
    for k, v in included_subjects.items(): img_paths += v['radiology']
    images, site, modality, prompts = [], [], [], []
    for img_path in img_paths:
        folds = str(img_path).split('/')
        project = folds[-3]
        img_mod = {'MR': 'MRI', 'CT': 'CT'}[folds[-4]]
        if PANCIA_PROJECT_SITE.get(project, False):
            images.append(img_path)
            img_site = PANCIA_PROJECT_SITE[project]
            site.append(img_site)
            modality.append(img_mod)
            if seg_obj == 'tumor':
                target = PANCIA_PROMPT_TEMPLETES[img_site]
                prompts.append(f"{target} on {img_mod}")
            else:
                prompts.append(f"{img_site} on {img_mod}")
    
    plan_paths = [None] * len(images)
    if planner_dir is not None:
        for i, img_path in enumerate(images):
            pp = _plan_path_for_image(img_path, planner_dir)
            if pp is not None:
                prompts[i] = load_plan_prompts(pp, plan_lists)
                plan_paths[i] = str(pp)
        print(f'planner prompts: {sum(p is not None for p in plan_paths)}/{len(images)} images have a plan')

    if lab_dir is not None:
        lab_paths = [str(p).split('/CPTAC_NIFTI/')[-1] for p in images]
        lab_paths = [f"{lab_dir}/{lab_mode}/{p}".replace('.nii.gz', f'_{seg_obj}.nii.gz') for p in lab_paths]
    else:
        lab_paths = [None]*len(img_paths)

    dataset_info = {
        'name': 'CPTAC-Radiology',
        'img_paths': images,
        'lab_paths': lab_paths,
        'text_prompts': prompts,
        'plan_paths': plan_paths,
        'modality': modality,
        'site': site,
        'meta_list': None,
        'img_format': [img_format] * len(images)
    }

    return dataset_info

def prepare_EndoMRI_info(img_dir, lab_dir=None, lab_mode=None, img_format='nifti'):
    if img_format == 'dicom':
        img_paths = pathlib.Path(img_dir).glob('*')
        img_paths = [p for p in img_paths if p.is_dir()]
        patient_ids = [p.name for p in img_paths]
    else:
        img_paths = sorted(pathlib.Path(f"{img_dir}/images").rglob('*.nii.gz'))
        if lab_dir is not None:
            lab_paths = sorted(pathlib.Path(f"{lab_dir}/{lab_mode}").glob('*.nii.gz'))
        else:
            lab_paths = [None]*len(img_paths)
    text_prompts = [
        [
            "endometrioma in pelvis",
            "ovary",
            "uterus",
            "tumor in pelvis and cervix",
            "tumor in pelvis and uterus",
            "tumor in pelvis and ovaries"
        ]
    ]*len(img_paths)
    
    dataset_info = {
        'name': 'EndoMRI',
        'img_paths': img_paths,
        'lab_paths': lab_paths,
        'text_prompts': text_prompts,
        'modality': ['MRI']*len(img_paths),
        'site': ['pelvis']*len(img_paths),
        'meta_list': None,
        'img_format': ['nifti']*len(img_paths)
    }
    return dataset_info

def prepare_OV04_endometriosis_radiology_info(img_json, lab_dir=None, lab_mode=None, img_format='nifti'):
    assert pathlib.Path(img_json).suffix == '.json', 'only support loading info from json file'
    with open(img_json, 'r') as f:
        data = json.load(f)
    img_paths = data['included nifti']
    modality = []
    for img_path in img_paths:
        folds = str(img_path).split('/')
        img_mod = {'MR': 'MRI', 'CT': 'CT'}[folds[-3].split('_')[-1]]
        modality.append(img_mod)
    
    if lab_dir is not None:
        lab_paths = [str(p).split('/OV04_endometriosis_NIFTI/')[-1] for p in images]
        lab_paths = [f"{lab_dir}/{lab_mode}/{p}".replace('.nii.gz', f'_{seg_obj}.nii.gz') for p in lab_paths]
    else:
        lab_paths = [None]*len(img_paths)

    text_prompts = [
        [
            "endometrioma in pelvis",
            "ovary",
            "uterus",
            "tumor in pelvis and cervix",
            "tumor in pelvis and uterus",
            "tumor in pelvis and ovaries"
        ]
    ]*len(img_paths)
    
    dataset_info = {
        'name': 'OV04_endometriosis',
        'img_paths': img_paths,
        'lab_paths': lab_paths,
        'text_prompts': text_prompts,
        'modality': modality,
        'site': ['pelvis']*len(img_paths),
        'meta_list': None,
        'img_format': ['nifti']*len(img_paths)
    }
    return dataset_info


# R2Seg (collections of public tumour-segmentation datasets): collection -> (modality, site).
# Used when R2Seg/datainfo.xlsx cannot be read; datainfo.xlsx is the primary source (columns dataset, modality, cancer_type).
R2SEG_INFO = {
    'Bladder_Tumor_00': ('MRI', 'bladder'), 'Breast_Tumor_00': ('MRI', 'breast'), 'Cervix_Tumor_00': ('MRI', 'cervix'),
    'Colon_Tumor_00': ('CT', 'colon'), 'Kidney_Tumor_00': ('CT', 'kidney'), 'Liver_Tumor_00': ('CT', 'liver'),
    'Lung_Tumor_00': ('CT', 'lung'), 'Lung_Tumor_01': ('CT', 'lung'), 'Pancreas_Tumor_00': ('CT', 'pancreas'),
    'Prostate_Tumor_00': ('MRI', 'prostate'), 'Uterus_Tumor_00': ('MRI', 'uterus'),
}


def _r2seg_table(root):
    """collection -> (modality 'MRI'/'CT', site) from <root>/datainfo.xlsx; {} if it cannot be read"""
    try:
        df = pd.read_excel(pathlib.Path(root) / 'datainfo.xlsx')
    except Exception:
        return {}
    out = {}
    for r in df.itertuples(index=False):
        ds, mod, ct = getattr(r, 'dataset', None), getattr(r, 'modality', None), getattr(r, 'cancer_type', None)
        if not isinstance(ds, str) or not isinstance(mod, str):
            continue
        site = str(ct).replace(' cancer', '').strip().lower() if isinstance(ct, str) else R2SEG_INFO.get(ds, (None, None))[1]
        site = {'uterine': 'uterus', 'ovarian': 'ovary', 'cervical': 'cervix', 'colorectal': 'colon', 'pancreatic': 'pancreas'}.get(site, site)
        out[ds] = ({'MR': 'MRI', 'MRI': 'MRI', 'CT': 'CT'}.get(mod.strip().upper(), mod.strip()), site)
    return out


def prepare_R2Seg_info(r2seg_dir, collections=None, seg_obj='organ', img_format='nifti', prostate_seq='t2'):
    """R2Seg images: <r2seg_dir>/<collection>/images/*.nii.gz (labels in <collection>/labels/<stem>_seg.nii.gz).
    collections: list of collection names (default: every folder with an images/ subfolder).
    prostate_seq: Prostate_Tumor_00 has a T2 and an ADC image per case; 't2' (default), 'adc' or 'both'.
    Output names are '<collection>/<stem>' (m_tumor_segmentation._image_name): stems repeat across collections."""
    root = pathlib.Path(r2seg_dir)
    table = _r2seg_table(root)
    cols = list(collections) if collections else sorted(d.name for d in root.iterdir() if (d / 'images').is_dir())
    images, labels, site, modality, prompts = [], [], [], [], []
    for c in cols:
        if not (root / c / 'images').is_dir():
            raise FileNotFoundError(f'{root / c / "images"} not found')
        mod, st = table.get(c) or R2SEG_INFO.get(c) or (None, None)
        if mod is None:
            raise ValueError(f'modality of {c} unknown: add it to R2Seg/datainfo.xlsx or R2SEG_INFO')
        fs = sorted(p for p in (root / c / 'images').glob('*.nii.gz') if not p.name.startswith('.'))
        if c == 'Prostate_Tumor_00' and prostate_seq != 'both':
            fs = [p for p in fs if p.name.endswith(f'_{prostate_seq}.nii.gz')]
        for p in fs:
            images.append(str(p)); site.append(st); modality.append(mod)
            labels.append(str(root / c / 'labels' / p.name.replace('.nii.gz', '_seg.nii.gz')))
            if seg_obj == 'tumor':
                prompts.append(f"{PANCIA_PROMPT_TEMPLETES.get(st, f'{st} tumor')} on {mod}")
            else:
                prompts.append(f"{st} on {mod}")
        print(f'R2Seg {c}: {len(fs)} images ({mod}, {st})')
    return {
        'name': 'R2Seg',
        'img_paths': images,
        'lab_paths': labels,
        'text_prompts': prompts,
        'plan_paths': [None] * len(images),
        'modality': modality,
        'site': site,
        'meta_list': None,
        'img_format': [img_format] * len(images),
    }
