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

def prepare_TCGA_radiology_info(img_json, lab_dir=None, lab_mode=None, img_format='nifti'):
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
            target = PANCIA_PROMPT_TEMPLETES[img_site]
            prompts.append(f"{target} on {img_mod}")
    
    if lab_dir is not None:
        lab_paths = [str(p).split('/TCGA_NIFTI/')[-1] for p in images]
        lab_paths = [f"{lab_dir}/{lab_mode}/{p}" for p in lab_paths]
    else:
        lab_paths = [None]*len(img_paths)

    dataset_info = {
        'name': 'TCGA-Radiology',
        'img_paths': images,
        'lab_paths': lab_paths,
        'text_prompts': prompts,
        'modality': modality,
        'site': site,
        'meta_list': None,
        'img_format': [img_format] * len(images)
    }

    return dataset_info