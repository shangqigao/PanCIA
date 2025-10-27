import ast
import json
import pathlib
import pandas as pd

from utilities.constants import PANCIA_PROJECT_SITE

radiomics_dims = {
    "None": None,
    "pyradiomics": 851,
    "SegVol": 768,
    "BiomedParse": 1024,
    "BayesBP": 64,
}

pathomics_dims = {
    "None": None,
    "CNN": 2048,
    "HIPT": 384,
    "UNI": 1024,
    "CONCH": 35,
    "CHIEF": 768
}

def prepare_MAMAMIA_omics_info(
        img_dir, 
        save_omics_dir, 
        phase='pre-contrast',
        segmentator="BiomedParse",
        radiomics_mode='Pyradiomics', 
        radiomics_suffix='_tumor_radiomics.json'
    ):
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

    save_radiomics_dir = f"{save_omics_dir}/radiomics/MAMAMIA_radiomic_features/{radiomics_mode}/segmentator_{segmentator}"
    parent_names = [pathlib.Path(p).parent.name for p in img_paths]
    radiomics_names = [pathlib.Path(p).name.replace("*.nii.gz", radiomics_suffix) for p in img_paths]
    radiomics_paths = [f"{save_radiomics_dir}/{p}/{r}" for p, r in zip(parent_names, radiomics_names)]
    data_paths, omics_paths, sites = {}, {}, {}
    for subject, radio, img_path in zip(parent_names, radiomics_paths, img_paths):
        if pathlib.Path(radio).exists(): 
            data_paths.update({subject: {"radiology": [img_path], "pathology": None}})
            omics_paths.update({subject: {"radiomics": [radio], "pathomics": None}})
            sites.update({subject: 'breast'})
    omics_dir = {"radiomics": save_radiomics_dir, "pathomics": None}        
    
    omics_info = {
        'name': 'MAMA-MIA',
        'data_paths': data_paths,
        'omics_dir': omics_dir,
        'omics_paths': omics_paths,
        'sites': sites,
    }
    return omics_info

def prepare_TCGA_omics_info(
        dataset_json, 
        save_omics_dir, 
        radiomics_mode='Pyradiomics', 
        segmentator="BiomedParse", 
        radiomics_suffix='_tumor_radiomics.json',
        pathomics_mode='UNI',
        pathomics_suffix='.json',
    ):
    assert pathlib.Path(dataset_json).suffix == '.json', 'only support loading info from json file'
    with open(dataset_json, 'r') as f:
        data = json.load(f)
    included_subjects = data['included subjects']

    save_radiomics_dir = f"{save_omics_dir}/radiomics/TCGA_radiomic_features/{radiomics_mode}/segmentator_{segmentator}"
    save_pathomics_dir = f"{save_omics_dir}/pathomics/TCGA_pathomic_features/{pathomics_mode}"
    data_paths, omics_paths, sites = {}, {}, {}
    for k, v in included_subjects.items(): 
        radiology_paths = v['radiology']
        project_id = str(radiology_paths[0]).split('/')[-3]
        parent_names = [pathlib.Path(r).parent.name for r in radiology_paths]
        radiomics_names = [pathlib.Path(r).name.replace('.nii.gz', radiomics_suffix) for r in radiology_paths]
        radiomics_paths = [f"{save_radiomics_dir}/{p}/{r}" for p, r in zip(parent_names, radiomics_names)]
        radiomics_existed_paths = [r for r in radiomics_paths if pathlib.Path(r).exists()]
        radiology_existed_paths = [r for r, o in zip(radiology_paths, radiomics_paths) if pathlib.Path(o).exists()]

        pathology_paths = v['pathology']
        pathomics_names = [pathlib.Path(p).name.replace('.svs', pathomics_suffix) for p in pathology_paths]
        pathomics_paths = [f"{save_pathomics_dir}/{p}" for p in pathomics_names]
        pathomics_existed_paths = [p for p in pathomics_paths if pathlib.Path(p).exists()]
        pathology_existed_paths = [p for p, o in zip(pathology_paths, pathomics_paths) if pathlib.Path(o).exists()]

        if len(radiomics_existed_paths) > 0 and pathomics_mode == "None":
            data_paths.update({k: {"radiology": radiology_existed_paths, "pathology": None}})
            omics_paths.update({k: {"radiomics": radiomics_existed_paths, "pathomics": None}})
            sites.update({k: PANCIA_PROJECT_SITE[project_id]})
        if radiomics_mode == "None" and len(pathomics_existed_paths) > 0:
            data_paths.update({k: {"radiology": None, "pathology": pathology_existed_paths}})
            omics_paths.update({k: {"radiomics": None, "pathomics": pathomics_existed_paths}})
            sites.update({k: PANCIA_PROJECT_SITE[project_id]})
        if len(radiomics_existed_paths) > 0 and len(pathomics_existed_paths) > 0:
            data_paths.update({k: {"radiology": radiology_existed_paths, "pathology": pathology_existed_paths}})
            omics_paths.update({k: {"radiomics": radiomics_existed_paths, "pathomics": pathomics_existed_paths}})
            sites.update({k: PANCIA_PROJECT_SITE[project_id]})

    omics_dir = {"radiomics": save_radiomics_dir, "pathomics": save_pathomics_dir}

    omics_info = {
        'name': 'TCGA',
        'data_paths': data_paths,
        'omics_dir': omics_dir,
        'omics_paths': omics_paths,
        'sites': sites
    }

    return omics_info