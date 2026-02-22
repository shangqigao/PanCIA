import ast
import json
import pathlib
import pandas as pd

from utilities.constants import PANCIA_PROJECT_SITE

radiomics_dims = {
    "None": None,
    "pyradiomics": {'child0': 851},
    "FMCIB": {'child0': 4096},
    "SegVol": {'child0': 768},
    "BiomedParse": {'child0': 512},
    "BayesBP": {'child0': 64},
    "LVMMed": {'child0': 64, 'child1': 256, 'child2': 512, 'child3': 1024, 'child4': 2048}
}

pathomics_dims = {
    "None": None,
    "CNN": {'child0': 2048},
    "HIPT": {'child0': 384},
    "UNI": {'child0': 1024},
    "CONCH": {'child0': 512},
    "CHIEF": {'child0': 768}
}

radiomics_pool_ratio = {
    "None": None,
    "pyradiomics": {'child0': 1},
    "FMCIB": {'child0': 1},
    "SegVol": {'child0': 0.7},
    "BiomedParse": {'child0': 0.7},
    "BayesBP": {'child0': 0.7},
    "LVMMed": {'child0': 0.7, 'child1': 0.9, 'child2': 0.9, 'child3': 0.9, 'child4': 0.9}
}

pathomics_pool_ratio = {
    "None": None,
    "CNN": {'child0': 0.7},
    "HIPT": {'child0': 0.7},
    "UNI": {'child0': 0.7},
    "CONCH": {'child0': 0.7},
    "CHIEF": {'child0': 0.7}
}

signatures_predictable = {
    "GeneProgrames": None,
    "HRDscore": None,
    "ImmuneSignatureScore": None,
    "StemnessScoreDNA": None,
    "StemScoreRNA": None,
    "AGE": None
}

radiomics_suffix = {
    "None": None,
    "pyradiomics": [
        "_tumor_radiomics.json"
    ],
    "FMCIB": [
        "_tumor_radiomics.npy"
    ],
    "SegVol": [
        "_tumor_graph_aggr_mean.npy"
    ],
    "BiomedParse": [
        "_tumor_graph_aggr_mean.npy"
    ],
    "BayesBP": [
        "_tumor_radiomics_pooled.json"
    ],
    "LVMMed": [
        "_tumor_layer0_graph_aggr_mean.npy",
        "_tumor_layer1_graph_aggr_mean.npy",
        "_tumor_layer2_graph_aggr_mean.npy",
        "_tumor_layer3_graph_aggr_mean.npy",
        "_tumor_layer4_graph_aggr_mean.npy",
    ]
}

pathomics_suffix = {
    "None": [
        "_graph_aggr_mean.npy"
    ],
    "CNN": [
        "_graph_aggr_mean.npy"
    ],
    "HIPT": [
        "_graph_aggr_mean.npy"
    ],
    "UNI": [
        "_graph_aggr_mean.npy"
    ],
    "CONCH": [
        "_graph_aggr_mean.npy"
    ],
    "CHIEF": [
        "_graph_aggr_mean.npy"
    ],
    "UNI2": [
        "_graph_aggr_mean.npy"
    ]
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
        radiology_existed_paths = []
        radiomics_existed_paths = []
        for r in radiology_paths:
            radiology_name = pathlib.Path(r).name
            parent_name = pathlib.Path(r).parent.name
            radiomics_names = [radiology_name.replace('.nii.gz', suffix) for suffix in radiomics_suffix]
            radiomics_children = [f"{save_radiomics_dir}/{parent_name}/{r}" for r in radiomics_names]
            radiomics_existed_children = [r for r in radiomics_children if pathlib.Path(r).exists()]
            radiomics_existed_children = {f"child{i}": r for i, r in enumerate(radiomics_existed_children)}
            if len(radiomics_existed_children) > 0:
                radiology_existed_paths.append(r)
                radiomics_existed_paths.append(radiomics_existed_children)

        pathology_paths = v['pathology']
        pathology_existed_paths = []
        pathomics_existed_paths = []
        for p in pathology_paths:
            pathomics_names = [pathlib.Path(p).name.replace('.svs', suffix) for suffix in pathomics_suffix]
            pathomics_children = [f"{save_pathomics_dir}/{p}" for p in pathomics_names]
            pathomics_existed_children = [p for p in pathomics_children if pathlib.Path(p).exists()]
            pathomics_existed_children = {f"child{i}": r for i, r in enumerate(pathomics_existed_children)}
            if len(pathomics_existed_children) > 0:
                pathology_existed_paths.append(p)
                pathomics_existed_paths.append(pathomics_existed_children)

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