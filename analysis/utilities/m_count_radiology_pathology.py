import os
import sys
import pathlib
import shutil

# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
relative_path = os.path.join(script_dir, '../../')
sys.path.append(relative_path)

import json
from utilities.constants import PANCIA_PROJECT_SITE

json_path = '/home/sg2162/rds/hpc-work/Experiments/clinical/TCGA_included_subjects.json'

with open(json_path, 'r') as f:
    data = json.load(f)

subjects = data['included subjects']

project_subjects = {}
project_radiology = {}
project_pathology = {}
included_radiology = []
included_pathology = []
for k, v in subjects.items():
    included_radiology += v['radiology']
    included_pathology += v['pathology']
    radiology = v['radiology'][0]
    project_id = str(radiology).split('/')[-3]
    site = PANCIA_PROJECT_SITE[project_id]
    if project_subjects.get(site, False):
        project_subjects[site] = project_subjects[site] + [k]
    else:
        project_subjects[site] = [k]
    if project_radiology.get(site, False):
        project_radiology[site] = project_radiology[site] + v['radiology']
    else:
        project_radiology[site] = v['radiology']
    if project_pathology.get(site, False):
        project_pathology[site] = project_pathology[site] + v['pathology']
    else:
        project_pathology[site] = v['pathology']

num_subjects = {k: len(v) for k, v in project_subjects.items()}
num_subjects = dict(sorted(num_subjects.items(), key=lambda item: item[1], reverse=True))
print("Num of subjects:", num_subjects)

num_radiology = {k: len(project_radiology[k]) for k in num_subjects.keys()}
print("Num of radiology:", num_radiology)

num_pathology = {k: len(project_pathology[k]) for k in num_subjects.keys()}
print("Num of pathology:", num_pathology)

# segmentations = pathlib.Path('/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/TCGA_Seg').rglob('*.nii.gz')
# segmentation_niis = [str(s).replace('/TCGA_Seg/', '/TCGA_NIFTI/') for s in segmentations]
# print(f"{len(included_radiology)} radiology and {len(included_pathology)} pathology are included")
# filter_niis = list(set(included_radiology) - set(segmentation_niis))
# assert len(filter_niis) == 0, "Not all radiology have segmentations"
# filter_niis = list(set(segmentation_niis) - set(included_radiology))
# print(f"Found {len(filter_niis)} segmentations without images")
# filter_niis = [str(s).replace('/TCGA_NIFTI/', '/TCGA_Seg/') for s in filter_niis]
# for p in filter_niis:
#     parent_dir = os.path.dirname(p)
#     shutil.rmtree(parent_dir)
