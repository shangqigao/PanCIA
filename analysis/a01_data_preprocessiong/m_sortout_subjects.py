import os
import sys

# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
relative_path = os.path.join(script_dir, '../../')
sys.path.append(relative_path)

from joblib import Parallel, delayed
from pathlib import Path
import json
import argparse
import pandas as pd

from tiatoolbox import logger
from utilities.constants import PANCIA_PROJECT_SITE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--included_nifti', default="")
    parser.add_argument('--included_wsi', default="")
    parser.add_argument('--meta_data', default="")
    parser.add_argument('--clinical_data', default="")
    parser.add_argument('--dataset', default="TCGA", type=str)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    parser.add_argument('--pre_diagnosis', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.meta_data)

    with open(f"{args.included_nifti}", "r") as f:
        niftis = json.load(f)
    in_niftis = niftis["included nifti"]
    in_nifti_ids = [Path(p).parent.name for p in in_niftis]
    df_nifti = pd.DataFrame({'Series ID': in_nifti_ids, 'Series Path': in_niftis})

    with open(f"{args.included_wsi}", "r") as f:
        wsis = json.load(f)
    in_wsis = wsis["included series"]
    in_wsi_ids = [Path(p).stem[:12] for p in in_wsis]
    df_wsi = pd.DataFrame({'Subject ID': in_wsi_ids, 'WSI Path': in_wsis})

    if args.pre_diagnosis:
        df_clinical = pd.read_csv(args.clinical_data)

    subject_ids = df['Subject ID'].unique().tolist()
    def _inclusion_exclusion(idx, subject_id):
        logger.info(f"Processing [{idx + 1} / {len(subject_ids)}] ...")
        df_subject = df[df['Subject ID'] == subject_id].copy()
        project_id = df_subject['Collection Name'].unique().tolist()
        assert len(project_id) == 1, "Found one subject belongs to multiple projects"
        in_projects = PANCIA_PROJECT_SITE.get(project_id[0], False)
        df_subject['Study Date'] = pd.to_datetime(df['Study Date'], format='%Y-%m-%d')

        # filter out studies after diagnosis year
        if args.pre_diagnosis:
            dx_year = df_clinical.loc[df_clinical['_PATIENT'] == subject_id, 'initial_pathologic_dx_year'].values[0]
            df_subject = df_subject[df_subject['Study Date'].dt.year < dx_year].copy()
            
        df_subject = df_subject.sort_values(by='Study Date')

        pathology = df_wsi[df_wsi['Subject ID'] == subject_id]
        common = df_subject['Series ID'][df_subject['Series ID'].isin(df_nifti['Series ID'])]
        radiology = df_nifti[df_nifti['Series ID'].isin(common)].set_index('Series ID').loc[common].reset_index()
        subject_dict = {
                'radiology': radiology['Series Path'].tolist(),
                'pathology': pathology['WSI Path'].tolist()
        }
        if in_projects and not pathology.empty and not radiology.empty: 
            return ("included", subject_id, subject_dict)
        else:
            logger.info(f"Excluding subject {subject_id}")
            return ("excluded", subject_id, subject_dict)
    # process in parallel
    results = Parallel(n_jobs=32, backend="threading")(
        delayed(_inclusion_exclusion)(idx, subject_id)
        for idx, subject_id in enumerate(subject_ids)
    )
    # Merge results
    included_subjects = {s : d for t, s, d in results if t == "included"}
    excluded_subjects = {s : d for t, s, d in results if t == "excluded"}
    logger.info(f"Totally {len(included_subjects)} subjects included")
    logger.info(f"Totally {len(excluded_subjects)} subjects excluded")
    if args.pre_diagnosis:
        save_path = f"{args.save_dir}/{args.dataset}_included__prediagnosis_subjects.json"
    else:
        save_path = f"{args.save_dir}/{args.dataset}_included_subjects.json"
    data_dict = {"included subjects": included_subjects, "excluded subjects": excluded_subjects}
    with open(save_path, "w") as f:
        json.dump(data_dict, f, indent=4)