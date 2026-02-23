import sys
import os
# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
relative_path = os.path.join(script_dir, '../../')
sys.path.append(relative_path)

import pathlib
import logging
import argparse
import warnings
warnings.filterwarnings('ignore')

from analysis.a01_data_preprocessiong.m_prepare_dataset_info import prepare_MAMAMIA_info
from analysis.a01_data_preprocessiong.m_prepare_dataset_info import prepare_TCGA_radiology_info
from analysis.a01_data_preprocessiong.m_prepare_dataset_info import prepare_CPTAC_radiology_info

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+', required=True, help='Path(s) to the config file(s).')
    args = parser.parse_args()
    
    from utilities.arguments import load_opt_from_config_files
    opt = load_opt_from_config_files(args.config_files)
    if opt.get('META_INFO', False):
        meta_info = opt['META_INFO']
    else:
        meta_info = None

    ## get image and label paths
    if opt['DATASET'] == 'MAMAMIA':
        dataset_info = prepare_MAMAMIA_info(
            img_dir=opt['RADIOLOGY'],
            lab_dir=opt['LABEL_DIR'],
            lab_mode=opt['RADIOMICS']['SEGMENTATOR']['VALUE'],
            img_format=opt['DATA_FORMAT'],
            phase='pre-contrast',
            meta_info=meta_info
        )
    elif opt['DATASET'] == 'TCGA':
        dataset_info = prepare_TCGA_radiology_info(
            img_json=opt['RADIOLOGY'],
            lab_dir=opt['LABEL_DIR'],
            lab_mode=opt['RADIOMICS']['SEGMENTATOR']['VALUE'],
            img_format=opt['DATA_FORMAT'],
            seg_obj=opt['RADIOMICS']['SEGMENTATOR']['OBJECT']
        )
    elif opt['DATASET'] == 'CPTAC':
        dataset_info = prepare_CPTAC_radiology_info(
            img_json=opt['RADIOLOGY'],
            lab_dir=opt['LABEL_DIR'],
            lab_mode=opt['RADIOMICS']['SEGMENTATOR']['VALUE'],
            img_format=opt['DATA_FORMAT'],
            seg_obj=opt['RADIOMICS']['SEGMENTATOR']['OBJECT']
        )
    else:
        raise NotImplementedError
    
    ## set save dir
    save_feature_dir = pathlib.Path(opt['SAVE_DIR'])
    save_feature_dir = save_feature_dir / f"{opt['DATASET']}_radiomic_features"
    save_feature_dir = save_feature_dir / f"{opt['RADIOMICS']['MODE']['VALUE']}"
    save_feature_dir = save_feature_dir / f"segmentator_{opt['RADIOMICS']['SEGMENTATOR']['VALUE']}"
    
    # extract radiomics
    # warning: do not run this function in a loop
    if opt['RADIOMICS']['TASKS']['EXTRACTION']:
        from analysis.a03_feature_extraction.m_feature_extraction import extract_radiomic_feature
        print("Number of images", len(dataset_info['img_paths']))
        extract_radiomic_feature(
            img_paths=dataset_info['img_paths'],
            lab_paths=dataset_info['lab_paths'],
            feature_mode=opt['RADIOMICS']['MODE']['VALUE'],
            save_dir=save_feature_dir,
            target=opt['RADIOMICS']['TARGET'],
            prompts=dataset_info['text_prompts'],
            format=dataset_info['img_format'],
            modality=dataset_info['modality'],
            site=dataset_info['site'],
            batch_size=opt['RADIOMICS']['BATCH_SIZE'],
            dilation_mm=opt['RADIOMICS']['DILATION_MM'],
            layer_method=opt['RADIOMICS']['LAYER_METHOD']['VALUE'],
            resolution=opt['RADIOMICS']['RESOLUTION'],
            units=opt['RADIOMICS']['UNITS'],
            n_jobs=opt['N_JOBS'],
            skip_exist=opt['RADIOMICS']['SKIP_EXITS']
        )

    # cluster radiomics
    if opt['RADIOMICS']['TASKS']['CLUSTERING']:
        from analysis.a04_feature_aggregation.m_spatial_feature_clustering import cluster_radiomic_feature 
        cluster_radiomic_feature(
            img_paths=dataset_info['img_paths'], 
            feature_mode=opt['RADIOMICS']['MODE']['VALUE'], 
            save_dir=save_feature_dir, 
            target=opt['RADIOMICS']['TARGET'],
            n_clusters=6,
            n_jobs=opt['N_JOBS'],
            skip_exist=opt['RADIOMICS']['SKIP_EXITS']
        )

    # construct image graph
    if opt['RADIOMICS']['TASKS']['GRAPH_CONSTRUCTION']['VALUE']:
        from analysis.a04_feature_aggregation.m_graph_construction import construct_img_graph
        construct_img_graph(
            img_paths=dataset_info['img_paths'],
            save_dir=save_feature_dir,
            radiomics_suffix=opt['RADIOMICS']['MODE']['SUFFIX'],
            target=opt['RADIOMICS']['TARGET'],
            window_size=24**3,
            lambda_f=opt['RADIOMICS']['TASKS']['GRAPH_CONSTRUCTION']['FEATURE_DIS_WEIGHT'],
            n_jobs=opt['N_JOBS'],
            delete_npy=opt['RADIOMICS']['DELETE_NPY'],
            skip_exist=opt['RADIOMICS']['SKIP_EXITS']
        )

    # aggregate image graph
    if opt['RADIOMICS']['TASKS']['GRAPH_AGGREGATION']['VALUE']:
        from analysis.a04_feature_aggregation.m_graph_construction import aggregate_img_graph
        aggregate_img_graph(
            img_paths=dataset_info['img_paths'],
            save_dir=save_feature_dir,
            radiomics_suffix=opt['RADIOMICS']['MODE']['SUFFIX'],
            target=opt['RADIOMICS']['TARGET'],
            mode=opt['RADIOMICS']['TASKS']['GRAPH_AGGREGATION']['MODE'],
            n_jobs=opt['N_JOBS'],
            skip_exist=opt['RADIOMICS']['SKIP_EXITS']
        )

    # convert graph to npz
    if opt['RADIOMICS']['TASKS']['CONVERT_JSON2NPZ']:
        from analysis.a04_feature_aggregation.m_graph_construction import convert_img_graph_to_npz
        convert_img_graph_to_npz(
            img_paths=dataset_info['img_paths'],
            save_dir=save_feature_dir,
            radiomics_suffix=opt['RADIOMICS']['MODE']['SUFFIX'],
            target=opt['RADIOMICS']['TARGET'],
            n_jobs=opt['N_JOBS'],
            skip_exist=opt['RADIOMICS']['SKIP_EXITS']
        )

    # measure graph properties
    if opt['RADIOMICS']['TASKS']['MEASURE_GRAPH_PROPERTIES']:
        from analysis.a04_feature_aggregation.m_graph_construction import measure_graph_properties
        parent_names = [pathlib.Path(p).parent.name for p in dataset_info['img_paths']]
        graph_suffix = f"_{opt['RADIOMICS']['TARGET']}_graph.json"
        graph_names = [pathlib.Path(p).name.replace(".nii.gz", graph_suffix) for p in dataset_info['img_paths']]
        graph_paths = [save_feature_dir / p / g for p, g in zip(parent_names, graph_names)]
        measure_graph_properties(
            graph_paths=graph_paths,
            label_paths=[None] * len(graph_paths),
            save_dir=save_feature_dir,
            with_parent_name=True,
            subgraph_dict=None,
            n_jobs=32
        )

    # visualize radiomics
    if opt['RADIOMICS']['TASKS']['VISUALIZE_RADIOMICS']:
        from analysis.a04_feature_aggregation.m_graph_construction import radiomic_feature_visualization
        radiomic_feature_visualization(
            img_paths=dataset_info['img_paths'][0:10],
            save_feature_dir=save_feature_dir,
            target=opt['RADIOMICS']['TARGET'],
            mode="umap",
            graph=True
        )

    # visualize radiomic graph
    if opt['RADIOMICS']['TASKS']['VISUALIZE_GRAPH']:
        from analysis.a04_feature_aggregation.m_graph_construction import visualize_radiomic_graph
        visualize_radiomic_graph(
            img_path=dataset_info['img_paths'][0],
            lab_path=dataset_info['lab_paths'][0],
            save_graph_dir=save_feature_dir,
            target=opt['RADIOMICS']['TARGET'],
            spacing=tuple([opt['RADIOMICS']['RESOLUTION']]*3),
            feature_extractor=opt['RADIOMICS']['MODE']['VALUE'],
            remove_front_corner=False
        )

