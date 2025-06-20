import sys
sys.path.append('./')

import pathlib
import logging
import argparse
from collections import defaultdict
import pandas as pd
import ast

from analysis.feature_extraction.m_feature_extraction import extract_radiomic_feature
from analysis.feature_aggregation.m_graph_construction import construct_img_graph
from analysis.feature_aggregation.m_graph_construction import radiomic_feature_visualization
from analysis.feature_aggregation.m_graph_construction import visualize_radiomic_graph
from analysis.feature_aggregation.m_graph_construction import measure_graph_properties

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--lab_dir', default=None)
    parser.add_argument('--meta_info', default=None)
    parser.add_argument('--modality', default="MRI", type=str)
    parser.add_argument('--format', default="rgb", choices=["dicom", "nifti", "rgb"], type=str)
    parser.add_argument('--site', default="breast", type=str)
    parser.add_argument('--target', default="tumor", type=str)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    parser.add_argument('--feature_mode', default="BiomedParse", choices=["pyradiomics", "SegVol", "M3D-CLIP", "BiomedParse"], type=str)
    parser.add_argument('--feature_dim', default=768, choices=[2048, 768, 768], type=int)
    parser.add_argument('--resolution', default=1.024, type=float)
    parser.add_argument('--units', default="mm", type=str)
    args = parser.parse_args()

    ## get image and label paths
    pixel_spacings = None
    if args.format == 'dicom':
        dicom_cases = pathlib.Path(args.img_dir).glob('*')
        dicom_cases = [p for p in dicom_cases if p.is_dir()]
        img_paths = [sorted(p.glob('*.dcm')) for p in dicom_cases]
    elif args.format == 'rgb':
        assert args.meta_info is not None, 'Expect meta data with pixel spacing info'
        meta_info = pd.read_excel(args.meta_info)
        meta_info['pixel_spacing'] = meta_info['pixel_spacing'].apply(ast.literal_eval)
        rgb_images = sorted(pathlib.Path(args.img_dir).glob('*.png'))
        grouped = defaultdict(list)
        pixel_spacings = []
        for p in rgb_images:
            idx = '_'.join(p.name.split('_')[:2])
            grouped[idx].append(p)
            pixel_spacing = meta_info[meta_info['patient_id'] == idx]['pixel_spacing']
            pixel_spacings.append(pixel_spacing)
        img_paths = list(grouped.values())
    elif args.format == 'nifti':
        img_paths = sorted(pathlib.Path(args.img_dir).glob('*.nii.gz'))
        if args.lab_dir is not None:
            lab_paths = sorted(pathlib.Path(args.lab_dir).glob('*.nii.gz'))
        else:
            lab_paths = [None]*len(img_paths)
    logging.info("The number of images on {}: {}".format(args.site, len(img_paths)))
    text_prompts = [[f'{args.site} {args.target}']]*len(img_paths)
    
    ## set save dir
    save_feature_dir = pathlib.Path(f"{args.save_dir}/{args.site}_{args.modality}_radiomic_features/{args.feature_mode}")
    
    # extract radiomics
    bs = 40000
    nb = len(img_paths) // bs if len(img_paths) % bs == 0 else len(img_paths) // bs + 1
    for i in range(0, nb):
        logging.info(f"Processing images of batch [{i+1}/{nb}] ...")
        start = i * bs
        end = min(len(img_paths), (i + 1) * bs)
        batch_img_paths = img_paths[start:end]
        batch_lab_paths = lab_paths[start:end]
        extract_radiomic_feature(
            img_paths=batch_img_paths,
            lab_paths=batch_lab_paths,
            feature_mode=args.feature_mode,
            save_dir=save_feature_dir,
            class_name=args.target,
            prompts=text_prompts,
            format=args.format,
            modality=args.modality,
            site=args.site,
            pixel_spacings=pixel_spacings
        )

    # construct image graph
    # construct_img_graph(
    #     img_paths=img_paths,
    #     save_dir=save_feature_dir,
    #     class_name=class_name,
    #     patch_size=(30, 30, 30),
    #     n_jobs=16
    # )

    # measure graph properties
    # graph_paths = [save_feature_dir / pathlib.Path(p).name.replace(".nii.gz", f"_{class_name}.json") for p in img_paths]
    # measure_graph_properties(
    #     graph_paths=graph_paths,
    #     label_paths=lab_paths,
    #     save_dir=save_feature_dir,
    #     subgraph_dict=None,
    #     n_jobs=32
    # )

    # visualize radiomics
    # radiomic_feature_visualization(
    #     img_paths=img_paths[0:1],
    #     save_feature_dir=save_feature_dir,
    #     class_name=class_name,
    #     mode="tsne",
    #     graph=False
    # )

    # visualize radiomic graph
    # for i in range(0, 200, 20):
    #     visualize_radiomic_graph(
    #         img_path=img_paths[i],
    #         lab_path=lab_paths[i],
    #         save_graph_dir=save_feature_dir,
    #         class_name=class_name,
    #         save_name=i
    #     )

