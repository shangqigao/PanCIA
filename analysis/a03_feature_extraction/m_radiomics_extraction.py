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

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--lab_dir', default=None)
    parser.add_argument('--lab_mode', default="expert", choices=["expert", "nnUNet", "BiomedParse"], type=str)
    parser.add_argument('--meta_info', default=None)
    parser.add_argument('--modality', default="MRI", type=str)
    parser.add_argument('--format', default="nifti", choices=["dicom", "nifti"], type=str)
    parser.add_argument('--phase', default="1st-contrast", choices=["pre-contrast", "1st-contrast", "2nd-contrast", "multiple"], type=str)
    parser.add_argument('--site', default="breast", type=str)
    parser.add_argument('--target', default="tumor", type=str)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    parser.add_argument('--feature_mode', default="BayesBP", choices=["pyradiomics", "SegVol", "BiomedParse", "BayesBP"], type=str)
    parser.add_argument('--feature_dim', default=768, choices=[2048, 768, 512], type=int)
    parser.add_argument('--dilation_mm', default=10, type=float)
    parser.add_argument('--layer_method', default="peeling")
    parser.add_argument('--resolution', default=1, type=float)
    parser.add_argument('--units', default="mm", type=str)
    args = parser.parse_args()

    ## get image and label paths
    if args.format == 'dicom':
        dicom_cases = pathlib.Path(args.img_dir).glob('*')
        dicom_cases = [p for p in dicom_cases if p.is_dir()]
        img_paths = [sorted(p.glob('*.dcm')) for p in dicom_cases]
    elif args.format == 'nifti':
        if args.phase == "pre-contrast":
            img_paths = sorted(pathlib.Path(args.img_dir).rglob('*_0000.nii.gz'))
        elif args.phase == "1st-contrast":
            img_paths = sorted(pathlib.Path(args.img_dir).rglob('*_0001.nii.gz'))
        elif args.phase == "2nd-contrast":
            img_paths = sorted(pathlib.Path(args.img_dir).rglob('*_0002.nii.gz'))
        else:
            case_paths = sorted(pathlib.Path(args.img_dir).glob('*'))
            case_paths = [p for p in case_paths if p.is_dir()]
            img_paths = []
            for path in case_paths:
                nii_paths = path.glob("*.nii.gz")
                multiphase_keys = ["_0000.nii.gz", "_0001.nii.gz", "_0002.nii.gz"]
                nii_paths = [p for p in nii_paths if any(k in p.name for k in multiphase_keys)]
                img_paths.append(sorted(nii_paths))
        if args.lab_dir is not None:
            lab_paths = sorted(pathlib.Path(f"{args.lab_dir}/{args.lab_mode}").glob('*.nii.gz'))
        else:
            lab_paths = [None]*len(img_paths)
    logging.info("The number of images on {}: {}".format(args.site, len(img_paths)))
    text_prompts = [[f'{args.site} {args.target}']]*len(img_paths)
    
    ## set save dir
    save_feature_dir = pathlib.Path(f"{args.save_dir}/{args.site}_{args.modality}_radiomic_features/{args.feature_mode}/{args.phase}/{args.lab_mode}")
    
    # extract radiomics
    # warning: do not run this function in a loop
    # from analysis.a03_feature_extraction.m_feature_extraction import extract_radiomic_feature

    # extract_radiomic_feature(
    #     img_paths=img_paths,
    #     lab_paths=lab_paths,
    #     feature_mode=args.feature_mode,
    #     save_dir=save_feature_dir,
    #     class_name=args.target,
    #     prompts=text_prompts,
    #     format=args.format,
    #     modality=args.modality,
    #     site=args.site,
    #     dilation_mm=args.dilation_mm,
    #     layer_method=args.layer_method,
    #     resolution=args.resolution,
    #     units=args.units,
    #     skip_exist=True
    # )

    # cluster radiomics
    from analysis.a04_feature_aggregation.m_spatial_feature_clustering import cluster_radiomic_feature 

    cluster_radiomic_feature(
        img_paths=img_paths, 
        feature_mode=args.feature_mode, 
        save_dir=save_feature_dir, 
        class_name=args.target,
        n_jobs=32,
        skip_exist=True
    )

    # construct image graph
    # from analysis.a04_feature_aggregation.m_graph_construction import construct_img_graph
    
    # construct_img_graph(
    #     img_paths=img_paths,
    #     save_dir=save_feature_dir,
    #     class_name=args.target,
    #     window_size=30**3,
    #     n_jobs=32,
    #     delete_npy=True,
    #     skip_exist=True
    # )

    # measure graph properties
    # from analysis.a04_feature_aggregation.m_graph_construction import measure_graph_properties
    
    # graph_paths = [save_feature_dir / pathlib.Path(p).name.replace(".nii.gz", f"_{class_name}.json") for p in img_paths]
    # measure_graph_properties(
    #     graph_paths=graph_paths,
    #     label_paths=lab_paths,
    #     save_dir=save_feature_dir,
    #     subgraph_dict=None,
    #     n_jobs=32
    # )

    # visualize radiomics
    # from analysis.a04_feature_aggregation.m_graph_construction import radiomic_feature_visualization
    
    # radiomic_feature_visualization(
    #     img_paths=img_paths[0:10],
    #     save_feature_dir=save_feature_dir,
    #     class_name=args.target,
    #     mode="umap",
    #     graph=True
    # )

    # visualize radiomic graph
    # from analysis.a04_feature_aggregation.m_graph_construction import visualize_radiomic_graph
    
    # visualize_radiomic_graph(
    #     img_path=img_paths[0],
    #     lab_path=lab_paths[0],
    #     save_graph_dir=save_feature_dir,
    #     class_name=args.target,
    #     spacing=tuple([args.resolution]*3),
    #     feature_extractor=args.feature_mode,
    #     remove_front_corner=False
    # )

