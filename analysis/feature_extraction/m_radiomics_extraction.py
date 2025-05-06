import sys
sys.path.append('../')

import pathlib
import logging
import argparse

from .m_feature_extraction import extract_radiomic_feature
from feature_aggregation.m_graph_construction import construct_img_graph
from feature_aggregation.m_graph_construction import radiomic_feature_visualization
from feature_aggregation.m_graph_construction import visualize_radiomic_graph
from feature_aggregation.m_graph_construction import measure_graph_properties

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--lab_dir', default="/home/s/sg2162/projects/TCIA_NIFTI/binary_label")
    parser.add_argument('--dataset', default="TCGA-RCC", type=str)
    parser.add_argument('--modality', default="CT", type=str)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    parser.add_argument('--feature_mode', default="SegVol", choices=["pyradiomics", "SegVol", "M3D-CLIP"], type=str)
    parser.add_argument('--feature_dim', default=768, choices=[2048, 768, 768], type=int)
    parser.add_argument('--resolution', default=1.024, type=float)
    parser.add_argument('--units', default="mm", type=str)
    args = parser.parse_args()

    ## get image and label paths
    class_name = ["kidney_and_mass", "mass", "tumour"][2]
    lab_dir = pathlib.Path(f"{args.lab_dir}/{args.dataset}/{args.modality}")
    lab_paths = lab_dir.rglob(f"{class_name}.nii.gz")
    lab_paths = [f"{p}" for p in lab_paths]
    img_paths = [p.replace(args.lab_dir, args.img_dir) for p in lab_paths]
    img_paths = [p.replace(f"_ensemble/{class_name}.nii.gz", ".nii.gz") for p in img_paths]
    logging.info("The number of images on {}: {}".format(args.dataset, len(img_paths)))
    
    ## set save dir
    save_feature_dir = pathlib.Path(f"{args.save_dir}/{args.dataset}_{args.modality}_radiomic_features/{args.feature_mode}")
    
    # extract radiomics
    # bs = 8
    # nb = len(img_paths) // bs if len(img_paths) % bs == 0 else len(img_paths) // bs + 1
    # for i in range(0, nb):
    #     logging.info(f"Processing images of batch [{i+1}/{nb}] ...")
    #     start = i * bs
    #     end = min(len(img_paths), (i + 1) * bs)
    #     batch_img_paths = img_paths[start:end]
    #     batch_lab_paths = lab_paths[start:end]
    #     extract_radiomic_feature(
    #         img_paths=batch_img_paths,
    #         lab_paths=batch_lab_paths,
    #         feature_mode=args.feature_mode,
    #         save_dir=save_feature_dir,
    #         class_name=class_name,
    #         label=1,
    #         n_jobs=8,
    #         resolution=args.resolution
    #     )

    # construct image graph
    # construct_img_graph(
    #     img_paths=img_paths,
    #     save_dir=save_feature_dir,
    #     class_name=class_name,
    #     patch_size=(30, 30, 30),
    #     n_jobs=16
    # )

    # measure graph properties
    graph_paths = [save_feature_dir / pathlib.Path(p).name.replace(".nii.gz", f"_{class_name}.json") for p in img_paths]
    measure_graph_properties(
        graph_paths=graph_paths,
        label_paths=lab_paths,
        save_dir=save_feature_dir,
        subgraph_dict=None,
        n_jobs=32
    )

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

