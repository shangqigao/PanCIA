import os
import sys

# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
relative_path = os.path.join(script_dir, '../../')
sys.path.append(relative_path)

import json
import pathlib
import logging
import torch
torch.multiprocessing.set_sharing_strategy("file_system")

import argparse
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathology', default="/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/TCGA/WSI")
    parser.add_argument('--dataset', default="TCGA-RCC", type=str)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/pathomics", type=str)
    parser.add_argument('--mask_method', default='otsu', choices=["otsu", "morphological"], help='method of tissue masking')
    parser.add_argument('--feature_mode', default="conch", choices=["cnn", "vit", "uni", "conch", "chief", "uni2"], type=str)
    parser.add_argument('--node_features', default=37, choices=[2048, 384, 1024, 35, 768, 1536], type=int)
    parser.add_argument('--resolution', default=20, type=float)
    parser.add_argument('--units', default="power", type=str)
    args = parser.parse_args()

    ## get wsi path
    assert pathlib.Path(args.pathology).suffix == '.json', 'only support loading info from json file'
    with open(args.pathology, 'r') as f:
        data = json.load(f)
    included_subjects = data['included subjects']
    wsi_paths = []
    for k, v in included_subjects.items(): wsi_paths += v['pathology']
    logging.info("The number of selected WSIs on {}: {}".format(args.dataset, len(wsi_paths)))
    
    ## set save dir
    save_msk_dir = pathlib.Path(f"{args.save_dir}/{args.dataset}_pathomic_masks")
    save_feature_dir = pathlib.Path(f"{args.save_dir}/{args.dataset}_pathomic_features/{args.feature_mode}")
    
    # generate wsi tissue mask batch by batch
    # from analysis.a01_data_preprocessiong.m_tissue_masking import generate_wsi_tissue_mask
    # bs = 32
    # nb = len(wsi_paths) // bs if len(wsi_paths) % bs == 0 else len(wsi_paths) // bs + 1
    # for i in range(0, nb):
    #     logging.info(f"Processing WSIs of batch [{i+1}/{nb}] ...")
    #     start = i * bs
    #     end = min(len(wsi_paths), (i + 1) * bs)
    #     batch_wsi_paths = wsi_paths[start:end]
    #     generate_wsi_tissue_mask(
    #         wsi_paths=batch_wsi_paths,
    #         save_msk_dir=save_msk_dir,
    #         n_jobs=1,
    #         method=args.mask_method,
    #         resolution=1.25,
    #         units="power"
    #     )

    # extract wsi feature patch by patch
    # from analysis.a03_feature_extraction.m_feature_extraction import extract_pathomic_feature
    # msk_paths = [save_msk_dir / f"{p.stem}.jpg" for p in wsi_paths]
    # logging.info("The number of extracted tissue masks on {}: {}".format(args.dataset, len(msk_paths)))
    # bs = 32
    # nb = len(wsi_paths) // bs if len(wsi_paths) % bs == 0 else len(wsi_paths) // bs + 1
    # for i in range(0, nb):
    #     logging.info(f"Processing WSIs of batch [{i+1}/{nb}] ...")
    #     start = i * bs
    #     end = min(len(wsi_paths), (i + 1) * bs)
    #     batch_wsi_paths = wsi_paths[start:end]
    #     batch_msk_paths = msk_paths[start:end]
    #     extract_pathomic_feature(
    #         wsi_paths=batch_wsi_paths,
    #         wsi_msk_paths=batch_msk_paths,
    #         feature_mode=args.feature_mode,
    #         save_dir=save_feature_dir,
    #         resolution=args.resolution,
    #         units=args.units,
    #         skip_exist=True
    #     )

    # construct wsi graph
    # from analysis.a04_feature_aggregation.m_graph_construction import construct_wsi_graph
    # bs = 32
    # nb = len(wsi_paths) // bs if len(wsi_paths) % bs == 0 else len(wsi_paths) // bs + 1
    # for i in range(0, nb):
    #     logging.info(f"Processing WSIs of batch [{i+1}/{nb}] ...")
    #     start = i * bs
    #     end = min(len(wsi_paths), (i + 1) * bs)
    #     batch_wsi_paths = wsi_paths[start:end]
    #     construct_wsi_graph(
    #         wsi_paths=batch_wsi_paths,
    #         save_dir=save_feature_dir,
    #         n_jobs=1,
    #         delete_npy=False,
    #         skip_exist=True
    #     )

    # # extract minimum spanning tree
    # from analysis.a04_feature_aggregation.m_graph_construction import extract_minimum_spanning_tree
    # wsi_graph_paths = [save_feature_dir / f"{p.stem}.json" for p in wsi_paths]
    # extract_minimum_spanning_tree(
    #     wsi_graph_paths=wsi_graph_paths,
    #     save_dir=save_feature_dir,
    #     n_jobs=8
    # )

    # measure graph properties
    # from analysis.a04_feature_aggregation.m_graph_construction import measure_graph_properties
    # wsi_graph_paths = [save_feature_dir / f"{p.stem}.json" for p in wsi_paths]
    # wsi_label_paths = [save_feature_dir / f"{p.stem}.label.npy" for p in wsi_paths]
    # subgraph_dict = {
    #     "ADI": [0, 4],
    #     "BACK": [5, 8],
    #     "DEB": [9, 11],
    #     "LYM": [12, 16],
    #     "MUC": [17, 20],
    #     "MUS": [21, 25],
    #     "NORM": [26, 26],
    #     "STR": [27, 31],
    #     "TUM": [32, 34]
    # }
    # measure_graph_properties(
    #     graph_paths=wsi_graph_paths,
    #     label_paths=wsi_label_paths,
    #     save_dir=save_feature_dir,
    #     subgraph_dict=None,
    #     n_jobs=32
    # )

    # visualize feature
    # from analysis.a04_feature_aggregation.m_graph_construction import pathomic_feature_visualization
    # graph_feature = True
    # if graph_feature:
    #     save_label_dir = save_feature_dir
    # else:
    #     save_label_dir = save_classification_dir
    # pathomic_feature_visualization(
    #     wsi_paths=wsi_paths[0:900:90],
    #     save_feature_dir=save_feature_dir,
    #     mode="umap",
    #     save_label_dir=save_label_dir,
    #     graph=graph_feature
    # )

    # visualize graph properties
    # from analysis.a04_feature_aggregation.m_graph_construction import plot_graph_properties
    # graph_prop_paths = [save_feature_dir / f"{p.stem}.MST.graph.properties.json" for p in wsi_paths]
    # subgraph_dict = None
    # graph_prop_paths = [save_feature_dir / f"{p.stem}.MST.subgraphs.properties.json" for p in wsi_paths]
    # subgraph_dict = {
    #     "ADI": [0, 4],
    #     "BACK": [5, 8],
    #     "DEB": [9, 11],
    #     "LYM": [12, 16],
    #     "MUC": [17, 20],
    #     "MUS": [21, 25],
    #     "NORM": [26, 26],
    #     "STR": [27, 31],
    #     "TUM": [32, 34]
    # }
    # graph_properties = [
    #     "num_nodes", 
    #     "num_edges", 
    #     "num_components", 
    #     "degree", 
    #     "closeness", 
    #     "graph_diameter",
    #     "graph_assortativity",
    #     "mean_neighbor_degree"
    # ]
    # plot_types = ["bar", "stem", "hist", "box", "voilin", "plot"]
    # percentile = [90, 90, 90, 100, 90, 90, 100, 100]
    # for i in range(len(graph_properties)):
    #     plot_graph_properties(
    #         prop_paths=graph_prop_paths,
    #         subgraph_dict=subgraph_dict,
    #         prop_key=graph_properties[i],
    #         plotted=plot_types[4],
    #         min_percentile=0,
    #         max_percentile=percentile[i]
    #     )


    ## visualize graph on wsi
    # from analysis.a04_feature_aggregation.m_graph_construction import visualize_pathomic_graph
    # for wsi_path in wsi_paths:
    #     wsi_name = pathlib.Path(wsi_path).stem 
    #     logging.info(f"Visualizing graph of {wsi_name}...")
    #     graph_path = save_feature_dir / f"{wsi_name}.json"
    #     label_path = save_feature_dir / f"{wsi_name}.label.npy"
    #     # subgraph can be {key: int, ..., key: int}, a dict of mutiple classes
    #     # {key: [int, int], ..., key: [int, int]}, a dict of mutiple class ranges
    #     # subgraph = None
    #     # subgraph = {'immune': [12, 17], 'stroma': [27, 32], 'tumor': [32, 35]}
    #     subgraph = {'psammoma': [35, 37]}
    #     prompts = load_prompts(args.prompts, index=0)
    #     if subgraph is not None: 
    #         assert len(subgraph) > 0, "Empty subgraph!"
    #         class_name = ",".join(list(subgraph.keys()))
    #         values = list(subgraph.values())
    #         if isinstance(values[0], list):
    #             assert len(values[0]) == 2
    #             subgraph_id = subgraph
    #         else:
    #             subgraph_id = {k: [v, v + 1] for k, v in subgraph.items()}
    #         logging.info(f"Visualizing subgraph for {class_name} of {wsi_name}...")
    #     else:
    #         class_name = "pathomics"
    #         subgraph_id = None
    #         logging.info(f"Visualizing slide graph for {wsi_name}...")
    #     visualize_pathomic_graph(
    #         wsi_path=wsi_path,
    #         graph_path=graph_path,
    #         label=label_path,
    #         label_min=0,
    #         label_max=len(prompts) - 1,
    #         subgraph_id=subgraph_id,
    #         show_map=True,
    #         magnify=False,
    #         save_title=f"{wsi_name}:{class_name}",
    #         save_name=wsi_name,
    #         cmap_type='husl',
    #         resolution=args.resolution,
    #         units=args.units
    #     )