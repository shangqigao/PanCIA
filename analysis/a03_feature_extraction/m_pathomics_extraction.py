import os
import sys

# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
relative_path = os.path.join(script_dir, '../../')
sys.path.append(relative_path)

import json
import pathlib
import torch
torch.multiprocessing.set_sharing_strategy("file_system")

import argparse
import warnings
warnings.filterwarnings('ignore')

from tiatoolbox import logger

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', nargs='+', required=True, help='Path(s) to the config file(s).')
    args = parser.parse_args()

    from utilities.arguments import load_opt_from_config_files
    opt = load_opt_from_config_files(args.config_files)

    ## get wsi path
    assert pathlib.Path(opt['PATHOLOGY']).suffix == '.json', 'only support loading info from json file'
    with open(opt['PATHOLOGY'], 'r') as f:
        data = json.load(f)
    included_subjects = data['included subjects']
    wsi_paths = []
    for k, v in included_subjects.items(): wsi_paths += v['pathology']
    logger.info("The number of selected WSIs on {}: {}".format(opt['DATASET'], len(wsi_paths)))
    
    ## set save dir
    save_msk_dir = pathlib.Path(f"{opt['SAVE_DIR']}/{opt['DATASET']}_pathomic_masks")
    pathomics_mode = opt['PATHOMICS']['MODE']['VALUE']
    save_feature_dir = pathlib.Path(f"{opt['SAVE_DIR']}/{opt['DATASET']}_pathomic_features/{pathomics_mode}")
    
    # generate wsi tissue mask batch by batch
    if opt['PATHOMICS']['TASKS']['TISSUE_MASKING']:
        from analysis.a01_data_preprocessiong.m_tissue_masking import generate_wsi_tissue_mask
        bs = 32
        nb = len(wsi_paths) // bs if len(wsi_paths) % bs == 0 else len(wsi_paths) // bs + 1
        for i in range(0, nb):
            logger.info(f"Processing WSIs of batch [{i+1}/{nb}] ...")
            start = i * bs
            end = min(len(wsi_paths), (i + 1) * bs)
            batch_wsi_paths = wsi_paths[start:end]
            generate_wsi_tissue_mask(
                wsi_paths=batch_wsi_paths,
                save_msk_dir=save_msk_dir,
                n_jobs=8,
                method=opt['PATHOMICS']['MASKING_METHOD']['VALUE'],
                resolution=1.25,
                units="power"
            )

    # extract wsi feature patch by patch
    if opt['PATHOMICS']['TASKS']['EXTRACTION']:
        from analysis.a03_feature_extraction.m_feature_extraction import extract_pathomic_feature
        msk_paths = [save_msk_dir / f"{pathlib.Path(p).stem}.jpg" for p in wsi_paths]
        logger.info("The number of extracted tissue masks on {}: {}".format(opt['DATASET'], len(msk_paths)))
        extract_pathomic_feature(
            wsi_paths=wsi_paths,
            wsi_msk_paths=msk_paths,
            feature_mode=pathomics_mode,
            save_dir=save_feature_dir,
            resolution=opt['PATHOMICS']['RESOLUTION'],
            units=opt['PATHOMICS']['UNITS'],
            skip_exist=opt['PATHOMICS']['SKIP_EXITS']
        )

    # construct wsi graph
    if opt['PATHOMICS']['TASKS']['GRAPH_CONSTRUCTION']:
        from analysis.a04_feature_aggregation.m_graph_construction import construct_wsi_graph
        construct_wsi_graph(
            wsi_paths=wsi_paths,
            save_dir=save_feature_dir,
            n_jobs=opt['N_JOBS'],
            delete_npy=opt['PATHOMICS']['DELETE_NPY'],
            skip_exist=opt['PATHOMICS']['SKIP_EXITS']
        )

    # aggregate wsi graph
    if opt['PATHOMICS']['TASKS']['GRAPH_AGGREGATION']['VALUE']:
        from analysis.a04_feature_aggregation.m_graph_construction import aggregate_wsi_graph
        aggregate_wsi_graph(
            wsi_paths=wsi_paths,
            save_dir=save_feature_dir,
            mode=opt['PATHOMICS']['TASKS']['GRAPH_AGGREGATION']['MODE'],
            n_jobs=opt['N_JOBS'],
            skip_exist=opt['PATHOMICS']['SKIP_EXITS']
        )

    # convert graph to npz
    if opt['PATHOMICS']['TASKS']['CONVERT_JSON2NPZ']:
        from analysis.a04_feature_aggregation.m_graph_construction import convert_wsi_graph_to_npz
        convert_wsi_graph_to_npz(
            wsi_paths=wsi_paths,
            save_dir=save_feature_dir,
            n_jobs=opt['N_JOBS'],
            skip_exist=opt['PATHOMICS']['SKIP_EXITS']
        )

    # measure graph properties
    if opt['PATHOMICS']['TASKS']['MEASURE_GRAPH_PROPERTIES']:
        from analysis.a04_feature_aggregation.m_graph_construction import measure_graph_properties
        wsi_graph_paths = [save_feature_dir / f"{p.stem}_graph.json" for p in wsi_paths]
        wsi_label_paths = [save_feature_dir / f"{p.stem}_label.npy" for p in wsi_paths]
        subgraph_dict = {
            "ADI": [0, 4],
            "BACK": [5, 8],
            "DEB": [9, 11],
            "LYM": [12, 16],
            "MUC": [17, 20],
            "MUS": [21, 25],
            "NORM": [26, 26],
            "STR": [27, 31],
            "TUM": [32, 34]
        }
        measure_graph_properties(
            graph_paths=wsi_graph_paths,
            label_paths=wsi_label_paths,
            save_dir=save_feature_dir,
            subgraph_dict=None,
            n_jobs=opt['N_JOBS']
        )

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
    #     logger.info(f"Visualizing graph of {wsi_name}...")
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
    #         logger.info(f"Visualizing subgraph for {class_name} of {wsi_name}...")
    #     else:
    #         class_name = "pathomics"
    #         subgraph_id = None
    #         logger.info(f"Visualizing slide graph for {wsi_name}...")
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