import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import json
import pathlib
import joblib

from tiatoolbox import logger

def cluster_radiomic_feature(
        img_paths, 
        feature_mode, 
        save_dir, 
        class_name,
        n_clusters,
        n_jobs=32,
        skip_exist=False
    ):
    """cluster spatial radiomic features
    Args:
        img_paths (list): a list of image paths
        fature_mode (str): mode of extracting features, 
            "pyradiomics" for extracting radiomics
        save_dir (str): directory of saving features
        label (int): value for which to extract features
        resolution (int): the resolution of extacting features
        units (str): the units of resolution, e.g., mpp  

    """
    if feature_mode == "BayesBP":
        _ = cluster_Bayes_BiomedParse_radiomics(
            img_paths=img_paths, 
            save_dir=save_dir, 
            class_name=class_name, 
            n_clusters=n_clusters, 
            n_jobs=n_jobs, 
            skip_exist=skip_exist
        )
    else:
        raise ValueError(f"Invalid feature mode: {feature_mode}")
    return

def Bayes_radiomics_pooling(feature_path, save_path, n_clusters=3):
    """pool layer-wise Bayesian features into fixed clusters
    """
    with open(feature_path, 'r') as f:
        data = json.load(f)

    df_ori = pd.DataFrame(data['radiomics'][1:])
    df = df_ori.dropna().reset_index(drop=True)
    df['layer_index'] = range(len(df))
    assert len(df) >= n_clusters, 'sample size less than the number of clusters'

    pooled_data = {f"global.{k}": v for k, v in data['radiomics'][0].items() if k != "layer_index"}

    # Step 1: Standardize features for clustering (exclude max/min)
    features_for_clustering = ['mean','var','skewness','kurtosis','entropy']
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features_for_clustering] = scaler.fit_transform(df[features_for_clustering])

    # Step 2: Cluster layers
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_scaled['cluster'] = kmeans.fit_predict(df_scaled[features_for_clustering])

    # Step 3: Pool features per cluster and store in DataFrame
    for c in range(n_clusters):
        cluster_df = df[df_scaled['cluster']==c]
        if cluster_df.empty:
            # handle empty cluster
            pooled_data[f'cluster{c}.n_voxels'] = 0
            pooled_data[f'cluster{c}.mean'] = 0
            pooled_data[f'cluster{c}.max'] = 0
            pooled_data[f'cluster{c}.min'] = 0
            pooled_data[f'cluster{c}.var'] = 0
            pooled_data[f'cluster{c}.skewness'] = 0
            pooled_data[f'cluster{c}.kurtosis'] = 0
            pooled_data[f'cluster{c}.entropy'] = 0
        else:
            pooled_data[f'cluster{c}.n_voxels'] = cluster_df['n_voxels'].sum()
            pooled_data[f'cluster{c}.mean'] = cluster_df['mean'].mean()
            pooled_data[f'cluster{c}.max'] = cluster_df['max'].max()
            pooled_data[f'cluster{c}.min'] = cluster_df['min'].min()
            pooled_data[f'cluster{c}.var'] = cluster_df['var'].mean()
            pooled_data[f'cluster{c}.skewness'] = cluster_df['skewness'].mean()
            pooled_data[f'cluster{c}.kurtosis'] = cluster_df['kurtosis'].mean()
            pooled_data[f'cluster{c}.entropy'] = cluster_df['entropy'].mean()

    pooled_data = {k: float(v) for k, v in pooled_data.items()}
    with open(save_path, "w") as f:
        json.dump(pooled_data, f, indent=4)

    return

def cluster_Bayes_BiomedParse_radiomics(img_paths, save_dir, class_name="tumour", n_clusters=3, n_jobs=32, skip_exist=False):
    """pool layer-wise Bayesian features by clustering
    Args:
        img_paths (list): a list of image paths
        save_dir (str): directory of reading feature and saving results
    """
    def _feature_clustering(idx, img_path):
        img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")
        save_path = pathlib.Path(f"{save_dir}/{img_name}_{class_name}_radiomics_pooled.json")
        if save_path.exists() and skip_exist:
            logger.info(f"{save_path.name} has existed, skip!")
            return

        feature_path = pathlib.Path(f"{save_dir}/{img_name}_{class_name}_radiomics.json")
        if not feature_path.exists(): 
            logger.info(f"{feature_path.name} doesn't exist, skip!")
            return
            
        logger.info("clustering Bayesian radiomics: {}/{}...".format(idx + 1, len(img_paths)))
        Bayes_radiomics_pooling(feature_path, save_path, n_clusters)
        return
    
    # construct graphs in parallel
    joblib.Parallel(n_jobs=n_jobs, backend="threading")(
        joblib.delayed(_feature_clustering)(idx, img_path)
        for idx, img_path in enumerate(img_paths)
    )
    return 
