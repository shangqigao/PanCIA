# **Pan Cancer Image Analysis**

---

## Dependencies

For pan-cancer image analysis
- conda create -n PanCIA python=3.9.19
- conda activate PanCIA
- conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
- conda install -c conda-forge openslide
- conda install -c conda-forge gcc_linux-64 gxx_linux-64 (optional)
- pip install --no-build-isolation git+https://github.com/MaureenZOU/detectron2-xyz.git
- pip install -r requirements-pancia.txt
- pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
- pip install torch-sparse  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

If GCC is too old (need GCC 9 or later):
- conda install -c conda-forge gcc_linux-64 gxx_linux-64

For MLLM 
- conda create --prefix /path/to/Qwen python=3.9 -y
- pip install -r requirements-mllm.txt

---

## Remark
- Loads LoRA-based BiomedParse multiple times in a loop can get different results due to the nondeterminism in language encoder, so just loads it once.

---

## Pipeline
### Step 1: data preprocessing
#### Radiology
- Requirements of radiology path
- Radiology exclusion and inclusion (only for DICOM)

```bash
data_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer"
save_dir="/home/sg2162/rds/hpc-work/Experiments/radiomics"

python analysis/a01_data_preprocessiong/m_inclusion_exclusion.py \
            --data_dir $data_dir \
            --dataset TCGA \
            --modality radiology \
            --save_dir $save_dir
```

- dicom to nifit (save to /parent/to/dataset/dataset_NIFTI)

```bash
series="/home/sg2162/rds/hpc-work/Experiments/radiomics/TCGA_included_raw_series.json"

python analysis/a01_data_preprocessiong/m_dicom2nii.py \
            --series $series \
            --dataset TCGA
```

#### Pathology
pathology exclusion and inclusion

```bash
data_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer"
save_dir="/home/sg2162/rds/hpc-work/Experiments/pathomics"

python analysis/a01_data_preprocessiong/m_inclusion_exclusion.py \
            --data_dir $data_dir \
            --dataset TCGA \
            --modality pathology \
            --save_dir $save_dir
```

#### subject exclusion and inclusion

```bash
included_nifti="/home/sg2162/rds/hpc-work/Experiments/radiomics/TCGA_included_nifti.json"
included_wsi="/home/sg2162/rds/hpc-work/Experiments/pathomics/TCGA_included_wsi.json"
meta_data="/home/sg2162/rds/hpc-work/Experiments/clinical/TCGA_pathology_has_radiology.csv"
save_dir="/home/sg2162/rds/hpc-work/Experiments/clinical"

python analysis/a01_data_preprocessiong/m_sortout_subjects.py \
            --included_nifti $included_nifti \
            --included_wsi $included_wsi \
            --meta_data $meta_data \
            --dataset TCGA \
            --save_dir $save_dir
```

### Step 2: radiology tumor segmentation

```bash
radiology="/home/sg2162/rds/hpc-work/Experiments/clinical/CPTAC_included_subjects.json"
save_dir="/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/CPTAC_Seg"
srun python analysis/a02_tumor_segmentation/m_tumor_segmentation.py \
            --radiology $radiology \
            --dataset CPTAC \
            --save_dir $save_dir
```

### Step 3: feature extraction

#### Radiomic feature extraction

```bash
radiomics_config="/home/sg2162/rds/hpc-work/PanCIA/configs/feature_extraction/radiomics_extraction.yaml"
python analysis/a03_feature_extraction/m_radiomics_extraction.py --config_files $radiomics_config
```

Exact whole slice features
    - TARGET: "slice"
    - DILATION_MM: 0 # only select slices with tumors, no dilation applied to tumor mask
    - SAMPLING_RATE: 0.01 # randomly sample spatial features to reduce feature size
    - MODE: choose one from ["BiomedParse", "LVMMed"], both can extract volumetric features (4D tensor), and LVMMed (ResNet architecture) can extract multi-scale volumetric features.

Extract intra-tumor features
    - TARGET: "tumor"
    - DILATION_MM: 10 # dialate tumor mask to include marginal information
    - SAMPLING_RATE: 1 # no sampling required since tumor size is small in general
    - MODE: choose one from ["pyradiomics", "FMCIB", "BiomedParse", "LVMMed"], the former two can only extract scan-level embeddings, while the latter two can extract volumetric features (4D tensor)

Suffix used to save radiomic features
    - BiomedParse extract single scale features, and its SUFFIX is
        - "radiomics.npy"
    - LVMMed extracts multi-scale features, and its SUFFIX is
        - "layer0_radiomics.npy"
        - "layer1_radiomics.npy"
        - "layer2_radiomics.npy"
        - "layer3_radiomics.npy"
        - "layer4_radiomics.npy"

Graph construction
    - VALUE: True
    - FEATURE_DIS_WEIGHT: 0.01 # 0.01 for LVMMed else 0.1, since LVMMed (ResNet backbone) has larger feature scales, used for feature clustering in graph construction. If you want to use other feature extractors, this should be adjusted. In general, if backbone is ViT, use 0.1.
    - SAVE_CLUSTER_POINTS: False # only true for visualization purpose, default false to reduce json file size

Convert json to npz for accelerating data loading
    - CONVERT_JSON2NPZ: True

#### Pathomic feature extraction
```bash
pathomics_config="/home/sg2162/rds/hpc-work/PanCIA/configs/feature_extraction/pathomics_extraction.yaml"
python analysis/a03_feature_extraction/m_pathomics_extraction.py --config_files $pathomics_config
```

- Whole slide features
    - MODE: choose one from ["UNI", "CONCH", "CHIEF"], all extract features patch-by-patch
- Graph construction
    - VALUE: True
    - SAVE_CLUSTER_POINTS: False # only true for visualization purpose, default false to reduce json file size

### Step 4: outcome prediction
#### Multimodal multi-scale multi-task learning

This aims to aggreagte multiple graphs of each patient by multi-task learning and obtain universal patient-level embedding.

```bash
multitask_config="/home/sg2162/rds/hpc-work/PanCIA/configs/outcome_prediction/multitask_learning.yaml"
python analysis/a05_outcome_prediction/m_multitask_learning.py --config_files $multitask_config
```

The setting of RADIOMICS SUFFIX directly dertermines what kinds of radiomics will be used.

For BiomedParse, "_slice_graph.npz" only includes slice radiomics, "_tumor_graph.npz" only includes tumor radiomics, while both would include slice and tumor radiomics

For LVMMed, the first setting would only include multi-scale slice radiomics;
SUFFIX:
    - "_slice_layer0_graph.npz"
    - "_slice_layer1_graph.npz"
    - "_slice_layer2_graph.npz"
Raiomics feature dimensions should be set as "LVMMed": {'child0': 64, 'child1': 256, 'child2': 512} in 'analysis/a05_outcome_prediction/m_prepare_omics_info.py'

The seond setting would only include multi-scale tumor radiomics.
SUFFIX:
    - "_tumor_layer0_graph.npz"
    - "_tumor_layer1_graph.npz"
    - "_tumor_layer2_graph.npz"
Raiomics feature dimensions should be set as "LVMMed": {'child0': 64, 'child1': 256, 'child2': 512} in 'analysis/a05_outcome_prediction/m_prepare_omics_info.py'
    
And the third setting would include both multi-scale slice and tumor radiomics
SUFFIX:
    - "_slice_layer0_graph.npz"
    - "_slice_layer1_graph.npz"
    - "_slice_layer2_graph.npz"
    - "_tumor_layer0_graph.npz"
    - "_tumor_layer1_graph.npz"
    - "_tumor_layer2_graph.npz"
Raiomics feature dimensions should be set as "LVMMed": {'child0': 64, 'child1': 256, 'child2': 512, 'child3': 64, 'child4': 256, 'child5': 512} in 'analysis/a05_outcome_prediction/m_prepare_omics_info.py'

TRAIN
    - SAMPLING_RATE: 0.1 # propertly set this if graph is very large, we used 0.1 for pathological graph, 0.01 for BiomedParse slice graph, 0.1 BiomedParse tumor graph and LVMMed tumor graph, 1 for LVMMed slice graph
INFERENCE
    - SAMPLING_RATE: 0.01 # this keep the same as the TRAIN SAMPLING_RATE

#### Task-specific Machine learning
#### Survival prediction

```bash
survival_config="/home/sg2162/rds/hpc-work/PanCIA/configs/outcome_prediction/survival_analysis.yaml"
python analysis/a05_outcome_prediction/m_survival_analysis.py --config_files $survival_config
```

- Overall survival (OS):
- Disease specfic survival (DSS):
- Disease free interval (DFI):
- Progression free interval (PFI):

#### Phenotype prediction 

```bash
phenotype_config="/home/sg2162/rds/hpc-work/PanCIA/configs/outcome_prediction/phenotype_prediction.yaml"
python analysis/a05_outcome_prediction/m_phenotype_prediction.py --config_files $phenotype_config
```

- Immune subtype classification
- Molecular subtype classification
- Primary disease classification

#### Signature prediction

```bash
signature_config="/home/sg2162/rds/hpc-work/PanCIA/configs/outcome_prediction/signature_prediction.yaml"
python analysis/a05_outcome_prediction/m_signature_prediction.py --config_files $signature_config
```

- Gene programes regression
- HRD score regression
- Immune signature score regression
- Stemness score (DNA) regression
- Stem score (RNA) regression
- AGE regression

---

## Acknowlegement
This is based on [BiomedParse](https://github.com/microsoft/BiomedParse) and [tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox)