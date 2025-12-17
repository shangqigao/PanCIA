# **Pan Cancer Image Analysis**

---

## Dependencies

- conda create -n PanCIA python=3.9.19
- conda activate PanCIA
- conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
- conda install -c conda-forge openslide
- conda install -c conda-forge gcc_linux-64 gxx_linux-64 (optional)
- pip install --no-build-isolation git+https://github.com/MaureenZOU/detectron2-xyz.git
- pip install -r requirements.txt

If GCC is too old (need GCC 9 or later):
- conda install -c conda-forge gcc_linux-64 gxx_linux-64

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

### Step 2: tumor segmentation

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

#### Pathomic feature extraction
```bash
pathomics_config="/home/sg2162/rds/hpc-work/PanCIA/configs/feature_extraction/pathomics_extraction.yaml"
python analysis/a03_feature_extraction/m_pathomics_extraction.py --config_files $pathomics_config
```

### Step 4: outcome prediction
#### Multi-omics multi-task learning

---

## Acknowlegement
This is based on [BiomedParse](https://github.com/microsoft/BiomedParse) and [tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox)