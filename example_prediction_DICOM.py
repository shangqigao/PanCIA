from PIL import Image
import torch
import json
import pathlib
import matplotlib.pyplot as plt
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES

from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import check_mask_stats
from inference_utils.processing_utils import read_dicom

import huggingface_hub

# Build model config
opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)

# Load model from pretrained weights
pretrained_pth = 'hf_hub:microsoft/BiomedParse'

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

def plot_segmentation_masks(original_image, segmentation_masks, texts):
    ''' Plot a list of segmentation mask over an image.
    '''
    original_image = original_image[:, :, :3]
    fig, ax = plt.subplots(1, len(segmentation_masks) + 1, figsize=(10, 5))
    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title('Original Image')
    # grid off
    for a in ax:
        a.axis('off')

    for i, mask in enumerate(segmentation_masks):
        
        ax[i+1].set_title(texts[i])
        mask_temp = original_image.copy()
        mask_temp[mask > 0.5] = [255, 0, 0]
        mask_temp[mask <= 0.5] = [0, 0, 0, ]
        ax[i+1].imshow(mask_temp, alpha=0.9)
        ax[i+1].imshow(original_image, cmap='gray', alpha=0.5)
        
    
    plt.show()

def inference_dicom(file_path, text_prompts, is_CT, site=None):
    image = read_dicom(file_path, is_CT, site=site)
    
    pred_mask = interactive_infer_image(model, Image.fromarray(image), text_prompts)

    # Plot feature over image
    plot_segmentation_masks(image, pred_mask, text_prompts)
    
    return image, pred_mask

image_path = 'examples/CT_lung_nodule.dcm'
text_prompt = ['nodule']

image, pred_mask = inference_dicom(image_path, text_prompt, is_CT=True, site='lung')

# P-value (adjusted) that the segmentation belongs to "nodule" in the CT-Chest class
# Lower p-value indicates it is likely the segmentation not belongs to the class. Recommended threshold is 0.05
adj_pvalue = check_mask_stats(image, pred_mask[0]*255, 'CT-Chest', 'nodule')
print(f'P-value: {adj_pvalue}')

data_dir = "/home/sg2162/rds/hpc-work/TCIA"
text_prompt = ['tumor']
project_site = {
    'TCGA-BLCA': 'pelvis',
    'TCGA-BRCA': None,
    'TCGA-CESC': 'pelvis',
    'TCGA-COAD': 'colon',
    'TCGA-KICH': 'abdomen',
    'TCGA-KIRC': 'abdomen',
    'TCGA-KIRP': 'abdomen',
    'TCGA-LIHC': 'liver',
    'TCGA-LUAD': 'lung',
    'TCGA-LUSC': 'lung',
    'TCGA-OV': 'pelvis',
    'TCGA-STAD': 'abdomen',
    'TCGA-UCEC': 'pelvis'
}

for modality in ["CT", "MR"]:
    is_CT = modality == "CT"
    project_dir = pathlib.Path(data_dir) / modality
    projects = project_dir.glob('TCGA-*')
    projects = [p.name for p in projects]
    included_projects = list(project_site.keys())
    projects = list(projects.intersection(included_projects))
    print(f"Totally {len(projects)} projects included for {modality}")
    for project in projects:
        site = project_site[project]
        series_dir = project_dir / project
        serieses = series_dir.glob('1.3.6*')
        serieses = [s.name for s in serieses]
        print(f"Totally {len(serieses)} serieses found for {project}")
        for series in serieses:
            image_dir = series_dir / series
            images = image_dir.glob('*.dcm')
            print(f"Totally {len(images)} images found for {series}")
            for image in images:
                image, pred_mask = inference_dicom(image_path, text_prompt, is_CT=is_CT, site=site) 
                


