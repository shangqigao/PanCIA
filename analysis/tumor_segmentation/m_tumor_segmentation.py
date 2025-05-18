import torch
import logging
import pathlib
import argparse
import numpy as np
import nibabel as nib
from PIL import Image
from skimage.morphology import disk
from scipy.ndimage import binary_dilation
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES

from inference_utils.inference import interactive_infer_image
from inference_utils.processing_utils import read_dicom
from inference_utils.processing_utils import read_nifti_inplane

def extract_radiology_segmentation(
        img_paths, 
        text_prompts,
        class_name,
        model_mode, 
        save_dir,
        is_CT=True,
        site='kidney',
        img_format='nifti'
    ):
    """extract segmentation from radiology images
    Args:
        img_paths (list): a list of image paths
        text_prompts (list): a list of text prompts
        class_name (str): target of segmentation
        model_mode (str): name of segmentation model
        save_dir (str): directory of saving masks
        is_CT (bool): if the modality is CT
        site (str): the site of scan, such as kidney
        img_format (str): only support nifti or dicom
    """
    if model_mode == "BiomedParse":
        _ = extract_BiomedParse_segmentation(
            img_paths,
            text_prompts,
            save_dir,
            class_name,
            format=img_format,
            is_CT=is_CT,
            site=site
        )
    else:
        raise ValueError(f"Invalid model mode: {model_mode}")
    return

def extract_BiomedParse_segmentation(img_paths, text_prompts, save_dir, class_name, 
                                  format='nifti', is_CT=True, site=None, device="gpu"):
    """extracting radiomic features slice by slice in a size of (1024, 1024)
        img_paths: a list of paths for single-phase images
            or a list of lists, where each list has paths of multi-phase images.
            For multi-phase images, only nifti format is allowed.
        text_prompts: a list of strings with the same length as the img_paths
    """

    # Build model config
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)

    # Load model from pretrained weights
    pretrained_pth = '../checkpoints/BiomedParse/biomedparse_v1.pt'

    if device == 'gpu':
        model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    else:
        model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cpu()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
    
    for img_path, text_prompt in zip(img_paths, text_prompts):
        # read slices from dicom or nifti
        if format == 'dicom':
            dicom_dir = pathlib.Path(img_path)
            assert pathlib.Path(img_path).is_dir()
            dicom_paths = sorted(dicom_dir.glob('*.dcm'))
            images = [read_dicom(p, is_CT, site, keep_size=True, return_spacing=True) for p in dicom_paths]
        elif format == 'nifti':
            images, slice_axis, affine = read_nifti_inplane(img_path, is_CT, site, keep_size=True, return_spacing=True)
        else:
            raise ValueError(f'Only support DICOM or NIFTI, but got {format}')

        masks = []
        for i, element in enumerate(images):
            assert len(element) == 2
            img, spacing = element

            # resize_mask=False would keep mask size to be (1024, 1024)
            pred_prob = interactive_infer_image(model, Image.fromarray(img), text_prompt, resize_mask=True, return_feature=False)
            pred_mask = (1*(pred_prob > 0.5)).astype(np.uint8)
            masks.append(pred_mask)
        final_mask = np.stack(masks, axis=0)
        final_mask = np.moveaxis(final_mask, 0, slice_axis)
        
        img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")
        save_mask_path = f"{save_dir}/{img_name}_{class_name}_mask.nii.gz"
        logging.info(f"Saving predicted segmentation to {save_mask_path}")
        nifti_img = nib.Nifti1Image(final_mask, affine)
        nib.save(nifti_img, save_mask_path)
    return

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--dataset', default="TCGA-RCC", type=str)
    parser.add_argument('--modality', default="CT", choices=["CT", "MRI"], type=str)
    parser.add_argument('--format', default="dicom", choices=["dicom", "nifti"], type=str)
    parser.add_argument('--site', default="kidney", type=str)
    parser.add_argument('--target', default="tumor", type=str)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    parser.add_argument('--model_mode', default="BiomedParse", choices=["SegVol", "BiomedParse"], type=str)
    args = parser.parse_args()

    if args.format == 'dicom':
        img_paths = pathlib.Path(args.img_dir).glob('*')
        img_paths = [p for p in img_paths if p.is_dir()]
    else:
        img_paths = pathlib.Path(args.img_dir).glob('*.nii.gz')
    text_prompts = [[f'{args.site} {args.target}']]*len(img_paths)
    save_dir = pathlib.Path(args.save_dir) / args.modality / args.dataset

    # extract radiology segmentation
    bs = 8
    nb = len(img_paths) // bs if len(img_paths) % bs == 0 else len(img_paths) // bs + 1
    for i in range(0, nb):
        logging.info(f"Processing images of batch [{i+1}/{nb}] ...")
        start = i * bs
        end = min(len(img_paths), (i + 1) * bs)
        batch_img_paths = img_paths[start:end]
        batch_txt_prompts = text_prompts[start:end]
        extract_radiology_segmentation(
            img_paths=batch_img_paths,
            text_prompts=batch_txt_prompts,
            class_name=args.target,
            model_mode=args.model_mode,
            save_dir=save_dir,
            is_CT=args.modality == 'CT',
            site=args.site,
            img_format=args.format
        )