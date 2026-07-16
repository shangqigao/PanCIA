import os
import sys

# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
relative_path = os.path.join(script_dir, '../../')
sys.path.append(relative_path)

import json
import torch
import pathlib
import argparse
import logging
import numpy as np
import nibabel as nib
from PIL import Image

logging.getLogger("modeling").setLevel(logging.ERROR)
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES, CT_SITES

from inference_utils.inference import interactive_infer_image
from inference_utils.processing_utils import read_dicom
from inference_utils.processing_utils import read_nifti_inplane

from analysis.a01_data_preprocessiong.m_prepare_dataset_info import prepare_EndoMRI_info
from peft import LoraConfig, get_peft_model

from tiatoolbox import logger

def extract_radiology_segmentation(
        dataset,
        seg_obj,
        img_paths, 
        text_prompts,
        model_mode, 
        save_dir,
        modality='CT',
        site='kidney',
        meta_list=None,
        img_format='nifti',
        beta_params=None,
        keep_largest=False,
        prompt_ensemble=False,
        save_radiomics=False,
        zoom_in=False,
        skip_exist=False
    ):
    """extract segmentation from radiology images
    Args:
        dataset (str): name of dataset
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
            dataset,
            seg_obj,
            img_paths,
            text_prompts,
            save_dir,
            format=img_format,
            modality=modality,
            site=site,
            meta_list=meta_list,
            beta_params=beta_params,
            keep_largest=keep_largest,
            prompt_ensemble=prompt_ensemble,
            save_radiomics=save_radiomics,
            zoom_in=zoom_in,
            skip_exist=skip_exist
        )
    else:
        raise ValueError(f"Invalid model mode: {model_mode}")
    return

def extract_BiomedParse_segmentation(dataset, seg_obj, img_paths, text_prompts, save_dir,
                                  format='nifti', modality='MR', site='breast', 
                                  meta_list=None, beta_params=None, keep_largest=False,
                                  prompt_ensemble=False, save_radiomics=False,
                                  zoom_in=False, device="gpu", skip_exist=False):
    """extracting radiomic features slice by slice in a size of (1024, 1024)
        dataset: name of dataset
        img_paths: a list of paths for single-phase images
            or a list of lists, where each list has paths of multi-phase images.
            For multi-phase images, only nifti format is allowed.
        text_prompts: a list of strings with the same length as the img_paths
        meta_list (list): a list of imaging metadata, 
            such as 'field_strength', 'bilateral', 'scanner_manufacturer'
        prompt_ensemble: if true, use prompt ensemble
        beta_params: the parameters of Beta distribution, 
            if provided, it would be used to compute p-values of segmented objects
            if p-value is less than alpha, i.e., 0.05, the object would be removed
    """

    # Build model config
    opt = load_opt_from_config_files([os.path.join(relative_path, "configs/radiology_segmentation/biomedparse_inference.yaml")])
    opt = init_distributed(opt)

    # Load model from pretrained weights
    if seg_obj == 'endometrioma':
        opt['LoRA'] = True
        pretrained_pth = os.path.join(relative_path, 'checkpoints/BiomedParse/Endometriosis_LoRA')
    else:
        opt['LoRA'] = False
        pretrained_pth = os.path.join(relative_path, 'checkpoints/BiomedParse/biomedparse_v1.pt')
    # pretrained_pth = os.path.join(relative_path, 'checkpoints/Bayes_BiomedParse/Bayes_PanCancer/model_state_dict.pt')

    if device == 'gpu':
        if not opt.get('LoRA', False):
            model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
        else:
            with open(f'{pretrained_pth}/adapter_config.json', 'r') as f:
                config = json.load(f)
            model = get_peft_model(BaseModel(opt, build_model(opt)), LoraConfig(**config)).cuda()
            ckpt = torch.load(os.path.join(pretrained_pth, 'module_training_states.pt'))['module']
            ckpt = {key.replace('module.',''): ckpt[key] for key in ckpt.keys() if 'criterion' not in key}
            model.load_state_dict(ckpt)
            model = model.model.eval()
    else:
        raise ValueError(f'Require gpu, but got {device}')
    
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

    if isinstance(format, str): format = [format] * len(img_paths)
    if isinstance(modality, str): modality = [modality] * len(img_paths)
    if isinstance(site, str): site = [site] * len(img_paths)

    for idx, (img_path, text_prompt) in enumerate(zip(img_paths, text_prompts)):
        logger.info("Segmenting image: {}/{}...".format(idx + 1, len(img_paths)))

        img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")
        save_prob_path = pathlib.Path(f"{save_dir}/{img_name}_{seg_obj}.npy")
        if save_prob_path.exists() and skip_exist:
            logger.info(f"{save_prob_path.name} has existed, skip!")
            continue

        # read slices from dicom or nifti
        is_CT = modality[idx] == 'CT'
        ct_site = CT_SITES[site[idx]]
        if format[idx] == 'dicom':
            dicom_dir = pathlib.Path(img_path)
            assert pathlib.Path(img_path).is_dir()
            dicom_paths = sorted(dicom_dir.glob('*.dcm'))
            images = [read_dicom(p, is_CT, ct_site, keep_size=True, return_spacing=True) for p in dicom_paths]
            slice_axis, affine = 0, np.eye(4)
        elif format[idx] == 'nifti':
            images, slice_axis, affine = read_nifti_inplane(img_path, is_CT, ct_site, keep_size=True, return_spacing=True)
        else:
            raise ValueError(f'Only support DICOM or NIFTI, but got {format[idx]}')
        
        prob_4d = []
        feat_4d = []
        meta_data = {} if meta_list is None else meta_list[idx]
        for i, element in enumerate(images):
            assert len(element) == 3
            img, spacing, phase = element

            # use prompt ensemble
            if isinstance(text_prompt, list):
                prompts = text_prompt
            else:
                prompts = [text_prompt]

            # resize_mask=False would keep mask size to be (1024, 1024)
            ensemble_prob = []
            ensemble_feat = []
            for prompt in prompts:
                if save_radiomics:
                    pred_prob, feature = interactive_infer_image(model, Image.fromarray(img), prompt, resize_mask=True, return_feature=True)
                    ensemble_feat.append(np.transpose(feature, (1, 2, 0)))
                else:
                    pred_prob = interactive_infer_image(model, Image.fromarray(img), prompt, resize_mask=True, return_feature=False)
                ensemble_prob.append(pred_prob)
            pred_prob = np.concatenate(ensemble_prob, axis=0)
            prob_4d.append(np.transpose(pred_prob, (1, 2, 0)))

            if save_radiomics: 
                slice_feat = np.mean(np.stack(ensemble_feat, axis=0), axis=0, keepdims=True)
                feat_4d.append(slice_feat)

        final_prob = np.concatenate(prob_4d, axis=0)
        final_prob = np.moveaxis(final_prob, 0, slice_axis)
        if save_radiomics: feat_4d = np.concatenate(feat_4d, axis=0)
        
        logger.info(f"Saving predicted prob to {save_prob_path}")
        os.makedirs(os.path.dirname(save_prob_path), exist_ok=True)
        np.save(final_prob, save_prob_path)
        if save_radiomics:
            radiomic_feat = np.moveaxis(feat_4d, 0, slice_axis)
            save_feat_path = f"{save_dir}/{img_name}_radiomics.npy"
            logger.info(f"Saving radiomic features to {save_feat_path}")
            np.save(save_feat_path, radiomic_feat)       

    return

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--radiology', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--dataset', default="EndoMRI_All", type=str)
    parser.add_argument('--seg_obj', default="endometrioma", choices=["tumor", "endometrioma"], type=str)
    parser.add_argument("--keep_largest", action="store_true")
    parser.add_argument('--phase', default="single", choices=["single", "multiple"], type=str)
    parser.add_argument('--format', default="nifti", choices=["dicom", "nifti"], type=str)
    parser.add_argument('--meta_info', default=None)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    parser.add_argument('--model', default="BiomedParse", choices=["SegVol", "BiomedParse"], type=str)
    args = parser.parse_args()

    save_dir = pathlib.Path(args.save_dir) / args.model

    if args.dataset == 'EndoMRI_All':
        dataset_info = prepare_EndoMRI_info(
            img_dir=args.radiology,
            img_format=args.format
        )
    else:
        raise ValueError(f'Dataset {args.dataset} is currently unsupported')

    # extract radiology segmentation
    # warning: do not run this function in a loop
    logger.info(f"starting segmentation on {dataset_info['name']}...")
    extract_radiology_segmentation(
        dataset=args.dataset,
        seg_obj=args.seg_obj,
        img_paths=dataset_info['img_paths'],
        text_prompts=dataset_info['text_prompts'],
        model_mode=args.model,
        save_dir=save_dir,
        modality=dataset_info['modality'],
        site=dataset_info['site'],
        meta_list=dataset_info['meta_list'],
        img_format=dataset_info['img_format'],
        beta_params=None,
        keep_largest=args.keep_largest,
        prompt_ensemble=False,
        save_radiomics=False,
        zoom_in=False,
        skip_exist=True
    )