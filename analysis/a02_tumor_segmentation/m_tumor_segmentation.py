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

from analysis.a01_data_preprocessiong.m_prepare_dataset_info import prepare_MAMAMIA_info
from analysis.a01_data_preprocessiong.m_prepare_dataset_info import prepare_TCGA_radiology_info
from analysis.a02_tumor_segmentation.m_post_processing import remove_inconsistent_objects
from peft import LoraConfig, get_peft_model

from tiatoolbox import logger

def extract_radiology_segmentation(
        dataset,
        img_paths, 
        text_prompts,
        model_mode, 
        save_dir,
        modality='CT',
        site='kidney',
        meta_list=None,
        img_format='nifti',
        beta_params=None,
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
            img_paths,
            text_prompts,
            save_dir,
            format=img_format,
            modality=modality,
            site=site,
            meta_list=meta_list,
            beta_params=beta_params,
            prompt_ensemble=prompt_ensemble,
            save_radiomics=save_radiomics,
            zoom_in=zoom_in,
            skip_exist=skip_exist
        )
    else:
        raise ValueError(f"Invalid model mode: {model_mode}")
    return

def extract_BiomedParse_segmentation(dataset, img_paths, text_prompts, save_dir,
                                  format='nifti', modality='MR', site='breast', 
                                  meta_list=None, beta_params=None, 
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
    pretrained_pth = os.path.join(relative_path, 'checkpoints/BiomedParse/PanCancer_LoRA')
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

        if isinstance(img_path, list):
            img_name = pathlib.Path(img_path[0]).name.replace("_0000.nii.gz", "")
        else:
            if '/MAMA-MIA/' in str(img_path):
                img_name = pathlib.Path(img_path).name.replace("_0001.nii.gz", "")
            elif f'/{dataset}_NIFTI/' in str(img_path):
                img_name = str(img_path).split(f'/{dataset}_NIFTI/')[-1].replace(".nii.gz", "")
            else:
                img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")
        save_mask_path = pathlib.Path(f"{save_dir}/{img_name}.nii.gz")
        if save_mask_path.exists() and skip_exist:
            logger.info(f"{save_mask_path.name} has existed, skip!")
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

        mask_3d = []
        image_4d = []
        prob_3d = []
        feat_4d = []
        meta_data = {} if meta_list is None else meta_list[idx]
        for i, element in enumerate(images):
            assert len(element) == 3
            img, spacing, phase = element

            # use prompt ensemble
            if prompt_ensemble:
                assert isinstance(meta_data, dict)
                meta_data['view'] = phase
                meta_data['slice_index'] = f'{i:03}'
                meta_data['modality'] = modality[idx]
                meta_data['site'] = site[idx]
                meta_data['target'] = text_prompt
                if len(spacing) == 2:
                    meta_data['pixel_spacing'] = spacing
                else:
                    assert len(spacing) == 3
                    pixel_index = list(set([0, 1, 2]) - {slice_axis})
                    pixel_spacing = [spacing[i] for i in pixel_index]
                    meta_data['pixel_spacing'] = pixel_spacing
                prompts = create_prompts(meta_data)
                # prompts = [prompts[2], prompts[9]]
                prompts = [prompts[2]]
            else:
                prompts = [text_prompt]
            # print(f"Segmenting slice [{i+1}/{len(images)}] ...")

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
            pred_prob = np.max(np.concatenate(ensemble_prob, axis=0), axis=0, keepdims=True)
            if beta_params is not None:
                image_4d.append(img)
                prob_3d.append(pred_prob)
            pred_mask = (1*(pred_prob > 0.5)).astype(np.uint8)

            if zoom_in:
                # Find coordinates where mask == 1
                ys, xs = np.where(np.squeeze(pred_mask) == 1)
                min_size = 256
                if len(xs) > 0 and len(ys) > 0:
                    x_min, x_max = np.min(xs), np.max(xs)
                    y_min, y_max = np.min(ys), np.max(ys)
                    H, W = img.shape[:2]

                    # Current width and height
                    box_w = x_max - x_min + 1
                    box_h = y_max - y_min + 1

                    # How much to expand
                    pad_w = max(0, min_size - box_w)
                    pad_h = max(0, min_size - box_h)

                    # Pad equally on both sides
                    x_min = max(0, x_min - pad_w // 2)
                    x_max = min(W - 1, x_max + (pad_w - pad_w // 2))
                    y_min = max(0, y_min - pad_h // 2)
                    y_max = min(H - 1, y_max + (pad_h - pad_h // 2))
                    zoom_img = img[y_min:y_max+1, x_min:x_max+1]

                    ensemble_prob = []
                    for prompt in prompts:
                        zoom_pred_prob = interactive_infer_image(model, Image.fromarray(zoom_img), prompt, resize_mask=True, return_feature=False)
                        ensemble_prob.append(zoom_pred_prob)
                    zoom_pred_prob = np.max(np.concatenate(ensemble_prob, axis=0), axis=0, keepdims=True)
                    zoom_pred_prob = np.maximum(pred_prob[:, y_min:y_max+1, x_min:x_max+1], zoom_pred_prob)
                    zoom_pred_mask = (1*(zoom_pred_prob > 0.5)).astype(np.uint8)
                    pred_mask[:, y_min:y_max+1, x_min:x_max+1] = zoom_pred_mask
            mask_3d.append(pred_mask)

            if save_radiomics: 
                slice_feat = np.mean(np.stack(ensemble_feat, axis=0), axis=0, keepdims=True)
                feat_4d.append(slice_feat)
        
        # post-processing predicted masks
        mask_3d = np.concatenate(mask_3d, axis=0)

        # crop by breast mask if available
        if meta_data.get("breast_coordinates", False):
            coords = meta_data["breast_coordinates"]
            x_min, x_max = coords["x_min"], coords["x_max"]
            y_min, y_max = coords["y_min"], coords["y_max"]
            z_min, z_max = coords["z_min"], coords["z_max"]

            mask_nib = np.moveaxis(mask_3d, 0, slice_axis) # to nib array
            mask_sitk = np.transpose(mask_nib, (2, 1, 0)) # to sitk array
            mask_new = np.zeros_like(mask_sitk)
            mask_new[x_min:x_max, y_min:y_max, z_min:z_max] = mask_sitk[x_min:x_max, y_min:y_max, z_min:z_max] 
            mask_nib = np.transpose(mask_new, (2, 1, 0)) # to nib array
            mask_3d = np.moveaxis(mask_nib, slice_axis, 0)

        if save_radiomics: feat_4d = np.concatenate(feat_4d, axis=0)
        if beta_params is not None:
            prob_3d = np.concatenate(prob_3d, axis=0)
            image_4d = np.stack(image_4d, axis=0)
            logger.info("Post-processing by removing both unconfident predictions and spatially inconsistent objects")
            mask_3d = remove_inconsistent_objects(mask_3d, prob_3d=prob_3d, image_4d=image_4d, beta_params=beta_params)
        else:
            logger.info("Post-processing by removing spatially inconsistent objects")
            if format[idx] == 'dicom':
                voxel_spacing = None
            else:
                voxel_spacing = spacing.tolist()
                z_spacing = voxel_spacing.pop(slice_axis)
                voxel_spacing.insert(0, z_spacing)
            mask_3d = remove_inconsistent_objects(mask_3d, spacing=voxel_spacing)
        final_mask = np.moveaxis(mask_3d, 0, slice_axis)
        logger.info(f"Saving predicted segmentation to {save_mask_path}")
        nifti_img = nib.Nifti1Image(final_mask, affine)
        os.makedirs(os.path.dirname(save_mask_path), exist_ok=True)
        nib.save(nifti_img, save_mask_path)
        if save_radiomics:
            radiomic_feat = np.moveaxis(feat_4d, 0, slice_axis)
            ndim = np.squeeze(radiomic_feat).ndim
            if ndim == 3:
                radiomic_feat = np.squeeze(radiomic_feat) * final_mask
                save_feat_path = f"{save_dir}/{img_name}_radiomics.nii.gz"
                nifti_img = nib.Nifti1Image(radiomic_feat, affine)
                logger.info(f"Saving radiomic features to {save_feat_path}")
                nib.save(nifti_img, save_feat_path)
            else:
                radiomic_feat = radiomic_feat[final_mask > 0]
                save_feat_path = f"{save_dir}/{img_name}_radiomics.npy"
                logger.info(f"Saving radiomic features to {save_feat_path}")
                np.save(save_feat_path, radiomic_feat)       

    return

def load_beta_params(modality, site, target):
    beta_path = os.path.join(relative_path, 'analysis/tumor_segmentation/Beta_params.json')
    with open(beta_path, 'r') as f:
        data = json.load(f)
        beta_params = data[f"{modality}-{site}"][target]

    return beta_params

def create_prompts(meta_data):
    keys = ['view', 'slice_index', 'modality', 'site', 'target']
    assert all(meta_data.get(k) is not None for k in keys), f"all basic info {keys} should be provided"
    view = meta_data['view']
    slice_index = meta_data['slice_index']
    modality = meta_data['modality']
    site = meta_data['site']
    target_name = meta_data['target']
    # target = 'tumor' if 'tumor' in target_name else target_name
    # target = "tumor located within fibroglandular tissue of the breast"
    target = "tumor located within the breast, adjacent to the chest wall"

    # basic_prompts = [
    #     f"{target_name} in {site} {modality}",
    #     f"{view} slice {slice_index} showing {target} in {site}",
    #     f"{target} located in the {site} on {modality}",
    #     f"{view} {site} {modality} with {target}",
    #     f"{target} visible in slice {slice_index} of {modality}",
    # ]
    basic_prompts = [
        f"{target_name} in {site} {modality}",
        f"{view} slice {slice_index} showing {target}",
        f"{target} on {modality}",
        f"{view} {modality} with {target}",
        f"{target} visible in slice {slice_index} of {modality}",
    ]

    # meta information
    keys = ['pixel_spacing', 'field_strength', 'bilateral', 'scanner_manufacturer']
    meta_prompts = []
    if all(meta_data.get(k) is not None for k in keys):
        pixel_spacing = meta_data['pixel_spacing']
        x_spacing, y_spacing = pixel_spacing[0], pixel_spacing[1]
        field_strength = meta_data['field_strength']
        bilateral_mri = meta_data['bilateral']
        lateral = 'bilateral' if bilateral_mri == 1 else 'unilateral'
        manufacturer = meta_data['scanner_manufacturer']
        meta_prompts = [
            f"a {modality} scan of the {lateral} {site}, {view} view, slice {slice_index}, pixel spacing {x_spacing:.2f}x{y_spacing:.2f} mm, showing {target}",
            f"{lateral} {site} {modality} in {view} view at slice {slice_index} with spacing {x_spacing:.2f}x{y_spacing:.2f} mm, includes {target}",
            f"{view} slice {slice_index} from a {field_strength}T {manufacturer} {modality} of the {lateral} {site}, pixel spacing {x_spacing:.2f}x{y_spacing:.2f} mm, showing {target}",
            f"{lateral} {site} {modality} in {view} view, slice {slice_index}, using {field_strength}T {manufacturer} scanner, spacing {x_spacing:.2f}x{y_spacing:.2f} mm, showing {target}",
            f"{modality} of the {lateral} {site} at slice {slice_index}, {view} view, spacing: {x_spacing:.2f}x{y_spacing:.2f} mm, scanned by {field_strength}T {manufacturer} scanner, shows {target}"
        ]
    
    return basic_prompts + meta_prompts

if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--radiology', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--dataset', default="MAMAMIA", type=str)
    parser.add_argument('--phase', default="single", choices=["single", "multiple"], type=str)
    parser.add_argument('--format', default="nifti", choices=["dicom", "nifti"], type=str)
    parser.add_argument('--meta_info', default=None)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    parser.add_argument('--model', default="BiomedParse", choices=["SegVol", "BiomedParse"], type=str)
    args = parser.parse_args()

    save_dir = pathlib.Path(args.save_dir) / args.model

    if args.dataset == 'MAMAMIA':
        dataset_info = prepare_MAMAMIA_info(
            img_dir=args.radiology,
            img_format=args.format,
            phase=args.phase,
            meta_info=args.meta_info
        )
    elif args.dataset == 'TCGA':
        dataset_info = prepare_TCGA_radiology_info(
            img_json=args.radiology,
            img_format=args.format
        )
    elif args.dataset == 'CPTAC':
        dataset_info = prepare_TCGA_radiology_info(
            img_json=args.radiology,
            img_format=args.format
        )
    else:
        raise ValueError(f'Dataset {args.dataset} is currently unsupported')

    # extract radiology segmentation
    # warning: do not run this function in a loop
    logger.info(f"starting segmentation on {dataset_info['name']}...")
    extract_radiology_segmentation(
        dataset=args.dataset,
        img_paths=dataset_info['img_paths'],
        text_prompts=dataset_info['text_prompts'],
        model_mode=args.model,
        save_dir=save_dir,
        modality=dataset_info['modality'],
        site=dataset_info['site'],
        meta_list=dataset_info['meta_list'],
        img_format=dataset_info['img_format'],
        beta_params=None,
        prompt_ensemble=False,
        save_radiomics=False,
        zoom_in=False,
        skip_exist=True
    )