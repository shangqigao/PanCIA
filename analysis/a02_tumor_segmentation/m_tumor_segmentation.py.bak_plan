import os
import sys

# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
relative_path = os.path.join(script_dir, '../../')
sys.path.append(relative_path)

import pathlib
import argparse
import logging
import numpy as np

from analysis.a01_data_preprocessiong.m_prepare_dataset_info import prepare_MAMAMIA_info
from analysis.a01_data_preprocessiong.m_prepare_dataset_info import prepare_TCGA_radiology_info
from analysis.a01_data_preprocessiong.m_prepare_dataset_info import prepare_CPTAC_radiology_info
logger = logging.getLogger(__name__)

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
        voxtell_model_root=None,
        device=None,
        skip_exist=False,
        ts_model_root=None,
        ts_task=None,
        ts_fast=False,
        ts_extra_tasks=None,
        ts_roi_subset=None,
        ts_threads=4,
        ts_body_seg=False,
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
            device=device or "gpu",
            skip_exist=skip_exist
        )
    elif model_mode == "VoxTell":
        extract_VoxTell_segmentation(
            dataset=dataset,
            seg_obj=seg_obj,
            img_paths=img_paths,
            save_dir=save_dir,
            format=img_format,
            site=site,
            model_root=voxtell_model_root,
            keep_largest=keep_largest,
            prompt_ensemble=prompt_ensemble,
            device=device,
            skip_exist=skip_exist,
        )
    elif model_mode == "TotalSegmentator":
        extract_TotalSegmentator_segmentation(
            dataset=dataset,
            seg_obj=seg_obj,
            img_paths=img_paths,
            save_dir=save_dir,
            format=img_format,
            modality=modality,
            site=site,
            keep_largest=keep_largest,
            prompt_ensemble=prompt_ensemble,
            device=device,
            skip_exist=skip_exist,
            model_root=ts_model_root,
            task=ts_task,
            fast=ts_fast,
            extra_tasks=ts_extra_tasks,
            roi_subset=ts_roi_subset,
            nr_threads_resample=ts_threads,
            nr_threads_save=ts_threads,
            body_seg=ts_body_seg,
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

    # BiomedParse is installed in a separate environment. Keep its imports local
    # so selecting VoxTell does not require any BiomedParse dependencies.
    import json
    import nibabel as nib
    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model

    logging.getLogger("modeling").setLevel(logging.ERROR)
    from modeling.BaseModel import BaseModel
    from modeling import build_model
    from utilities.distributed import init_distributed
    from utilities.arguments import load_opt_from_config_files
    from utilities.constants import BIOMED_CLASSES, CT_SITES
    from inference_utils.inference import interactive_infer_image
    from inference_utils.processing_utils import read_dicom, read_nifti_inplane
    from analysis.a02_tumor_segmentation.m_post_processing import remove_inconsistent_objects

    # Build model config
    opt = load_opt_from_config_files([os.path.join(relative_path, "configs/radiology_segmentation/biomedparse_inference.yaml")])
    opt = init_distributed(opt)

    # Load model from pretrained weights
    if seg_obj == 'tumor':
        opt['LoRA'] = True
        pretrained_pth = os.path.join(relative_path, 'checkpoints/BiomedParse/PanCancer_LoRA')
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

        if isinstance(img_path, list):
            img_name = pathlib.Path(img_path[0]).name.replace("_0000.nii.gz", "")
        else:
            if '/MAMA-MIA/' in str(img_path):
                img_name = pathlib.Path(img_path).name.replace("_0001.nii.gz", "")
            elif f'/{dataset}_NIFTI/' in str(img_path):
                img_name = str(img_path).split(f'/{dataset}_NIFTI/')[-1].replace(".nii.gz", "")
            else:
                img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")
        save_mask_path = pathlib.Path(f"{save_dir}/{img_name}_{seg_obj}.nii.gz")
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
                ys, xs = np.where(np.squeeze(pred_mask) == 1)
                min_size = 256
                if len(xs) > 0 and len(ys) > 0:
                    x_min, x_max = np.min(xs), np.max(xs)
                    y_min, y_max = np.min(ys), np.max(ys)
                    H, W = img.shape[:2]
                    box_w = x_max - x_min + 1
                    box_h = y_max - y_min + 1
                    pad_w = max(0, min_size - box_w)
                    pad_h = max(0, min_size - box_h)
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
        
        logger.info("Using BiomedParse prompt: %s", prompts[0])

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
        # keep_largest = seg_obj == 'tumor'
        if beta_params is not None:
            prob_3d = np.concatenate(prob_3d, axis=0)
            image_4d = np.stack(image_4d, axis=0)
            logger.info("Post-processing by removing both unconfident predictions and spatially inconsistent objects")
            mask_3d = remove_inconsistent_objects(mask_3d, prob_3d=prob_3d, image_4d=image_4d, beta_params=beta_params, keep_largest=False)
        else:
            logger.info("Post-processing by removing spatially inconsistent objects")
            if format[idx] == 'dicom':
                voxel_spacing = None
            else:
                voxel_spacing = spacing.tolist()
                z_spacing = voxel_spacing.pop(slice_axis)
                voxel_spacing.insert(0, z_spacing)
            mask_3d = remove_inconsistent_objects(mask_3d, spacing=voxel_spacing, keep_largest=False)
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
    import json

    beta_path = os.path.join(relative_path, 'analysis/tumor_segmentation/Beta_params.json')
    with open(beta_path, 'r') as f:
        data = json.load(f)
        beta_params = data[f"{modality}-{site}"][target]

    return beta_params


def extract_VoxTell_segmentation(
        dataset,
        seg_obj,
        img_paths,
        save_dir,
        format='nifti',
        site='kidney',
        model_root=None,
        keep_largest=False,
        prompt_ensemble=False,
        device=None,
        skip_exist=False,
    ):
    """Segment NIfTI volumes with VoxTell v1.1 using free-text prompts.

    ``model_root`` is the directory that contains the ``voxtell_v1.1`` model
    directory. The checkpoint is downloaded there only when the required model
    files are missing.
    """

    # VoxTell is installed in a separate environment. Keep every backend-specific
    # import local so BiomedParse can run without VoxTell (and vice versa).
    import torch
    from huggingface_hub import snapshot_download
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    from voxtell.inference.predictor import VoxTellPredictor
    from analysis.a02_tumor_segmentation.m_post_processing import keep_largest_components

    if prompt_ensemble:
        raise ValueError("prompt_ensemble is currently supported only by BiomedParse")

    if model_root is None:
        model_root = os.path.join(relative_path, "checkpoints", "VoxTell")
    model_root = pathlib.Path(model_root).expanduser().resolve()
    model_dir = model_root / "voxtell_v1.1"
    required_files = (
        model_dir / "plans.json",
        model_dir / "fold_0" / "checkpoint_final.pth",
    )

    if all(path.is_file() for path in required_files):
        logger.info("Using existing VoxTell v1.1 model at %s", model_dir)
    else:
        logger.info("Downloading VoxTell v1.1 model to %s", model_root)
        model_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="mrokuss/VoxTell",
            allow_patterns="voxtell_v1.1/*",
            local_dir=str(model_root),
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise RuntimeError(
                "VoxTell v1.1 download completed without required files: "
                + ", ".join(missing)
            )

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    predictor = VoxTellPredictor(model_dir=str(model_dir), device=torch_device)
    reader_writer = NibabelIOWithReorient()

    if isinstance(format, str):
        format = [format] * len(img_paths)
    if isinstance(site, str):
        site = [site] * len(img_paths)

    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(img_paths):
        logger.info("Segmenting image: %s/%s...", idx + 1, len(img_paths))
        if format[idx] != "nifti":
            raise ValueError("VoxTell currently supports only NIfTI input")
        if isinstance(img_path, list):
            if len(img_path) != 1:
                raise ValueError(
                    "VoxTell currently supports one NIfTI volume per case, not multi-phase input"
                )
            img_path = img_path[0]

        if '/MAMA-MIA/' in str(img_path):
            img_name = pathlib.Path(img_path).name.replace("_0001.nii.gz", "")
        elif f'/{dataset}_NIFTI/' in str(img_path):
            img_name = str(img_path).split(f'/{dataset}_NIFTI/')[-1].replace(".nii.gz", "")
        else:
            img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")

        save_mask_path = save_dir / f"{img_name}_{seg_obj}.nii.gz"
        if save_mask_path.exists() and skip_exist:
            logger.info("%s has existed, skip!", save_mask_path.name)
            continue

        if seg_obj == "tumor":
            text_prompt = f"{site[idx]} {seg_obj}"
        else:
            text_prompt = f"{site[idx]}"
        logger.info("Using VoxTell prompt: %s", text_prompt)
        image, properties = reader_writer.read_images([str(img_path)])
        prediction = predictor.predict_single_image(image, [text_prompt])
        mask = np.asarray(prediction[0], dtype=np.uint8)
        if keep_largest:
            mask = keep_largest_components(mask)

        logger.info("Saving predicted segmentation to %s", save_mask_path)
        os.makedirs(os.path.dirname(save_mask_path), exist_ok=True)
        reader_writer.write_seg(mask, str(save_mask_path), properties)

    return

def extract_TotalSegmentator_segmentation(
        dataset,
        seg_obj,
        img_paths,
        save_dir,
        format='nifti',
        modality='CT',
        site='kidney',
        keep_largest=False,
        prompt_ensemble=False,
        device=None,
        skip_exist=False,
        model_root=None,
        task=None,
        fast=False,
        extra_tasks=None,
        roi_subset=None,
        nr_threads_resample=4,
        nr_threads_save=4,
        body_seg=False,
    ):
    """Segment anatomy in NIfTI volumes with TotalSegmentator (multi-label, no prompts).

    Mirrors ``extract_VoxTell_segmentation``. TotalSegmentator is installed in its own
    environment, so every backend import stays inside this function.

    Per image the function writes, next to each other in ``save_dir``:
      ``{img_name}_{seg_obj}.nii.gz``          multi-label mask in the input image grid (task ``total``/``total_mr``)
      ``{img_name}_{seg_obj}_labels.json``     label id -> class name for that task
      ``{img_name}_{seg_obj}_structures.csv``  one row per found class: volume_ml, centroid (RAS, mm),
                                               zmin/zmax (mm), touches_boundary — the input the KB planner needs
      ``{img_name}_{seg_obj}_{extra_task}.nii.gz`` for each ``extra_tasks`` entry (e.g. ``body``, ``tissue_types``)

    Args:
        model_root: directory holding the TotalSegmentator weights (``<model_root>/nnunet/results/...``).
            Defaults to ``checkpoints/TotalSegmentator`` next to the repo, mirroring VoxTell. Weights missing
            for the requested task are downloaded there on first use (needs internet; pre-download on a login
            node with ``TOTALSEG_HOME_DIR=<model_root> totalseg_download_weights -t total``).
        task: TotalSegmentator task. ``None`` picks ``total`` for CT and ``total_mr`` for MR per image.
        fast: use the 3 mm low-resolution model (``--fast``); much quicker, coarser masks.
        extra_tasks: optional list of additional TS tasks run on the same image (``body``, ``lung_vessels``,
            ``tissue_types`` [licensed], ``abdominal_muscles`` [licensed], ...).
        device: ``gpu``, ``gpu:N``, ``cuda:N``, ``cpu`` or ``mps`` (default: auto).
        roi_subset: optional list of TS class names; TS then runs only the sub-models covering them
            (big speed-up when a handful of anchors is enough).
        nr_threads_resample / nr_threads_save: CPU threads for TS resampling and saving (TS defaults 1 / 6).
        body_seg: crop to the body region first (faster on scans with a lot of air / table).

    Speed notes: ``total`` = 5 nnU-Net models at 1.5 mm plus CPU resampling, so ~1 min/scan on a GPU is
    normal; ``fast=True`` (3 mm, one model) is ~5-10x quicker; ``roi_subset`` skips unneeded sub-models.
    """
    import json
    import csv
    import nibabel as nib

    # Weights location: TotalSegmentator reads TOTALSEG_HOME_DIR (default ~/.totalsegmentator).
    # Set it before importing the package so every download / lookup uses the repo checkpoint folder.
    if model_root is None:
        model_root = os.path.join(relative_path, "checkpoints", "TotalSegmentator")
    model_root = pathlib.Path(model_root).expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    os.environ["TOTALSEG_HOME_DIR"] = str(model_root)
    logger.info("TotalSegmentator weights directory: %s", model_root)

    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map

    if prompt_ensemble:
        raise ValueError("prompt_ensemble is not applicable to TotalSegmentator (no text prompts)")
    if seg_obj != "organ":
        logger.warning("TotalSegmentator segments anatomy; seg_obj=%r is unusual (expected 'organ')", seg_obj)
    if keep_largest:
        logger.warning("keep_largest is ignored for TotalSegmentator multi-label output")

    # TotalSegmentator device vocabulary: gpu | gpu:N | cpu | mps
    if device is None:
        try:
            import torch
            ts_device = "gpu" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        except Exception:
            ts_device = "cpu"
    else:
        ts_device = str(device).replace("cuda", "gpu")
    logger.info("TotalSegmentator device: %s", ts_device)

    if isinstance(format, str):
        format = [format] * len(img_paths)
    if isinstance(modality, str):
        modality = [modality] * len(img_paths)
    if isinstance(site, str):
        site = [site] * len(img_paths)
    extra_tasks = list(extra_tasks or [])

    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def _structure_table(mask_img, label_names):
        """Per-label volume / centroid / z-extent / boundary contact in patient (RAS) coordinates.
        Single pass over the volume: bincount (volumes), find_objects (bboxes), center_of_mass (centroids)."""
        from scipy import ndimage
        data = np.asanyarray(mask_img.dataobj)
        if data.dtype.kind not in "iu":
            data = np.rint(data).astype(np.int32)
        affine = mask_img.affine
        vox_ml = float(abs(np.linalg.det(affine[:3, :3]))) / 1000.0
        counts = np.bincount(data.ravel())
        labels = np.nonzero(counts)[0]
        labels = labels[labels > 0]
        if labels.size == 0:
            return []
        objs = ndimage.find_objects(data)                      # index i -> slices for label i+1 (None if absent)
        coms = ndimage.center_of_mass(np.ones_like(data, dtype=np.uint8), data, labels.tolist())
        shape = np.array(data.shape)
        rows = []
        for lab, com in zip(labels, coms):
            sl = objs[int(lab) - 1]
            if sl is None:
                continue
            mins = np.array([x.start for x in sl]); maxs = np.array([x.stop - 1 for x in sl])
            centroid_ras = affine[:3, :3] @ np.asarray(com, dtype=float) + affine[:3, 3]
            corners = np.array([[x, y, z] for x in (mins[0], maxs[0]) for y in (mins[1], maxs[1]) for z in (mins[2], maxs[2])], dtype=float)
            corners_ras = (affine[:3, :3] @ corners.T).T + affine[:3, 3]
            touches = bool(np.any(mins == 0) or np.any(maxs == shape - 1))
            n = int(counts[lab])
            rows.append(dict(label=int(lab), name=label_names.get(int(lab), str(int(lab))), n_voxels=n,
                             volume_ml=round(n * vox_ml, 3),
                             centroid_x_mm=round(float(centroid_ras[0]), 2), centroid_y_mm=round(float(centroid_ras[1]), 2), centroid_z_mm=round(float(centroid_ras[2]), 2),
                             zmin_mm=round(float(corners_ras[:, 2].min()), 2), zmax_mm=round(float(corners_ras[:, 2].max()), 2),
                             touches_boundary=touches))
        return rows

    for idx, img_path in enumerate(img_paths):
        logger.info("Segmenting image: %s/%s...", idx + 1, len(img_paths))
        if format[idx] != "nifti":
            raise ValueError("TotalSegmentator currently supports only NIfTI input")
        if isinstance(img_path, list):
            if len(img_path) != 1:
                raise ValueError("TotalSegmentator supports one NIfTI volume per case, not multi-phase input")
            img_path = img_path[0]

        if '/MAMA-MIA/' in str(img_path):
            img_name = pathlib.Path(img_path).name.replace("_0001.nii.gz", "")
        elif f'/{dataset}_NIFTI/' in str(img_path):
            img_name = str(img_path).split(f'/{dataset}_NIFTI/')[-1].replace(".nii.gz", "")
        else:
            img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")

        save_mask_path = save_dir / f"{img_name}_{seg_obj}.nii.gz"
        labels_path = save_dir / f"{img_name}_{seg_obj}_labels.json"
        table_path = save_dir / f"{img_name}_{seg_obj}_structures.csv"
        if save_mask_path.exists() and table_path.exists() and skip_exist:
            logger.info("%s has existed, skip!", save_mask_path.name)
            continue

        img_task = task or ("total_mr" if str(modality[idx]).upper().startswith("MR") else "total")
        if img_task not in class_map:
            raise ValueError(f"Unknown TotalSegmentator task {img_task!r}")
        logger.info("Using TotalSegmentator task: %s (fast=%s) for %s", img_task, fast, img_path)
        os.makedirs(os.path.dirname(save_mask_path), exist_ok=True)

        # multi-label output written directly to save_mask_path (ml=True -> single file)
        totalsegmentator(
            str(img_path), str(save_mask_path),
            ml=True, task=img_task, fast=fast, device=ts_device,
            roi_subset=roi_subset, body_seg=body_seg,
            nr_thr_resamp=nr_threads_resample, nr_thr_saving=nr_threads_save,
            quiet=True, verbose=False, skip_saving=False,
        )

        label_names = {int(k): v for k, v in class_map[img_task].items()}
        with open(labels_path, "w") as f:
            json.dump(dict(task=img_task, fast=fast, modality=str(modality[idx]), site=str(site[idx]), labels=label_names), f, indent=1)

        mask_img = nib.load(str(save_mask_path))
        rows = _structure_table(mask_img, label_names)
        with open(table_path, "w", newline="") as f:
            fieldnames = ["label", "name", "n_voxels", "volume_ml", "centroid_x_mm", "centroid_y_mm", "centroid_z_mm", "zmin_mm", "zmax_mm", "touches_boundary"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        logger.info("Saved %s (%d structures) and %s", save_mask_path.name, len(rows), table_path.name)

        for extra in extra_tasks:
            if extra not in class_map:
                logger.warning("Skipping unknown extra task %r", extra)
                continue
            extra_path = save_dir / f"{img_name}_{seg_obj}_{extra}.nii.gz"
            if extra_path.exists() and skip_exist:
                continue
            logger.info("Running extra TotalSegmentator task: %s", extra)
            try:
                totalsegmentator(str(img_path), str(extra_path), ml=True, task=extra, fast=fast, device=ts_device,
                                 nr_thr_resamp=nr_threads_resample, nr_thr_saving=nr_threads_save, quiet=True, verbose=False)
            except Exception as exc:  # licensed tasks raise without a licence key
                logger.warning("Extra task %s failed for %s: %s", extra, img_name, exc)

    return

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--radiology', default="/home/s/sg2162/projects/TCIA_NIFTI/image")
    parser.add_argument('--dataset', default="MAMAMIA", type=str)
    parser.add_argument('--seg_obj', default="tumor", choices=["tumor", "organ"], type=str)
    parser.add_argument("--keep_largest", action="store_true")
    parser.add_argument('--phase', default="single", choices=["single", "multiple"], type=str)
    parser.add_argument('--format', default="nifti", choices=["dicom", "nifti"], type=str)
    parser.add_argument('--meta_info', default=None)
    parser.add_argument('--save_dir', default="/home/sg2162/rds/hpc-work/Experiments/radiomics", type=str)
    parser.add_argument('--model', default="BiomedParse", choices=["SegVol", "BiomedParse", "VoxTell", "TotalSegmentator"], type=str)
    parser.add_argument(
        '--voxtell_model_root',
        default=os.path.join(relative_path, 'checkpoints', 'VoxTell'),
        type=str,
        help='Directory containing voxtell_v1.1; downloads the model here if missing',
    )
    parser.add_argument(
        '--device',
        default=None,
        type=str,
        help='Torch device for VoxTell/TotalSegmentator, e.g. cuda:0 or cpu (default: auto-detect)',
    )
    parser.add_argument(
        '--ts_model_root',
        default=os.path.join(relative_path, 'checkpoints', 'TotalSegmentator'),
        type=str,
        help='Directory holding TotalSegmentator weights (nnunet/results); downloads missing tasks here',
    )
    parser.add_argument(
        '--ts_task',
        default=None,
        type=str,
        help='TotalSegmentator task; default picks total (CT) or total_mr (MR) per image',
    )
    parser.add_argument('--ts_fast', action='store_true', help='TotalSegmentator 3 mm fast model')
    parser.add_argument('--ts_roi_subset', default=None, nargs='*', help='Only these TS classes (runs only the needed sub-models)')
    parser.add_argument('--ts_threads', default=4, type=int, help='CPU threads for TS resampling/saving (match your Slurm --cpus-per-task)')
    parser.add_argument('--ts_body_seg', action='store_true', help='Crop to body region before segmenting')
    parser.add_argument(
        '--ts_extra_tasks',
        default=None,
        nargs='*',
        help='Additional TotalSegmentator tasks run per image, e.g. body lung_vessels tissue_types',
    )
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
            img_format=args.format,
            seg_obj=args.seg_obj
        )
    elif args.dataset == 'CPTAC':
        dataset_info = prepare_CPTAC_radiology_info(
            img_json=args.radiology,
            img_format=args.format,
            seg_obj=args.seg_obj
        )
    else:
        raise ValueError(f'Dataset {args.dataset} is currently unsupported')

    # extract radiology segmentation
    # warning: do not run this function in a loop
    logger.info(f"starting segmentation on {dataset_info['name']}...")
    extract_radiology_segmentation(
        dataset=args.dataset,
        seg_obj=args.seg_obj,
        img_paths=dataset_info['img_paths'][3200:],
        text_prompts=dataset_info['text_prompts'][3200:],
        model_mode=args.model,
        save_dir=save_dir,
        modality=dataset_info['modality'][3200:],
        site=dataset_info['site'][3200:],
        meta_list=dataset_info['meta_list'],
        img_format=dataset_info['img_format'][3200:],
        beta_params=None,
        keep_largest=args.keep_largest,
        prompt_ensemble=False,
        save_radiomics=False,
        zoom_in=False,
        voxtell_model_root=args.voxtell_model_root,
        device=args.device,
        skip_exist=True,
        ts_model_root=args.ts_model_root,
        ts_task=args.ts_task,
        ts_fast=args.ts_fast,
        ts_extra_tasks=args.ts_extra_tasks,
        ts_roi_subset=args.ts_roi_subset,
        ts_threads=args.ts_threads,
        ts_body_seg=args.ts_body_seg,
    )
