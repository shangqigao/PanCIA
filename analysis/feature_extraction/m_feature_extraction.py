import sys
sys.path.append('../')

import random
import torch
import os
import pathlib
import joblib
import argparse
import pathlib
import timm
import cv2
import logging
import json
import PIL
import skimage

import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from utilities.m_utils import recur_find_ext, rmdir, select_checkpoints, mkdir
from tiatoolbox.models import DeepFeatureExtractor, IOSegmentorConfig, NucleusInstanceSegmentor
from tiatoolbox.models.architecture.vanilla import CNNBackbone, CNNModel
from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.data import stain_norm_target
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.utils.misc import imwrite

from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree
from pprint import pprint
from scipy.ndimage import zoom

from radiomics import featureextractor
import radiomics

from monai.transforms.utils import generate_spatial_bounding_box
from monai.transforms.utils import get_largest_connected_component_mask
from inference_utils.processing_utils import get_orientation

SEED = 5
random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Get the directory where the current script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add a relative subdirectory to sys.path
root_dir = os.path.join(script_dir, '../../')

def extract_pathomic_feature(
        wsi_paths, 
        wsi_msk_paths, 
        feature_mode, 
        save_dir, 
        mode, 
        resolution=0.5, 
        units="mpp",
        skip_exist=False
    ):
    """extract pathomic feature from wsi
    Args:
        wsi_paths (list): a list of wsi paths
        wsi_msk_paths (list): a list of tissue mask paths of wsi
        fature_mode (str): mode of extracting features, 
            "composition" for extracting features by segmenting and counting nucleus
            "cnn" for extracting features by deep neural networks
        save_dir (str): directory of saving features
        mode (str): 'wsi' or 'tile', if 'wsi', extracting features of wsi
            could be slow if feature mode if 'composition'
        resolution (int): the resolution of extacting features
        units (str): the units of resolution, e.g., mpp  
    """
    if feature_mode == "CNN":
        _ = extract_cnn_pathomics(
            wsi_paths=wsi_paths,
            msk_paths=wsi_msk_paths,
            save_dir=save_dir,
            mode=mode,
            resolution=resolution,
            units=units,
            skip_exist=skip_exist
        )
    elif feature_mode == "HIPT":
        _ = extract_vit_pathomics(
            wsi_paths=wsi_paths,
            msk_paths=wsi_msk_paths,
            save_dir=save_dir,
            mode=mode,
            resolution=resolution,
            units=units,
            skip_exist=skip_exist
        )
    elif feature_mode == "UNI":
        _ = extract_uni_pathomics(
            wsi_paths=wsi_paths,
            msk_paths=wsi_msk_paths,
            save_dir=save_dir,
            mode=mode,
            resolution=resolution,
            units=units,
            skip_exist=skip_exist
        )
    elif feature_mode == "CONCH":
        _ = extract_conch_pathomics(
            wsi_paths=wsi_paths,
            msk_paths=wsi_msk_paths,
            save_dir=save_dir,
            mode=mode,
            resolution=resolution,
            units=units,
            skip_exist=skip_exist
        )
    elif feature_mode == "CHIEF":
        _ = extract_chief_pathomics(
            wsi_paths=wsi_paths,
            msk_paths=wsi_msk_paths,
            save_dir=save_dir,
            mode=mode,
            resolution=resolution,
            units=units,
            skip_exist=skip_exist
        )
    else:
        raise NotImplementedError
    return

def extract_radiomic_feature(
        img_paths, 
        lab_paths, 
        feature_mode, 
        save_dir, 
        class_name,
        prompts=None,
        format='nifit',
        modality='CT',
        site='breast',
        label=1,
        dilation_mm=0,
        resolution=None, 
        units="mm",
        n_jobs=32,
        device="cuda",
        skip_exist=False
    ):
    """extract pathomic feature from wsi
    Args:
        img_paths (list): a list of image paths
        lab_paths (list): a list of label paths
        fature_mode (str): mode of extracting features, 
            "pyradiomics" for extracting radiomics
        save_dir (str): directory of saving features
        label (int): value for which to extract features
        resolution (int): the resolution of extacting features
        units (str): the units of resolution, e.g., mpp  

    """
    if feature_mode == "pyradiomics":
        _ = extract_pyradiomics(
            img_paths=img_paths,
            lab_paths=lab_paths,
            save_dir=save_dir,
            class_name=class_name,
            label=label,
            dilation_mm=dilation_mm,
            resolution=resolution,
            units=units,
            n_jobs=n_jobs,
            skip_exist=skip_exist
        )
    elif feature_mode == "SegVol":
        _ = extract_SegVolViT_radiomics(
            img_paths=img_paths,
            lab_paths=lab_paths,
            save_dir=save_dir,
            class_name=class_name,
            label=label,
            dilation_mm=dilation_mm,
            resolution=resolution,
            units=units,
            device=device,
            skip_exist=skip_exist
        )
    elif feature_mode == "M3D-CLIP":
        _ = extract_M3DCLIP_radiomics(
            img_paths=img_paths,
            lab_paths=lab_paths,
            save_dir=save_dir,
            class_name=class_name,
            label=label,
            resolution=resolution,
            units=units,
            device=device,
            skip_exist=skip_exist
        )
    elif feature_mode == "BiomedParse":
        _ = extract_BiomedParse_radiomics(
            img_paths=img_paths,
            lab_paths=lab_paths,
            prompts=prompts,
            save_dir=save_dir,
            class_name=class_name,
            label=label,
            format=format,
            is_CT=modality == 'CT',
            dilation_mm=dilation_mm,
            site=site,
            resolution=resolution,
            units=units,
            device=device,
            skip_exist=skip_exist
        )
    else:
        raise ValueError(f"Invalid feature mode: {feature_mode}")
    return

def extract_cnn_pathomics(wsi_paths, msk_paths, save_dir, mode, resolution=0.5, units="mpp", skip_exist=False):
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": units, "resolution": resolution},],
        output_resolutions=[{"units": units, "resolution": resolution},],
        patch_input_shape=[224, 224],
        patch_output_shape=[224, 224],
        stride_shape=[224, 224],
        save_resolution={"units": "mpp", "resolution": 8.0}
    )
    
    model = CNNBackbone("resnet50")
    ## define preprocessing function
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    TS = A.Compose([A.Normalize(mean, std), ToTensorV2()])
    def _preproc_func(img):
        return TS(image=img)["image"]
    def _postproc_func(img):
        return img
    model.preproc_func = _preproc_func
    model.postproc_func = _postproc_func
    
    extractor = DeepFeatureExtractor(
        batch_size=128, 
        model=model, 
        num_loader_workers=32, 
    )

    if skip_exist:
        new_wsi_paths, new_msk_paths = [], []
        for wsi_path, msk_path in zip(wsi_paths, msk_paths):
            wsi_name = pathlib.Path(wsi_path).name
            feature_path = pathlib.Path(f"{save_dir}/{wsi_name}_pathomics.npy")
            if feature_path.exists() and skip_exist:
                logging.info(f"{feature_path.name} has existed, skip!")
            else:
                new_wsi_paths.append(wsi_path)
                new_msk_paths.append(msk_path)
    else:
        new_wsi_paths = wsi_paths
        new_msk_paths = msk_paths

    # create temporary dir
    tmp_save_dir = pathlib.Path(f"{save_dir}/tmp")
    rmdir(tmp_save_dir)
    output_map_list = extractor.predict(
        new_wsi_paths,
        new_msk_paths,
        mode=mode,
        ioconfig=ioconfig,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=tmp_save_dir,
    )
    
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent.parent

        src_path = pathlib.Path(f"{output_path}_coordinates.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_coordinates.npy")
        src_path.rename(new_path)

        src_path = pathlib.Path(f"{output_path}_pathomics.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_pathomics.npy")
        src_path.rename(new_path)

    # remove temporary dir
    rmdir(tmp_save_dir)

    return output_map_list

class CNNClassifier(CNNModel):
    def __init__(self, backbone, num_classes=1):
        super().__init__(backbone, num_classes)
    
    def forward(self, imgs):
        feat = self.feat_extract(imgs)
        gap_feat = self.pool(feat)
        return torch.flatten(gap_feat, 1)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        image = batch_data.to(device).type(torch.float32)
        model.eval()
        with torch.inference_mode():
            output = model(image)
        return [output.cpu().numpy()]    

    def load(self, feature_path, classifier_path):
        feature_state_dict = torch.load(feature_path)
        self.feat_extract.load_state_dict(feature_state_dict)
        classifier_state_dict = torch.load(classifier_path)
        self.classifier.load_state_dict(classifier_state_dict)

class ViT(torch.nn.Module):
    def __init__(self, model256_path):
        super().__init__()
        from tiatoolbox.models.architecture.hipt import get_vit256
        self.model256 = get_vit256(pretrained_weights=model256_path)
    
    def forward(self, imgs):
        feat = self.model256(imgs)
        return torch.flatten(feat, 1)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        image = batch_data.to(device).type(torch.float32)
        model.eval()
        with torch.inference_mode():
            output = model(image)
        return [output.cpu().numpy()]
    
def extract_vit_pathomics(wsi_paths, msk_paths, save_dir, mode, resolution=0.5, units="mpp", skip_exist=False):
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": units, "resolution": resolution},],
        output_resolutions=[{"units": units, "resolution": resolution},],
        patch_input_shape=[256, 256],
        patch_output_shape=[256, 256],
        stride_shape=[256, 256],
        save_resolution={"units": "mpp", "resolution": 8.0}
    )
    
    pretrained_path = os.path.join(root_dir, 'checkpoints/HIPT/vit256_small_dino.pth')
    model = ViT(pretrained_path)
    ## define preprocessing function
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    TS = A.Compose([A.Normalize(mean, std), ToTensorV2()])
    def _preproc_func(img):
        return TS(image=img)["image"]
    def _postproc_func(img):
        return img
    model.preproc_func = _preproc_func
    model.postproc_func = _postproc_func

    extractor = DeepFeatureExtractor(
        batch_size=128, 
        model=model, 
        num_loader_workers=32, 
    )

    if skip_exist:
        new_wsi_paths, new_msk_paths = [], []
        for wsi_path, msk_path in zip(wsi_paths, msk_paths):
            wsi_name = pathlib.Path(wsi_path).name
            feature_path = pathlib.Path(f"{save_dir}/{wsi_name}_pathomics.npy")
            if feature_path.exists() and skip_exist:
                logging.info(f"{feature_path.name} has existed, skip!")
            else:
                new_wsi_paths.append(wsi_path)
                new_msk_paths.append(msk_path)
    else:
        new_wsi_paths = wsi_paths
        new_msk_paths = msk_paths

    # create temporary dir
    tmp_save_dir = pathlib.Path(f"{save_dir}/tmp")
    rmdir(tmp_save_dir)
    output_map_list = extractor.predict(
        new_wsi_paths,
        new_msk_paths,
        mode=mode,
        ioconfig=ioconfig,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=tmp_save_dir,
    )
    
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent.parent

        src_path = pathlib.Path(f"{output_path}_coordinates.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_coordinates.npy")
        src_path.rename(new_path)

        src_path = pathlib.Path(f"{output_path}_pathomics.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_pathomics.npy")
        src_path.rename(new_path)

    # remove temporary dir
    rmdir(tmp_save_dir)

    return output_map_list

class UNI(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = timm.create_model(
            model_name="vit_large_patch16_224", 
            img_size=224, 
            patch_size=16, 
            init_values=1e-5, 
            num_classes=0, 
            dynamic_img_size=True,
            checkpoint_path=model_path
        )
    
    def forward(self, imgs):
        feat = self.model(imgs)
        return torch.flatten(feat, 1)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        image = batch_data.to(device).type(torch.float32)
        model.eval()
        with torch.inference_mode():
            output = model(image)
        return [output.cpu().numpy()]
    
def extract_uni_pathomics(wsi_paths, msk_paths, save_dir, mode, resolution=0.5, units="mpp", skip_exist=False):
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": units, "resolution": resolution},],
        output_resolutions=[{"units": units, "resolution": resolution},],
        patch_input_shape=[256, 256],
        patch_output_shape=[256, 256],
        stride_shape=[256, 256],
        save_resolution={"units": "mpp", "resolution": 8.0}
    )
    
    pretrained_path = os.path.join(root_dir, 'checkpoints/UNI/pytorch_model.bin')
    model = UNI(pretrained_path)
    ## define preprocessing function
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    TS = A.Compose([A.Resize(224, 224, cv2.INTER_CUBIC), A.Normalize(mean, std), ToTensorV2()])
    def _preproc_func(img):
        return TS(image=img)["image"]
    def _postproc_func(img):
        return img
    model.preproc_func = _preproc_func
    model.postproc_func = _postproc_func

    extractor = DeepFeatureExtractor(
        batch_size=128, 
        model=model, 
        num_loader_workers=32, 
    )

    if skip_exist:
        new_wsi_paths, new_msk_paths = [], []
        for wsi_path, msk_path in zip(wsi_paths, msk_paths):
            wsi_name = pathlib.Path(wsi_path).name
            feature_path = pathlib.Path(f"{save_dir}/{wsi_name}_pathomics.npy")
            if feature_path.exists() and skip_exist:
                logging.info(f"{feature_path.name} has existed, skip!")
            else:
                new_wsi_paths.append(wsi_path)
                new_msk_paths.append(msk_path)
    else:
        new_wsi_paths = wsi_paths
        new_msk_paths = msk_paths

    # create temporary dir
    tmp_save_dir = pathlib.Path(f"{save_dir}/tmp")
    rmdir(tmp_save_dir)
    output_map_list = extractor.predict(
        new_wsi_paths,
        new_msk_paths,
        mode=mode,
        ioconfig=ioconfig,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=tmp_save_dir,
    )
    
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent.parent

        src_path = pathlib.Path(f"{output_path}_coordinates.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_coordinates.npy")
        src_path.rename(new_path)

        src_path = pathlib.Path(f"{output_path}_pathomics.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_pathomics.npy")
        src_path.rename(new_path)

    # remove temporary dir
    rmdir(tmp_save_dir)

    return output_map_list

class CONCH(torch.nn.Module):
    def __init__(self, cfg, ckpt_path, device):
        super().__init__()
        from tiatoolbox.models.architecture.conch.open_clip_custom import create_model_from_pretrained
        self.model, self.preprocess = create_model_from_pretrained(cfg, ckpt_path, device)

    def forward(self, images):
        img_embeddings = self.model.encode_image(images, proj_contrast=False, normalize=False)
        return torch.flatten(img_embeddings, 1)

    @staticmethod
    def infer_batch(model, images, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        images = images.to(device).type(torch.float32)
        model.eval()
        with torch.inference_mode():
            output = model(images)
        return [output.cpu().numpy()]
    
def extract_conch_pathomics(wsi_paths, msk_paths, save_dir, mode, resolution=0.5, units="mpp", skip_exist=False):
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": units, "resolution": resolution},],
        output_resolutions=[{"units": units, "resolution": resolution},],
        patch_input_shape=[256, 256],
        patch_output_shape=[256, 256],
        stride_shape=[256, 256],
        save_resolution={"units": "mpp", "resolution": 8.0}
    )
    
    model_cfg = 'conch_ViT-B-16'
    ckpt_path = os.path.join(root_dir, 'checkpoints/CONCH/pytorch_model.bin')
    model = CONCH(model_cfg, ckpt_path, 'cuda')
    ## define preprocessing function
    TS = model.preprocess
    def _preproc_func(img):
        img = PIL.Image.fromarray(img)
        return TS(img)
    def _postproc_func(img):
        return img
    model.preproc_func = _preproc_func
    model.postproc_func = _postproc_func

    extractor = DeepFeatureExtractor(
        batch_size=128, 
        model=model, 
        num_loader_workers=32, 
    )

    if skip_exist:
        new_wsi_paths, new_msk_paths = [], []
        for wsi_path, msk_path in zip(wsi_paths, msk_paths):
            wsi_name = pathlib.Path(wsi_path).name
            feature_path = pathlib.Path(f"{save_dir}/{wsi_name}_pathomics.npy")
            if feature_path.exists() and skip_exist:
                logging.info(f"{feature_path.name} has existed, skip!")
            else:
                new_wsi_paths.append(wsi_path)
                new_msk_paths.append(msk_path)
    else:
        new_wsi_paths = wsi_paths
        new_msk_paths = msk_paths

    # create temporary dir
    tmp_save_dir = pathlib.Path(f"{save_dir}/tmp")
    rmdir(tmp_save_dir)
    output_map_list = extractor.predict(
        new_wsi_paths,
        new_msk_paths,
        mode=mode,
        ioconfig=ioconfig,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=tmp_save_dir,
    )
    
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent.parent

        src_path = pathlib.Path(f"{output_path}_coordinates.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_coordinates.npy")
        src_path.rename(new_path)

        src_path = pathlib.Path(f"{output_path}_pathomics.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_pathomics.npy")
        src_path.rename(new_path)

    # remove temporary dir
    rmdir(tmp_save_dir)

    return output_map_list

class CHIEF(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        from tiatoolbox.models.architecture.chief.ctran import ConvStem
        from tiatoolbox.models.architecture.chief.timm.timm import create_model
        self.model = create_model(
            'swin_tiny_patch4_window7_224', 
            embed_layer=ConvStem, 
            pretrained=False
        )
        self.model.head = torch.nn.Identity()
        td = torch.load(model_path)
        self.model.load_state_dict(td['model'], strict=True)
    
    def forward(self, imgs):
        feat = self.model(imgs)
        return torch.flatten(feat, 1)

    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        device = "cuda" if on_gpu else "cpu"
        image = batch_data.to(device).type(torch.float32)
        model.eval()
        with torch.inference_mode():
            output = model(image)
        return [output.cpu().numpy()]

def extract_chief_pathomics(wsi_paths, msk_paths, save_dir, mode, resolution=0.5, units="mpp", skip_exist=False):
    ioconfig = IOSegmentorConfig(
        input_resolutions=[{"units": units, "resolution": resolution},],
        output_resolutions=[{"units": units, "resolution": resolution},],
        patch_input_shape=[256, 256],
        patch_output_shape=[256, 256],
        stride_shape=[256, 256],
        save_resolution={"units": "mpp", "resolution": 8.0}
    )
    
    pretrained_path = os.path.join(root_dir, 'checkpoints/CHIEF/CHIEF_CTransPath.pth')
    model = CHIEF(pretrained_path)
    ## define preprocessing function
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    TS = A.Compose([A.Resize(224, 224, cv2.INTER_CUBIC), A.Normalize(mean, std), ToTensorV2()])
    def _preproc_func(img):
        return TS(image=img)["image"]
    def _postproc_func(img):
        return img
    model.preproc_func = _preproc_func
    model.postproc_func = _postproc_func

    extractor = DeepFeatureExtractor(
        batch_size=128, 
        model=model, 
        num_loader_workers=32, 
    )

    if skip_exist:
        new_wsi_paths, new_msk_paths = [], []
        for wsi_path, msk_path in zip(wsi_paths, msk_paths):
            wsi_name = pathlib.Path(wsi_path).name
            feature_path = pathlib.Path(f"{save_dir}/{wsi_name}_pathomics.npy")
            if feature_path.exists() and skip_exist:
                logging.info(f"{feature_path.name} has existed, skip!")
            else:
                new_wsi_paths.append(wsi_path)
                new_msk_paths.append(msk_path)
    else:
        new_wsi_paths = wsi_paths
        new_msk_paths = msk_paths

    # create temporary dir
    tmp_save_dir = pathlib.Path(f"{save_dir}/tmp")
    rmdir(tmp_save_dir)
    output_map_list = extractor.predict(
        new_wsi_paths,
        new_msk_paths,
        mode=mode,
        ioconfig=ioconfig,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=tmp_save_dir,
    )
    
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent.parent

        src_path = pathlib.Path(f"{output_path}_coordinates.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_coordinates.npy")
        src_path.rename(new_path)

        src_path = pathlib.Path(f"{output_path}_pathomics.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_pathomics.npy")
        src_path.rename(new_path)

    # remove temporary dir
    rmdir(tmp_save_dir)

    return output_map_list

def extract_chief_wsi_level_features(patch_feature_paths, anatomic=13, on_gpu=True):
    from tiatoolbox.models.architecture.chief.CHIEF import CHIEF
    text_embedding_path = os.path.join(root_dir, 'checkpoints/CHIEF/Text_emdding.pth')
    model = CHIEF(size_arg="small", dropout=True, n_classes=2, text_embedding_path=text_embedding_path)
    td = torch.load(os.path.join(root_dir, 'checkpoints/CHIEF/CHIEF_pretraining.pth'))
    model.load_state_dict(td, strict=True)
    device = "cuda" if on_gpu else "cpu"
    model.to(device)
    model.eval()

    for i, path in enumerate(patch_feature_paths):
        features = np.load(path)
        with torch.no_grad():
            x = torch.tensor(features).to(device)
            anatomic = torch.tensor([anatomic]).to(device)
            result = model(x, anatomic)
            wsi_feature_emb = result['WSI_feature'].squeeze().cpu().numpy()
        save_path = f"{path}".replace("_pathomics.npy", "_WSI_pathomics.npy")
        save_name = save_path.split("/")[-1]
        logging.info(f"Saving [{i+1}/{len(patch_feature_paths)}] WSI-level features as {save_name} ...")
        np.save(save_path, wsi_feature_emb)
    return

def extract_composition_features(wsi_paths, msk_paths, save_dir, mode, resolution=0.5, units="mpp"):
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        batch_size=16,
        num_loader_workers=8,
        num_postproc_workers=8,
    )
    if mode == "wsi":
        inst_segmentor.ioconfig.tile_shape = (5120, 5120)
    
    ## define preprocessing function
    target_image = stain_norm_target()
    stain_normaliser = get_normalizer("reinhard")
    stain_normaliser.fit(target_image)
    def _stain_norm_func(img):
        return stain_normaliser.transform(img)
    inst_segmentor.model.preproc_func = _stain_norm_func

    # create temporary dir
    tmp_save_dir = pathlib.Path(f"{save_dir}/tmp")
    rmdir(tmp_save_dir)
    output_map_list = inst_segmentor.predict(
        wsi_paths,
        msk_paths,
        mode=mode,
        on_gpu=True,
        crash_on_exception=True,
        save_dir=save_dir,
        resolution=resolution,
        units=units,
    )

    output_paths = []
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem
        output_parent_dir = pathlib.Path(output_path).parent

        src_path = pathlib.Path(f"{output_path}.dat")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.dat")
        src_path.rename(new_path)
        output_paths.append(new_path)
    for idx, path in enumerate(output_paths):
        if msk_paths is not None:
            get_cell_compositions(wsi_paths[idx], msk_paths[idx], path, save_dir, resolution=resolution, units=units)
        else:
            get_cell_compositions(wsi_paths[idx], None, path, save_dir, resolution=resolution, units=units)

    # remove temporary dir
    rmdir(tmp_save_dir)

    return output_paths


def get_cell_compositions(
        wsi_path,
        mask_path,
        inst_pred_path,
        save_dir,
        num_types = 2,
        patch_input_shape = (512, 512),
        stride_shape = (512, 512),
        resolution = 0.5,
        units = "mpp",
):
    if pathlib.Path(wsi_path).suffix == ".jpg":
        reader = WSIReader.open(wsi_path, mpp=(resolution, resolution))
    else:
        reader = WSIReader.open(wsi_path)
    inst_pred = joblib.load(inst_pred_path)
    inst_pred = {i: v for i, (_, v) in enumerate(inst_pred.items())}
    inst_boxes = [v["box"] for v in inst_pred.values()]
    inst_boxes = np.array(inst_boxes)

    geometries = [shapely_box(*bounds) for bounds in inst_boxes]
    spatial_indexer = STRtree(geometries)
    wsi_shape = reader.slide_dimensions(resolution=resolution, units=units)

    (patch_inputs, _) = PatchExtractor.get_coordinates(
        image_shape=wsi_shape,
        patch_input_shape=patch_input_shape,
        patch_output_shape=patch_input_shape,
        stride_shape=stride_shape,
    )

    if mask_path is not None:
        mask_reader = WSIReader.open(mask_path)
        mask_reader.info = reader.info
        selected_coord_indices = PatchExtractor.filter_coordinates(
            mask_reader,
            patch_inputs,
            wsi_shape=wsi_shape,
            min_mask_ratio=0.5,
        )
        patch_inputs = patch_inputs[selected_coord_indices]

    bounds_compositions = []
    for bounds in patch_inputs:
        bounds_ = shapely_box(*bounds)
        indices = [geo for geo in spatial_indexer.query(bounds_) if bounds_.contains(geometries[geo])]
        insts = [inst_pred[v]["type"] for v in indices]
        _, freqs = np.unique(insts, return_counts=True)
        holder = np.zeros(num_types, dtype=np.int16)
        holder[0] = freqs.sum()
        bounds_compositions.append(holder)
    bounds_compositions = np.array(bounds_compositions)

    base_name = pathlib.Path(wsi_path).stem
    np.save(f"{save_dir}/{base_name}_coordinates.npy", patch_inputs)
    np.save(f"{save_dir}/{base_name}_pathomics.npy", bounds_compositions)

def extract_pyradiomics(img_paths, lab_paths, save_dir, class_name, label=None, dilation_mm=0, resolution=None, units="mm", n_jobs=32, skip_exist=False):
    import nibabel as nib

    # Get the PyRadiomics logger (default log-level = INFO)
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

    # Write out all log entries to a file
    handler = logging.FileHandler(filename=f"testLog.{class_name}.txt", mode='w')
    formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    settings = {}
    settings['resampledPixelSpacing'] = [resolution, resolution, resolution]
    settings['correctMask'] = True
    settings['maskDilation'] = int(dilation_mm)

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableImageTypeByName('Wavelet')
    os.makedirs(save_dir, exist_ok=True)
    def _extract_radiomics(idx, img_path, lab_path):
        img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")
        save_path = pathlib.Path(f"{save_dir}/{img_name}_{class_name}_radiomics.json")
        if save_path.exists() and skip_exist:
            logging.info(f"{save_path.name} has existed, skip!")
            return
        
        logging.info("extracting radiomics: {}/{}...".format(idx + 1, len(img_paths)))

        # skip if mask is empty
        nii = nib.load(lab_path)
        label_arr = nii.get_fdata()
        if np.sum(label_arr > 0) < 1: 
            lab_name = pathlib.Path(lab_path).name
            logging.info(f"Skip case {lab_name}, because no foreground found!")
            return
        del label_arr

        features = extractor.execute(str(img_path), str(lab_path), label)
        for k, v in features.items():
            if isinstance(v, np.ndarray):
                features[k] = v.tolist()

        logging.info(f"Saving radiomic features to {save_path}")
        with save_path.open("w") as handle:
            json.dump(features, handle, indent=4)

        return

    # extract radiomics in parallel
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_extract_radiomics)(idx, img_path, lab_path)
        for idx, (img_path, lab_path) in enumerate(zip(img_paths, lab_paths))
    )
    return

def extract_VOI(image, label, patch_size, padding, output_shape=None):
    assert image.ndim == 3
    label = get_largest_connected_component_mask(label)
    s, e = generate_spatial_bounding_box(np.expand_dims(label, 0))
    s = np.array(s) - np.array(padding)
    e = np.array(e) + np.array(padding)
    image = image[s[0]:e[0], s[1]:e[1], s[2]:e[2]]
    if output_shape is not None:
        shape = output_shape
    else:
        shape = image.shape * np.array(patch_size, np.int32)
    image = skimage.transform.resize(image, output_shape=shape)
    bbox = [s, e]
    return image, bbox

def SegVol_image_transforms(keys, spacing):
    from monai import transforms

    class MinMaxNormalization(transforms.Transform):
        def __init__(self, keys):
            self.keys = keys

        def __call__(self, data):
            d = dict(data)
            for k in self.keys:
                d[k] = d[k] - d[k].min()
                d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)
            return d

    class ForegroundNormalization(transforms.Transform):
        def __init__(self, keys):
            self.keys = keys
        
        def __call__(self, data):
            d = dict(data)
            
            for key in self.keys:
                d[key] = self.normalize(d[key])
            return d
        
        def normalize(self, ct_narray):
            ct_voxel_ndarray = ct_narray.copy()
            ct_voxel_ndarray = ct_voxel_ndarray.flatten()
            thred = np.mean(ct_voxel_ndarray)
            voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
            upper_bound = np.percentile(voxel_filtered, 99.95)
            lower_bound = np.percentile(voxel_filtered, 00.05)
            mean = np.mean(voxel_filtered)
            std = np.std(voxel_filtered)
            ### transform ###
            ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
            ct_narray = (ct_narray - mean) / max(std, 1e-8)
            return ct_narray

    class DimTranspose(transforms.Transform):
        def __init__(self, keys):
            self.keys = keys
        
        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                d[key] = np.swapaxes(d[key], -1, -3)
            return d

    transform = transforms.Compose(
            [
                transforms.LoadImaged(keys, ensure_channel_first=True, allow_missing_keys=True),
                transforms.Spacingd(keys, pixdim=spacing, mode=('bilinear', 'nearest')),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                ForegroundNormalization(keys=["image"]),
                MinMaxNormalization(keys=["image"]),
                DimTranspose(keys=["image", "label"])
            ]
        )
    return transform


def extract_SegVolViT_radiomics(img_paths, lab_paths, save_dir, class_name, label=1, dilation_mm=0, resolution=1, units="mm", device="cuda", skip_exist=False):
    from monai.networks.nets import ViT
    from monai.inferers import SlidingWindowInferer
    from scipy.ndimage import binary_dilation
    import nibabel as nib
    
    roi_size = (32, 256, 256)
    patch_size = (4, 16, 16)
    vit = ViT(
        in_channels=1,
        img_size=roi_size,
        patch_size=patch_size,
        pos_embed="perceptron",
        )
    # print(vit)
    vit_checkpoint = os.path.join(root_dir, 'checkpoints/SegVol/ViT_pretrain.ckpt')
    with open(vit_checkpoint, "rb") as f:
        state_dict = torch.load(f, map_location='cpu')['state_dict']
        encoder_dict = {k.replace('model.encoder.', ''): v for k, v in state_dict.items() if 'model.encoder.' in k}
    vit.load_state_dict(encoder_dict)
    vit.to(device)
    vit.eval()
    print(f'Loaded SegVol encoder param: {vit_checkpoint}')

    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=8,
        sw_device=device,
        device='cpu',
        progress=False
    )
    print("Set sliding window for model inference.")

    spacing = (resolution, resolution, resolution)
    keys = ["image", "label"]
    transform = SegVol_image_transforms(keys, spacing)
    fs = (np.array(roi_size) / np.array(patch_size)).astype(np.int32)
    mkdir(save_dir)
    
    for idx, (img_path, lab_path) in enumerate(zip(img_paths, lab_paths)):
        img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")
        feature_path = pathlib.Path(f"{save_dir}/{img_name}_{class_name}_radiomics.npy")
        if feature_path.exists() and skip_exist:
            logging.info(f"{feature_path.name} has existed, skip!")
            continue
        
        logging.info("extracting radiomics: {}/{}...".format(idx + 1, len(img_paths)))
        case_dict = {"image": img_path, "label": lab_path}
        data = transform(case_dict)
        image = data["image"].squeeze()
        label = data["label"].squeeze()
        img_shape = image.shape

        # skip empty mask
        if np.sum(label > 0) < 1: 
            lab_name = pathlib.Path(lab_path).name
            logging.info(f"Skip case {lab_name}, because no foreground found!")
            continue

        # get scanning phase and move slice axis to the first 
        # SAR: (axial: RA-xy), (sagittal: AS-yz), (coronal: RS-xz)
        nii = nib.load(lab_path)
        affine = nii.affine
        phase, _, _ = get_orientation(affine)
        if phase == "axial":
            slice_axis = 0
        elif phase == "sagittal":
            slice_axis = 2
        elif phase == "coronal":
            slice_axis == 1
        image = np.moveaxis(image, slice_axis, 0)

        # pad if smaller than roi size
        pad_after = [max(0, s1 - s2) for s1, s2 in zip(roi_size, image.shape)]
        padding = [(0, p) for p in pad_after]
        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)

        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to('cpu')
        with torch.no_grad():
            feature = inferer(image, lambda x: vit(x)[0].transpose(1, 2).reshape(-1, 768, fs[0], fs[1], fs[2]))
        feature = feature.squeeze().permute(1, 2, 3, 0).cpu().numpy().astype(np.float16)

        crop_pad = [s / p for s, p in zip(pad_after, patch_size)]
        crop_roi = [int(s - p) for s, p in zip(feature.shape[:3], crop_pad)]
        feature = feature[:crop_roi[0], :crop_roi[1], :crop_roi[2], :]
        feature = np.moveaxis(feature, 0, slice_axis)
        feat_shape = feature.shape
        feat_memory = feature.nbytes / 1024**3
        logging.info(f"Got image of shape {img_shape}, image feature of shape {feat_shape} ({feat_memory:.2f}GiB)")

        # dilate label
        if dilation_mm > 0:
            logging.info(f"Dilating mask by {int(dilation_mm)}mm")
            radius_voxels = int(dilation_mm / resolution)
            kernel = skimage.morphology.ball(radius_voxels)
            label = binary_dilation(label, structure=kernel).astype(np.uint8)

        # downsample label
        new_shape = [feat_shape[0], feat_shape[1], feat_shape[2]]
        if slice_axis == 0:
            new_patch_size = patch_size
        elif slice_axis == 1:
            new_patch_size = (patch_size[1], patch_size[0], patch_size[2])
        elif slice_axis == 2:
            new_patch_size = (patch_size[1], patch_size[2], patch_size[0])
        cropped_shape = [i*j for i, j in zip(new_shape, new_patch_size)]
        zoom_factors = tuple(ns/os for ns, os in zip(new_shape, cropped_shape))
        ds_label = label[:cropped_shape[0], :cropped_shape[1], :cropped_shape[2]]
        ds_label = zoom(ds_label, zoom=zoom_factors, order=0)
        
        # extract ROI features
        feature = feature[ds_label > 0]
        feat_memory = feature.nbytes / 1024**2
        coordinates = np.argwhere(ds_label > 0) * np.array(new_patch_size).reshape(1, 3)
        coordinates += np.array(new_patch_size).reshape(1, 3) // 2
        logging.info(f"Extracted ROI feature of shape {feature.shape} ({feat_memory:.2f}MiB)")
        assert len(feature) == len(coordinates)
        logging.info(f"Saving radiomics in the resolution of {spacing}...")
        logging.info(f"Saving radiomic features to {feature_path}")
        np.save(feature_path, feature)
        coordinates_path = f"{save_dir}/{img_name}_{class_name}_coordinates.npy"
        logging.info(f"Saving feature coordinates to {coordinates_path}")
        np.save(coordinates_path, coordinates)
    return

def M3DCLIP_image_transforms(keys, padding):
    from monai import transforms

    class MinMaxNormalization(transforms.Transform):
        def __init__(self, keys):
            self.keys = keys

        def __call__(self, data):
            d = dict(data)
            for k in self.keys:
                d[k] = d[k] - d[k].min()
                d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)
            return d

    class ForegroundNormalization(transforms.Transform):
        def __init__(self, keys):
            self.keys = keys
        
        def __call__(self, data):
            d = dict(data)
            
            for key in self.keys:
                d[key] = self.normalize(d[key])
            return d
        
        def normalize(self, ct_narray):
            ct_voxel_ndarray = ct_narray.copy()
            ct_voxel_ndarray = ct_voxel_ndarray.flatten()
            thred = np.mean(ct_voxel_ndarray)
            voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
            upper_bound = np.percentile(voxel_filtered, 99.95)
            lower_bound = np.percentile(voxel_filtered, 00.05)
            mean = np.mean(voxel_filtered)
            std = np.std(voxel_filtered)
            ### transform ###
            ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
            ct_narray = (ct_narray - mean) / max(std, 1e-8)
            return ct_narray

    class DimTranspose(transforms.Transform):
        def __init__(self, keys):
            self.keys = keys
        
        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                d[key] = np.swapaxes(d[key], -1, -3)
            return d

    transform = transforms.Compose(
            [
                transforms.LoadImaged(keys, ensure_channel_first=True, allow_missing_keys=True),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                ForegroundNormalization(keys=["image"]),
                MinMaxNormalization(keys=["image"]),
                DimTranspose(keys=["image", "label"]),
                transforms.BorderPadd(keys=["image", "label"], spatial_border=padding)
            ]
        )
    return transform

def extract_M3DCLIP_radiomics(img_paths, lab_paths, save_dir, class_name, label=1, resolution=1.024, units="mm", device="cpu", skip_exist=False):
    from transformers import AutoTokenizer, AutoModel
    
    roi_size = (32, 256, 256)
    tokenizer = AutoTokenizer.from_pretrained(
        "GoodBaiBai88/M3D-CLIP",
        model_max_length=512,
        padding_side="right",
        use_fast=False
    )
    model = AutoModel.from_pretrained(
        "GoodBaiBai88/M3D-CLIP",
        trust_remote_code=True
    )
    model = model.to(device=device)

    keys = ["image", "label"]
    padding = (4, 8, 8)
    transform = M3DCLIP_image_transforms(keys, padding)
    case_dicts = [
        {"image": img_path, "label": lab_path} for img_path, lab_path in zip(img_paths, lab_paths)
    ]
    data_dicts = transform(case_dicts)
    mkdir(save_dir)
    
    for idx, (case, data) in enumerate(zip(case_dicts, data_dicts)):
        img_name = pathlib.Path(case["image"]).name.replace(".nii.gz", "")
        feature_path = pathlib.Path(f"{save_dir}/{img_name}_{class_name}_radiomics.npy")
        if feature_path.exists() and skip_exist:
            logging.info(f"{feature_path.name} has existed, skip!")
            continue
        
        logging.info("extracting radiomics: {}/{}...".format(idx + 1, len(case_dicts)))
        image = data["image"].squeeze().numpy()
        label = data["label"].squeeze().numpy()
        voi, bbox = extract_VOI(image, label, None, padding, roi_size)
        img_shape, voi_shape = image.shape, voi.shape
        voi = torch.from_numpy(voi).unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode():
            feature = model.encode_image(voi)[:, 0]
        feat_shape = feature.shape
        logging.info(f"Got image of shape {img_shape}, VOI of shape {voi_shape}, feature of shape {feat_shape}")
        feature = feature.squeeze().cpu().numpy()
        logging.info(f"Saving radiomics...")
        np.save(feature_path, feature)
        coordinates_path = f"{save_dir}/{img_name}_{class_name}_coordinates.npy"
        np.save(coordinates_path, np.array(bbox))
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
    target = "tumor located within fibroglandular tissue of the breast"

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

def extract_BiomedParse_radiomics(img_paths, lab_paths, text_prompts, save_dir, class_name, 
                                  label=1, format='nifti', is_CT=True, site=None,
                                  meta_list=None, prompt_ensemble=False,
                                  dilation_mm=0, resolution=None, units="mm", device="gpu", skip_exist=False):
    """extracting radiomic features slice by slice in a size of (1024, 1024)
        if no label provided, directly use model segmentation, else use give labels
    """
    from PIL import Image
    import nibabel as nib
    from skimage.morphology import disk
    from scipy.ndimage import binary_dilation

    logging.getLogger("modeling").setLevel(logging.ERROR)
    from modeling.BaseModel import BaseModel
    from modeling import build_model
    from utilities.distributed import init_distributed
    from utilities.arguments import load_opt_from_config_files
    from utilities.constants import BIOMED_CLASSES

    from inference_utils.inference import interactive_infer_image
    from inference_utils.processing_utils import read_dicom
    from inference_utils.processing_utils import read_nifti_inplane
    from peft import LoraConfig, get_peft_model

    # Build model config
    opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)

    # Load model from pretrained weights
    pretrained_pth = os.path.join(root_dir, 'checkpoints/BiomedParse/MP_heart_LoRA_sqrt')

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

    mkdir(save_dir)
    for idx, (img_path, lab_path, target_name) in enumerate(zip(img_paths, lab_paths, text_prompts)):
        if format == 'dicom':
            img_name = pathlib.Path(img_path[0]).parent.name
        elif format == 'nifti':
            if isinstance(img_path, list):
                img_name = pathlib.Path(img_path[0]).name.replace(".nii.gz", "")
            else:
                img_name = pathlib.Path(img_path).name.replace(".nii.gz", "")
        feature_path = pathlib.Path(f"{save_dir}/{img_name}_{class_name}_radiomics.npy")
        if feature_path.exists() and skip_exist:
            logging.info(f"{feature_path.name} has existed, skip!")
            continue
        
        # read slices from dicom or nifti
        if format == 'dicom':
            dicom_dir = pathlib.Path(img_path)
            assert pathlib.Path(img_path).is_dir()
            dicom_paths = sorted(dicom_dir.glob('*.dcm'))
            images = [read_dicom(p, is_CT, site, keep_size=True, return_spacing=True) for p in dicom_paths]
            slice_axis, affine = 0, np.eye(4)
        elif format == 'nifti':
            images, slice_axis, affine = read_nifti_inplane(img_path, is_CT, site, keep_size=True, return_spacing=True, resolution=resolution)
        else:
            raise ValueError(f'Only support DICOM or NIFTI, but got {format}')

        if format == 'nifti' and lab_path is not None:
            assert f'{lab_path}'.endswith(('.nii', '.nii.gz'))
            nii = nib.load(lab_path)
            affine = nii.affine
            _, _, voxel_spacing = get_orientation(affine)
            labels = nii.get_fdata()

            # resample to given resolution
            if resolution is not None:
                new_spacing = (resolution, resolution, resolution)
                zoom_factors = tuple(os/ns for os, ns in zip(voxel_spacing, new_spacing))
                labels = zoom(labels, zoom=zoom_factors, order=0)

            # move the slice axis to the first
            labels = np.moveaxis(labels, slice_axis, 0) 
        else:
            labels = None

        # select slice range of interest
        if labels is not None:
            if np.sum(labels > 0) < 1: 
                lab_name = pathlib.Path(lab_path).name
                logging.info(f"Skip case {lab_name}, because no foreground found!")
                continue
            coords = np.array(np.where(labels))
            zmin, _, _ = coords.min(axis=1)
            zmax, _, _ = coords.max(axis=1)
            zmin = max(zmin - int(dilation_mm), 0)
            zmax = min(zmax + int(dilation_mm), labels.shape[0] - 1)
            assert len(images) == len(labels)
            images = images[zmin:zmax+1]
            labels = labels[zmin:zmax+1, ...]
            logging.info(f"Selected {zmax - zmin + 1} slices from index {zmin} to {zmax} for feature extraction")

        radiomic_feat, masks = [], []
        meta_data = {} if meta_list is None else meta_list[idx]
        for i, element in enumerate(images):
            assert len(element) == 3
            img, spacing, phase = element

            if len(spacing) == 2:
                    pixel_spacing = spacing
            else:
                assert len(spacing) == 3
                pixel_index = list(set([0, 1, 2]) - {slice_axis})
                pixel_spacing = [spacing[i] for i in pixel_index]

            # use prompt ensemble
            if prompt_ensemble:
                assert isinstance(meta_data, dict)
                meta_data['view'] = phase
                meta_data['slice_index'] = f'{i:03}'
                meta_data['modality'] = 'CT' if is_CT else 'MRI'
                meta_data['site'] = site
                meta_data['target'] = target_name
                meta_data['pixel_spacing'] = pixel_spacing
                text_prompts = create_prompts(meta_data)
                # text_prompts = [text_prompts[2], text_prompts[9]]
                text_prompts = [text_prompts[2]]
            else:
                text_prompts = [target_name]
            # print(f"Segmenting slice [{i+1}/{len(images)}] ...")

            # resize_mask=False would keep mask size to be (1024, 1024)
            ensemble_prob = []
            ensemble_feat = []
            for text_prompt in text_prompts:
                pred_prob, feature = interactive_infer_image(model, Image.fromarray(img), text_prompt, resize_mask=True, return_feature=True)
                ensemble_feat.append(np.transpose(feature, (1, 2, 0)))
                ensemble_prob.append(pred_prob)
            pred_prob = np.max(np.concatenate(ensemble_prob, axis=0), axis=0, keepdims=True)
            slice_feat = np.mean(np.stack(ensemble_feat, axis=0), axis=0, keepdims=True)
            radiomic_feat.append(slice_feat.astype(np.float16))
            if labels is None:
                pred_mask = (1*(pred_prob > 0.5)).astype(np.uint8)
            else:
                pred_mask = labels[i, ...]
            masks.append(pred_mask)

        # Get feature array with shape [X, Y, Z, C]
        masks = np.stack(masks, axis=0)
        radiomic_feat = np.concatenate(radiomic_feat, axis=0)
        radiomic_memory = radiomic_feat.nbytes / 1024**3
        logging.info(f"Got image of shape {masks.shape}, image feature of shape {radiomic_feat.shape} ({radiomic_memory:.2f}GiB)")
        final_mask = np.moveaxis(masks, 0, slice_axis)
        # mask dilation based on physical size
        if dilation_mm > 0:
            logging.info(f"Dilating mask by {int(dilation_mm)}mm")
            mean_spacing = np.mean(spacing)
            radius_voxels = int(dilation_mm / mean_spacing)
            kernel = skimage.morphology.ball(radius_voxels)
            final_mask = binary_dilation(final_mask, structure=kernel).astype(np.uint8)
        radiomic_feat = np.moveaxis(radiomic_feat, 0, slice_axis)

        # skip empty mask
        if np.sum(final_mask) < 1: continue

        # extract radiomic features of tumor regions
        radiomic_feat = radiomic_feat[final_mask > 0]
        radiomic_memory = radiomic_feat.nbytes / 1024**2
        radiomic_coord = np.argwhere(final_mask > 0)
        radiomic_coord[:, slice_axis] += zmin
        logging.info(f"Extracted ROI feature of shape {radiomic_feat.shape} ({radiomic_memory:.2f}MiB)")
        logging.info(f"Saving radiomic features to {feature_path}")
        np.save(feature_path, radiomic_feat)
        coordinates_path = f"{save_dir}/{img_name}_{class_name}_coordinates.npy"
        logging.info(f"Saving feature coordinates to {coordinates_path}")
        np.save(coordinates_path, radiomic_coord)

        # save final mask
        # save_mask_path = f"{save_dir}/{img_name}.nii.gz"
        # print(f"Saving predicted segmentation to {save_mask_path}")
        # nifti_img = nib.Nifti1Image(final_mask, affine)
        # nib.save(nifti_img, save_mask_path)

    return



if __name__ == "__main__":
    ## argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', default="/well/rittscher/shared/datasets/KiBla/cases/3923_21/3923_21_G_HE.isyntax")
    parser.add_argument('--mask_method', default='otsu', help='method of tissue masking')
    parser.add_argument('--tile_location', default=[50000, 50000], type=list)
    parser.add_argument('--level', default=0, type=int)
    parser.add_argument('--tile_size', default=[1024, 1024], type=list)
    parser.add_argument('--save_dir', default="a_04feature_extraction/wsi_features", type=str)
    parser.add_argument('--mode', default="tile", type=str)
    parser.add_argument('--feature_mode', default="cnn", type=str)
    args = parser.parse_args()

    wsi = WSIReader.open(args.slide_path)
    mask = wsi.tissue_mask(method=args.mask_method, resolution=1.25, units="power")
    pprint(wsi.info.as_dict())
    if args.mode == "tile":
        tile = wsi.read_region(args.tile_location, args.level, args.tile_size)
        wsi_path = os.path.join("a_04feature_extraction", 'tile_sample.jpg')
        imwrite(wsi_path, tile)
        tile_mask = mask.read_region(args.tile_location, args.level, args.tile_size)
        msk_path = os.path.join("a_04feature_extraction", 'tile_mask.jpg')
        imwrite(msk_path, np.uint8(tile_mask*255))
    elif args.mode == "wsi":
        wsi_mask = mask.slide_thumbnail(resolution=1.25, units="power")
        msk_path = os.path.join("a_04feature_extraction", 'wsi_mask.jpg')
        imwrite(msk_path, wsi_mask)
        wsi_path = args.slide_path
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    wsi_feature_dir = os.path.join(args.save_dir, args.feature_mode)
    if args.feature_mode == "composition":
        output_list = extract_composition_features(
            [wsi_path],
            [msk_path],
            wsi_feature_dir,
            args.mode,
        )
    elif args.feature_mode == "cnn":
        output_list = extract_cnn_pathomics(
            [wsi_path],
            [msk_path],
            wsi_feature_dir,
            args.mode,
        )
    else:
        raise ValueError(f"Invalid feature mode: {args.feature_mode}")






