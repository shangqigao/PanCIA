import os
import shutil
import random
import pathlib
import joblib
import json
import nibabel as nib
import numpy as np
from PIL import Image
from skimage import transform
import pandas as pd

CT_WINDOWS = {
    'abdomen': [-150, 250],
    'lung': [-1000, 1000],
    'pelvis': [-55, 200],
    'liver': [-25, 230],
    'colon': [-68, 187],
    'pancreas': [-100, 200]
}

CT_sites = {
    "bladder": 'pelvis',
    "breast": 'breast',
    "cervix": 'pelvis',
    "colon": 'colon',
    "kidney": 'abdomen',
    "liver": 'liver',
    "lung": 'lung',
    "ovary": 'pelvis',
    "pancreas": 'pancreas',
    "prostate": 'pelvis',
    "uterus": 'pelvis'
}

def process_intensity_image(image_data, is_CT, site=None):
    # process intensity-based image. If CT, apply site specific windowing
    
    # image_data: 2D numpy array of shape (H, W)
    
    # return: 3-channel numpy array of shape (H, W, 3) as model input
    
    if is_CT:
        # process image with windowing
        if site and site in CT_WINDOWS:
            window = CT_WINDOWS[site]
        else:
            raise ValueError(f'Please choose CT site from {CT_WINDOWS.keys()}')
        lower_bound, upper_bound = window
    else:
        # process image with intensity range 0.5-99.5 percentile
        if image_data.max() > 0:
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
        else:
            lower_bound, upper_bound = 0, 0

    if lower_bound == 0 and upper_bound == 0:
        image_data_pre = np.zeros_like(image_data)  
    else:
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - image_data_pre.min())
            / (image_data_pre.max() - image_data_pre.min() + 1e-10)
            * 255.0
        )
    
    # pad to square with equal padding on both sides
    shape = image_data_pre.shape
    if shape[0] > shape[1]:
        pad = (shape[0]-shape[1])//2
        pad_width = ((0,0), (pad, pad))
    elif shape[0] < shape[1]:
        pad = (shape[1]-shape[0])//2
        pad_width = ((pad, pad), (0,0))
    else:
        pad_width = None
    
    if pad_width is not None:
        image_data_pre = np.pad(image_data_pre, pad_width, 'constant', constant_values=0)
        
    # resize image to 1024x1024
    image_size = 1024
    resize_image = transform.resize(image_data_pre, (image_size, image_size), order=3, 
                                    mode='constant', preserve_range=True, anti_aliasing=True)
    
    # convert to 3-channel image
    resize_image = np.stack([resize_image]*3, axis=-1)
        
    return resize_image.astype(np.uint8)

def get_orientation(nii_img):
    affine = nii_img.affine
    ornt = nib.orientations.io_orientation(affine)
    axcodes = nib.orientations.ornt2axcodes(ornt)

    # Compute voxel spacing from affine
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    # Find the two axes with the closest spacing (likely in-plane)
    diffs = np.abs(voxel_sizes[:, None] - voxel_sizes[None, :])
    diffs[np.eye(3, dtype=bool)] = np.inf  # ignore diagonal
    i1, i2 = np.unravel_index(np.argmin(diffs), diffs.shape)
    in_plane_axes = {i1, i2}
    slice_axis = (set([0, 1, 2]) - in_plane_axes).pop()
    slice_dir = axcodes[slice_axis]

    if slice_dir in ('I', 'S'):
        return 'axial', slice_axis
    elif slice_dir in ('A', 'P'):
        return 'coronal', slice_axis
    elif slice_dir in ('L', 'R'):
        return 'sagittal', slice_axis
    else:
        return 'unknown', slice_axis

def save_slice(slice_img, save_path, is_CT, site):
    slice_img = process_intensity_image(slice_img, is_CT, site)
    slice_img = Image.fromarray(slice_img)
    slice_img.save(save_path)

def save_mask(slice_msk, save_path):
    slice_msk = (slice_msk * 255).astype(np.uint8)
    shape = slice_msk.shape
    if shape[0] > shape[1]:
        pad = (shape[0]-shape[1])//2
        pad_width = ((0,0), (pad, pad))
    elif shape[0] < shape[1]:
        pad = (shape[1]-shape[0])//2
        pad_width = ((pad, pad), (0,0))
    else:
        pad_width = None
    
    if pad_width is not None:
        slice_msk = np.pad(slice_msk, pad_width, 'constant', constant_values=0)
    slice_msk = Image.fromarray(slice_msk)
    slice_msk = slice_msk.resize((1024, 1024), Image.NEAREST)
    slice_msk.save(save_path)


def convert_nifti_to_png(img_path, lab_path, output_img_dir, output_msk_dir, prefix_dict):
    img_name = prefix_dict['img_name']
    modality = prefix_dict['modality']
    site = prefix_dict['site']
    target = prefix_dict['target']
    tumor_label = prefix_dict['tumor_label']

    os.makedirs(output_dir, exist_ok=True)
    
    nii_img = nib.load(img_path)
    phase, slice_axis = get_orientation(nii_img)
    img = nii_img.get_fdata()
    nii_lab = nib.load(lab_path)
    lab = nii_lab.get_fdata()
    lab = np.array(lab, np.float32)
    assert img.shape == lab.shape
    img_slices = []
    if phase in ['axial', 'sagittal', 'coronal']:
        for i in range(img.shape[slice_axis]):
            if slice_axis == 0:
                slice_lab = lab[i, :, :]
                slice_img = img[i, :, :]
            elif slice_axis == 1:
                slice_lab = lab[:, i, :]
                slice_img = img[:, i, :]
            elif slice_axis == 2:
                slice_lab = lab[:, :, i]
                slice_img = img[:, :, i]
            else:
                raise ValueError(f"Slice axis is {slice_axis}, maximum shoud be 2")
            # save slice
            save_img_path = f'{output_img_dir}/{img_name}_{phase}_slice_{i:03d}_{modality}_{site}.png'
            # save_slice(slice_img, save_img_path, modality == 'CT', CT_sites[site])
            # save mask
            mask = slice_lab == int(tumor_label)
            if np.sum(mask) > 0:
                slice_name = f'{img_name}_{phase}_slice_{i:03d}_{modality}_{site}'     
                save_msk_path = f'{output_msk_dir}/{slice_name}_{target}.png'
                img_slices.append({'img_name': img_name, 'slice_name': slice_name, 'class': 'tumor'})
            else:
                slice_name = f'{img_name}_{phase}_slice_{i:03d}_{modality}_{site}'
                save_msk_path = f'{output_msk_dir}/{slice_name}_background.png'
                img_slices.append({'img_name': img_name, 'slice_name': slice_name, 'class': 'background'})
            # save_mask(mask, save_msk_path)
    else:
        raise ValueError("Unsupported or unknown scanning phase")

    return img_slices

def split_list(data, train_ratio=0.8, seed=42):
    random.seed(seed)
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

if __name__ == "__main__":
    # dir = pathlib.Path('Uterus_Tumor_00/labels')
    # lab_paths = dir.glob('*.nii.gz')
    # for path in lab_paths:
    #     try:
    #         nib.load(path)
    #     except:
    #         old_name = path.name
    #         old_path = f'{path}'
    #         new_path = path.with_name(old_name.replace('.nii.gz', '.nii'))
    #         path.rename(new_path)
    #         img = nib.load(new_path)
    #         nib.save(img, old_path)

    root_dir = '/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/TumorSegmentation'
    dataset_names = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    output_dir = '/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/BiomedParse_TumorSegmentation'
    print(f"Found {len(dataset_names)} datasets")
    for dataset in dataset_names:
        # if 'Bladder' not in dataset: continue
        dataset_dir = f'{root_dir}/{dataset}'
        dataset_info = f'{dataset_dir}/dataset.json'
        with open(dataset_info, "r") as f:
            data_info = json.load(f)
        prefix_dict = {}
        if 'MR' in data_info['modality'].values():
            prefix_dict['modality'] = 'MRI'
        elif 'CT' in data_info['modality'].values():
            prefix_dict['modality'] = 'CT'
        prefix_dict['site'] = data_info['site']
        prefix_dict['target'] = data_info['site'] + '+' + 'tumor'
        for k, v in data_info['labels'].items():
            if v == 'tumor': 
                prefix_dict['tumor_label'] = k

        # json_dir = f"{root_dir}/../TumorSegmentation_split/{dataset}"
        # os.makedirs(json_dir, exist_ok=True)
        # shutil.copy(dataset_info, json_dir)

        # img_dir = f'{dataset_dir}/images'
        # lab_dir = f'{dataset_dir}/labels'
        # img_paths = pathlib.Path(img_dir).glob('*.nii.gz')
        # # img_names = sorted([p.name.replace('.nii.gz', '') for p in img_paths])
        # img_names = [p.name.replace('.nii.gz', '') for p in img_paths]
        # print(f'Found {len(img_names)} images in {dataset}')
        # train_img_names, test_img_names = split_list(img_names)

        # train_img_paths = [f"images/{name}.nii.gz" for name in train_img_names]
        # train_lab_paths = [f"labels/{name}_seg.nii.gz" for name in train_img_names]
        # train_split = [{"image": img, "label": lab} for img, lab in zip(train_img_paths, train_lab_paths)]

        # test_img_paths = [f"images/{name}.nii.gz" for name in test_img_names]
        # test_lab_paths = [f"labels/{name}_seg.nii.gz" for name in test_img_names]
        # test_split = [{"image": img, "label": lab} for img, lab in zip(test_img_paths, test_lab_paths)]

        # data_info['train'] = train_split
        # data_info['test'] = test_split
        # with open(dataset_info, "w") as f:
        #     json.dump(data_info, f, indent=4)

        output_dataset = dataset[:-3] + '+Background'
        # if os.path.isdir(f'{output_dir}/{output_dataset}'): continue

        # train_img_dir = f'{output_dir}/{output_dataset}/train'
        # os.makedirs(train_img_dir, exist_ok=True)
        # train_msk_dir = f'{output_dir}/{output_dataset}/train_mask'
        # os.makedirs(train_msk_dir, exist_ok=True)
        # train_img_names = [d['image'] for d in data_info['train']]
        # train_lab_names = [d['label'] for d in data_info['train']]
        # def _extract_train_slices(img_name, lab_name):
        #     # if name not in ['case_00027']: continue
        #     print(f'Extracting slices from {img_name} in {dataset}...')
        #     prefix_dict['img_name'] = pathlib.Path(img_name).name.replace(".nii.gz", "")
        #     img_path = f'{dataset_dir}/{img_name}'
        #     lab_path = f'{dataset_dir}/{lab_name}'
        #     convert_nifti_to_png(
        #         img_path=img_path,
        #         lab_path=lab_path,
        #         output_img_dir=train_img_dir,
        #         output_msk_dir=train_msk_dir,
        #         prefix_dict=prefix_dict
        #     )
        # joblib.Parallel(n_jobs=32)(
        #     joblib.delayed(_extract_train_slices)(img_name, lab_name)
        #     for img_name, lab_name in zip(train_img_names, train_lab_names)
        # )
        
        test_img_dir = f'{output_dir}/{output_dataset}/test'
        os.makedirs(test_img_dir, exist_ok=True)
        test_msk_dir = f'{output_dir}/{output_dataset}/test_mask'
        os.makedirs(test_msk_dir, exist_ok=True)
        test_img_names = [d['image'] for d in data_info['test']]
        test_lab_names = [d['label'] for d in data_info['test']]
        def _extract_test_slices(img_name, lab_name):
            # if 'FLARE23_1336' not in name: continue
            print(f'Extracting slices from {img_name} in {dataset}...')
            prefix_dict['img_name'] = pathlib.Path(img_name).name.replace(".nii.gz", "")
            img_path = f'{dataset_dir}/{img_name}'
            lab_path = f'{dataset_dir}/{lab_name}'
            img_slices = convert_nifti_to_png(
                img_path=img_path,
                lab_path=lab_path,
                output_img_dir=test_img_dir,
                output_msk_dir=test_msk_dir,
                prefix_dict=prefix_dict
            )
            return img_slices

        outputs = joblib.Parallel(n_jobs=32)(
            joblib.delayed(_extract_test_slices)(img_name, lab_name)
            for img_name, lab_name in zip(test_img_names, test_lab_names)
        )
        all_slices = []
        for o in outputs: all_slices += o
        df = pd.DataFrame(all_slices)
        df.to_csv(f'{output_dir}/{output_dataset}/test_slices.csv')








