
import nibabel as nib
import numpy as np
import json
import pathlib
from PIL import Image

def get_orientation(affine):
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
    pixel_spacing = voxel_sizes

    if slice_dir in ('I', 'S'):
        return 'axial', slice_axis, pixel_spacing
    elif slice_dir in ('A', 'P'):
        return 'coronal', slice_axis, pixel_spacing
    elif slice_dir in ('L', 'R'):
        return 'sagittal', slice_axis, pixel_spacing
    else:
        print(affine)
        valid_codes = [c for c in axcodes if c is not None]
        code_set = set(valid_codes)
        if {'L', 'R'} & code_set and {'A', 'P'} & code_set:
            return 'axial', slice_axis, pixel_spacing
        elif {'L', 'R'} & code_set and {'S', 'I'} & code_set:
            return 'coronal', slice_axis, pixel_spacing
        elif {'A', 'P'} & code_set and {'S', 'I'} & code_set:
            return 'sagittal', slice_axis, pixel_spacing
        else:
            return 'unknown', slice_axis, pixel_spacing

def load_valid_affine(img, verbose=False):
    hdr = img.header
    affine = img.affine

    # Step 1 — validate the existing affine
    def affine_is_valid(A):
        if A is None or A.shape != (4, 4): 
            return False
        if not np.isfinite(A).all(): 
            return False
        A3 = A[:3, :3]
        if np.linalg.matrix_rank(A3) < 3:  # flat or singular
            return False
        voxel_sizes = np.sqrt((A3 ** 2).sum(axis=0))
        if np.any(voxel_sizes <= 0):
            return False
        if None in nib.aff2axcodes(A):  # orientation unknown
            return False
        return True

    if affine_is_valid(affine):
        if verbose:
            print("✅ Using nii.affine (already valid)")
        return affine

    if verbose:
        print("⚠️ nii.affine invalid or missing — trying sform/qform...")

    # Step 2 — try sform
    affine = img.get_sform(coded=True)[0]
    if affine is None or not np.any(affine):
        affine = img.get_qform(coded=True)[0]

    # Step 3 — if both missing, build from zooms
    if affine is None or not np.any(affine):
        zooms = hdr.get_zooms()[:3]
        affine = np.diag(zooms + (1,))
        if verbose:
            print(f"⚙️ Constructed affine from header zooms: {zooms}")

    return affine

def prepare_TCGA_info(img_json, img_format='nifti'):
    assert pathlib.Path(img_json).suffix == '.json', 'only support loading info from json file'

    project_site = {
        "TCGA-KIRC": 'kidney',
        "TCGA-BRCA": 'breast',
        "TCGA-LIHC": 'liver',
        "TCGA-BLCA": 'bladder',
        "TCGA-UCEC": 'uterus',
        "TCGA-OV":   'ovary',
        "TCGA-LUAD": 'lung',
        "TCGA-CESC": 'cervix',
        "TCGA-KIRP": 'kidney',
        "TCGA-STAD": 'stomach',
        "TCGA-LUSC": 'lung',
        "TCGA-PRAD": 'prostate',
        "TCGA-ESCA": 'esophagus',
        "TCGA-KICH": 'kidney',
        "TCGA-COAD": 'colon',
    }
    prompt_template = {
        'kidney': 'tumor located within the kidney, near the liver',
        'breast': 'tumor located within the breast, adjacent to the chest wall',
        'liver': 'tumor located within the liver, adjacent to the right kidney',
        'bladder': 'tumor located within the bladder, adjacent to the prostate',
        'uterus': 'tumor within the uterus, adjacent to the bladder',
        'ovary': 'tumor located within the ovary, near the uterus',
        'lung': 'tumor within the lung, adjacent to the heart',
        'cervix': 'tumor located within the cervix, adjacent to the bladder',
        'stomach': 'tumor located within the stomach, adjacent to the pancreas and liver',
        'prostate': 'tumor within the prostate, adjacent to the bladder',
        'esophagus': 'tumor located within the esophagus, adjacent to the trachea and stomach',
        'colon': 'tumor located within the colon, near the small intestine',
    }
    with open(img_json, 'r') as f:
        data = json.load(f)
    included_subjects = data['included subjects']
    img_paths = []
    for k, v in included_subjects.items(): img_paths += v['radiology']
    images, site, modality, prompts = [], [], [], []
    for img_path in img_paths:
        folds = str(img_path).split('/')
        project = folds[-3]
        img_mod = {'MR': 'MRI', 'CT': 'CT'}[folds[-4]]
        if project_site.get(project, False):
            images.append(img_path)
            img_site = project_site[project]
            site.append(img_site)
            modality.append(img_mod)
            target = prompt_template[img_site]
            prompts.append(f"{target} on {img_mod}")
    
    dataset_info = {
        'name': 'TCGA',
        'img_paths': images,
        'text_prompts': prompts,
        'modality': modality,
        'site': site,
        'meta_list': None,
        'img_format': [img_format] * len(images)
    }
    return dataset_info

# nifit_path = '/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/TCGA_NIFTI/Radiology/CT/TCGA-OV/1.3.6.1.4.1.14519.5.2.1.7777.4007.233005020700722095334542706028/_3.nii.gz'
# nifit_path = '/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/TCGA_NIFTI/Radiology/CT/TCGA-UCEC/1.3.6.1.4.1.14519.5.2.1.3344.4020.198095390502821890860366659919/6.1_02_0_ABDOMEN_PELVIS_3.nii.gz'

data = prepare_TCGA_info('/home/sg2162/rds/hpc-work/Experiments/clinical/TCGA_included_subjects.json')
nifit_path = data['img_paths'][2500+447]

for text in data['text_prompts'][::50]:
    print(text)

nii = nib.load(nifit_path)
affine = load_valid_affine(nii, verbose=False)
phase, axis, spacing = get_orientation(affine)
image_array = nii.get_fdata()
print('Image shape before squeeze', image_array.shape)
image_array = np.squeeze(image_array)
print('Image shape after squeeze', image_array.shape)
print(f"Phase: {phase}, Axis: {axis}, Spacing: {spacing}")

image_data = image_array[:, :, 0]
if image_data.max() > 0:
    lower_bound, upper_bound = np.percentile(
        image_data[image_data > 0], 0.5
    ), np.percentile(image_data[image_data > 0], 99.5)
else:
    lower_bound, upper_bound = 0, 0
image_data_pre = np.clip(image_data, lower_bound, upper_bound)
image_data_pre = (
    (image_data_pre - image_data_pre.min())
    / (image_data_pre.max() - image_data_pre.min())
    * 255.0
)
image_array = np.stack([image_data_pre]*3, axis=-1).astype(np.uint8)
print(image_array.shape)
print(Image.fromarray(image_array).size)

json_path = nifit_path.replace('.nii.gz', '.json')
with open(json_path, 'r') as f:
    data = json.load(f)
for k, v in data.items(): print(k, v)