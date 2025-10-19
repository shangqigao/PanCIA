import nibabel as nib
import numpy as np
import json

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

nifit_path = '/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/TCGA_NIFTI/Radiology/CT/TCGA-OV/1.3.6.1.4.1.14519.5.2.1.7777.4007.233005020700722095334542706028/_3.nii.gz'
# nifit_path = '/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/TCGA_NIFTI/Radiology/CT/TCGA-UCEC/1.3.6.1.4.1.14519.5.2.1.3344.4020.198095390502821890860366659919/6.1_02_0_ABDOMEN_PELVIS_3.nii.gz'
nii = nib.load(nifit_path)
affine = load_valid_affine(nii, verbose=False)
phase, axis, spacing = get_orientation(affine)
image_array = nii.get_fdata()
print('Image shape is', image_array.shape)

json_path = nifit_path.replace('.nii.gz', '.json')
with open(json_path, 'r') as f:
    data = json.load(f)
for k, v in data.items(): print(k, v)