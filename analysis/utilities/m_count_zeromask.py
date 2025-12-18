import os
import nibabel as nib
import numpy as np

folder = "/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/TCGA_Seg/BiomedParse"

total_files = 0
empty_files = []
zero_mask_files = []

for root, _, files in os.walk(folder):
    for f in files:
        if f.endswith(".nii.gz"):
            total_files += 1
            path = os.path.join(root, f)
            
            # Case 1: empty file (0 bytes)
            if os.path.getsize(path) == 0:
                empty_files.append(path)
                continue  # skip further checks

            # Case 2: check if all voxel values are 0
            try:
                img = nib.load(path)
                data = img.get_fdata()

                if np.all(data == 0):
                    zero_mask_files.append(path)
                    print('Found zero mask!')

                if np.sum(data) == 1:
                    print('mask only contains 1 segmented voxel!')

            except Exception as e:
                print(f"Error reading {path}: {e}")

print("Total .nii.gz files:", total_files)
print("Empty files:", len(empty_files))
print("Zero mask files:", len(zero_mask_files))
