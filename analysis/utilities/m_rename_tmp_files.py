# import pickle
# import pathlib

# file_map = "/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/Experiments/pathomics/TCGA_pathomic_features/UNI/tmp_2/file_map.dat"
# with open(file_map, "rb") as f:
#     file_map = pickle.load(f)

# for input_path, output_path in file_map:
#     input_name = pathlib.Path(input_path).stem
#     output_parent_dir = pathlib.Path(output_path).parent.parent

#     src_path = pathlib.Path(f"{output_path}.position.npy")
#     new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_coordinates.npy")
#     src_path.rename(new_path)

#     src_path = pathlib.Path(f"{output_path}.features.0.npy")
#     new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_pathomics.npy")
#     src_path.rename(new_path)

# from pathlib import Path

# # Set your root folder
# root_folder = Path("/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/TCGA_Seg/BiomedParse")

# for file_path in root_folder.rglob("*.nii.gz"):
#     # Remove both suffixes
#     base_name = file_path.name[:-7]  # len(".nii.gz") == 7
#     new_name = file_path.with_name(base_name + "_tumor.nii.gz")
    
#     file_path.rename(new_name)
#     print(f"Renamed: {file_path} -> {new_name}")

from pathlib import Path

root = Path("/home/sg2162/rds/rds-ge-sow2-imaging-MRNJucHuBik/PanCancer/Experiments/radiomics/TCGA_radiomic_features")

for file in root.rglob("*.json"):
    file.unlink()