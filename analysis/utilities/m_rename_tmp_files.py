import pickle
import pathlib

file_map = "/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/Experiments/pathomics/TCGA_pathomic_features/UNI/tmp_2/file_map.dat"
with open(file_map, "rb") as f:
    file_map = pickle.load(f)

for input_path, output_path in file_map:
    input_name = pathlib.Path(input_path).stem
    output_parent_dir = pathlib.Path(output_path).parent.parent

    src_path = pathlib.Path(f"{output_path}.position.npy")
    new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_coordinates.npy")
    src_path.rename(new_path)

    src_path = pathlib.Path(f"{output_path}.features.0.npy")
    new_path = pathlib.Path(f"{output_parent_dir}/{input_name}_pathomics.npy")
    src_path.rename(new_path)