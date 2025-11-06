import os

folder = "/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/BiomedParse_TumorSegmentation/Uterus_Tumor+Background/test_mask"

for filename in os.listdir(folder):
    if filename.endswith(".png") and "uterus_background" in filename:
        new_name = filename.replace("uterus_background", "uterus_uterus+tumor")
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
