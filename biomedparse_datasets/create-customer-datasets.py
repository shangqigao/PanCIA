import glob
import ast
from tqdm import tqdm
import pandas as pd

from create_annotations import *


# provide the path to the dataset. There should be train, train_mask, test, test_mask under this folder
targetpath = '/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/BiomedParse_TumorSegmentation/Multiphase_Breast'
if 'Breast_Tumor' in targetpath:
    clinical_info_path = 'clinical_and_imaging_info.xlsx'
    df_clinic = pd.read_excel(clinical_info_path, sheet_name='dataset_info')
    df_clinic['pixel_spacing'] = df_clinic['pixel_spacing'].apply(ast.literal_eval)
else:
    df_clinic = None

image_size = 1024


### Load Biomed Label Base
# provide path to predefined label base
with open('label_base.json', 'r') as f:
    label_base = json.load(f)
    
    
    
# get parent class for the names
parent_class = {}
for i in label_base:
    subnames = [label_base[i]['name']] + label_base[i].get('child', [])
    for label in subnames:
        parent_class[label] = int(i)
    
# Label ids of the dataset
category_ids = {label_base[i]['name']: int(i) for i in label_base if 'name' in label_base[i]}

# create descriptive prompts
def create_prompts(filename, targetname, df_meta=None):
    parts = filename.split("_")
    # slice information
    view = parts[-5]
    slice_index = parts[-3]
    modality = parts[-2]
    site = parts[-1].split(".")[0]
    target = 'tumor' if 'tumor' in targetname else targetname

    basic_prompts = [
        # f"{target} in {site} {modality}",
        f"{view} slice {slice_index} showing {target} in {site}",
        f"{target} located in the {site} on {modality}",
        f"{view} {site} {modality} with {target}",
        f"{target} visible in slice {slice_index} of {modality}",
    ]

    # meta information
    if df_meta is not None:
        excluded = '_' + "_".join(parts[-5:])
        patient_id = filename.replace(excluded, "")
        pixel_spacing = df_meta.loc[df_meta["patient_id"] == patient_id, 'pixel_spacing'].values[0]
        x_spacing, y_spacing = pixel_spacing[0], pixel_spacing[1]
        field_strength = df_meta.loc[df_meta["patient_id"] == patient_id, 'field_strength'].values[0]
        bilateral_mri = df_meta.loc[df_meta["patient_id"] == patient_id, 'bilateral_mri'].values[0]
        lateral = 'bilateral' if bilateral_mri == 1 else 'unilateral'
        manufacturer = df_meta.loc[df_meta["patient_id"] == patient_id, 'manufacturer'].values[0]
        meta_prompts = [
            f"a {modality} scan of the {lateral} {site}, {view} view, slice {slice_index}, pixel spacing {x_spacing:.2f}x{y_spacing:.2f} mm, showing {target}",
            f"{lateral} {site} {modality} in {view} view at slice {slice_index} with spacing {x_spacing:.2f}x{y_spacing:.2f} mm, includes {target}",
            f"{view} slice {slice_index} from a {field_strength}T {manufacturer} {modality} of the {lateral} {site}, pixel spacing {x_spacing:.2f}x{y_spacing:.2f} mm, showing {target}",
            f"{lateral} {site} {modality} in {view} view, slice {slice_index}, using {field_strength}T {manufacturer} scanner, spacing {x_spacing:.2f}x{y_spacing:.2f} mm, showing {target}",
            f"{modality} of the {lateral} {site} at slice {slice_index}, {view} view, spacing: {x_spacing:.2f}x{y_spacing:.2f} mm, scanned by {field_strength}T {manufacturer} scanner, shows {target}"
        ]
    else:
        meta_prompts = []
    
    return basic_prompts + meta_prompts

# Get "images" and "annotations" info 
def images_annotations_info(maskpath):
    
    imagepath = maskpath.replace('_mask', '')
    # This id will be automatically increased as we go
    annotation_id = 0
    
    sent_id = 0
    ref_id = 0
    
    annotations = []
    images = []
    image_to_id = {}
    n_total = len(glob.glob(maskpath + "*.png"))
    n_errors = 0
    
    def extra_annotation(ann, file_name, target):
        nonlocal sent_id, ref_id
        ann['file_name'] = file_name
        ann['split'] = keyword
        
        ### modality
        mod = file_name.split('.')[0].split('_')[-2]
        ### site
        site = file_name.split('.')[0].split('_')[-1]
        
        task = {'target': target, 'modality': mod, 'site': site}
        if 'T1' in mod or 'T2' in mod or 'FLAIR' in mod or 'ADC' in mod:
            task['modality'] = 'MRI'
            if 'MRI' not in mod:
                task['sequence'] = mod
            else:
                task['sequence'] = mod[4:]
            
        prompts = [f'{target} in {site} {mod}']
        prompts += create_prompts(filename=file_name, targetname=target, df_meta=df_clinic)
        
        ann['sentences'] = []
        for p in prompts:
            ann['sentences'].append({'raw': p, 'sent': p, 'sent_id': sent_id})
            sent_id += 1
        ann['sent_ids'] = [s['sent_id'] for s in ann['sentences']]
        
        ann['ann_id'] = ann['id']
        ann['ref_id'] = ref_id
        ref_id += 1
        
        return ann
    
    for mask_image in tqdm(glob.glob(maskpath + "*.png")):
        # The mask image is *.png but the original image is *.jpg.
        # We make a reference to the original file in the COCO JSON file
        filename_parsed = os.path.basename(mask_image).split("_")
        target_name = filename_parsed[-1].split(".")[0].replace("+", " ")
        
        original_file_name = "_".join(filename_parsed[:-1]) + ".png"
        
        if original_file_name not in os.listdir(imagepath):
            print("Original file not found: {}".format(original_file_name))
            n_errors += 1
            continue
        
        if original_file_name not in image_to_id:
            image_to_id[original_file_name] = len(image_to_id)

            # "images" info 
            image_id = image_to_id[original_file_name]
            image = create_image_annotation(original_file_name, image_size, image_size, image_id)
            images.append(image)
            
        
        annotation = {
            "mask_file": os.path.basename(mask_image),
            "iscrowd": 0,
            "image_id": image_to_id[original_file_name],
            "category_id": parent_class[target_name],
            "id": annotation_id,
        }

        annotation = extra_annotation(annotation, original_file_name, target_name)
                
        annotations.append(annotation)
        annotation_id += 1
            
    #print(f"Number of errors in conversion: {n_errors}/{n_total}")
    return images, annotations, annotation_id




if __name__ == "__main__":
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()

    for keyword in ['train', 'test']:
        mask_path = os.path.join(targetpath, "{}_mask/".format(keyword))
        
        # Create category section
        coco_format["categories"] = create_category_annotation(category_ids)
    
        # Create images and annotations sections
        coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

        # post-process file
        images_with_ann = set()
        for ann in coco_format['annotations']:
            images_with_ann.add(ann['file_name'])
        for im in coco_format['images']:
            if im["file_name"] not in images_with_ann:
                coco_format['images'].remove(im)

        with open(os.path.join(targetpath, "{}.json".format(keyword)),"w") as outfile:
            json.dump(coco_format, outfile)
        
        print("Created %d annotations for %d images in folder: %s" % (annotation_cnt, len(coco_format['images']), mask_path))
