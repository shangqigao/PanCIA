import glob
import ast
from tqdm import tqdm
import pandas as pd

from create_annotations import *


# provide the path to the dataset. There should be train, train_mask, test, test_mask under this folder
targetpath = '/home/sg2162/rds/rds-pion-p3-3b78hrFsASU/PanCancer/BiomedParse_TumorSegmentation/AMOS22MR_Abdomen'
if 'MP_Breast' in targetpath:
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

# design pan-cancer prompts
def structured_pancancer_prompts(targetname):
    if targetname == 'bladder tumor':
        prompts = [
            "tumor located within the bladder, adjacent to the prostate",
            "mass lesion in the bladder near the uterus",
            "bladder tumor close to the rectum",
            "abnormal growth in the bladder extending toward the pelvic wall",
            "lesion in the bladder adjacent to surrounding peritoneum"
        ]
    elif targetname == 'cervix tumor':
        prompts = [
            "tumor located within the cervix, adjacent to the bladder",
            "mass lesion in the cervical canal near the uterus",
            "cervical tumor close to the rectum",
            "abnormal growth in the cervix extending toward the vaginal canal",
            "lesion in the cervix adjacent to surrounding pelvic tissues"
        ]
    elif targetname == 'colon tumor':
        prompts = [
            "tumor located within the colon, near the small intestine",
            "mass lesion in the ascending colon adjacent to the liver",
            "colon tumor close to the sigmoid colon and bladder",
            "abnormal growth in the transverse colon near the pancreas",
            "lesion in the descending colon adjacent to the spleen"
        ]
    elif targetname == 'kidney tumor':
        prompts = [
            "tumor located within the kidney, near the liver",
            "mass lesion in the left kidney adjacent to the spleen",
            "kidney tumor close to the pancreas",
            "abnormal growth in the right kidney near the inferior vena cava",
            "lesion in the kidney extending toward perinephric fat"
        ]
    elif targetname == 'liver tumor':
        prompts = [
            "tumor located within the liver, adjacent to the right kidney",
            "mass lesion in the liver near the stomach",
            "liver tumor close to the diaphragm",
            "abnormal growth in the liver adjacent to the portal vein and gallbladder",
            "lesion in the left lobe of the liver near the spleen"
        ]
    elif targetname == 'lung tumor':
        prompts = [
            "tumor within the lung, adjacent to the heart",
            "mass lesion in the upper lobe near the mediastinum",
            "lung tumor close to the diaphragm",
            "abnormal growth in the lower lobe near the liver",
            "lesion in the lung adjacent to the pleura and ribs"
        ]
    elif targetname == 'ovary tumor':
        prompts = [
            "tumor located within the ovary, near the uterus",
            "mass lesion in the ovary adjacent to the bladder",
            "ovarian tumor close to the fallopian tube",
            "abnormal growth in the ovary near the pelvic wall",
            "lesion in the ovary extending toward peritoneal cavity"
        ]
    elif targetname == 'pancreas tumor':
        prompts = [
            "tumor within the pancreas, adjacent to the stomach",
            "mass lesion in the pancreatic head near the duodenum",
            "pancreatic tumor close to the left kidney",
            "abnormal growth in the pancreas near the portal vein",
            "lesion in the pancreatic tail adjacent to the spleen"
        ]
    elif targetname == 'prostate tumor':
        prompts = [
            "tumor within the prostate, adjacent to the bladder",
            "mass lesion in the prostate near the rectum",
            "prostate tumor close to the seminal vesicles",
            "abnormal growth in the prostate extending toward the pelvic wall",
            "lesion in the prostate adjacent to the urethra"
        ]
    elif targetname == 'uterus tumor':
        prompts = [
            "tumor within the uterus, adjacent to the bladder",
            "mass lesion in the uterine wall near the rectum",
            "uterine tumor close to the ovaries",
            "abnormal growth in the uterus extending toward the pelvic cavity",
            "lesion in the uterus adjacent to surrounding peritoneum"
        ]
    elif targetname == 'breast tumor':
        prompts = [
            "tumor located within the breast, adjacent to the chest wall",
            "mass lesion in the upper breast near the pectoral muscle",
            "breast tumor close to the axilla",
            "abnormal growth in the lower breast extending toward subcutaneous fat",
            "lesion in the breast adjacent to surrounding fibroglandular tissue"
        ]
    elif targetname == 'fibroglandular tissue':
        prompts = [
            "fibroglandular tissue of the breast",
            "glandular tissue located inside the breast",
            "dense breast tissue excluding blood vessels",
            "breast parenchyma composed of glandular tissue"
        ]
    elif targetname == 'breast tissue':
        prompts = [
            "entire breast tissue including vessels and glandular structures",
            "complete breast region with glandular and vascular components",
            "breast organ including tumor, glands, and vessels",
            "whole breast tissue with internal anatomical features"
        ]
    elif targetname == 'blood vessel':
        prompts = [
            "blood vessels located within the breast",
            "vascular structures running through breast tissue",
            "veins and arteries inside breast region",
            "breast vasculature system"
        ]
    else:
        prompts = None

    return prompts

# create descriptive prompts
def create_prompts(filename, targetname, df_meta=None):
    parts = filename.split("_")
    # slice information
    view = parts[-5]
    slice_index = parts[-3]
    modality = parts[-2]
    site = parts[-1].split(".")[0]
    prompts = structured_pancancer_prompts(targetname)
    if prompts is None:
        prompts = ['tumor'] if 'tumor' in targetname else [targetname]

    basic_prompts = []
    for target in prompts:
        basic_prompts += [
            # f"{target} in {site} {modality}",
            f"{view} slice {slice_index} showing {target}",
            f"{target} on {modality}",
            f"{view} {modality} with {target}",
            f"{target} visible in slice {slice_index} of {modality}",
        ]

    # meta information
    meta_prompts = []
    if df_meta is not None:
        # excluded = '_0001_' + "_".join(parts[-5:])
        excluded = '_' + "_".join(parts[-5:])
        patient_id = filename.replace(excluded, "")
        pixel_spacing = df_meta.loc[df_meta["patient_id"] == patient_id, 'pixel_spacing'].values[0]
        x_spacing, y_spacing = pixel_spacing[0], pixel_spacing[1]
        field_strength = df_meta.loc[df_meta["patient_id"] == patient_id, 'field_strength'].values[0]
        bilateral_mri = df_meta.loc[df_meta["patient_id"] == patient_id, 'bilateral_mri'].values[0]
        lateral = 'bilateral' if bilateral_mri == 1 else 'unilateral'
        manufacturer = df_meta.loc[df_meta["patient_id"] == patient_id, 'manufacturer'].values[0]
        for target in prompts:
            meta_prompts += [
                f"a {modality} scan of the {lateral} {site}, {view} view, slice {slice_index}, pixel spacing {x_spacing:.2f}x{y_spacing:.2f} mm, showing {target}",
                f"{lateral} {site} {modality} in {view} view at slice {slice_index} with spacing {x_spacing:.2f}x{y_spacing:.2f} mm, includes {target}",
                f"{view} slice {slice_index} from a {field_strength}T {manufacturer} {modality} of the {lateral} {site}, pixel spacing {x_spacing:.2f}x{y_spacing:.2f} mm, showing {target}",
                f"{lateral} {site} {modality} in {view} view, slice {slice_index}, using {field_strength}T {manufacturer} scanner, spacing {x_spacing:.2f}x{y_spacing:.2f} mm, showing {target}",
                f"{modality} of the {lateral} {site} at slice {slice_index}, {view} view, spacing: {x_spacing:.2f}x{y_spacing:.2f} mm, scanned by {field_strength}T {manufacturer} scanner, shows {target}"
            ]
    
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

    # for keyword in ['train', 'test']:
    for keyword in ['test']:
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
            json.dump(coco_format, outfile, indent=4)
        
        print("Created %d annotations for %d images in folder: %s" % (annotation_cnt, len(coco_format['images']), mask_path))
