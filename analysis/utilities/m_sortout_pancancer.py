import os
import re
import shutil
import pathlib
from pathlib import Path
import pydicom
import hashlib
import json
import pandas as pd
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder, ds_helper, image_helper
from dcmrtstruct2nii import dcmrtstruct2nii
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct

import tempfile
from collections import Counter, defaultdict, namedtuple

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("run.log"),
        logging.StreamHandler()
    ]
)

logging.info("Starting conversion")


def check_empty_nii(folder_path):
    empty_files = []
    total_files = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                total_files += 1
                file_path = os.path.join(root, file)
                try:
                    nii = nib.load(file_path)
                    data = nii.get_fdata()
                    if not data.any():  # True if all zeros
                        empty_files.append(file_path)
                except Exception as e:
                    logging.info(f"Failed to read {file_path}: {e}")

    logging.info(f"Total NIfTI files checked: {total_files}")
    logging.info(f"Empty files found: {len(empty_files)}")
    if empty_files:
        logging.info("List of empty files:")
        for f in empty_files:
            logging.info(f)

def remove_empty_labels(labels_dir, images_dir):
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)
    removed_count = 0

    for label_path in labels_dir.glob("*.nii.gz"):
        try:
            label_nii = nib.load(str(label_path))
            label_data = label_nii.get_fdata()

            if not label_data.any():  # empty label
                # Construct corresponding image path
                # e.g., D2-062_T2_seg.nii.gz → D2-062_T2.nii.gz
                image_name = label_path.stem.replace("_seg.nii", "") + ".nii.gz"
                image_path = images_dir / image_name

                if image_path.exists():
                    image_path.unlink()
                    label_path.unlink()
                    removed_count += 1
                    logging.info(f"Removed empty label and image: {label_path.name}, {image_name}")
        except Exception as e:
            logging.info(f"Failed to process {label_path}: {e}")

    logging.info(f"Total empty labels removed: {removed_count}")

def process_UMD(input_dir, output_dir):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    cases_paths = list(pathlib.Path(input_dir).glob('UMD*'))
    logging.info(f"Found {len(cases_paths)} cases")
    logging.info(f"Coping to {output_dir}")
    for case_path in cases_paths:
        case_id = case_path.name
        img_path = f"{case_path}/{case_id}_t2.nii.gz"
        lab_path = f"{case_path}/{case_id}_t2_seg.nii.gz"
        if os.path.exists(img_path) and os.path.exists(lab_path):
            logging.info(img_path, lab_path)
            shutil.copy(img_path, img_dir)
            shutil.copy(lab_path, lab_dir)
        else:
            logging.info(f"No segmentation for {case_id}")
    logging.info("Done!")

def fuse_multiple_segmentations(nifti_paths, save_path):
    # Initialize a fused segmentation with zeros (same shape as the first seg)
    ref_img = nib.load(nifti_paths[0])
    fused_data = np.zeros(ref_img.shape, dtype=np.uint8)

    # Loop over segmentations and add them into the fused volume
    for path in nifti_paths:
        seg_img = nib.load(path)
        seg_data = seg_img.get_fdata()

        # Add label where segmentation is non-zero
        fused_data[seg_data > 0] = 1

    # Save the fused segmentation
    fused_img = nib.Nifti1Image(fused_data, ref_img.affine, ref_img.header)
    nib.save(fused_img, save_path)


def process_FedBCa(input_dir, output_dir):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    center_paths = list(pathlib.Path(input_dir).glob('Center*'))
    logging.info(f"Found {len(center_paths)} centers")
    for center in center_paths:
        center_name = center.name
        img_paths = list(pathlib.Path(f"{center}/{center_name}/T2WI").glob('*nii.gz'))
        label_file = f"{center}/{center_name}/{center_name}_label.xlsx"
        df = pd.read_excel(label_file)
        logging.info(f"Found {len(img_paths)} images in {center_name}")
        for img_path in img_paths:
            img_name = img_path.name
            if center_name == "Center1":
                lab_names = df[df['image'] == img_name]['mask_new'].to_list()
            else:
                lab_names = df[df['image_name'] == img_name]['mask_name'].to_list()
            img_save_name = f"{center_name}_{img_name}"
            lab_save_name = f"{img_save_name}".replace(".nii.gz", "_seg.nii.gz")
            if len(lab_names) > 1:
                logging.info(f"Found multiple segmentations for {img_name}")
                lab_paths = [f"{center}/{center_name}/Annotation/{name}" for name in lab_names]
                save_path = f"{lab_dir}/{lab_save_name}"
                logging.info(f"Fusing multiple segmentation into one")
                fuse_multiple_segmentations(lab_paths, save_path)
            elif len(lab_names) == 1:
                lab_path = f"{center}/{center_name}/Annotation/{lab_names[0]}"
                shutil.copy(lab_path, f"{lab_dir}/{lab_save_name}")
            else:
                logging.info(f"No segmentation for {img_name} in {center_name}")
            shutil.copy(img_path, f"{img_dir}/{img_save_name}")
    logging.info("Done!")

def process_Prostate158(input_dir, output_dir):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    df_train = pd.read_csv(f"{input_dir}/train.csv")
    df_valid = pd.read_csv(f"{input_dir}/valid.csv")
    df_test = pd.read_csv(f"{input_dir}/test.csv")
    df = pd.concat([df_train, df_valid, df_test])
    logging.info(f"Found {len(df)} cases")
    for index, row in df.iterrows():
        ID = str(row['ID']).zfill(3)
        logging.info(f"Processing case {ID}")
        t2 = row['t2']
        t2_anatomy = row['t2_anatomy_reader1']
        t2_tumor = row['t2_tumor_reader1']
        save_img_path = f"{img_dir}/{ID}_t2.nii.gz"
        save_seg_path = f"{lab_dir}/{ID}_t2_seg.nii.gz"
        if pd.notna(t2_anatomy) and pd.notna(t2_tumor):
            anatomy = nib.load(f"{input_dir}/{t2_anatomy}")
            seg = anatomy.get_fdata()
            tumor = nib.load(f"{input_dir}/{t2_tumor}").get_fdata()
            seg[tumor > 0] = 3
            fused_img = nib.Nifti1Image(seg, anatomy.affine, anatomy.header) 
            nib.save(fused_img, save_seg_path)
        elif pd.notna(t2_anatomy):
            seg_path = f"{input_dir}/{t2_anatomy}"
            shutil.copy(seg_path, save_seg_path)
        else:
            logging.info(f"No segmentation for {t2}")
        shutil.copy(f"{input_dir}/{t2}", save_img_path)

        adc = row['adc']
        adc_tumor = row['adc_tumor_reader1']
        save_img_path = f"{img_dir}/{ID}_adc.nii.gz"
        save_seg_path = f"{lab_dir}/{ID}_adc_seg.nii.gz"
        if pd.notna(adc_tumor):
            tumor = nib.load(f"{input_dir}/{adc_tumor}")
            tumor_data = tumor.get_fdata()
            tumor_data[tumor_data > 0] = 3
            fused_img = nib.Nifti1Image(tumor_data, tumor.affine, tumor.header) 
            nib.save(fused_img, save_seg_path)
        else:
            logging.info(f"No segmentation for {adc}")
        shutil.copy(f"{input_dir}/{adc}", save_img_path)
    logging.info("Done!")

def process_MAMAMIA(input_dir, output_dir):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    DUKE_paths = list(pathlib.Path(f"{input_dir}/images").glob('DUKE*'))
    logging.info(f"Found {len(DUKE_paths)} cases from DUKE dataset")
    ISPY1_paths = list(pathlib.Path(f"{input_dir}/images").glob('ISPY1*'))
    logging.info(f"Found {len(ISPY1_paths)} cases from ISPY1 dataset")
    ISPY2_paths = list(pathlib.Path(f"{input_dir}/images").glob('ISPY2*'))
    logging.info(f"Found {len(ISPY2_paths)} cases from ISPY2 dataset")
    NACT_paths = list(pathlib.Path(f"{input_dir}/images").glob('NACT*'))
    logging.info(f"Found {len(NACT_paths)} cases from NACT dataset")
    cases_paths = DUKE_paths + ISPY1_paths + ISPY2_paths + NACT_paths
    logging.info(f"Totally found {len(cases_paths)} cases")
    logging.info(f"Coping to {output_dir}")
    for case_path in cases_paths:
        case_id = case_path.name
        img_path = f"{case_path}/{case_id}_0001.nii.gz" # the first post-contrast phase
        lab_path = f"{input_dir}/segmentations/expert/{case_id}.nii.gz"
        if os.path.exists(img_path) and os.path.exists(lab_path):
            # logging.info(img_path, lab_path)
            shutil.copy(img_path, img_dir)
            save_lab_path = f"{lab_dir}/{case_id}_0001_seg.nii.gz"
            shutil.copy(lab_path, save_lab_path)
        else:
            logging.info(f"No segmentation for {case_id}")
    logging.info("Done!")

def CC_DICOMRTSTUCT2NIFTI(dicom_series_path, rtstruct_path, output_img_path, output_seg_path):
    # Read the image series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Write as NIfTI
    sitk.WriteImage(image, output_img_path)

    # Create RTStruct with reference DICOM series
    rtstruct = RTStructBuilder.create_from(dicom_series_path, rtstruct_path)

    # Get list of structures
    x, y, z = image.GetSize()
    fused_mask = np.zeros((z, y, x), dtype=np.uint8)
    final_mask = np.zeros((z, y, x), dtype=np.uint8)
    for i, roi in enumerate(rtstruct.ds.StructureSetROISequence):
        contour_sequence = ds_helper.get_contour_sequence_by_roi_number(
            rtstruct.ds, roi.ROINumber
        )
        mask = image_helper.create_series_mask_from_contour_sequence(
            rtstruct.series_data, contour_sequence
        )
        mask = np.transpose(mask, (2, 1, 0))
        fused_mask[mask > 0] = i + 1
    fused_mask = np.flip(fused_mask, axis=2)
    fused_mask = np.rot90(fused_mask, axes=(1,2))
    z, y1, x1 = fused_mask.shape
    y, x = min(y, y1), min(x, x1)
    final_mask[:, :y, :x] = fused_mask[:, :y, :x]

    # Convert to NIfTI
    mask_image = sitk.GetImageFromArray(final_mask.astype(np.uint8))
    mask_image.SetSpacing(image.GetSpacing())
    mask_image.SetOrigin(image.GetOrigin())
    mask_image.SetDirection(image.GetDirection())
    sitk.WriteImage(mask_image, output_seg_path)


def process_CC(input_dir, output_dir):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    df = pd.read_csv(f"{input_dir}/metadata.csv")
    df = df[df['Series Description'].isin(['SAG T2', 'SAG T2 TSE', 'Sag T2 TSER', 'Sag T2', 'ROI'])]
    subjects_ids = df['Subject ID'].unique().tolist()
    logging.info(f"Found {len(subjects_ids)} cases")
    logging.info(f"Coping to {output_dir}")
    for subject_id in subjects_ids:
        df_subject = df[df['Subject ID'] == subject_id]
        df_subject.loc[:, 'Study Date'] = df_subject['Study Date'].str.replace('-', '/')
        df_subject.loc[:, 'Study Date'] = pd.to_datetime(df_subject['Study Date'], format='%m/%d/%Y')
        df_subject = df_subject.sort_values(by='Study Date')
        study_ids = df_subject['Study UID'].unique().tolist()
        for i, study_id in enumerate(study_ids):
            logging.info(f"Converting {subject_id} MR{i+1} to NIFTI")
            df_mr = df_subject[df_subject['Study UID'] == study_id]
            df_t2 = df_mr[df_mr['Series Description'] != 'ROI']
            df_roi = df_mr[df_mr['Series Description'] == 'ROI']
            if not df_t2.empty and not df_roi.empty:
                sag_t2_path = df_t2['File Location'].tolist()[0]
                sag_t2_path = f"{input_dir}/{sag_t2_path}"
                roi_path = df_roi['File Location'].tolist()[0]
                roi_path = f"{input_dir}/{roi_path}/1-1.dcm"
                save_img_path = f"{img_dir}/{subject_id}_MR{i+1}_SAG_T2.nii.gz"
                save_seg_path = f"{lab_dir}/{subject_id}_MR{i+1}_SAG_T2_seg.nii.gz"
                CC_DICOMRTSTUCT2NIFTI(sag_t2_path, roi_path, save_img_path, save_seg_path)
            else:
                logging.info(f"{subject_id} does not have MR{i+1} SAG T2")
    logging.info("Done!")


def process_FLARE23(input_dir, output_dir, dataset):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    df = pd.read_excel(f"{input_dir}/FLARE23-DataInfoTraining.xlsx")
    df = df[df['Dataset'] == dataset]
    cases_names = df['public_name'].tolist()
    logging.info(f"Found {len(cases_names)} cases for {dataset}")
    logging.info(f"Coping to {output_dir}")
    for case_name in cases_names:
        case_id = f"{case_name}".replace("_0000.nii.gz", "")
        img_path = f"{input_dir}/imagesTr2200/{case_name}"
        lab_path = f"{input_dir}/labelsTr2200/{case_id}.nii.gz"
        if os.path.exists(img_path) and os.path.exists(lab_path):
            logging.info(img_path, lab_path)
            img_save_path = f"{img_dir}/{case_id}.nii.gz"
            lab_save_path = f"{lab_dir}/{case_id}_seg.nii.gz"
            shutil.copy(img_path, img_save_path)
            shutil.copy(lab_path, lab_save_path)
        else:
            logging.info(f"No segmentation for {case_id}")
    logging.info("Done!")

def process_MSD(input_dir, output_dir, task):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    data_json = f"{input_dir}/{task}/dataset.json"
    with open(data_json, 'r') as f:
        data_info = json.load(f)
    cases_paths = data_info["training"]
    logging.info(f"Found {len(cases_paths)} cases for {task}")
    logging.info(f"Coping to {output_dir}")
    for case_path in cases_paths:
        case_id = pathlib.Path(case_path["image"]).name.replace(".nii.gz", "")
        img_path = f"{input_dir}/{task}/imagesTr/{case_id}.nii.gz"
        lab_path = f"{input_dir}/{task}/labelsTr/{case_id}.nii.gz"
        if os.path.exists(img_path) and os.path.exists(lab_path):
            logging.info(img_path, lab_path)
            img_save_path = f"{img_dir}/{case_id}.nii.gz"
            lab_save_path = f"{lab_dir}/{case_id}_seg.nii.gz"
            shutil.copy(img_path, img_save_path)
            shutil.copy(lab_path, lab_save_path)
        else:
            logging.info(f"No segmentation for {case_id}")
    logging.info("Done!")

def NSCLC_DICOMRTSTUCT2NIFTI(dicom_series_path, rtstruct_path, output_img_path, output_seg_path):
    # Read the image series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Write as NIfTI
    sitk.WriteImage(image, output_img_path)

    # Create RTStruct with reference DICOM series
    rtstruct = RTStructBuilder.create_from(dicom_series_path, rtstruct_path)
    roi_names = rtstruct.get_roi_names()
    # logging.info("Available ROIs:", roi_names)

    # Get list of structures
    x, y, z = image.GetSize()
    fused_mask = np.zeros((z, y, x), dtype=np.uint8)
    final_mask = np.zeros((z, y, x), dtype=np.uint8)
    lab_dict = {'Lung': 1, 'Heart': 2, 'Esophagus': 3, 'Spinal-Cord': 4, 'GTV': 5}
    for roi in rtstruct.ds.StructureSetROISequence:
        roi_name = roi.ROIName
        contour_sequence = ds_helper.get_contour_sequence_by_roi_number(
            rtstruct.ds, roi.ROINumber
        )
        mask = image_helper.create_series_mask_from_contour_sequence(
            rtstruct.series_data, contour_sequence
        )
        mask = np.transpose(mask, (2, 1, 0))
        if 'GTV' in f"{roi_name}".upper():
            fused_mask[mask > 0] = lab_dict['GTV']
        elif 'lung' in f"{roi_name}".lower():
            fused_mask[mask > 0] = lab_dict['Lung']
        elif roi_name in list(lab_dict.keys()):
            fused_mask[mask > 0] = lab_dict[roi_name]
        else:
            logging.info(f"{roi_name} is not in given {lab_dict}")
    fused_mask = np.flip(fused_mask, axis=2)
    fused_mask = np.rot90(fused_mask, axes=(1,2))
    z, y1, x1 = fused_mask.shape
    y, x = min(y, y1), min(x, x1)
    final_mask[:, :y, :x] = fused_mask[:, :y, :x]

    # Convert to NIfTI
    mask_image = sitk.GetImageFromArray(final_mask.astype(np.uint8))
    mask_image.SetSpacing(image.GetSpacing())
    mask_image.SetOrigin(image.GetOrigin())
    mask_image.SetDirection(image.GetDirection())
    sitk.WriteImage(mask_image, output_seg_path)

def process_NSCLC(input_dir, output_dir):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    df = pd.read_csv(f"{input_dir}/metadata.csv")
    df = df[df['Modality'].isin(['RTSTRUCT', 'CT'])]
    subjects_ids = df['Subject ID'].unique().tolist()
    logging.info(f"Found {len(subjects_ids)} cases")
    logging.info(f"Coping to {output_dir}")
    for subject_id in subjects_ids:
        df_subject = df[df['Subject ID'] == subject_id]
        logging.info(f"Converting {subject_id} DICOM to NIFTI")
        df_ct = df_subject[df_subject['Modality'] == 'CT']
        df_roi = df_subject[df_subject['Modality'] == 'RTSTRUCT']
        if not df_ct.empty and not df_roi.empty:
            ct_path = df_ct['File Location'].tolist()[0]
            ct_path = f"{input_dir}/{ct_path}"
            roi_path = df_roi['File Location'].tolist()[0]
            roi_path = f"{input_dir}/{roi_path}/1-1.dcm"
            save_img_path = f"{img_dir}/{subject_id}_CT.nii.gz"
            save_seg_path = f"{lab_dir}/{subject_id}_CT_seg.nii.gz"
            try:
                NSCLC_DICOMRTSTUCT2NIFTI(ct_path, roi_path, save_img_path, save_seg_path)
            except:
                logging.info(f"Error loading RTSTRUC: {roi_path}")
                os.remove(save_img_path)
                logging.info(f"{save_img_path} has been removed")
                continue
        else:
            logging.info(f"{subject_id} does not have annotation")
    logging.info("Done!")

def process_KiTS23(input_dir, output_dir):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    cases_paths = list(pathlib.Path(input_dir).glob('case*'))
    logging.info(f"Found {len(cases_paths)} cases")
    logging.info(f"Coping to {output_dir}")
    for case_path in cases_paths:
        case_id = case_path.name
        img_path = f"{case_path}/imaging.nii.gz"
        lab_path = f"{case_path}/segmentation.nii.gz"
        if os.path.exists(img_path) and os.path.exists(lab_path):
            logging.info(img_path, lab_path)
            save_img_path = f"{img_dir}/{case_id}.nii.gz"
            save_lab_path = f"{lab_dir}/{case_id}_seg.nii.gz"
            shutil.copy(img_path, save_img_path)
            shutil.copy(lab_path, save_lab_path)
        else:
            logging.info(f"No segmentation for {case_id}")
    logging.info("Done!")

def process_OV04(input_dir, output_dir):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    cases_paths = list(pathlib.Path(f"{input_dir}/images").glob('case*'))
    logging.info(f"Found {len(cases_paths)} cases")
    logging.info(f"Coping to {output_dir}")
    for case_path in cases_paths:
        case_id = case_path.name.replace('.nii.gz', '')
        img_path = case_path
        lab_path = f"{case_path}".replace('/images/', '/segmentations/')
        if os.path.exists(img_path) and os.path.exists(lab_path):
            logging.info(img_path, lab_path)
            save_img_path = f"{img_dir}/{case_id}.nii.gz"
            save_lab_path = f"{lab_dir}/{case_id}_seg.nii.gz"
            shutil.copy(img_path, save_img_path)
            shutil.copy(lab_path, save_lab_path)
        else:
            logging.info(f"No segmentation for {case_id}")
    logging.info("Done!")

def normalize_roi_name(roi_name):
    """Normalize ROI name for filenames."""
    roi_name = roi_name.strip()
    roi_name = re.sub(r"\s+", "-", roi_name)
    roi_name = re.sub(r"-+", "-", roi_name)
    roi_name = re.sub(r"[^A-Za-z0-9\-]", "", roi_name)
    return roi_name

def dicom_series_to_nifti(dicom_series_path, output_img_path):
    """
    Convert a DICOM series to NIfTI and return the SimpleITK image.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_series_path))

    if len(dicom_names) == 0:
        raise RuntimeError(f"No DICOM files found in {dicom_series_path}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    sitk.WriteImage(image, str(output_img_path))
    return image


def create_empty_mask(reference_image, output_mask_path):
    """
    Create an all-zero mask matching the reference image geometry.
    """
    mask = sitk.Image(reference_image.GetSize(), sitk.sitkUInt8)
    mask.CopyInformation(reference_image)

    sitk.WriteImage(mask, str(output_mask_path))
    return str(output_mask_path)


def CPTAC_DICOMRTSTRUCT2NIFTI(
    dicom_series_path,
    rtstruct_path,
    output_img_dir,
    output_seg_dir,
    annotation_type,
):
    """
    Returns:
        (image_path, list_of_mask_paths)

    Behavior:
        - DICOM missing      -> return None, []
        - No Findings        -> image + empty mask
        - Segmentation valid -> image + masks
        - Segmentation bad   -> image + []
    """

    dicom_series_path = Path(dicom_series_path)
    rtstruct_path = Path(rtstruct_path)

    output_img_dir = Path(output_img_dir)
    output_seg_dir = Path(output_seg_dir)

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_seg_dir.mkdir(parents=True, exist_ok=True)

    output_img_path = output_img_dir / "image.nii.gz"

    # ==========================================================
    # DICOM series must exist
    # ==========================================================
    if (
        not dicom_series_path.exists()
        or not dicom_series_path.is_dir()
        or len(list(dicom_series_path.glob("*.dcm"))) == 0
    ):
        logging.info(
            f"⚠️ Skipping case because DICOM series is missing or empty:\n"
            f"  DICOM: {dicom_series_path}\n"
            f"  RTSTRUCT: {rtstruct_path}"
        )
        return None, []

    # ==========================================================
    # RTSTRUCT exists -> try conversion
    # ==========================================================
    if rtstruct_path.exists():

        mask_dir = output_seg_dir / "_tmp_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        try:
            dcmrtstruct2nii(
                dicom_file=str(dicom_series_path),
                rtstruct_file=str(rtstruct_path),
                output_path=str(mask_dir),
                mask_foreground_value=1,
            )

            generated_image = mask_dir / "image.nii.gz"

            if generated_image.exists():
                shutil.move(
                    str(generated_image),
                    str(output_img_path)
                )
            else:
                raise RuntimeError(
                    "image.nii.gz not produced by dcmrtstruct2nii"
                )

            mask_paths = []

            for mask_file in mask_dir.glob("mask*.nii.gz"):
                dst = output_seg_dir / mask_file.name
                shutil.move(str(mask_file), str(dst))
                mask_paths.append(str(dst))

            shutil.rmtree(mask_dir, ignore_errors=True)

            # --------------------------------------------------
            # Successful segmentation masks
            # --------------------------------------------------
            if len(mask_paths) > 0:
                return str(output_img_path), mask_paths

            # --------------------------------------------------
            # No masks produced
            # --------------------------------------------------
            if annotation_type == "No Findings":

                image = sitk.ReadImage(str(output_img_path))

                empty_mask_path = output_seg_dir / "mask.nii.gz"

                create_empty_mask(
                    reference_image=image,
                    output_mask_path=empty_mask_path,
                )

                return str(output_img_path), [str(empty_mask_path)]

            logging.info(
                f"⚠️ Segmentation RTSTRUCT produced no masks:\n"
                f"  {rtstruct_path}"
            )

            return str(output_img_path), []

        except Exception as e:

            logging.info(
                f"⚠️ RTSTRUCT conversion failed:\n"
                f"  RTSTRUCT: {rtstruct_path}\n"
                f"  Error: {e}"
            )

            shutil.rmtree(mask_dir, ignore_errors=True)

    # ==========================================================
    # Fallback:
    #   - RTSTRUCT missing
    #   - RTSTRUCT conversion failed
    # ==========================================================
    try:
        image = dicom_series_to_nifti(
            dicom_series_path=dicom_series_path,
            output_img_path=output_img_path,
        )
    except Exception as e:
        logging.info(
            f"⚠️ Failed to convert DICOM series:\n"
            f"  DICOM: {dicom_series_path}\n"
            f"  Error: {e}"
        )
        return None, []

    # ----------------------------------------------------------
    # No Findings -> create empty mask
    # ----------------------------------------------------------
    if annotation_type == "No Findings":

        empty_mask_path = output_seg_dir / "mask.nii.gz"

        create_empty_mask(
            reference_image=image,
            output_mask_path=empty_mask_path,
        )

        return str(output_img_path), [str(empty_mask_path)]

    # ----------------------------------------------------------
    # Segmentation case -> image only, no mask
    # ----------------------------------------------------------
    logging.info(
        f"⚠️ Segmentation case without valid RTSTRUCT:\n"
        f"  {rtstruct_path}"
    )

    return str(output_img_path), []


def process_CPTAC(input_meta_dir, input_img_dir, output_dir, json_output="output_index2.json"):
    input_meta_dir = Path(input_meta_dir)
    output_dir = Path(output_dir)

    # Dictionary to collect patient → list of images
    results = {}

    csv_files = list(input_meta_dir.glob("*.csv"))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = df[df["Annotation Type"].isin(["Segmentation", "No Findings"])]
        df = df[df["ReferencedSeriesModality"].isin(['CT', 'MR'])]

        logging.info(f"Found {len(df)} segmentation annotations in {csv_file.name}")

        project_id = "-".join(csv_file.name.split("_")[:2])
        subjects = df["SeriesInstanceUID"].unique().tolist()

        for subject_id in subjects:
            df_sub = df[df["SeriesInstanceUID"] == subject_id]
            patient_id = df_sub["PatientID"].iloc[0]
            modality = df_sub["ReferencedSeriesModality"].iloc[0]
            img_id = df_sub["ReferencedSeriesInstanceUID"].iloc[0]

            logging.info(f"\nProcessing Patient {patient_id}, Instance {subject_id}")

            # Input paths
            img_path = Path(input_img_dir) / modality / project_id / img_id
            rt_path = Path(input_img_dir) / "RTSTRUCT" / project_id / subject_id / "1-1.dcm"

            # Output paths
            img_outdir = output_dir / "images" / modality / project_id / img_id
            seg_outdir = output_dir / "labels" / modality / project_id / subject_id
            img_outdir.mkdir(parents=True, exist_ok=True)
            seg_outdir.mkdir(parents=True, exist_ok=True)

            # Output filenames
            annotation_type = df_sub["Annotation Type"].iloc[0]

            img_out, seg_out = CPTAC_DICOMRTSTRUCT2NIFTI(
                img_path,
                rt_path,
                img_outdir,
                seg_outdir,
                annotation_type=annotation_type,
            )

            if img_out is None:
                continue

            if len(seg_out) != 1:
                continue

            # Convert to relative paths
            img_rel = str(Path(img_out).relative_to(output_dir))
            seg_rel = str(Path(seg_out[0]).relative_to(output_dir))

            # Store in results structure
            if patient_id not in results:
                results[patient_id] = []

            results[patient_id].append({
                "image": img_rel,
                "label": seg_rel
            })

    # Save JSON summary
    json_path = output_dir / json_output
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info("\nDone!")
    logging.info(f"JSON index saved to: {json_path}")

CLASS_MAP = {
    "em": 1,
    "cy": 2,
    "ov": 3,
    "ut": 4,
    "cds": 5,
}

# Priority: earlier = stronger
PRIORITY = ["em", "cy", "ov", "ut", "cds"]


def load_nifti(path):
    nii = nib.load(str(path))
    return nii.get_fdata(), nii.affine


def fuse_raters(volumes):
    """Majority voting"""
    vols = np.stack(volumes, axis=0)
    return (np.sum(vols, axis=0) >= (len(volumes) / 2)).astype(np.uint8)


def process_UT_EndoMRI(input_dir, output_dir):
    from collections import defaultdict

    img_dir = pathlib.Path(output_dir) / "images"
    lab_dir = pathlib.Path(output_dir) / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    center_paths = list(pathlib.Path(input_dir).glob('D*'))
    logging.info(f"Found {len(center_paths)} centers")

    for center in center_paths:
        case_paths = list(center.glob('D*'))
        logging.info(f"Found {len(case_paths)} cases in {center.name}")

        for case_path in case_paths:
            nii_paths = list(case_path.glob('*.nii.gz'))

            scans = [p for p in nii_paths if any(t in p.name for t in ['T1', 'T1FS', 'T2', 'T2FS'])]
            annotations = [p for p in nii_paths if any(t in p.name for t in CLASS_MAP.keys())]

            # Load scans
            scan_data = {}
            for scan_path in scans:
                data, affine = load_nifti(scan_path)
                scan_data[scan_path] = {
                    "data": data,
                    "shape": data.shape,
                    "affine": affine
                }

            # Group annotations by class
            ann_by_class = defaultdict(list)
            for ann_path in annotations:
                for cls in CLASS_MAP:
                    if f"_{cls}_" in ann_path.name:
                        ann_by_class[cls].append(ann_path)

            # Process each scan
            for scan_path, scan_info in scan_data.items():
                shape = scan_info["shape"]

                # Store fused masks per class
                fused_masks = {}

                for cls, ann_paths in ann_by_class.items():
                    matched_vols = []

                    for ann_path in ann_paths:
                        ann_data, _ = load_nifti(ann_path)

                        if ann_data.shape == shape:
                            matched_vols.append(ann_data > 0)

                    if len(matched_vols) > 0:
                        fused_masks[cls] = fuse_raters(matched_vols)

                # 🔥 Priority-based label assignment (NO overwrite)
                label_volume = np.zeros(shape, dtype=np.uint8)

                for cls in PRIORITY:
                    if cls not in fused_masks:
                        continue

                    mask = fused_masks[cls] > 0

                    # Only fill empty voxels → prevents overwrite
                    label_volume[(mask) & (label_volume == 0)] = CLASS_MAP[cls]

                # Save outputs
                scan_name = scan_path.stem.replace(".nii", "")

                img_save = img_dir / f"{scan_name}.nii.gz"
                lab_save = lab_dir / f"{scan_name}_seg.nii.gz"

                nib.save(
                    nib.Nifti1Image(scan_info["data"], scan_info["affine"]),
                    str(img_save)
                )

                nib.save(
                    nib.Nifti1Image(label_volume, scan_info["affine"]),
                    str(lab_save)
                )

    logging.info("Done!")



# -------------------------------------------------
# 1. Unified DICOM slice loader
# -------------------------------------------------
SliceInfo = namedtuple("SliceInfo", ["file", "z", "sop", "acq"])

def load_ct_slices(files):
    slices = []

    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)

            if not hasattr(ds, "ImagePositionPatient"):
                continue

            slices.append(
                SliceInfo(
                    file=f,
                    z=float(ds.ImagePositionPatient[2]),
                    sop=getattr(ds, "SOPInstanceUID", None),
                    acq=getattr(ds, "AcquisitionNumber", None),
                )
            )

        except Exception:
            continue

    return slices


# -------------------------------------------------
# 2. Unified z-clustering
# -------------------------------------------------
def split_z_clusters(slice_info, gap_factor=2.5):
    if len(slice_info) < 3:
        return [slice_info]

    slice_info = sorted(slice_info, key=lambda x: x.z)

    zs = np.array([s.z for s in slice_info])
    dz = np.abs(np.diff(zs))

    median_spacing = np.median(dz)
    if median_spacing <= 0:
        return [slice_info]

    breaks = np.where(dz > gap_factor * median_spacing)[0]

    clusters = []
    start = 0

    for b in breaks:
        clusters.append(slice_info[start:b + 1])
        start = b + 1

    clusters.append(slice_info[start:])
    return clusters


# -------------------------------------------------
# 3. SEG SOP extraction (cleaned + safer)
# -------------------------------------------------
def extract_seg_sops(seg):
    sops = set()

    frames = getattr(seg, "PerFrameFunctionalGroupsSequence", None)
    if not frames:
        return sops

    for frame in frames:
        try:
            deriv_seq = getattr(frame, "DerivationImageSequence", None)
            if not deriv_seq:
                continue

            for deriv in deriv_seq:
                src_seq = getattr(deriv, "SourceImageSequence", None)
                if not src_seq:
                    continue

                for src in src_seq:
                    sop = getattr(src, "ReferencedSOPInstanceUID", None)
                    if sop:
                        sops.add(sop)

        except Exception:
            continue

    return sops


# -------------------------------------------------
# 4. Unified cluster scoring
# -------------------------------------------------
def score_clusters(
    clusters,
    seg_sops=None,
    seg_z_range=None,
    use_size=True
):
    best = None
    best_score = (-1, -1, -1)

    for cluster in clusters:
        sops = {c.sop for c in cluster if c.sop}

        sop_hits = len(sops & seg_sops) if seg_sops else 0

        zs = [c.z for c in cluster]
        zmin, zmax = min(zs), max(zs)

        overlap = 0
        if seg_z_range:
            szmin, szmax = seg_z_range
            overlap = max(0, min(zmax, szmax) - max(zmin, szmin))

        size = len(cluster)

        score = (sop_hits, overlap, size if use_size else 0)

        if score > best_score:
            best_score = score
            best = cluster

    return best


# -------------------------------------------------
# 5. Main: SEG-based CT selection (clean)
# -------------------------------------------------
def filter_ct_by_seg_extent(ct_files, seg_path, gap_factor=2.5):
    seg = pydicom.dcmread(str(seg_path))

    seg_sops = extract_seg_sops(seg)

    seg_z = []
    frames = getattr(seg, "PerFrameFunctionalGroupsSequence", None)

    if frames:
        for f in frames:
            try:
                ipp = f.PlanePositionSequence[0].ImagePositionPatient
                seg_z.append(float(ipp[2]))
            except Exception:
                continue

    seg_z_range = (min(seg_z), max(seg_z)) if seg_z else None

    slices = load_ct_slices(ct_files)
    clusters = split_z_clusters(slices, gap_factor)

    best = score_clusters(
        clusters,
        seg_sops=seg_sops,
        seg_z_range=seg_z_range
    )

    if not best:
        return ct_files, None

    filtered = [c.file for c in best]

    # majority acquisition vote
    acq_counter = Counter(c.acq for c in best if c.acq)
    best_acq = acq_counter.most_common(1)[0][0] if acq_counter else None

    return filtered, best_acq


# -------------------------------------------------
# 6. RTSTRUCT-style selection (simplified)
# -------------------------------------------------
def keep_best_z_cluster(files, referenced_sops, gap_factor=2.5):
    slices = load_ct_slices(files)

    if len(slices) < 3:
        return files

    clusters = split_z_clusters(slices, gap_factor)

    best = max(
        clusters,
        key=lambda c: (
            len({x.sop for x in c if x.sop} & referenced_sops),
            len(c)
        )
    )

    return [c.file for c in best]

def select_seg_or_rtstruct_acquisition(dicom_series_path, seg_or_rtstruct_path):
    """
    Robust acquisition selector supporting BOTH:
        - RTSTRUCT (ROIContourSequence → SOP-based mapping)
        - DICOM SEG (ReferencedSeriesInstanceUID → CT series mapping)

    Returns:
        temp_dir, best_acq, n_slices
    """

    dicom_series_path = Path(dicom_series_path)

    # =========================================================
    # STEP 1: Index CT series (by Acquisition + Series UID)
    # =========================================================
    acquisition_files = defaultdict(list)
    sop_to_acq = {}
    series_to_files = defaultdict(list)
    series_to_acq = {}

    for f in dicom_series_path.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)

            acq = getattr(ds, "AcquisitionNumber", None)
            sop = getattr(ds, "SOPInstanceUID", None)
            sid = getattr(ds, "SeriesInstanceUID", None)

            acquisition_files[acq].append(f)

            if sop:
                sop_to_acq[sop] = acq

            if sid:
                series_to_files[sid].append(f)
                series_to_acq[sid] = acq

        except Exception:
            continue

    if not acquisition_files:
        raise RuntimeError("No valid DICOM files found in CT series.")

    # =========================================================
    # STEP 2: Load segmentation object
    # =========================================================
    seg = pydicom.dcmread(str(seg_or_rtstruct_path))

    is_rtstruct = hasattr(seg, "ROIContourSequence")
    is_seg = hasattr(seg, "SegmentSequence")

    referenced_sops = set()
    referenced_series_uids = set()
    contour_points = []

    # =========================================================
    # CASE 1: RTSTRUCT
    # =========================================================
    if is_rtstruct:

        for roi in seg.ROIContourSequence:

            if not hasattr(roi, "ContourSequence"):
                continue

            for contour in roi.ContourSequence:

                # SOP-linked mode
                if hasattr(contour, "ContourImageSequence"):
                    for img in contour.ContourImageSequence:
                        if hasattr(img, "ReferencedSOPInstanceUID"):
                            referenced_sops.add(img.ReferencedSOPInstanceUID)

                # Geometry fallback (optional)
                if hasattr(contour, "ContourData"):
                    pts = np.array(contour.ContourData).reshape(-1, 3)
                    contour_points.append(pts)

    # =========================================================
    # CASE 2: DICOM SEG (IMPORTANT FIX HERE)
    # =========================================================
    elif is_seg:

        # SEG → CT linkage via SeriesInstanceUID (correct DICOM behavior)
        if hasattr(seg, "ReferencedSeriesSequence"):
            for rs in seg.ReferencedSeriesSequence:
                if hasattr(rs, "SeriesInstanceUID"):
                    referenced_series_uids.add(rs.SeriesInstanceUID)

        if not referenced_series_uids:
            raise ValueError("SEG has no ReferencedSeriesSequence (cannot locate CT)")

        contour_points = []  # not used for SEG

    else:
        raise ValueError("Input file is neither RTSTRUCT nor DICOM SEG")

    # =========================================================
    # STEP 3: RTSTRUCT → SOP-based matching
    # =========================================================
    acq_counter = Counter()
    selected_files = None
    best_acq = None

    if is_rtstruct and referenced_sops:

        for sop in referenced_sops:
            if sop in sop_to_acq:
                acq_counter[sop_to_acq[sop]] += 1

        if acq_counter:
            best_acq, _ = acq_counter.most_common(1)[0]
            selected_files = acquisition_files[best_acq]

            # only keep the largest contiguous block if there are multiple blocks
            selected_files = keep_best_z_cluster(
                selected_files,
                referenced_sops
            )

            logging.info(f"[RTSTRUCT SOP MODE] Selected acquisition {best_acq} "
                  f"with {len(selected_files)} slices")

    # =========================================================
    # STEP 4: SEG → SeriesInstanceUID matching (FIXED LOGIC)
    # =========================================================
    if is_seg:

        best_series = None

        for uid in referenced_series_uids:
            if uid in series_to_files:
                best_series = uid
                break

        if best_series is None:
            raise ValueError(
                "Could not match SEG SeriesInstanceUID to CT series"
            )

        raw_files = series_to_files[best_series]

        selected_files, best_acq = filter_ct_by_seg_extent(
            raw_files,
            seg_or_rtstruct_path
        )

        logging.info(f"[SEG MODE] Selected CT series {best_series} "
              f"({len(selected_files)} slices)")

    # =========================================================
    # STEP 5: Fallback (RTSTRUCT only)
    # =========================================================
    if selected_files is None:

        if contour_points:

            contour_points = np.vstack(contour_points)

            rt_zmin, rt_zmax = (
                float(np.min(contour_points[:, 2])),
                float(np.max(contour_points[:, 2]))
            )

            best_overlap = -1

            for acq, files in acquisition_files.items():

                zs = []
                for f in files:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                    zs.append(float(ds.ImagePositionPatient[2]))

                ct_zmin, ct_zmax = min(zs), max(zs)

                overlap = max(
                    0,
                    min(ct_zmax, rt_zmax) - max(ct_zmin, rt_zmin)
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_acq = acq

        else:
            best_acq = max(acquisition_files.items(),
                           key=lambda x: len(x[1]))[0]

        selected_files = acquisition_files[best_acq]

        logging.info(f"[FALLBACK MODE] Selected acquisition {best_acq} "
              f"with {len(selected_files)} slices")

    # =========================================================
    # STEP 6: Create temp CT series
    # =========================================================
    temp_dir = Path(tempfile.mkdtemp(prefix="seg_rtstruct_series_"))

    for f in selected_files:
        shutil.copy2(f, temp_dir / f.name)

    return temp_dir, best_acq, len(selected_files)

def image_fingerprint(img: sitk.Image) -> str:
    arr = sitk.GetArrayFromImage(img)

    h = hashlib.sha256()

    # voxel content (core identity)
    h.update(arr.tobytes())

    # geometry (critical for medical imaging)
    h.update(np.array(img.GetSpacing(), dtype=np.float64).tobytes())
    h.update(np.array(img.GetOrigin(), dtype=np.float64).tobytes())
    h.update(np.array(img.GetDirection(), dtype=np.float64).tobytes())

    return h.hexdigest()[:16]

def EAY131_DICOMRTSTRUCT2NIFTI(
    dicom_series_path,
    rtstruct_path,
    output_img_dir,
    output_seg_dir,
    annotation_type,
):
    """
    Returns:
        (image_path, list_of_mask_paths)

    Behavior:
        - DICOM missing      -> return None, []
        - No Findings        -> image + empty mask
        - Segmentation valid -> image + masks
        - Segmentation bad   -> image + []
    """

    dicom_series_path = Path(dicom_series_path)
    rtstruct_path = Path(rtstruct_path)

    output_img_dir = Path(output_img_dir)
    output_seg_dir = Path(output_seg_dir)

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_seg_dir.mkdir(parents=True, exist_ok=True)

    output_img_path = output_img_dir / "image.nii.gz"

    # ==========================================================
    # DICOM series must exist
    # ==========================================================
    if (
        not dicom_series_path.exists()
        or not dicom_series_path.is_dir()
        or len(list(dicom_series_path.glob("*.dcm"))) == 0
    ):
        logging.info(
            f"⚠️ Skipping case because DICOM series is missing or empty:\n"
            f"  DICOM: {dicom_series_path}\n"
            f"  RTSTRUCT: {rtstruct_path}"
        )
        return None, []

    # ==========================================================
    # RTSTRUCT exists -> try conversion with Platipy
    # ==========================================================
    if rtstruct_path.exists():
        ds = pydicom.dcmread(rtstruct_path)
        modality = ds.Modality

        if not modality in ['RTSTRUCT', 'SEG']:
            raise ValueError(f'only support RTSTRUCT or SEG, but modality is {modality}!')

        mask_dir = output_seg_dir / "_tmp_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        selected_series_dir = None

        try:

            (
                selected_series_dir,
                selected_acq,
                n_slices,
            ) = select_seg_or_rtstruct_acquisition(
                dicom_series_path,
                rtstruct_path,
            )

            logging.info(
                f"Using acquisition {selected_acq} "
                f"({n_slices} slices)"
            )

            try:
                if modality == "RTSTRUCT":

                    # ======================================================
                    # RTSTRUCT → Platipy pipeline (image + masks)
                    # ======================================================
                    convert_rtstruct(
                        dcm_img=str(selected_series_dir),
                        dcm_rt_file=str(rtstruct_path),
                        prefix="mask_",
                        output_dir=str(mask_dir),
                        output_img=str(output_seg_dir / "image"),

                    )

                else:

                    logging.info("🧊 Detected DICOM SEG — using direct voxel conversion")

                    # ======================================================
                    # STEP 1: ALWAYS write CT NIfTI (selected series)
                    # ======================================================
                    ct_nifti_path = output_seg_dir / "image.nii.gz"

                    ct_image = dicom_series_to_nifti(
                        dicom_series_path=selected_series_dir,
                        output_img_path=ct_nifti_path,
                    )

                    # ======================================================
                    # STEP 2: Convert labels → binary masks
                    # ======================================================
                    ct_image = sitk.ReadImage(str(ct_nifti_path))
                    seg_img = sitk.ReadImage(str(rtstruct_path))

                    # Resample SEG into CT geometry
                    seg_resampled = sitk.Resample(
                        seg_img,
                        ct_image,                    # reference image
                        sitk.Transform(),
                        sitk.sitkNearestNeighbor,    # NEVER linear for labels
                        0,
                        seg_img.GetPixelID()
                    )

                    out_path = mask_dir / f"mask.nii.gz"
                    sitk.WriteImage(seg_resampled, str(out_path))

                generated_image_path = output_seg_dir / "image.nii.gz"

                if not generated_image_path.exists():
                    # generate image ourselves
                    image = dicom_series_to_nifti(
                        dicom_series_path=selected_series_dir,
                        output_img_path=generated_image_path,
                    )

                image = sitk.ReadImage(str(generated_image_path))
                series_key = image_fingerprint(image)
                output_img_path = output_img_dir / f"image_{series_key}.nii.gz"

                shutil.move(
                    str(generated_image_path),
                    str(output_img_path),
                )

                mask_paths = []

                for mask_file in mask_dir.glob("*.nii.gz"):
                    dst = output_seg_dir / mask_file.name

                    shutil.move(
                        str(mask_file),
                        str(dst),
                    )

                    mask_paths.append(str(dst))

                shutil.rmtree(mask_dir, ignore_errors=True)

                if len(mask_paths) > 0:
                    return str(output_img_path), mask_paths

                if annotation_type == "No Findings":

                    image = sitk.ReadImage(
                        str(output_img_path)
                    )

                    empty_mask_path = (
                        output_seg_dir / "mask.nii.gz"
                    )

                    create_empty_mask(
                        reference_image=image,
                        output_mask_path=empty_mask_path,
                    )

                    return (
                        str(output_img_path),
                        [str(empty_mask_path)],
                    )

                return str(output_img_path), []

            except Exception as e:

                logging.info(
                    f"⚠️ RTSTRUCT conversion failed:\n"
                    f"  RTSTRUCT: {rtstruct_path}\n"
                    f"  Error: {e}"
                )

                shutil.rmtree(mask_dir, ignore_errors=True)

        finally:

            if selected_series_dir is not None:
                shutil.rmtree(
                    selected_series_dir,
                    ignore_errors=True,
                )

    # ==========================================================
    # Fallback:
    #   - RTSTRUCT missing
    #   - RTSTRUCT conversion failed
    # ==========================================================
    try:
        image = dicom_series_to_nifti(
            dicom_series_path=dicom_series_path,
            output_img_path=output_img_path,
        )
    except Exception as e:
        logging.info(
            f"⚠️ Failed to convert DICOM series:\n"
            f"  DICOM: {dicom_series_path}\n"
            f"  Error: {e}"
        )
        return None, []

    # ----------------------------------------------------------
    # No Findings -> create empty mask
    # ----------------------------------------------------------
    if annotation_type == "No Findings":

        empty_mask_path = output_seg_dir / "mask.nii.gz"

        create_empty_mask(
            reference_image=image,
            output_mask_path=empty_mask_path,
        )

        return str(output_img_path), [str(empty_mask_path)]

    # ----------------------------------------------------------
    # Segmentation case -> image only, no mask
    # ----------------------------------------------------------
    logging.info(
        f"⚠️ Segmentation case without valid RTSTRUCT:\n"
        f"  {rtstruct_path}"
    )

    return str(output_img_path), []

def process_EAY131(input_meta_dir, input_img_dir, output_dir, json_output="output_index.json"):
    input_meta_dir = Path(input_meta_dir)
    output_dir = Path(output_dir)

    # Dictionary to collect patient → list of images
    results = {}

    csv_files = list(input_meta_dir.glob("*.csv"))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = df[df["AnnotationType"].isin(["Segmentation", "No Findings"])]
        df = df[df["ReferencedSeriesModality"].isin(['CT', 'MR'])]

        logging.info(f"Found {len(df)} segmentation annotations in {csv_file.name}")

        subjects = df["SeriesInstanceUID"].unique().tolist()

        for subject_id in subjects:
            df_sub = df[df["SeriesInstanceUID"] == subject_id]
            patient_id = df_sub["PatientID"].iloc[0]
            study_id = df_sub["StudyInstanceUID"].iloc[0]
            img_id = df_sub["ReferencedSeriesInstanceUID"].iloc[0]

            logging.info(f"\nProcessing Patient {patient_id}, Instance {subject_id}")

            # Input paths
            img_path = Path(input_img_dir) / "EAY131_SourceImages" / "eay131" / patient_id / study_id / img_id
            rt_path = Path(input_img_dir) / "EAY131_Segmentations" / "EAY131" / patient_id / study_id / subject_id / "1-1.dcm"

            # Output paths
            img_outdir = output_dir / "images" / patient_id / study_id / img_id
            seg_outdir = output_dir / "labels" / patient_id / study_id / subject_id
            img_outdir.mkdir(parents=True, exist_ok=True)
            seg_outdir.mkdir(parents=True, exist_ok=True)

            # Output filenames
            annotation_type = df_sub["AnnotationType"].iloc[0]

            img_out, seg_out = EAY131_DICOMRTSTRUCT2NIFTI(
                img_path,
                rt_path,
                img_outdir,
                seg_outdir,
                annotation_type=annotation_type,
            )

            if img_out is None:
                continue

            if len(seg_out) != 1:
                continue

            # Convert to relative paths
            img_rel = str(Path(img_out).relative_to(output_dir))
            seg_rel = str(Path(seg_out[0]).relative_to(output_dir))

            # Store in results structure
            if patient_id not in results:
                results[patient_id] = []

            results[patient_id].append({
                "image": img_rel,
                "label": seg_rel
            })

    # Save JSON summary
    json_path = output_dir / json_output
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info("\nDone!")
    logging.info(f"JSON index saved to: {json_path}")

if __name__ == "__main__":

    # UMD_input_dir = "./UMD"
    # UMD_output_dir = "../Pan-cancerSegmentation/UterusCancer"
    # process_UMD(UMD_input_dir, UMD_output_dir)

    # FedBCa_input_dir = "./FedBCa"
    # FedBCa_output_dir = "../TumorSegmentation/Bladder_Tumor_00"
    # process_FedBCa(FedBCa_input_dir, FedBCa_output_dir)

    # Prostate158_input_dir = "./Prostate158"
    # Prostate158_output_dir = "../Pan-cancerSegmentation/ProstateCancer_00"
    # process_Prostate158(Prostate158_input_dir, Prostate158_output_dir)

    # MAMAMIA_input_dir = "./MAMA-MIA"
    # MAMAMIA_output_dir = "../Pan-cancerSegmentation/BreastCancer_00"
    # process_MAMAMIA(MAMAMIA_input_dir, MAMAMIA_output_dir)

    # CC_input_dir = "./CC-Tumor-Heterogeneity/manifest-1744123832555"
    # CC_output_dir = "../Pan-cancerSegmentation/CervixCancer_00"
    # process_CC(CC_input_dir, CC_output_dir)

    # FLARE23_input_dir = "./FLARE23"
    # FLARE23_output_dir = "../Pan-cancerSegmentation/PancreasCancer_00"
    # process_FLARE23(FLARE23_input_dir, FLARE23_output_dir, 'MSD-Pancreas')

    # FLARE23_input_dir = "./FLARE23"
    # FLARE23_output_dir = "../Pan-cancerSegmentation/LiverCancer_00"
    # process_FLARE23(FLARE23_input_dir, FLARE23_output_dir, 'MSD-Liver')

    # FLARE23_input_dir = "./FLARE23"
    # FLARE23_output_dir = "../Pan-cancerSegmentation/ColonCancer_00"
    # process_FLARE23(FLARE23_input_dir, FLARE23_output_dir, 'MSD-Colon')

    # MSD_input_dir = "./MSD"
    # MSD_output_dir = "../Pan-cancerSegmentation/LungCancer_00"
    # process_MSD(MSD_input_dir, MSD_output_dir, "Task06_Lung")

    # NSCLC_input_dir = "./NSCLC/manifest-1744339505493"
    # NSCLC_output_dir = "../Pan-cancerSegmentation/LungCancer_01"
    # process_NSCLC(NSCLC_input_dir, NSCLC_output_dir)

    # KiTS23_input_dir = "./KiTS23"
    # KiTS23_output_dir = "../Pan-cancerSegmentation/KidneyCancer"
    # process_KiTS23(KiTS23_input_dir, KiTS23_output_dir)
    
    # OV04_input_dir = "./OV04_post_treatment"
    # OV04_output_dir = "../TumorSegmentation/Ovary_Tumor_00"
    # process_OV04(OV04_input_dir, OV04_output_dir)

    # CPTAC_input_dir = "/Users/sg2162/Datasets/CancerDatasets/CPTAC/Radiology"
    # CPTAC_output_dir = "/Users/sg2162/Datasets/CancerDatasets/CPTAC/Radiology_NIFTI"
    # CPTAC_meta_dir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/clinical/CPTAC_Annotation_Metadata_1"
    # process_CPTAC(CPTAC_meta_dir, CPTAC_input_dir, CPTAC_output_dir)

    EAY131_input_dir = "/Users/sg2162/Datasets/CancerDatasets/EAY131/Radiology"
    EAY131_output_dir = "/Users/sg2162/Datasets/CancerDatasets/EAY131/Radiology_NIFTI"
    EAY131_meta_dir = "/Users/sg2162/Datasets/CancerDatasets/EAY131/Radiology/Metadata"
    process_EAY131(EAY131_meta_dir, EAY131_input_dir, EAY131_output_dir)

    # Endo_input_dir = "/Users/sg2162/Datasets/CancerDatasets/UT-EndoMRI"
    # Endo_output_dir = "/Users/sg2162/Datasets/CancerDatasets/Endometriosis/EndoMRI"
    # process_UT_EndoMRI(Endo_input_dir, Endo_output_dir)

    # folder = "/Users/sg2162/Datasets/CancerDatasets/OV04CT/labels"
    # check_empty_nii(folder)

    # Example usage:
    # labels_folder = "/Users/sg2162/Datasets/CancerDatasets/Endometriosis/EndoMRI/labels"
    # images_folder = "/Users/sg2162/Datasets/CancerDatasets/Endometriosis/EndoMRI/images"
    # remove_empty_labels(labels_folder, images_folder)
    