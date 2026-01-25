import os
import re
import shutil
import pathlib
from pathlib import Path
import pydicom
import json
import pandas as pd
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from rt_utils import RTStructBuilder, ds_helper, image_helper
from dcmrtstruct2nii import dcmrtstruct2nii

def process_UMD(input_dir, output_dir):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    cases_paths = list(pathlib.Path(input_dir).glob('UMD*'))
    print(f"Found {len(cases_paths)} cases")
    print(f"Coping to {output_dir}")
    for case_path in cases_paths:
        case_id = case_path.name
        img_path = f"{case_path}/{case_id}_t2.nii.gz"
        lab_path = f"{case_path}/{case_id}_t2_seg.nii.gz"
        if os.path.exists(img_path) and os.path.exists(lab_path):
            print(img_path, lab_path)
            shutil.copy(img_path, img_dir)
            shutil.copy(lab_path, lab_dir)
        else:
            print(f"No segmentation for {case_id}")
    print("Done!")

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
    print(f"Found {len(center_paths)} centers")
    for center in center_paths:
        center_name = center.name
        img_paths = list(pathlib.Path(f"{center}/{center_name}/T2WI").glob('*nii.gz'))
        label_file = f"{center}/{center_name}/{center_name}_label.xlsx"
        df = pd.read_excel(label_file)
        print(f"Found {len(img_paths)} images in {center_name}")
        for img_path in img_paths:
            img_name = img_path.name
            if center_name == "Center1":
                lab_names = df[df['image'] == img_name]['mask_new'].to_list()
            else:
                lab_names = df[df['image_name'] == img_name]['mask_name'].to_list()
            img_save_name = f"{center_name}_{img_name}"
            lab_save_name = f"{img_save_name}".replace(".nii.gz", "_seg.nii.gz")
            if len(lab_names) > 1:
                print(f"Found multiple segmentations for {img_name}")
                lab_paths = [f"{center}/{center_name}/Annotation/{name}" for name in lab_names]
                save_path = f"{lab_dir}/{lab_save_name}"
                print(f"Fusing multiple segmentation into one")
                fuse_multiple_segmentations(lab_paths, save_path)
            elif len(lab_names) == 1:
                lab_path = f"{center}/{center_name}/Annotation/{lab_names[0]}"
                shutil.copy(lab_path, f"{lab_dir}/{lab_save_name}")
            else:
                print(f"No segmentation for {img_name} in {center_name}")
            shutil.copy(img_path, f"{img_dir}/{img_save_name}")
    print("Done!")

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
    print(f"Found {len(df)} cases")
    for index, row in df.iterrows():
        ID = str(row['ID']).zfill(3)
        print(f"Processing case {ID}")
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
            print(f"No segmentation for {t2}")
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
            print(f"No segmentation for {adc}")
        shutil.copy(f"{input_dir}/{adc}", save_img_path)
    print("Done!")

def process_MAMAMIA(input_dir, output_dir):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    DUKE_paths = list(pathlib.Path(f"{input_dir}/images").glob('DUKE*'))
    print(f"Found {len(DUKE_paths)} cases from DUKE dataset")
    ISPY1_paths = list(pathlib.Path(f"{input_dir}/images").glob('ISPY1*'))
    print(f"Found {len(ISPY1_paths)} cases from ISPY1 dataset")
    ISPY2_paths = list(pathlib.Path(f"{input_dir}/images").glob('ISPY2*'))
    print(f"Found {len(ISPY2_paths)} cases from ISPY2 dataset")
    NACT_paths = list(pathlib.Path(f"{input_dir}/images").glob('NACT*'))
    print(f"Found {len(NACT_paths)} cases from NACT dataset")
    cases_paths = DUKE_paths + ISPY1_paths + ISPY2_paths + NACT_paths
    print(f"Totally found {len(cases_paths)} cases")
    print(f"Coping to {output_dir}")
    for case_path in cases_paths:
        case_id = case_path.name
        img_path = f"{case_path}/{case_id}_0001.nii.gz" # the first post-contrast phase
        lab_path = f"{input_dir}/segmentations/expert/{case_id}.nii.gz"
        if os.path.exists(img_path) and os.path.exists(lab_path):
            # print(img_path, lab_path)
            shutil.copy(img_path, img_dir)
            save_lab_path = f"{lab_dir}/{case_id}_0001_seg.nii.gz"
            shutil.copy(lab_path, save_lab_path)
        else:
            print(f"No segmentation for {case_id}")
    print("Done!")

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
    print(f"Found {len(subjects_ids)} cases")
    print(f"Coping to {output_dir}")
    for subject_id in subjects_ids:
        df_subject = df[df['Subject ID'] == subject_id]
        df_subject.loc[:, 'Study Date'] = df_subject['Study Date'].str.replace('-', '/')
        df_subject.loc[:, 'Study Date'] = pd.to_datetime(df_subject['Study Date'], format='%m/%d/%Y')
        df_subject = df_subject.sort_values(by='Study Date')
        study_ids = df_subject['Study UID'].unique().tolist()
        for i, study_id in enumerate(study_ids):
            print(f"Converting {subject_id} MR{i+1} to NIFTI")
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
                print(f"{subject_id} does not have MR{i+1} SAG T2")
    print("Done!")


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
    print(f"Found {len(cases_names)} cases for {dataset}")
    print(f"Coping to {output_dir}")
    for case_name in cases_names:
        case_id = f"{case_name}".replace("_0000.nii.gz", "")
        img_path = f"{input_dir}/imagesTr2200/{case_name}"
        lab_path = f"{input_dir}/labelsTr2200/{case_id}.nii.gz"
        if os.path.exists(img_path) and os.path.exists(lab_path):
            print(img_path, lab_path)
            img_save_path = f"{img_dir}/{case_id}.nii.gz"
            lab_save_path = f"{lab_dir}/{case_id}_seg.nii.gz"
            shutil.copy(img_path, img_save_path)
            shutil.copy(lab_path, lab_save_path)
        else:
            print(f"No segmentation for {case_id}")
    print("Done!")

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
    print(f"Found {len(cases_paths)} cases for {task}")
    print(f"Coping to {output_dir}")
    for case_path in cases_paths:
        case_id = pathlib.Path(case_path["image"]).name.replace(".nii.gz", "")
        img_path = f"{input_dir}/{task}/imagesTr/{case_id}.nii.gz"
        lab_path = f"{input_dir}/{task}/labelsTr/{case_id}.nii.gz"
        if os.path.exists(img_path) and os.path.exists(lab_path):
            print(img_path, lab_path)
            img_save_path = f"{img_dir}/{case_id}.nii.gz"
            lab_save_path = f"{lab_dir}/{case_id}_seg.nii.gz"
            shutil.copy(img_path, img_save_path)
            shutil.copy(lab_path, lab_save_path)
        else:
            print(f"No segmentation for {case_id}")
    print("Done!")

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
    # print("Available ROIs:", roi_names)

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
            print(f"{roi_name} is not in given {lab_dict}")
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
    print(f"Found {len(subjects_ids)} cases")
    print(f"Coping to {output_dir}")
    for subject_id in subjects_ids:
        df_subject = df[df['Subject ID'] == subject_id]
        print(f"Converting {subject_id} DICOM to NIFTI")
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
                print(f"Error loading RTSTRUC: {roi_path}")
                os.remove(save_img_path)
                print(f"{save_img_path} has been removed")
                continue
        else:
            print(f"{subject_id} does not have annotation")
    print("Done!")

def process_KiTS23(input_dir, output_dir):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    cases_paths = list(pathlib.Path(input_dir).glob('case*'))
    print(f"Found {len(cases_paths)} cases")
    print(f"Coping to {output_dir}")
    for case_path in cases_paths:
        case_id = case_path.name
        img_path = f"{case_path}/imaging.nii.gz"
        lab_path = f"{case_path}/segmentation.nii.gz"
        if os.path.exists(img_path) and os.path.exists(lab_path):
            print(img_path, lab_path)
            save_img_path = f"{img_dir}/{case_id}.nii.gz"
            save_lab_path = f"{lab_dir}/{case_id}_seg.nii.gz"
            shutil.copy(img_path, save_img_path)
            shutil.copy(lab_path, save_lab_path)
        else:
            print(f"No segmentation for {case_id}")
    print("Done!")

def process_OV04(input_dir, output_dir):
    img_dir = pathlib.Path(output_dir) / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    lab_dir = pathlib.Path(output_dir) / "labels"
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    cases_paths = list(pathlib.Path(f"{input_dir}/images").glob('case*'))
    print(f"Found {len(cases_paths)} cases")
    print(f"Coping to {output_dir}")
    for case_path in cases_paths:
        case_id = case_path.name.replace('.nii.gz', '')
        img_path = case_path
        lab_path = f"{case_path}".replace('/images/', '/segmentations/')
        if os.path.exists(img_path) and os.path.exists(lab_path):
            print(img_path, lab_path)
            save_img_path = f"{img_dir}/{case_id}.nii.gz"
            save_lab_path = f"{lab_dir}/{case_id}_seg.nii.gz"
            shutil.copy(img_path, save_img_path)
            shutil.copy(lab_path, save_lab_path)
        else:
            print(f"No segmentation for {case_id}")
    print("Done!")

def normalize_roi_name(roi_name):
    """Normalize ROI name for filenames."""
    roi_name = roi_name.strip()
    roi_name = re.sub(r"\s+", "-", roi_name)
    roi_name = re.sub(r"-+", "-", roi_name)
    roi_name = re.sub(r"[^A-Za-z0-9\-]", "", roi_name)
    return roi_name

def CPTAC_DICOMRTSTRUCT2NIFTI(dicom_series_path, rtstruct_path, output_img_dir, output_seg_dir):
    """
    Convert a DICOM series + RTSTRUCT to NIfTI image and masks using dcmrtstruct2nii.
    
    Returns:
        (output_img_path, list_of_mask_paths)
    """
    dicom_series_path = Path(dicom_series_path)
    rtstruct_path = Path(rtstruct_path)
    output_img_dir = Path(output_img_dir)
    output_seg_dir = Path(output_seg_dir)

    # --- Ensure output dirs exist ---
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_seg_dir.mkdir(parents=True, exist_ok=True)

    # --- Convert RTSTRUCT to NIfTI masks ---
    mask_dir = output_seg_dir / "tmp_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    try:
        dcmrtstruct2nii(
            dicom_file=str(dicom_series_path),
            rtstruct_file=str(rtstruct_path),
            output_path=str(mask_dir),
            mask_foreground_value=1
        )
    except Exception as e:
        print(f"⚠️ Failed to convert RTSTRUCT: {rtstruct_path}\n  Error: {e}")
        return str(output_img_dir), []

    # --- Move or rename masks to match normalized ROI names ---
    mask_paths = []
    for f in mask_dir.glob("mask*.nii.gz"):
        new_mask_path = output_seg_dir / f.name
        shutil.move(str(f), str(new_mask_path))
        mask_paths.append(str(new_mask_path))

    # --- Copy DICOM image as NIfTI ---
    # dcmrtstruct2nii may also save the image as image.nii.gz
    possible_image = mask_dir / "image.nii.gz"
    output_img_path = output_img_dir / "image.nii.gz"
    if possible_image.exists():
        shutil.move(str(possible_image), str(output_img_path))
    else:
        # fallback: use sitk to convert DICOM to NIfTI
        import SimpleITK as sitk
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_series_path))
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        sitk.WriteImage(image, str(output_img_path))

    # --- Clean temporary mask folder ---
    try:
        mask_dir.rmdir()
    except OSError:
        pass

    return str(output_img_path), mask_paths


def process_CPTAC(input_meta_dir, input_img_dir, output_dir, json_output="output_index.json"):
    input_meta_dir = Path(input_meta_dir)
    output_dir = Path(output_dir)

    # Dictionary to collect patient → list of images
    results = {}

    csv_files = list(input_meta_dir.glob("*.csv"))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = df[df["Annotation Type"].isin(["Segmentation", "No Findings"])]
        df = df[df["ReferencedSeriesModality"].isin(['CT', 'MR'])]

        print(f"Found {len(df)} segmentation annotations in {csv_file.name}")

        project_id = "-".join(csv_file.name.split("_")[:2])
        subjects = df["SeriesInstanceUID"].unique().tolist()

        for subject_id in subjects:
            df_sub = df[df["SeriesInstanceUID"] == subject_id]
            patient_id = df_sub["PatientID"].iloc[0]
            modality = df_sub["ReferencedSeriesModality"].iloc[0]
            img_id = df_sub["ReferencedSeriesInstanceUID"].iloc[0]

            print(f"\nProcessing Patient {patient_id}, Instance {subject_id}")

            # Input paths
            img_path = Path(input_img_dir) / modality / project_id / img_id
            rt_path = Path(input_img_dir) / "RTSTRUCT" / project_id / subject_id / "1-1.dcm"

            # Output paths
            img_outdir = output_dir / "images" / modality / project_id / img_id
            seg_outdir = output_dir / "labels" / modality / project_id / subject_id
            img_outdir.mkdir(parents=True, exist_ok=True)
            seg_outdir.mkdir(parents=True, exist_ok=True)

            # Output filenames
            img_out, seg_out = CPTAC_DICOMRTSTRUCT2NIFTI(img_path, rt_path, img_outdir, seg_outdir)
            
            if len(seg_out) != 1: continue

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

    print("\nDone!")
    print(f"JSON index saved to: {json_path}")



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

    CPTAC_input_dir = "/Users/sg2162/Datasets/CancerDatasets/CPTAC/Radiology"
    CPTAC_output_dir = "/Users/sg2162/Datasets/CancerDatasets/CPTAC/Radiology_NIFTI"
    CPTAC_meta_dir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/clinical/CPTAC_Annotation_Metadata"
    process_CPTAC(CPTAC_meta_dir, CPTAC_input_dir, CPTAC_output_dir)
