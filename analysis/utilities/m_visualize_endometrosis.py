import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import imageio.v2 as imageio
import os

def rotate_90_ccw(x):
    return np.rot90(x, k=1)  # 90° anticlockwise

def load_array(path):
    """
    Supports:
    - .npy (recommended for pred/gt)
    - .png/.jpg (image or mask)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        return np.load(path)

    # image files
    img = imageio.imread(path)

    # if RGB but mask-like, keep as is
    return img


def visualize_from_paths(
    image_path,
    pred_path,
    gt_path,
    class_name="class",
    color="red",
    threshold=0.5,
    alpha=0.4,
    save_path=None
):

    # =========================
    # Load data
    # =========================
    image = load_array(image_path)
    image = rotate_90_ccw(image)

    pred_prob = load_array(pred_path)
    gt = load_array(gt_path)
    gt = rotate_90_ccw(gt)

    # ensure grayscale image compatibility
    if image.ndim == 2:
        img_disp = image
    else:
        img_disp = image

    # =========================
    # threshold prediction
    # =========================

    if pred_prob.ndim == 3:
        pred_prob = pred_prob[..., 0]  # or use np.mean(pred_prob, axis=-1)

    pred_mask = (pred_prob >= threshold).astype(np.uint8)

    if gt.ndim == 3:
        gt = gt[..., 0]

    gt_mask = (gt > 0).astype(np.uint8)

    pred_mask = rotate_90_ccw(pred_mask)

    rgb = to_rgb(color)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # =========================
    # 1. Image
    # =========================
    axes[0].imshow(img_disp, cmap="gray" if img_disp.ndim == 2 else None)
    axes[0].set_title("Image")
    axes[0].axis("off")

    # =========================
    # 2. Prediction overlay
    # =========================
    axes[1].imshow(img_disp, cmap="gray" if img_disp.ndim == 2 else None)

    pred_overlay = np.zeros((*pred_mask.shape, 4))
    pred_overlay[..., 0] = rgb[0]
    pred_overlay[..., 1] = rgb[1]
    pred_overlay[..., 2] = rgb[2]
    pred_overlay[..., 3] = pred_mask * alpha

    axes[1].imshow(pred_overlay)
    axes[1].set_title(f"Prediction (>{threshold}) - {class_name}")
    axes[1].axis("off")

    # =========================
    # 3. GT overlay
    # =========================
    axes[2].imshow(img_disp, cmap="gray" if img_disp.ndim == 2 else None)

    gt_mask = gt.astype(np.uint8)

    gt_overlay = np.zeros((*gt_mask.shape, 4))
    gt_overlay[..., 0] = rgb[0]
    gt_overlay[..., 1] = rgb[1]
    gt_overlay[..., 2] = rgb[2]
    gt_overlay[..., 3] = gt_mask * alpha

    axes[2].imshow(gt_overlay)
    axes[2].set_title(f"Ground Truth - {class_name}")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

root_dir = "/Users/sg2162/Datasets/CancerDatasets/EndoMRI_test_results"
img_name = 'D1-009_T1FS_axial_slice_016'
class_name = "endometrioma in pelvis"
# class_name = "uterus"
# class_name = "ovary"
color_dict = {
    "endometrioma in pelvis": "red",
    "uterus": "blue",
    "ovary": "green"
}

new_name = '+'.join(class_name.split(' '))
image_path = f"{root_dir}/test/{img_name}_MRI_pelvis.png"
pred_path = f"{root_dir}/test_BiomedParse_with_LoRA/{img_name}_MRI_pelvis_{new_name}+in+pelvis+MRI.png"
gt_path = f"{root_dir}/test_mask/{img_name}_MRI_pelvis_{new_name}.png"

output_dir = "/Users/sg2162/Library/CloudStorage/OneDrive-UniversityofCambridge/backup/project/Experiments/Endometriosis"
save_path = f"{output_dir}/{img_name}_{new_name}_segmentation_overlap.png"

visualize_from_paths(
    image_path=image_path,
    pred_path=pred_path,
    gt_path=gt_path,
    class_name=class_name,
    color=color_dict[class_name],
    threshold=0.5,
    save_path=save_path
)