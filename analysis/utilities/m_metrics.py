import numpy as np
from sklearn.metrics import f1_score


def find_best_dice_threshold(probs, labels, thresholds=np.linspace(0.0, 1.0, 101), epsilon=1e-6):
    """
    Find the best threshold for maximizing Dice score.

    Args:
        probs (np.ndarray): Prediction probabilities (shape: [B, H, W] or [H, W])
        labels (np.ndarray): Ground truth binary labels (same shape)
        thresholds (np.ndarray): List of thresholds to evaluate
        epsilon (float): Smoothing to prevent divide-by-zero

    Returns:
        best_threshold (float): Threshold yielding the best Dice
        best_dice (float): The highest Dice score
        all_scores (dict): {threshold: dice}
    """
    probs = probs.flatten()
    labels = labels.flatten().astype(np.uint8)

    best_threshold = 0.5
    best_dice = 0.0
    all_scores = {}

    for t in thresholds:
        pred = (probs >= t).astype(np.uint8)
        intersection = np.sum(pred * labels)
        union = np.sum(pred) + np.sum(labels)
        dice = (2 * intersection + epsilon) / (union + epsilon)
        all_scores[t] = dice

        if dice > best_dice:
            best_dice = dice
            best_threshold = t

    return best_threshold, best_dice, all_scores


def normalize_img(img):
    img_min = img.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    img_max = img.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    return (img - img_min) / (img_max - img_min + 1e-5)

def calculate_metrics(predictions, labels):
    """
    Calculate accuracy, F1 score, sensitivity, and specificity for binary classification.

    Args:
        predictions (np.ndarray): Binary predictions of shape (b, H, W).
        labels (np.ndarray): Binary ground truth labels of shape (b, H, W).

    Returns:
        dict: A dictionary containing accuracy, F1 score, sensitivity, and specificity.
    """
    # Flatten the arrays
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Calculate True Positives, True Negatives, False Positives, False Negatives
    TP = np.sum((predictions == 1) & (labels == 1))
    TN = np.sum((predictions == 0) & (labels == 0))
    FP = np.sum((predictions == 1) & (labels == 0))
    FN = np.sum((predictions == 0) & (labels == 1))

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # F1 Score
    f1 = f1_score(labels, predictions)

    # Sensitivity (Recall)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Specificity
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        # "ca": (TP + TN)/labels.numel()
    }


def calculate_dice(predictions, labels):
    """
    Calculate the Dice coefficient for binary classification.

    Args:
        predictions (np.ndarray): Binary predictions of shape (b, H, W).
        labels (np.ndarray): Binary ground truth labels of shape (b, H, W).

    Returns:
        float: Dice coefficient.
    """
    # Flatten the arrays
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Calculate intersection and union
    intersection = np.sum(predictions * labels)
    union = np.sum(predictions) + np.sum(labels)

    # Dice coefficient
    dice = (2 * intersection) / union if union > 0 else 0

    return dice


def calculate_soft_dice(predictions, labels, epsilon=1e-6):
    """
    Calculate the Soft Dice coefficient for binary classification.

    Args:
        predictions (np.ndarray): Predicted probabilities of shape (b, H, W).
        labels (np.ndarray): Binary ground truth labels of shape (b, H, W).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        float: Soft Dice coefficient.
    """
    # Flatten the arrays
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Calculate intersection and union
    intersection = np.sum(predictions * labels)
    union = np.sum(predictions) + np.sum(labels)

    # Soft Dice coefficient
    soft_dice = (2 * intersection + epsilon) / (union + epsilon)

    return soft_dice


import numpy as np
import torch
from sklearn.metrics import f1_score

def evaluate_segmentation_metrics(predictions, labels, from_logits=False, threshold=0.5, epsilon=1e-6):
    """
    Evaluate binary segmentation metrics from logits or probabilities.

    Args:
        predictions (torch.Tensor or np.ndarray): Model output logits or probabilities, shape [B, H, W]
        labels (torch.Tensor or np.ndarray): Ground truth labels, binary {0,1}, shape [B, H, W]
        from_logits (bool): Whether predictions are raw logits. If True, apply sigmoid.
        threshold (float): Threshold to convert probabilities to binary masks.
        epsilon (float): Smoothing term for dice computation.

    Returns:
        dict: Dictionary of metrics: accuracy, f1_score, sensitivity, specificity, dice, soft_dice
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if from_logits:
        probs = 1 / (1 + np.exp(-predictions))  # Sigmoid
    else:
        probs = predictions

    binary_preds = (probs >= threshold).astype(np.uint8)
    labels = labels.astype(np.uint8)

    # Flatten
    pred_flat = binary_preds.flatten()
    label_flat = labels.flatten()

    TP = np.sum((pred_flat == 1) & (label_flat == 1))
    TN = np.sum((pred_flat == 0) & (label_flat == 0))
    FP = np.sum((pred_flat == 1) & (label_flat == 0))
    FN = np.sum((pred_flat == 0) & (label_flat == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    f1 = f1_score(label_flat, pred_flat, zero_division=0)
    sensitivity = TP / (TP + FN + epsilon)
    specificity = TN / (TN + FP + epsilon)

    # Hard Dice
    intersection = np.sum(binary_preds * labels)
    union = np.sum(binary_preds) + np.sum(labels)
    dice = (2 * intersection + epsilon) / (union + epsilon)

    # Soft Dice
    soft_preds = probs.flatten()
    soft_labels = labels.flatten()
    soft_intersection = np.sum(soft_preds * soft_labels)
    soft_union = np.sum(soft_preds) + np.sum(soft_labels)
    soft_dice = (2 * soft_intersection + epsilon) / (soft_union + epsilon)
    # print(float((label_flat==0).sum()))
    best_t, soft_dice, scores = find_best_dice_threshold(probs, labels)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "dice": dice,
        "soft_dice": soft_dice,
        "ca": (TP/float((label_flat==1).sum()) + TN/float((label_flat==0).sum())) / 2
    }

# Example usage
if __name__ == "__main__":
    # Example binary predictions and labels
    predictions = np.random.randint(0, 2, (2, 256, 256))
    labels = np.random.randint(0, 2, (2, 256, 256))

    metrics = calculate_metrics(predictions, labels)
    print(metrics)