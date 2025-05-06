import json
from scipy import stats
import numpy as np

import huggingface_hub


def check_mask_stats(img, mask, modality_type, target):
    # img: np.array, shape=(H, W, 3) RGB image with pixel values in [0, 255]
    # mask: np.array, shape=(H, W, 1) mask probability scaled to [0,255] with pixel values in [0, 255]
    # modality_type: str, see target_dist.json for the list of modality types
    # target: str, see target_dist.json for the list of targets
    
    target_dist = get_target_dist()
    
    if modality_type not in target_dist:
        raise ValueError(f"Currently support modality types: {list(target_dist.keys())}")
    
    if target not in target_dist[modality_type]:
        raise ValueError(f"Currently support targets for {modality_type}: {list(target_dist[modality_type].keys())}")
    
    ms = mask_stats(mask, img)
    
    ps = [stats.ks_1samp([ms[i]], stats.beta(param[0], param[1]).cdf).pvalue for i, param in enumerate(target_dist[modality_type][target])]
    p_value = np.prod(ps)
    
    adj_p_value = p_value**0.25    # adjustment for four test products
    
    return adj_p_value
    
    

def mask_stats(mask, img):
    # mask is a prediction mask with pixel values in [0, 255] for probability in [0, 1]
    # img is a RGB image with pixel values in [0, 255]
    if mask.max() <= 127:
        return [0, 0, 0, 0]
    return [mask[mask>=128].mean()/256, img[:,:,0][mask>=128].mean()/256, 
            img[:,:,1][mask>=128].mean()/256, img[:,:,2][mask>=128].mean()/256]
    
    
    
def combine_masks(predicts):
    # predicts: a dictionary of pixel probability, {TARGET: pred_prob}
    pixel_preds = {}
    target_area = {}
    target_probs = {}
    for target in predicts:
        pred = predicts[target]
        pred_region = np.where(pred > 0.1)
        target_area[target] = 0
        target_probs[target] = 0
        for (i,j) in zip(*pred_region):
            if (i,j) not in pixel_preds:
                pixel_preds[(i,j)] = {}
            pixel_preds[(i,j)][target] = pred[i,j]
            target_area[target] += 1
            target_probs[target] += pred[i,j]
    for target in predicts:
        if target_area[target] == 0:
            continue
        target_probs[target] /= target_area[target]
    
    # generate combined masks
    combined_areas = {t: 0 for t in predicts}
    for index in pixel_preds:
        pred_target = sorted(pixel_preds[index].keys(), key=lambda t: pixel_preds[index][t], reverse=True)[0]
        combined_areas[pred_target] += 1

    # discard targets with small areas
    discard_targets = []
    for target in predicts:
        if combined_areas[target] < 0.5 * target_area[target]:
            discard_targets.append(target)

    # keep the most confident target
    most_confident_target = sorted(predicts.keys(), key=lambda t: target_probs[t], reverse=True)[0]

    discard_targets = [t for t in discard_targets if t != most_confident_target]
    
    masks = {t: np.zeros_like(predicts[t]).astype(np.uint8) for t in predicts if t not in discard_targets}
    for index in pixel_preds:
        candidates = [t for t in pixel_preds[index] if t not in discard_targets and pixel_preds[index][t] > 0.5]
        if len(candidates) == 0:
            continue
        pred_target = max(candidates, key=lambda t: pixel_preds[index][t])
        masks[pred_target][index[0], index[1]] = 1
    
    return masks

def get_target_dist():
    huggingface_hub.hf_hub_download('microsoft/BiomedParse', filename='target_dist.json', local_dir='./inference_utils')
    huggingface_hub.hf_hub_download('microsoft/BiomedParse', filename="config.yaml", local_dir="./configs")
    return json.load(open("inference_utils/target_dist.json"))