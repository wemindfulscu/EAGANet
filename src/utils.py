import json
import os
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color: color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else: color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]; mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1); ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]; neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax, label=None):
    x0, y0 = box[0], box[1]; w, h = box[2] - x0, box[3] - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    if label: ax.text(x0, y0 - 5, label, color='white', backgroundcolor='green', fontsize=10)

def create_visualization(image, mask, score, prompts, title, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    # Panel 1: Original image with prompts
    axes[0].imshow(image)
    if "box" in prompts: show_box(np.array(prompts["box"]), axes[0], label=prompts.get("text"))
    if "point_coords" in prompts: show_points(np.array(prompts["point_coords"]), np.array(prompts["point_labels"]), axes[0])
    axes[0].set_title("Input Image & Prompts")
    axes[0].axis('off')

    # Panel 2: Image with mask overlay
    axes[1].imshow(image); show_mask(mask, axes[1]);
    axes[1].set_title(f"Segmented Mask Overlay (Score: {score:.3f})")
    axes[1].axis('off')

    # Panel 3: Standalone binary mask
    axes[2].imshow(mask, cmap='gray'); axes[2].set_title("Final Binary Mask"); axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path)
    plt.show()
    plt.close(fig)

def load_tasks(config_path: str) -> List[Dict[str, Any]]:
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError: raise FileNotFoundError(f"Config file not found: {config_path}")
    except json.JSONDecodeError: raise ValueError(f"Invalid JSON format in {config_path}")

def save_binary_mask(mask: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    binary_mask = (mask.astype(np.uint8)) * 255
    cv2.imwrite(path, binary_mask)