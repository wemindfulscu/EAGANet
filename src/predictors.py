import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from typing import Dict, Any, Tuple, Optional
from groundingdino.util.inference import Model as GroundingDINO

class MedSAMPredictor:
    """Encapsulates SAM model interaction for segmentation from geometric prompts."""
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Initializing MedSAMPredictor on device: {self.device}")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path).to(self.device)
        self.predictor = SamPredictor(sam_model)

    def set_image(self, image_np: np.ndarray):
        self.predictor.set_image(image_np)
        self.original_image = image_np

    def predict(self, prompts: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[float]]:
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array(prompts["point_coords"]) if "point_coords" in prompts else None,
            point_labels=np.array(prompts["point_labels"]) if "point_labels" in prompts else None,
            box=np.array(prompts["box"]) if "box" in prompts else None,
            multimask_output=True,
        )
        best_mask_idx = np.argmax(scores)
        return masks[best_mask_idx], scores[best_mask_idx]

class TextToBoxPredictor:
    """Uses GroundingDINO to convert text prompts to bounding boxes."""
    def __init__(self, checkpoint_path: str, config_path: str, device: str):
        self.device = device
        print("Initializing GroundingDINO...")
        self.model = GroundingDINO(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path,
            device=self.device
        )
        print("GroundingDINO initialized.")

    def predict(self, image, text_prompt, box_thresh, text_thresh) -> Optional[np.ndarray]:
        detections = self.model.predict_with_classes(
            image=image, classes=[text_prompt],
            box_threshold=box_thresh, text_threshold=text_thresh
        )
        # Return the box with the highest confidence
        if len(detections) > 0:
            return detections.xyxy[0]
        return None