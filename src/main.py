# src/main.py
import os
import cv2
from predictors import MedSAMPredictor, TextToBoxPredictor
from utils import load_tasks, create_visualization, save_binary_mask

# --- Project Constants ---
# Assumes the script is run from the root directory `medsam-segmentation-pipeline/`
MODELS_DIR = "models"
CONFIG_PATH = "configs/segmentation_tasks.json"
DEVICE = "cuda"

# GroundingDINO requires cloning its repo for the config file.
# This structure assumes the user has done so.
# git clone https://github.com/IDEA-Research/GroundingDINO.git
# Then move the config file.
GROUNDING_DINO_CONFIG = "configs/GroundingDINO_SwinT_OGC.py"

def main():
    # --- Model Initialization ---
    medsam_checkpoint = os.path.join(MODELS_DIR, "medsam_vit_b.pth")
    grounding_dino_checkpoint = os.path.join(MODELS_DIR, "groundingdino_swint_ogc.pth")
    
    print("--- Initializing Models ---")
    medsam = MedSAMPredictor(checkpoint_path=medsam_checkpoint, device=DEVICE)
    # Lazy initialization for GroundingDINO
    text_to_box = None

    # --- Load and Process Tasks ---
    tasks = load_tasks(CONFIG_PATH)
    print(f"\nFound {len(tasks)} tasks to process.")

    for i, task in enumerate(tasks):
        task_name = task.get('task_name', f'Task_{i+1}')
        print(f"\n--- Running Task {i+1}/{len(tasks)}: {task_name} ---")

        try:
            image = cv2.cvtColor(cv2.imread(task["input_image"]), cv2.COLOR_BGR2RGB)
            prompts = task.get("prompts", {})

            if "text" in prompts:
                # --- Text-based Segmentation Workflow ---
                if text_to_box is None:
                    text_to_box = TextToBoxPredictor(
                        checkpoint_path=grounding_dino_checkpoint,
                        config_path=GROUNDING_DINO_CONFIG,
                        device=DEVICE
                    )
                
                detected_box = text_to_box.predict(
                    image, prompts["text"],
                    prompts.get("box_threshold", 0.35),
                    prompts.get("text_threshold", 0.25)
                )

                if detected_box is None:
                    print(f"Warning: No object found for text prompt '{prompts['text']}'. Skipping task.")
                    continue
                
                # Use the detected box as the new prompt for MedSAM
                medsam_prompts = {"box": detected_box}
                viz_prompts = {"box": detected_box, "text": prompts["text"]}

            else:
                # --- Geometric-based Segmentation Workflow ---
                medsam_prompts = prompts
                viz_prompts = prompts

            # --- MedSAM Segmentation and Output ---
            medsam.set_image(image)
            mask, score = medsam.predict(medsam_prompts)

            # Save and visualize results
            save_binary_mask(mask, task["output_mask"])
            print(f"Mask saved to {task['output_mask']}")
            
            create_visualization(
                image, mask, score, viz_prompts,
                task_name, task["output_visualization"]
            )
            print(f"Visualization saved to {task['output_visualization']}")

        except FileNotFoundError:
            print(f"Error: Input image not found at '{task['input_image']}'. Skipping task.")
        except Exception as e:
            print(f"An unexpected error occurred during task '{task_name}': {e}")
            
    print("\n--- All tasks completed. ---")


if __name__ == "__main__":
    main()