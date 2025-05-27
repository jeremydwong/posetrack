import torch
import numpy as np
import cv2 # Needed for visualization if using cv2 directly, or by supervision
from PIL import Image
import os
import argparse
import supervision as sv # For visualization consistency with original test/pose_detector

# Import the refactored functions from pose_detector.py
import sys
import os
path_to_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path_to_project_root)

from src.pose_detector import (
    load_models,
    detect_persons,
    estimate_poses,
    LOCAL_SP_DIR # Import the constant if needed for default paths
)

def run_verification(image_path, output_dir=os.path.join(path_to_project_root,"output/output_verify"), device_name="auto",
                     detector_dir=None, pose_model_dir=LOCAL_SP_DIR,
                     person_conf=0.3, show_result=True, save_result=True):
    """
    Runs the pose detection pipeline using refactored functions and visualizes.

    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the annotated output image.
        device_name (str): Device to use ('auto', 'cpu', 'mps', 'cuda').
        detector_dir (str, optional): Path to local detector model. Defaults to None (Hugging Face).
        pose_model_dir (str): Path to the local pose estimation model.
        person_conf (float): Confidence threshold for person detection.
        show_result (bool): Whether to display the annotated image.
        save_result (bool): Whether to save the annotated image.
    """

    # --- 1. Setup Device ---
    if device_name == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = device_name
        if device == "mps" and (not torch.backends.mps.is_available() or not torch.backends.mps.is_built()):
            print("Warning: MPS explicitly requested but unavailable/not built. Falling back to CPU.")
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA explicitly requested but unavailable. Falling back to CPU.")
            device = "cpu"
    print(f"Using device: {device}")

    # --- 2. Load Models ---
    print("Loading models using pose_detector.load_models...")
    try:
        person_processor, person_model, pose_processor, pose_model = load_models(
            detect_path=detector_dir,
            pose_model_path=pose_model_dir,
            device=device
        )
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # --- 3. Load Image ---
    print(f"Loading image from: {image_path}")
    try:
        image_pil = Image.open(image_path).convert("RGB")
        print(f"Image loaded successfully. Size: {image_pil.size}")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # --- 4. Detect Persons ---
    print("Detecting persons using pose_detector.detect_persons...")
    try:
        # Use the detect_persons function
        person_boxes_voc, person_boxes_coco, person_scores = detect_persons(
            image_pil, person_processor, person_model, device, confidence_threshold=person_conf
        )
        print(f"Detected {len(person_boxes_voc)} persons.")
        if len(person_boxes_voc) > 0:
            print(f"  Example VOC box (xyxy): {person_boxes_voc[0]}")
            print(f"  Example COCO box (xywh): {person_boxes_coco[0]}")
            print(f"  Example score: {person_scores[0]}")

    except Exception as e:
        print(f"Error during person detection: {e}")
        # It might be useful to continue to see if pose estimation crashes too
        person_boxes_voc = np.empty((0, 4), dtype=np.float32)
        person_boxes_coco = np.empty((0, 4), dtype=np.float32)
        person_scores = np.empty((0,), dtype=np.float32)


    # --- 5. Estimate Poses ---
    # Only proceed if persons were detected
    all_keypoints = []
    all_keypoint_scores = []
    if len(person_boxes_coco) > 0:
        print("Estimating poses using pose_detector.estimate_poses...")
        try:
            # Make sure to pass the COCO boxes
            all_keypoints, all_keypoint_scores = estimate_poses(
                image_pil, person_boxes_coco, pose_processor, pose_model, device,
                debug_plot=True # Set to True and provide prefix if needed
                # debug_save_prefix=os.path.join(output_dir, "debug_pose_estimate")
            )
            print(f"Estimated poses for {len(all_keypoints)} persons.")
            if all_keypoints:
                # Check the structure and values of the first keypoint set
                print(f"  Example keypoints shape (first person): {all_keypoints[0].shape if all_keypoints else 'N/A'}")
                print(f"  Example keypoints values (first person, first 5 points): \n{all_keypoints[0][:5] if all_keypoints else 'N/A'}")
                print(f"  Example scores shape (first person): {all_keypoint_scores[0].shape if all_keypoint_scores else 'N/A'}")
                print(f"  Example scores values (first person, first 5 points): {all_keypoint_scores[0][:5] if all_keypoint_scores else 'N/A'}")

                # ---------> DEBUGGING FOCUS <---------
                # Check for NaN or unexpected large/small values here
                if all_keypoints and np.isnan(all_keypoints[0]).any():
                    print("!!!!!! WARNING: NaN values detected in the first person's keypoints !!!!!")

                if isinstance(all_keypoints, list) and all_keypoints:
                    # Assuming all_keypoints is a list of numpy arrays, we can visualize them
                    xy = torch.stack([pose_result['keypoints'] for pose_result in all_keypoints]).cpu().numpy()
                    scores = torch.stack([pose_result['scores'] for pose_result in all_keypoint_scores]).cpu().numpy()
                    key_points = sv.KeyPoints(xy=xy, confidence=scores)

                    vertex_annotator = sv.VertexAnnotator(
                        color=sv.Color.RED,
                        radius=2
                    )

                    annotated_frame = vertex_annotator.annotate(
                        scene=image_pil.copy(),
                        key_points=key_points
                    )
                    annotated_frame.show()
                    print("Keypoints visualized successfully.")
                else:   
                    print("Warning: all_keypoints is not a list or is empty. No keypoints to visualize.")
            

            else:
                print("No keypoints detected. Skipping visualization.")
        except Exception as e:
            print(f"Error during pose estimation: {e}")


                # assume it is a list of numpy arrays. loop across and add to image.

                # image_pose_result = pose_results[0]  # results for first image
                # key_points = sv.KeyPoints(
                #     xy=all_keypoints, confidence=all_keypoint_scores)

                # vertex_annotator = sv.VertexAnnotator(
                #     color=sv.Color.RED,
                #     radius=2
                # )

                # annotated_frame = vertex_annotator.annotate(
                #     scene=image.copy(),
                #     key_points=key_points
                # )
                # annotated_frame.show()

        except Exception as e:
            print(f"Error during pose estimation: {e}")
            all_keypoints = [] # Ensure lists are empty on error
            all_keypoint_scores = []

    # --- 6. Visualize Results (using Supervision) ---
    
    


    if save_result or show_result:

        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, f"verify_{os.path.basename(image_path)}")
            # print out the output filename for clarity
            print(f"Output filename: {output_filename}")
            try:
                annotated_image.save(output_filename)
                print(f"Saved annotated image to: {output_filename}")
            except Exception as e:
                print(f"Error saving annotated image: {e}")

        if show_result:
            print("Displaying annotated image...")
            

    print("Verification script finished.")
    return annotated_image # Return for potential further use/testing

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verification script for refactored pose detection.")

    # Use paths relative to the project structure assumed
    # Adjust these defaults if your structure is different
    parser.add_argument("--image", default="test/image/chanel.png", help="Path to the input image.")
    parser.add_argument("--output_dir", default=os.path.join(path_to_project_root,"output/output_verify"), help="Directory to save the output annotated image.")
    parser.add_argument("--pose_model", default=LOCAL_SP_DIR, help="Path to the local SynthPose model directory.")
    parser.add_argument("--detector_model", default=None, help="Path to the local RT-DETR model directory (optional, uses HF if None).")
    parser.add_argument("--device", default="mps", choices=['auto', 'cpu', 'mps', 'cuda'], help="Computation device.")
    parser.add_argument("--person_conf", type=float, default=0.3, help="Confidence threshold for person detection.")
    parser.add_argument("--hide", action="store_true", help="Do not display the annotated image.")
    parser.add_argument("--no_save", action="store_true", help="Do not save the annotated image.")

    args = parser.parse_args()

    run_verification(
        image_path=args.image,
        output_dir=args.output_dir,
        device_name=args.device,
        detector_dir=args.detector_model,
        pose_model_dir=args.pose_model,
        person_conf=args.person_conf,
        show_result=True,
        save_result=not args.no_save
    )