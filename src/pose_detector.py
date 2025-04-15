
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation
)
import time
from tqdm import tqdm
import supervision as sv # Added for visualization types if needed later

def load_models(detect_path=None, pose_model_path="./checkpoints", device="cpu"):
    """Loads the detection and pose estimation models."""
    print(f"Loading models to device: {device}")

    # --- Person Detection Model ---
    print("Loading detection model...")
    if detect_path and detect_path != "PekingU/rtdetr_r50vd_coco_o365":
        print(f"Loading local detector from: {detect_path}")
        person_image_processor = AutoProcessor.from_pretrained(detect_path, local_files_only=True)
        person_model = RTDetrForObjectDetection.from_pretrained(detect_path, local_files_only=True, device_map=device)
    else:
        print("Loading detector from Hugging Face: PekingU/rtdetr_r50vd_coco_o365")
        # Ensure use_fast=True is compatible or remove if causing issues
        person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

    # --- Pose Estimation Model ---
    print(f"Loading pose estimation model from: {pose_model_path}")
    # Ensure use_fast=True is compatible or remove if causing issues
    pose_image_processor = AutoProcessor.from_pretrained(pose_model_path, local_files_only=True)
    pose_model = VitPoseForPoseEstimation.from_pretrained(pose_model_path, local_files_only=True, device_map=device)

    print("Models loaded successfully.")
    return person_image_processor, person_model, pose_image_processor, pose_model

def detect_persons(image, person_image_processor, person_model, device, confidence_threshold=0.3):
    """Detects persons in a PIL Image."""
    inputs = person_image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=confidence_threshold #PIL size is (width, height)
    )
    result = results[0]

    # Get boxes for label 0 (person in COCO)
    person_boxes_voc = result["boxes"][result["labels"] == 0]
    person_scores = result["scores"][result["labels"] == 0]

    if len(person_boxes_voc) == 0:
        return [], [], [] # No persons found

    person_boxes_voc = person_boxes_voc.cpu().numpy()
    person_scores = person_scores.cpu().numpy()

    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h)
    person_boxes_coco = person_boxes_voc.copy()
    person_boxes_coco[:, 2] = person_boxes_coco[:, 2] - person_boxes_coco[:, 0]
    person_boxes_coco[:, 3] = person_boxes_coco[:, 3] - person_boxes_coco[:, 1]

    return person_boxes_voc, person_boxes_coco, person_scores

def estimate_poses(image, person_boxes_coco, pose_image_processor, pose_model, device):
    """Estimates poses for given person bounding boxes."""
    if not person_boxes_coco.any(): # Check if the numpy array is non-empty
         return []

    # ViTPose processor expects list of boxes, even if only one image
    inputs = pose_image_processor(image, boxes=[person_boxes_coco.tolist()], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = pose_model(**inputs)

    # Post-process requires target_sizes and boxes as lists
    pose_results = pose_image_processor.post_process_pose_estimation(
        outputs,
        target_sizes=[image.size[::-1]], # Use target_sizes
        boxes=[person_boxes_coco.tolist()] # Pass boxes used for processing
    )

    # pose_results is a list (per image), get the results for the first (only) image
    image_pose_results = pose_results[0]

    # Extract keypoints and scores for each person
    all_keypoints = []
    all_keypoint_scores = []
    for person_result in image_pose_results:
        # Ensure keypoints are tensors before converting to numpy
        keypoints = person_result['keypoints']
        scores = person_result['scores']
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        if isinstance(scores, torch.Tensor):
             scores = scores.cpu().numpy()

        all_keypoints.append(keypoints) # Shape: (17, 2 or 3)
        all_keypoint_scores.append(scores) # Shape: (17,)

    return all_keypoints, all_keypoint_scores # List of arrays per person


# --- Keep the visualization part from test.py if needed for debugging ---
# def draw_poses(image, person_boxes_voc, all_keypoints, all_keypoint_scores):
#     # ... (visualization code using supervision or cv2) ...
#     pass

# --- Example Usage (similar to the original test.py) ---
if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_built():
         print("MPS not available, falling back to CPU")
         device = "cpu"

    # Load models
    person_processor, person_model, pose_processor, pose_model = load_models(device=device)

    # Load image
    try:
        image = Image.open("test/chanel.png").convert("RGB")
    except FileNotFoundError:
        print("Error: test/chanel.png not found. Please provide a valid image.")
        exit()

    # Detect persons
    person_boxes_voc, person_boxes_coco, person_scores = detect_persons(
        image, person_processor, person_model, device
    )
    print(f"Detected {len(person_boxes_voc)} persons.")

    # Estimate poses
    all_keypoints, all_keypoint_scores = estimate_poses(
        image, person_boxes_coco, pose_processor, pose_model, device
    )
    print(f"Estimated poses for {len(all_keypoints)} persons.")

    # --- Optional: Visualization using supervision (as in original test.py) ---
    if all_keypoints:
        import supervision as sv

        # Prepare KeyPoints object for supervision
        # Stack keypoints and scores from all detected persons
        xy = np.array([kp[:, :2] for kp in all_keypoints]) # Take only x, y
        confidence = np.array(all_keypoint_scores)

        if xy.ndim == 3 and confidence.ndim == 2 and xy.shape[0] == confidence.shape[0] and xy.shape[1] == confidence.shape[1]:
             keypoints_sv = sv.KeyPoints(xy=xy, confidence=confidence)

             # Annotate
             frame_np = np.array(image) # Convert PIL to numpy for annotation
             vertex_annotator = sv.VertexAnnotator(color=sv.Color.RED, radius=4)
             box_annotator = sv.BoxAnnotator(
                color=sv.Color.GREEN,
                color_lookup=sv.ColorLookup.INDEX) #,text_color=sv.Color.WHITE, Add this line 

             annotated_frame = vertex_annotator.annotate(scene=frame_np.copy(), key_points=keypoints_sv)

             # Create Detections object for bounding boxes
             detections = sv.Detections(
                  xyxy=person_boxes_voc, # Supervision uses xyxy (voc)
                  confidence=person_scores,
                  # class_id=np.zeros(len(person_boxes_voc), dtype=int) # Assign class ID if needed
             )
             # Annotate boxes
             annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)


             # Display or save
             annotated_image = Image.fromarray(annotated_frame)
             annotated_image.show() # Display the image
             # annotated_image.save("test/output_pose.png")
             print("Saved annotated image to test/output_pose.png")
        else:
            print("Warning: Could not format keypoints for supervision visualization.")
            print(f"xy shape: {xy.shape}, confidence shape: {confidence.shape}")

# --- END OF FILE pose_detector.py ---