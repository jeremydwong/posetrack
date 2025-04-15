#%%
#conda install pytorch torchvision torchaudio -c pytorch-nightly
#pip install supervision
#pip install transformers
# this test script loads a video file and processes the keypoints from synthpose. 

import torch
import requests
import numpy as np

from PIL import Image

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# url = "http://farm4.staticflickr.com/3300/3416216247_f9c6dfc939_z.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("test/chanel.png").convert("RGB")

# ------------------------------------------------------------------------
# Stage 1. Detect humans on the image
# ------------------------------------------------------------------------

person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365",use_fast=True)
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

inputs = person_image_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = person_model(**inputs)

results = person_image_processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
)
result = results[0]  # take first image results

# Human label refers 0 index in COCO dataset
person_boxes = result["boxes"][result["labels"] == 0]
person_boxes = person_boxes.cpu().numpy()

# Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

#%%
# ------------------------------------------------------------------------
# Stage 2. Detect keypoints for each person found
# ------------------------------------------------------------------------
local_model_path = "./checkpoints"
image_processor = AutoProcessor.from_pretrained(local_model_path,local_files_only=True,use_fast=True)
model = VitPoseForPoseEstimation.from_pretrained(local_model_path, local_files_only=True, device_map=device)

inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
image_pose_result = pose_results[0]  # results for first image

import supervision as sv

xy = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu().numpy()
scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu().numpy()

key_points = sv.KeyPoints(
    xy=xy, confidence=scores
)

vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.RED,
    radius=2
)

annotated_frame = vertex_annotator.annotate(
    scene=image.copy(),
    key_points=key_points
)
annotated_frame
# %%
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

def process_video(
    input_video_path, 
    output_video_path, 
    device="mps",
    skip_frames=1,
    confidence_threshold=0.3,
    model_path="./checkpoints",
    detect_path="/Users/jeremy/Git/rtdetr_r50vd_coco_o365",
    draw_skeleton=True,
    visualize=False
):
    # Check device availability
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"Running inference on {device}")
    
    # Initialize models
    print("Loading detection model...")
    if detect_path is None:
        person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        person_model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365", 
            device_map=device
        )
    else:
        person_image_processor = AutoProcessor.from_pretrained(detect_path)
        person_model = RTDetrForObjectDetection.from_pretrained(detect_path,
            device_map=device)
        
    print("Loading pose estimation model...")
    image_processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True
    )
    pose_model = VitPoseForPoseEstimation.from_pretrained(
        model_path,
        local_files_only=True,
        device_map=device
    )
    
    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Define skeleton connections for visualization (COCO format)
    skeleton_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),      # Face and ears
        (5, 7), (7, 9), (6, 8), (8, 10),     # Arms
        (5, 6), (5, 11), (6, 12), (11, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16) # Legs
    ]
    
    # Define colors for visualization
    keypoint_colors = [
        (255, 0, 0),     # Nose
        (255, 85, 0),    # Left eye
        (255, 170, 0),   # Right eye
        (255, 255, 0),   # Left ear
        (170, 255, 0),   # Right ear
        (85, 255, 0),    # Left shoulder
        (0, 255, 0),     # Right shoulder
        (0, 255, 85),    # Left elbow
        (0, 255, 170),   # Right elbow
        (0, 255, 255),   # Left wrist
        (0, 170, 255),   # Right wrist
        (0, 85, 255),    # Left hip
        (0, 0, 255),     # Right hip
        (85, 0, 255),    # Left knee
        (170, 0, 255),   # Right knee
        (255, 0, 255),   # Left ankle
        (255, 0, 170)    # Right ankle
    ]
    
    # Colors for skeleton connections
    skeleton_colors = [
        (255, 128, 0), (255, 128, 0), (255, 128, 0), (255, 128, 0),  # Face
        (0, 255, 128), (0, 255, 128), (0, 255, 128), (0, 255, 128),  # Arms
        (0, 128, 255), (0, 128, 255), (0, 128, 255), (0, 128, 255),  # Body
        (128, 0, 255), (128, 0, 255), (128, 0, 255), (128, 0, 255)   # Legs
    ]
    
    # For performance tracking
    processing_stats = {
        "frames_processed": 0,
        "people_detected": 0,
        "start_time": time.time()
    }
    
    # Process first frame to determine keypoint format
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Convert first frame for testing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Detect humans in the first frame
    inputs = person_image_processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = person_model(**inputs)
    
    results = person_image_processor.post_process_object_detection(
        outputs, 
        target_sizes=torch.tensor([(pil_image.height, pil_image.width)]), 
        threshold=confidence_threshold
    )
    result = results[0]
    
    # Check if humans were detected
    person_boxes = result["boxes"][result["labels"] == 0]
    if len(person_boxes) > 0:
        person_boxes = person_boxes.cpu().numpy()
        coco_boxes = person_boxes.copy()
        coco_boxes[:, 2] = coco_boxes[:, 2] - coco_boxes[:, 0]
        coco_boxes[:, 3] = coco_boxes[:, 3] - coco_boxes[:, 1]
        
        # Detect keypoints to determine format
        inputs = image_processor(pil_image, boxes=[coco_boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = pose_model(**inputs)
        
        pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[coco_boxes])
        image_pose_result = pose_results[0]
        
        if len(image_pose_result) > 0:
            # Examine the keypoint format
            print("Found person in first frame, checking keypoint format...")
            if "keypoints" in image_pose_result[0]:
                keypoints = image_pose_result[0]["keypoints"]
                if len(keypoints) > 0:
                    # Get the shape of the first keypoint
                    print(f"First keypoint format: {keypoints[0]}")
                    keypoint_format = len(keypoints[0])
                    print(f"Keypoint format has {keypoint_format} values per point")
                else:
                    print("No keypoints found in the first person")
                    keypoint_format = 2  # Default to 2
            else:
                print("No 'keypoints' field in pose results")
                # Look at the direct structure of the result
                print(f"Pose result keys: {image_pose_result[0].keys()}")
                keypoint_format = 2  # Default to 2
        else:
            print("No pose results for the first frame")
            keypoint_format = 2  # Default to 2
    else:
        print("No person detected in the first frame")
        keypoint_format = 2  # Default to 2
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Process frames with progress bar
    frame_idx = 0
    pbar = tqdm(total=total_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        pbar.update(1)
        
        # Process every nth frame
        if (frame_idx - 1) % skip_frames != 0:
            out.write(frame)  # Write unmodified frame
            continue
        
        # Convert BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # STEP 1: Detect humans
        inputs = person_image_processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = person_model(**inputs)
        
        results = person_image_processor.post_process_object_detection(
            outputs, 
            target_sizes=torch.tensor([(pil_image.height, pil_image.width)]), 
            threshold=confidence_threshold
        )
        result = results[0]
        
        # Human label refers to 0 index in COCO dataset
        person_boxes = result["boxes"][result["labels"] == 0]
        
        if len(person_boxes) == 0:
            out.write(frame)  # Write unmodified frame
            continue
            
        person_boxes = person_boxes.cpu().numpy()
        processing_stats["people_detected"] += len(person_boxes)
        
        # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
        coco_boxes = person_boxes.copy()
        coco_boxes[:, 2] = coco_boxes[:, 2] - coco_boxes[:, 0]
        coco_boxes[:, 3] = coco_boxes[:, 3] - coco_boxes[:, 1]
        
        # Draw bounding boxes
        for box in person_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # STEP 2: Detect keypoints
        inputs = image_processor(pil_image, boxes=[coco_boxes], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = pose_model(**inputs)
        
        pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[coco_boxes])
        image_pose_result = pose_results[0]  # results for first image
        
        # Process keypoints based on the determined format
        for person_idx, person_data in enumerate(image_pose_result):
            if "keypoints" not in person_data:
                print(f"Warning: No keypoints in person data at frame {frame_idx}")
                continue
                
            keypoints = person_data["keypoints"]
            
            # Draw keypoints
            for i, keypoint in enumerate(keypoints):
                if keypoint_format == 3:  # Format: [x, y, confidence]
                    x, y, conf = keypoint
                elif keypoint_format == 2:  # Format: [x, y]
                    x, y = keypoint
                    conf = 1.0  # Assume high confidence
                
                if conf > 0.5:  # Confidence threshold
                    x_int, y_int = int(x), int(y)
                    cv2.circle(frame, (x_int, y_int), 5, keypoint_colors[i % len(keypoint_colors)], -1)
            
            # Draw skeleton
            if draw_skeleton:
                for i, (start_idx, end_idx) in enumerate(skeleton_connections):
                    if start_idx >= len(keypoints) or end_idx >= len(keypoints):
                        continue  # Skip if indices out of range
                        
                    # Get start and end keypoints
                    start_keypoint = keypoints[start_idx]
                    end_keypoint = keypoints[end_idx]
                    
                    # Extract coordinates and confidence based on format
                    if keypoint_format == 3:
                        start_x, start_y, start_conf = start_keypoint
                        end_x, end_y, end_conf = end_keypoint
                    elif keypoint_format == 2:
                        start_x, start_y = start_keypoint
                        end_x, end_y = end_keypoint
                        start_conf = end_conf = 1.0
                    
                    if start_conf > 0.5 and end_conf > 0.5:
                        start_point = (int(start_x), int(start_y))
                        end_point = (int(end_x), int(end_y))
                        cv2.line(frame, start_point, end_point, 
                                 skeleton_colors[i % len(skeleton_colors)], 2)
        
        # Write the processed frame
        out.write(frame)
        
        # Optional: display the frame
        if visualize:
            cv2.imshow('Processed Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        processing_stats["frames_processed"] += 1
    
    # Release resources
    cap.release()
    out.release()
    if visualize:
        cv2.destroyAllWindows()
    pbar.close()
    
    # Calculate and print stats
    elapsed_time = time.time() - processing_stats["start_time"]
    fps_processing = processing_stats["frames_processed"] / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\nProcessing complete!")
    print(f"Processed {processing_stats['frames_processed']} frames in {elapsed_time:.2f} seconds")
    print(f"Processing speed: {fps_processing:.2f} FPS")
    print(f"Detected {processing_stats['people_detected']} people in total")
    print(f"Output saved to {output_video_path}")
    
    return processing_stats

# Example usage
if __name__ == "__main__":
    # Initialize device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    input_dir = "test"
    input_video = "Camera_002_synchronized.mp4"    
    # Process video
    process_video(
        input_video_path=f"{input_dir}/{input_video}",
        output_video_path=f"{input_dir}/output_{input_video}",
        device=device,
        skip_frames=1,  # Process every 2nd frame for better performance
        confidence_threshold=0.3,
        model_path="./checkpoints",
        draw_skeleton=True,
        visualize=False  # Set to True to display frames during processing
    )