# --- START OF FILE process_synced_poses.py ---
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
import os
from tqdm import tqdm
import time
import argparse # For command-line arguments

# Import functions from our other modules
from sp_parse import parse_calibration_mwc, parse_calibration_fmc, calculate_projection_matrices, triangulate_keypoints
from pose_detector import load_models, detect_persons, estimate_poses

def process_synced_frames(
    csv_path,
    calibration_path,
    video_dir,
    output_path, # Where to save the 3D poses (e.g., a new CSV or JSON)
    model_dir="./checkpoints",
    detector_dir=None, # Optional path to local detector model
    calib_type='mwc', # 'mwc' or 'fmc'
    skip_sync_indices=1, # Process every Nth sync index
    person_confidence=0.3,
    keypoint_confidence=0.1, # Threshold for using a keypoint view in triangulation
    device_name="auto" # "auto", "cpu", "mps", "cuda"
    ):
    """
    Processes synchronized frames from multiple cameras to estimate 3D poses.

    Args:
        csv_path (str): Path to the frame_time_history.csv file.
        calibration_path (str): Path to the MWC or FMC calibration file.
        video_dir (str): Directory containing the camera video files.
                         Expected naming: Camera_{port}_synchronized.mp4
        output_path (str): Path to save the output 3D pose data.
        model_dir (str): Path to the ViTPose checkpoint directory.
        detector_dir (str, optional): Path to local RT-DETR model. Defaults to None (uses Hub).
        calib_type (str): Type of calibration file ('mwc' or 'fmc').
        skip_sync_indices (int): Process every Nth sync index.
        person_confidence (float): Confidence threshold for person detection.
        keypoint_confidence (float): Min confidence for a 2D keypoint view to be used.
        device_name (str): Device to run inference on ('auto', 'cpu', 'mps', 'cuda').
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
        # Add checks if the specified device is available
        if device == "mps" and (not torch.backends.mps.is_available() or not torch.backends.mps.is_built()):
            print("Warning: MPS requested but not available/built. Falling back to CPU.")
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
    print(f"Using device: {device}")


    # --- 2. Load Frame History ---
    print(f"Loading frame history from: {csv_path}")
    try:
        frame_history_df = pd.read_csv(csv_path)
        # Convert columns to appropriate types if necessary
        frame_history_df['sync_index'] = frame_history_df['sync_index'].astype(int)
        frame_history_df['port'] = frame_history_df['port'].astype(int)
        frame_history_df['frame_index'] = frame_history_df['frame_index'].astype(int)
        frame_history_df['frame_time'] = frame_history_df['frame_time'].astype(float)
    except FileNotFoundError:
        print(f"Error: Frame history file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error reading frame history CSV: {e}")
        return

    print(f"Found {frame_history_df['sync_index'].nunique()} unique sync indices.")
    print(f"Ports found: {sorted(frame_history_df['port'].unique())}")


    # --- 3. Load Calibration ---
    print(f"Loading calibration ({calib_type}) from: {calibration_path}")
    if calib_type.lower() == 'mwc':
        camera_params = parse_calibration_mwc(calibration_path)
    elif calib_type.lower() == 'fmc':
        camera_params = parse_calibration_fmc(calibration_path)
    else:
        print(f"Error: Invalid calibration type '{calib_type}'. Use 'mwc' or 'fmc'.")
        return

    if not camera_params:
        print("Error: Failed to load camera calibration parameters.")
        return
    print(f"Loaded parameters for {len(camera_params)} cameras.")

    # Create a mapping from port number to camera index
    # Assumes camera_params list is ordered consistently with ports, or uses 'port' key if available
    port_to_cam_index = {}
    calibration_ports = set()
    for idx, params in enumerate(camera_params):
        port = params.get('port')
        if port is not None:
            port_to_cam_index[port] = idx
            calibration_ports.add(port)
        else:
            # Attempt to infer port from name like 'cam_X' if 'port' key missing (common in FMC)
            try:
                 inferred_port = int(params.get('name', f'cam_{idx}').split('_')[-1])
                 port_to_cam_index[inferred_port] = idx
                 calibration_ports.add(inferred_port)
                 print(f"Info: Inferred port {inferred_port} for camera index {idx} from name '{params.get('name', '')}'.")
            except ValueError:
                 print(f"Warning: Cannot determine port for camera index {idx}. Check calibration file format or 'name' field.")

    # Check if ports in CSV match ports in calibration
    csv_ports = set(frame_history_df['port'].unique())
    if not csv_ports.issubset(calibration_ports):
         print(f"Warning: Ports in CSV {csv_ports} do not fully match ports in calibration {calibration_ports}.")
         print(f"Missing in calibration: {csv_ports - calibration_ports}")
         print(f"Extra in calibration: {calibration_ports - csv_ports}")
         # Decide whether to continue or exit
         # For now, we'll continue with the intersection
    common_ports = sorted(list(csv_ports.intersection(calibration_ports)))
    print(f"Using common ports for processing: {common_ports}")
    if len(common_ports) < 2:
        print("Error: Need at least 2 cameras with matching calibration and video data for triangulation.")
        return

    # Filter camera_params and create new mapping based *only* on common_ports used
    filtered_cam_params = []
    filtered_port_map = {}
    new_idx = 0
    for port in common_ports:
        original_idx = port_to_cam_index[port]
        filtered_cam_params.append(camera_params[original_idx])
        filtered_port_map[port] = new_idx
        new_idx += 1

    camera_params = filtered_cam_params
    port_to_cam_index = filtered_port_map
    num_cameras = len(camera_params)
    print(f"Filtered calibration to {num_cameras} cameras based on common ports.")


    # --- 4. Calculate Projection Matrices ---
    print("Calculating projection matrices...")
    projection_matrices = calculate_projection_matrices(camera_params)
    if len(projection_matrices) != num_cameras:
         print("Error: Number of projection matrices doesn't match number of cameras.")
         return


    # --- 5. Load Models ---
    print("Loading detection and pose estimation models...")
    try:
        person_processor, person_model, pose_processor, pose_model = load_models(
            detect_path=detector_dir,
            pose_model_path=model_dir,
            device=device
        )
    except Exception as e:
        print(f"Error loading models: {e}")
        return


    # --- 6. Open Video Files ---
    print("Opening video capture for required ports...")
    caps = {}
    video_file_paths = {}
    for port in common_ports:
        video_path = os.path.join(video_dir, f"port_{port}.mp4")
        video_file_paths[port] = video_path
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            # Clean up already opened captures
            for p, cap in caps.items():
                cap.release()
            return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            # Clean up already opened captures
            for p, cap_other in caps.items():
                cap_other.release()
            return
        caps[port] = cap
        print(f"Opened video for port {port}: {video_path}")


    # --- 7. Process Synchronized Frames ---
    all_sync_indices = sorted(frame_history_df['sync_index'].unique())
    results_3d = [] # Store results: [{'sync_index': s, 'person_id': p, 'keypoints_3d': kps}, ...]
    start_time = time.time()

    print(f"Starting processing for {len(all_sync_indices)} sync indices...")
    sync_index_counter = 0
    for sync_index in tqdm(all_sync_indices, desc="Processing Sync Indices"):

        sync_index_counter += 1
        if (sync_index_counter - 1) % skip_sync_indices != 0:
            continue

        # Get data for this sync index
        sync_data = frame_history_df[frame_history_df['sync_index'] == sync_index]

        # Check if we have data for all *required* common ports
        if set(sync_data['port']) != set(common_ports):
             # print(f"Warning: Sync index {sync_index} missing data for some required ports. Skipping.")
             # print(f"  Expected: {set(common_ports)}, Found: {set(sync_data['port'])}")
             continue

        # --- Read frames for this sync index ---
        current_frames_pil = {}
        frame_read_success = True
        for _, row in sync_data.iterrows():
            port = row['port']
            frame_idx = row['frame_index']
            cap = caps.get(port)

            if cap is None:
                 print(f"Critical Error: Video capture for port {port} not found (should not happen).")
                 frame_read_success = False
                 break

            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Failed to read frame {frame_idx} from port {port} for sync index {sync_index}. Skipping sync index.")
                frame_read_success = False
                break
            # Convert BGR to RGB PIL Image
            current_frames_pil[port] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not frame_read_success:
            continue # Skip to next sync index

        # --- Detect and Estimate Poses for each view ---
        # Stores {person_idx_in_frame: {port: keypoints_2d_array}}
        # *Assumption*: person_idx_in_frame (0, 1, 2...) corresponds across views.
        # This is a MAJOR simplification and likely incorrect for multiple people.
        # A robust solution needs cross-view matching.
        detected_persons_data = {} # Key: person_index (0, 1, ...), Value: {port: kps_2d}

        max_persons_detected = 0 # Track max persons in any view for indexing

        for port in common_ports:
            image = current_frames_pil[port]

            # Detect persons
            _, person_boxes_coco, _ = detect_persons(
                image, person_processor, person_model, device, person_confidence
            )

            if len(person_boxes_coco) == 0:
                # print(f"Debug: No persons detected in port {port} for sync {sync_index}")
                continue # No one to estimate pose for in this view

            max_persons_detected = max(max_persons_detected, len(person_boxes_coco))

            # Estimate poses for all detected persons
            all_keypoints_2d, _ = estimate_poses(
                image, person_boxes_coco, pose_processor, pose_model, device
            )

            # Store keypoints, assuming order corresponds across views
            for person_idx, kps_2d in enumerate(all_keypoints_2d):
                 # Check keypoint format and confidence
                 valid_kps_view = []
                 for kp in kps_2d:
                      if len(kp) >= 3 and kp[2] >= keypoint_confidence: # Check confidence if available
                           valid_kps_view.append(kp)
                      elif len(kp) == 2: # Assume confident if no score provided
                           valid_kps_view.append(kp)
                      else:
                           valid_kps_view.append([np.nan, np.nan, 0.0]) # Mark low confidence as NaN

                 if person_idx not in detected_persons_data:
                      detected_persons_data[person_idx] = {}
                 detected_persons_data[person_idx][port] = np.array(valid_kps_view)


        # --- Triangulate for each assumed person ---
        for person_idx in range(max_persons_detected):
            if person_idx not in detected_persons_data:
                # This person index wasn't detected in enough views or at all
                continue

            person_kp_dict = detected_persons_data[person_idx]

            # Check if person seen in enough views
            if len(person_kp_dict) < 2:
                 # print(f"Debug: Person {person_idx} in sync {sync_index} seen in < 2 views. Cannot triangulate.")
                 continue

            # Perform triangulation
            keypoints_3d = triangulate_keypoints(
                person_kp_dict,      # Dict: {port: keypoints_array (17, 2|3)}
                port_to_cam_index,   # Map: {port: cam_idx}
                camera_params,       # List of cam params
                projection_matrices  # List of proj matrices
            )

            if keypoints_3d and any(kp is not None for kp in keypoints_3d):
                # Convert list of potential Nones/arrays to a structured format
                # Replace None with [NaN, NaN, NaN] for consistency
                kps_3d_list = []
                for kp in keypoints_3d:
                    if kp is not None and kp.shape == (3,):
                         kps_3d_list.append(kp.tolist()) # Convert numpy array to list
                    else:
                         kps_3d_list.append([np.nan, np.nan, np.nan])

                results_3d.append({
                    'sync_index': sync_index,
                    'person_id': person_idx, # Index based on detection order (ASSUMPTION)
                    'keypoints_3d': kps_3d_list # List of 17 lists [x, y, z] or [nan, nan, nan]
                })
            # else:
            #      print(f"Debug: Triangulation failed or yielded no valid points for person {person_idx} in sync {sync_index}.")


    # --- 8. Cleanup and Save Results ---
    print("\nReleasing video captures...")
    for port, cap in caps.items():
        cap.release()

    end_time = time.time()
    print(f"Processing finished in {end_time - start_time:.2f} seconds.")

    if not results_3d:
        print("No 3D poses were successfully generated.")
        return

    print(f"Generated 3D poses for {len(results_3d)} person instances.")
    print(f"Saving results to {output_path}...")

    try:
        # Save as CSV
        results_df = pd.DataFrame(results_3d)
        # Convert list of lists into separate columns or keep as string representation
        # Keeping as string representation for simplicity in CSV
        results_df['keypoints_3d_str'] = results_df['keypoints_3d'].apply(lambda x: str(x))
        results_df[['sync_index', 'person_id', 'keypoints_3d_str']].to_csv(output_path, index=False)

        # Alternatively save as JSON (better for nested lists)
        # import json
        # output_json_path = os.path.splitext(output_path)[0] + ".json"
        # with open(output_json_path, 'w') as f:
        #     json.dump(results_3d, f, indent=4)
        # print(f"Saved detailed results to {output_json_path}")

        print("Results saved successfully.")

    except Exception as e:
        print(f"Error saving results: {e}")


# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process synchronized multi-camera video for 3D pose estimation.")
    parser.add_argument("--csv_path", default="test/caliscope/frame_time_history.csv",help="Path to the frame_time_history.csv file.")
    parser.add_argument("--calibration_path",default="test/caliscope/config.toml", help="Path to the camera calibration file (MWC or FMC format).")
    parser.add_argument("--video_dir", default="test/caliscope", help="Directory containing camera videos (e.g., Port_0.mp4).")
    parser.add_argument("--output_path", default="test/caliscope/output",help="Path to save the output 3D poses CSV file.")
    parser.add_argument("--model_dir", default="./checkpoints", help="Path to the ViTPose checkpoint directory (default: ./checkpoints).")
    parser.add_argument("--detector_dir", default=None, help="Path to local RT-DETR detector model (optional, default: use Hugging Face).")
    parser.add_argument("--calib_type", default="mwc", choices=['mwc', 'fmc'], help="Type of calibration file (mwc or fmc, default: mwc).")
    parser.add_argument("--skip", type=int, default=1, help="Process every Nth sync index (default: 1, process all).")
    parser.add_argument("--person_conf", type=float, default=0.3, help="Confidence threshold for person detection (default: 0.3).")
    parser.add_argument("--keypoint_conf", type=float, default=0.1, help="Min confidence for a 2D keypoint view to be used in triangulation (default: 0.1).")
    parser.add_argument("--device", default="auto", choices=['auto', 'cpu', 'mps', 'cuda'], help="Device for inference (auto, cpu, mps, cuda, default: auto).")

    args = parser.parse_args()

    # Run the main processing function
    process_synced_frames(
        csv_path=args.csv_path,
        calibration_path=args.calibration_path,
        video_dir=args.video_dir,
        output_path=args.output_path,
        model_dir=args.model_dir,
        detector_dir=args.detector_dir,
        calib_type=args.calib_type,
        skip_sync_indices=args.skip,
        person_confidence=args.person_conf,
        keypoint_confidence=args.keypoint_conf,
        device_name=args.device
    )
# --- END OF FILE process_synced_poses.py ---