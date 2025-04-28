# In process_synced_poses.py
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
import os
from tqdm import tqdm
import time
import argparse
import math # For distance calculation
import pickle

from cs_parse import parse_calibration_mwc, parse_calibration_fmc, calculate_projection_matrices, triangulate_keypoints
from pose_detector import load_models, detect_persons, estimate_poses, SynthPoseMarkers

LOCAL_SP_DIR = "/Users/jeremy/Git/ProjectKeypointInference/models/synthpose/checkpoints"

# --- Add the project_3d_to_2d helper function here or import it ---
def project_3d_to_2d(point_3d, P):
    """Projects a 3D point to 2D using a projection matrix."""
    if point_3d is None or np.isnan(point_3d).any(): return None
    point_4d = np.append(point_3d, 1.0)
    point_2d_hom = P @ point_4d
    if abs(point_2d_hom[2]) < 1e-6 : return None # Check for near-zero depth
    point_2d = point_2d_hom[:2] / point_2d_hom[2]
    return point_2d.flatten()
# ---

def process_synced_frames(
    # ... (arguments remain the same) ...
    csv_path, calibration_path, video_dir, output_path, model_dir=LOCAL_SP_DIR,
    detector_dir=None, calib_type='mwc', skip_sync_indices=1, person_confidence=0.3,
    keypoint_confidence=0.1, device_name="auto",
    # --- Add tracking parameters ---
    tracking_max_2d_dist=100, # Max pixels distance for head matching in 2D
    head_kp_index=0 # Index of the head keypoint (e.g., 0 for Nose)
    # ---
    ):
    """Processes synchronized frames with simplified 3D tracking."""

    # --- 1. Setup Device ---
    # (Keep as is)
    if device_name == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"
        elif torch.cuda.is_available(): device = "cuda"
        else: device = "cpu"
    else:
        device = device_name
        if device == "mps" and (not torch.backends.mps.is_available() or not torch.backends.mps.is_built()): print("Warn: MPS unavailable."); device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available(): print("Warn: CUDA unavailable."); device = "cpu"
    print(f"Using device: {device}")


    # --- 2. Load Frame History & Derive 0-Based Index ---
    # (Keep as is)
    print(f"Loading frame history from: {csv_path}")
    try:
        frame_history_df = pd.read_csv(csv_path)
        frame_history_df['sync_index'] = frame_history_df['sync_index'].astype(int)
        frame_history_df['port'] = frame_history_df['port'].astype(int)
        frame_history_df['frame_time'] = frame_history_df['frame_time'].astype(float)
        frame_history_df = frame_history_df.sort_values(by=['port', 'frame_time'])
        frame_history_df['derived_frame_index'] = frame_history_df.groupby('port')['frame_time'].rank(method='min').astype(int) - 1
    except Exception as e: print(f"Error reading/processing frame history CSV: {e}"); return
    print(f"Found {frame_history_df['sync_index'].nunique()} unique sync indices.")
    print(f"Ports found: {sorted(frame_history_df['port'].unique())}")


    # --- 3. Load Calibration & Filter ---
    # (Keep as is)
    print(f"Loading calibration ({calib_type}) from: {calibration_path}")
    if calib_type.lower() == 'mwc': camera_params = parse_calibration_mwc(calibration_path)
    elif calib_type.lower() == 'fmc': camera_params = parse_calibration_fmc(calibration_path)
    else: print(f"Error: Invalid calib type '{calib_type}'."); return
    if not camera_params: print("Error: Failed to load camera calibration."); return
    print(f"Loaded {len(camera_params)} cameras initially.")
    port_to_cam_index={}; calibration_ports=set()
    for idx, params in enumerate(camera_params):
        port = params.get('port')
        if port is not None: port_to_cam_index[port] = idx; calibration_ports.add(port)
        else:
            try: inferred_port = int(params.get('name', f'cam_{idx}').split('_')[-1]); port_to_cam_index[inferred_port] = idx; calibration_ports.add(inferred_port)
            except ValueError: print(f"Warn: Cannot determine port for camera index {idx}.")
    csv_ports = set(frame_history_df['port'].unique())
    if not csv_ports.issubset(calibration_ports): print(f"Warn: Ports mismatch CSV:{csv_ports} vs Calib:{calibration_ports}.")
    common_ports = sorted(list(csv_ports.intersection(calibration_ports)))
    print(f"Using common ports for processing: {common_ports}")
    if len(common_ports) < 2: print("Error: Need >= 2 common ports."); return
    filtered_cam_params=[]; filtered_port_map={}; new_idx=0
    for port in common_ports: original_idx = port_to_cam_index[port]; filtered_cam_params.append(camera_params[original_idx]); filtered_port_map[port] = new_idx; new_idx += 1
    camera_params = filtered_cam_params; port_to_cam_index = filtered_port_map; num_cameras = len(camera_params)
    print(f"Filtered calibration to {num_cameras} cameras.")

    # --- 4. Calculate Projection Matrices ---
    # (Keep as is)
    print("Calculating projection matrices...")
    projection_matrices = calculate_projection_matrices(camera_params)
    if len(projection_matrices) != num_cameras: print("Error: Proj matrix count mismatch."); return

    # --- 5. Load Models ---
    # (Keep as is)
    print("Loading detection and pose estimation models...")
    try:
        person_processor, person_model, pose_processor, pose_model = load_models(detect_path=detector_dir, pose_model_path=model_dir, device=device)
    except Exception as e: print(f"Error loading models: {e}"); return

    # --- 6. Open Video Files ---
    # (Keep as is, including getting video_lengths)
    print("Opening video capture for required ports...")
    caps = {}; video_lengths = {}
    for port in common_ports:
        video_path = os.path.join(video_dir, f"port_{port}.mp4")
        if not os.path.exists(video_path): print(f"Error: Vid not found: {video_path}"); [c.release() for c in caps.values()]; return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): print(f"Error: Cannot open vid: {video_path}"); [c.release() for c in caps.values()]; return
        caps[port] = cap; video_lengths[port] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Opened video for port {port}: {video_path} ({video_lengths[port]} frames)")

    # --- 7. Process Synchronized Frames ---
    all_sync_indices = sorted(frame_history_df['sync_index'].unique())
    results_3d = []
    start_time = time.time()

    # --- Tracking State ---
    tracked_person_3d_kps = None # Stores the list of 3D kps [ [x,y,z], [x,y,z], ...] or None
    tracked_person_id_counter = 0 # Simple counter for assigning IDs

    print(f"Starting processing for {len(all_sync_indices)} sync indices...")
    sync_index_counter = 0
    for sync_index in tqdm(all_sync_indices, desc="Processing Sync Indices"):

        sync_index_counter += 1
        if (sync_index_counter - 1) % skip_sync_indices != 0: continue

        sync_data = frame_history_df[frame_history_df['sync_index'] == sync_index]
        if set(sync_data['port']) != set(common_ports): continue

        # --- Read frames ---
        # (Keep frame reading logic as is)
        current_frames_pil = {}
        frame_read_success = True
        for _, row in sync_data.iterrows():
            port = row['port']
            frame_idx_to_read = row['derived_frame_index']
            try: frame_idx_int = int(frame_idx_to_read)
            except (ValueError, TypeError): print(f"Warn: Invalid derived index '{frame_idx_to_read}' p{port} s{sync_index}"); frame_read_success = False; break
            cap = caps.get(port); total_frames = video_lengths.get(port, -1)
            if cap is None or total_frames == -1: print(f"CritErr: No capture/len p{port}"); frame_read_success = False; break
            if not (0 <= frame_idx_int < total_frames): print(f"Warn: Idx {frame_idx_int} out of bounds (0-{total_frames-1}) p{port} s{sync_index}"); frame_read_success = False; break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_int)
            ret, frame = cap.read()
            if not ret: print(f"Warn: Failed read idx {frame_idx_int} p{port} s{sync_index}"); frame_read_success = False; break
            current_frames_pil[port] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not frame_read_success: continue

        # --- Project previous head position (if tracking) ---
        projected_heads_2d = {} # {port: [x, y]}
        can_track = False
        if tracked_person_3d_kps is not None:
            prev_head_3d = tracked_person_3d_kps[head_kp_index] # Get previous head 3D pos
            if prev_head_3d is not None and not np.isnan(prev_head_3d).any():
                can_track = True # We have a valid previous position
                for port in common_ports:
                    cam_idx = port_to_cam_index[port]
                    P = projection_matrices[cam_idx]
                    proj_2d = project_3d_to_2d(prev_head_3d, P)
                    if proj_2d is not None:
                        projected_heads_2d[port] = proj_2d
                    # else: print(f"Debug: Projection failed for port {port}") # Optional

        # --- Detect, Estimate Poses, and Store Per View ---
        view_results = {} # {port: [person0_kps(17,3), person1_kps(17,3), ...]}
        any_persons_detected_this_sync = False
        for port in common_ports:
            image = current_frames_pil[port]
            plot_this_frame = True; debug_prefix = None # Or your debug conditions
            if plot_this_frame: debug_prefix = os.path.join(os.path.dirname(output_path), "debug_poses", f"sync_{sync_index}_port_{port}")

            person_boxes_voc, person_boxes_coco, person_scores = detect_persons(image, person_processor, person_model, device, person_confidence)
            if person_boxes_coco.size == 0: view_results[port] = []; continue

            all_keypoints_2d, all_keypoint_scores = estimate_poses(image, person_boxes_coco, pose_processor, pose_model, device, debug_plot=plot_this_frame, debug_save_prefix=debug_prefix)

            persons_in_view = []
            if len(all_keypoints_2d) == len(all_keypoint_scores):
                for person_idx in range(len(all_keypoints_2d)):
                    kps_2d, scores_2d = all_keypoints_2d[person_idx], all_keypoint_scores[person_idx]
                    valid_kps_view = []; num_kps = kps_2d.shape[0] if isinstance(kps_2d, np.ndarray) else 0
                    if num_kps > 0 and num_kps == scores_2d.shape[0]:
                        for kp_idx in range(num_kps):
                            kp, score = kps_2d[kp_idx], scores_2d[kp_idx]
                            if score >= keypoint_confidence: valid_kps_view.append([kp[0], kp[1], score])
                            else: valid_kps_view.append([np.nan, np.nan, score])
                    else: valid_kps_view = [[np.nan, np.nan, 0.0]] * 17 # Placeholder if shapes mismatch or invalid
                    persons_in_view.append(np.array(valid_kps_view))
                any_persons_detected_this_sync = True
            else: print(f"Warn: Mismatch len kps/scores p{port}"); persons_in_view = []
            view_results[port] = persons_in_view

        # --- Identify Person to Triangulate ---
        person_idx_to_triangulate = {} # {port: person_idx_in_that_port}
        found_match_for_tracking = False

        if can_track and any_persons_detected_this_sync:
            # --- Attempt to match tracked person ---
            candidate_distances = {} # { person_idx_in_ref_view : total_distance }
            ref_port = common_ports[0] # Choose a reference port

            if ref_port in projected_heads_2d and ref_port in view_results and view_results[ref_port]:
                projected_ref_head = projected_heads_2d[ref_port]
                # Find the closest person in the reference view
                min_dist_ref = float('inf')
                best_person_idx_ref = -1
                for p_idx, person_kps in enumerate(view_results[ref_port]):
                    head_2d = person_kps[head_kp_index, :2] # Get head XY
                    if not np.isnan(head_2d).any():
                        dist = math.dist(head_2d, projected_ref_head)
                        if dist < min_dist_ref:
                            min_dist_ref = dist
                            best_person_idx_ref = p_idx

                # Check if the closest match is within threshold
                if best_person_idx_ref != -1 and min_dist_ref < tracking_max_2d_dist:
                    # Assume this is the tracked person
                    found_match_for_tracking = True
                    # Gather this person's index from all views (simplistic: use same index)
                    # A more robust method would check distances in other views too
                    for p in common_ports:
                        # Check if view has enough people and the specific head is valid
                        if p in view_results and len(view_results[p]) > best_person_idx_ref:
                             candidate_head = view_results[p][best_person_idx_ref][head_kp_index, :2]
                             if not np.isnan(candidate_head).any():
                                  person_idx_to_triangulate[p] = best_person_idx_ref
                        # else: print(f"Debug: Cannot get kps for person {best_person_idx_ref} in port {p}")

        if not found_match_for_tracking and any_persons_detected_this_sync:
            # --- Attempt to initialize track with person 0 ---
            # print(f"Debug: No track or lost track. Attempting to initialize with Person 0. Sync {sync_index}")
            tracked_person_3d_kps = None # Ensure previous track is cleared
            can_initialize = False
            for p in common_ports:
                if p in view_results and view_results[p]: # Check if view has any person
                     head_0 = view_results[p][0][head_kp_index, :2]
                     if not np.isnan(head_0).any():
                          person_idx_to_triangulate[p] = 0 # Use index 0
                          can_initialize = True # Mark that we can try initializing
                     # else: print(f"Debug: Person 0 head is NaN in port {p}")
            if not can_initialize:
                person_idx_to_triangulate = {} # Cannot initialize if no valid person 0 head found


        # --- Triangulation ---
        keypoints_3d = None # Reset for this sync index
        if len(person_idx_to_triangulate) >= 2: # Need data from at least 2 views
            kps_to_triangulate_dict = {}
            num_valid_views_for_tri = 0
            for port, p_idx in person_idx_to_triangulate.items():
                 # Double check data exists and is valid shape (52, 3)
                 if port in view_results and len(view_results[port]) > p_idx:
                      kps = view_results[port][p_idx]
                      if isinstance(kps, np.ndarray) and kps.shape == (52, 3):
                           kps_to_triangulate_dict[port] = kps
                           num_valid_views_for_tri += 1
                      # else: print(f"Debug: Invalid kps format for port {port}, person {p_idx}")
                 # else: print(f"Debug: Data missing for port {port}, person {p_idx}")


            if num_valid_views_for_tri >= 2:
                # print(f"Debug: Attempting triangulation s{sync_index} with {num_valid_views_for_tri} views. Indices: {person_idx_to_triangulate}")
                keypoints_3d = triangulate_keypoints(
                    kps_to_triangulate_dict, port_to_cam_index, camera_params, projection_matrices
                )

        # --- Update Track and Store Results ---
        if keypoints_3d and any(kp is not None for kp in keypoints_3d):
            # Successfully triangulated
            tracked_person_3d_kps = keypoints_3d # Update track state for next frame
            # print(f"Debug: Track UPDATED s{sync_index}")

            kps_3d_list = [[np.nan]*3 if kp is None else kp.tolist() for kp in keypoints_3d]
            results_3d.append({
                'sync_index': sync_index,
                'person_id': tracked_person_id_counter, # Use the consistent tracked ID
                'keypoints_3d': kps_3d_list
            })
        else:
            # Triangulation failed or no match found
            if tracked_person_3d_kps is not None:
                 print(f"Info: Track lost at sync_index {sync_index}")
            tracked_person_3d_kps = None # Lose track
            # Do not increment tracked_person_id_counter if track is lost/not found

        # Increment ID counter ONLY when starting a new track
        if found_match_for_tracking == False and tracked_person_3d_kps is not None:
             # This means we just successfully initialized
             tracked_person_id_counter += 1


    # --- 8. Cleanup and Save Results ---
    print("\nReleasing video captures...")
    [cap.release() for cap in caps.values()]
    end_time = time.time()
    print(f"Processing finished in {end_time - start_time:.2f} seconds.")

    if not results_3d:
        print("No 3D poses were successfully generated.")
        return

    print(f"Generated 3D poses for {len(results_3d)} tracked person instances.")

    # --- Define output filenames ---
    output_csv_path = output_path # Use the provided path for CSV
    output_pickle_path = os.path.splitext(output_path)[0] + ".pkl"

    # --- 1. Save as Pickle (Compact Python Object) ---
    print(f"Saving raw results to Pickle file: {output_pickle_path}...")
    try:
        with open(output_pickle_path, 'wb') as f_pkl:
            pickle.dump(results_3d, f_pkl)
        print("Pickle file saved successfully.")
    except Exception as e:
        print(f"Error saving results to pickle: {e}")

    # --- 2. Prepare and Save as Wide CSV ---
    print(f"Preparing data for wide CSV format...")
    csv_rows = []
    expected_num_kps = SynthPoseMarkers.num_markers # Should be 52

    for result in tqdm(results_3d, desc="Formatting CSV Data"):
        row_data = {
            'sync_index': result['sync_index'],
            'person_id': int(result['person_id']) # Ensure integer ID
        }

        kps_list = result['keypoints_3d']

        # Check if the keypoint list has the expected length
        if len(kps_list) != expected_num_kps:
            print(f"Warning: Sync {result['sync_index']}, Person {result['person_id']} has {len(kps_list)} keypoints, expected {expected_num_kps}. Padding with NaNs.")
            # Pad with NaNs if too short, truncate if too long (or handle differently)
            kps_list.extend([[np.nan, np.nan, np.nan]] * (expected_num_kps - len(kps_list)))
            kps_list = kps_list[:expected_num_kps]

        for kp_idx in range(expected_num_kps):
            marker_name = SynthPoseMarkers.markers.get(kp_idx, f"KP_{kp_idx}") # Get marker name
            coords = kps_list[kp_idx] # Should be [x, y, z] or [nan, nan, nan]

            # Ensure coords is a list/tuple of 3 elements, handle None if necessary
            if coords is None or not isinstance(coords, (list, tuple)) or len(coords) != 3:
                 x, y, z = np.nan, np.nan, np.nan
            else:
                 x, y, z = coords[0], coords[1], coords[2]

            row_data[f"{marker_name}_X"] = x
            row_data[f"{marker_name}_Y"] = y
            row_data[f"{marker_name}_Z"] = z

        csv_rows.append(row_data)

    if not csv_rows:
         print("No valid data formatted for CSV.")
         return

    print(f"Saving formatted results to CSV file: {output_csv_path}...")
    try:
        results_df_wide = pd.DataFrame(csv_rows)
        # Define column order (optional but good practice)
        cols = ['sync_index', 'person_id']
        for i in range(expected_num_kps):
            marker_name = SynthPoseMarkers.markers.get(i, f"KP_{i}")
            cols.extend([f"{marker_name}_X", f"{marker_name}_Y", f"{marker_name}_Z"])
        # Reindex DataFrame to ensure columns are in order, adding missing ones if any
        results_df_wide = results_df_wide.reindex(columns=cols)

        results_df_wide.to_csv(output_csv_path, index=False, float_format='%.4f') # Format floats
        print("CSV file saved successfully.")

    except Exception as e:
        print(f"Error saving results to CSV: {e}")

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process synchronized multi-camera video for 3D pose estimation with tracking.")
    # (Add the new arguments if needed, or keep defaults)
    testfolder = "recording_3by1"
    parser.add_argument("--csv_path", default=f"test/caliscope/{testfolder}/frame_time_history.csv",help="Path...")
    parser.add_argument("--calibration_path",default=f"test/caliscope/{testfolder}/config.toml", help="Path...")
    parser.add_argument("--video_dir", default=f"test/caliscope/{testfolder}", help="Dir...")
    parser.add_argument("--output_path", default=f"output/caliscope/{testfolder}/output_3d_poses_tracked.csv",help="Path...")
    parser.add_argument("--model_dir", default=LOCAL_SP_DIR, help="Path...")
    parser.add_argument("--detector_dir", default=None, help="Path...")
    parser.add_argument("--calib_type", default="mwc", choices=['mwc', 'fmc'], help="Type...")
    parser.add_argument("--skip", type=int, default=1, help="Skip N...")
    parser.add_argument("--person_conf", type=float, default=0.8, help="Conf...")
    parser.add_argument("--keypoint_conf", type=float, default=0.1, help="Conf...")
    parser.add_argument("--device", default="auto", choices=['auto', 'cpu', 'mps', 'cuda'], help="Device...")
    parser.add_argument("--track_max_dist", type=float, default=100.0, help="Max 2D pixel distance for head tracking match (default: 100).")
    parser.add_argument("--head_idx", type=int, default=0, help="Index of keypoint used for tracking (default: 0, Nose).")


    args = parser.parse_args()
    process_synced_frames(
        csv_path=args.csv_path, calibration_path=args.calibration_path, video_dir=args.video_dir,
        output_path=args.output_path, model_dir=args.model_dir, detector_dir=args.detector_dir,
        calib_type=args.calib_type, skip_sync_indices=args.skip, person_confidence=args.person_conf,
        keypoint_confidence=args.keypoint_conf, device_name=args.device,
        tracking_max_2d_dist=args.track_max_dist, head_kp_index=args.head_idx # Pass tracking params
    )