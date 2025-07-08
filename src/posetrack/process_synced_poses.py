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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import glob

from .cs_parse import parse_calibration_mwc, parse_calibration_fmc, calculate_projection_matrices, triangulate_keypoints
from .pose_detector import load_models, detect_persons, estimate_poses, SynthPoseMarkers, LOCAL_DET_DIR, LOCAL_SP_DIR
from .libwalk import quick_rotation_matrix

# --- Add the project_3d_to_2d helper function here or import it ---
def project_3d_to_2d(point_3d, P):
    """Projects a 3D point to 2D using a projection matrix."""
    if point_3d is None or np.isnan(point_3d).any(): return None
    point_4d = np.append(point_3d, 1.0)
    point_2d_hom = P @ point_4d
    if abs(point_2d_hom[2]) < 1e-6 : return None # Check for near-zero depth
    point_2d = point_2d_hom[:2] / point_2d_hom[2]
    return point_2d.flatten()

def process_synced_mwc_frames_multi_person(
    csv_path, calibration_path, video_dir, output_path, model_dir=LOCAL_SP_DIR,
    detector_dir=LOCAL_DET_DIR, calib_type='mwc', skip_sync_indices=1, person_confidence=0.3,
    keypoint_confidence=0.1, device_name="auto",
    # --- Tracking parameters ---
    max_persons=2,  # Maximum number of persons to track
    com_distance_threshold=0.3,  # meters - minimum distance between COMs to be different people
    track_lost_patience=10,  # frames to wait before considering a track lost
    min_keypoints_for_com=2,  # minimum valid hip keypoints needed to compute COM
    hip_indices=(11, 12),  # COCO format: left hip, right hip
    epipolar_threshold=30,  # pixels - max distance from epipolar line
    reprojection_error_threshold=50,  # pixels - max reprojection error after triangulation
    min_views_for_detection=2,  # minimum camera views to confirm a person
    iou_threshold=0.3,  # for matching bounding boxes across views
    temporal_smoothing_window=5,  # frames for temporal consistency
    min_track_length=5,  # minimum frames before considering a track valid
    max_com_velocity=2.0,  # m/s - maximum reasonable COM velocity between frames
    ):
    """Processes synchronized frames with multi-person 3D tracking using COM-based matching."""

    # --- 1. Setup Device (same as before) ---
    if device_name == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"
        elif torch.cuda.is_available(): device = "cuda"
        else: device = "cpu"
    else:
        device = device_name
        if device == "mps" and (not torch.backends.mps.is_available() or not torch.backends.mps.is_built()): 
            print("Warn: MPS unavailable."); device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available(): 
            print("Warn: CUDA unavailable."); device = "cpu"
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

    # --- 6.5. Validate Sync Range vs Video Length ---
    all_sync_indices = sorted(frame_history_df['sync_index'].unique())
    sync_range = all_sync_indices[-1] - all_sync_indices[0] + 1  # +1 because both endpoints inclusive
    
    print(f"\n--- Sync Validation ---")
    print(f"Sync index range: {all_sync_indices[0]} to {all_sync_indices[-1]} ({len(all_sync_indices)} unique indices)")
    print(f"Expected sync range: {sync_range} frames")
    
    # Check if sync range matches video lengths (should be exact or very close)
    sync_matches_video = True
    for port in common_ports:
        video_len = video_lengths[port]
        if abs(sync_range - video_len) > 1:  # Allow only 1 frame difference
            print(f"WARNING: Port {port} video length ({video_len}) doesn't match sync range ({sync_range})")
            sync_matches_video = False
        else:
            print(f"Port {port}: sync range ({sync_range}) matches video length ({video_len}) ✓")
    
    if sync_matches_video:
        print("✓ Frame alignment validation PASSED - videos appear to share first and last frames with sync data")
    else:
        print("⚠ Frame alignment validation FAILED - cannot assume sync_index corresponds to video frame positions!")
        print("  This may indicate missing frames at start/end or sync timing issues.")


    # --- Multi-Person Tracking State ---
    class PersonTrack:
        def __init__(self, track_id, keypoints_3d, sync_index):
            self.id = track_id
            self.keypoints_3d_history = [keypoints_3d]  # List of 3D keypoints over time
            self.last_seen_sync = sync_index
            self.frames_since_seen = 0
            self.is_active = True
            
        def update(self, keypoints_3d, sync_index):
            self.keypoints_3d_history.append(keypoints_3d)
            self.last_seen_sync = sync_index
            self.frames_since_seen = 0
            
        def increment_lost_counter(self):
            self.frames_since_seen += 1
            if self.frames_since_seen > track_lost_patience:
                self.is_active = False
                
        def get_com_3d(self):
            """Get 3D center of mass from last known keypoints."""
            if not self.keypoints_3d_history:
                return None
            last_kps = self.keypoints_3d_history[-1]
            if last_kps is None:
                return None
            
            # Get hip keypoints
            left_hip = last_kps[hip_indices[0]] if hip_indices[0] < len(last_kps) else None
            right_hip = last_kps[hip_indices[1]] if hip_indices[1] < len(last_kps) else None
            
            valid_hips = []
            if left_hip is not None and not np.isnan(left_hip).any():
                valid_hips.append(left_hip)
            if right_hip is not None and not np.isnan(right_hip).any():
                valid_hips.append(right_hip)
                
            if len(valid_hips) >= min_keypoints_for_com:
                return np.mean(valid_hips, axis=0)
            return None

    active_tracks = []  # List of PersonTrack objects
    next_track_id = 0
    all_results_by_person = {}  # {person_id: [results]}

    # --- 7. Process Synchronized Frames ---
    all_sync_indices = sorted(frame_history_df['sync_index'].unique())
    start_time = time.time()

    print(f"Starting multi-person processing for {len(all_sync_indices)} sync indices...")
    sync_index_counter = 0
    
    for sync_index in tqdm(all_sync_indices, desc="Processing Sync Indices"):
        sync_index_counter += 1
        if (sync_index_counter - 1) % skip_sync_indices != 0: 
            continue

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

        # --- Detect and estimate poses in all views ---
        view_detections = {}  # {port: [(confidence, person_data), ...]} sorted by confidence
        
        for port in common_ports:
            image = current_frames_pil[port]
            person_boxes_voc, person_boxes_coco, person_scores = detect_persons(
                image, person_processor, person_model, device, person_confidence
            )
            
            if person_boxes_coco.size == 0: 
                view_detections[port] = []
                continue

            all_keypoints_2d, all_keypoint_scores = estimate_poses(
                image, person_boxes_coco, pose_processor, pose_model, device
            )

            port_detections = []
            
            for person_idx in range(len(all_keypoints_2d)):
                kps_2d, scores_2d = all_keypoints_2d[person_idx], all_keypoint_scores[person_idx]
                detection_confidence = person_scores[person_idx]  # Use detection confidence
                
                valid_kps_view = []
                
                # Process keypoints
                for kp_idx in range(len(kps_2d)):
                    kp, score = kps_2d[kp_idx], scores_2d[kp_idx]
                    if score >= keypoint_confidence:
                        valid_kps_view.append([kp[0], kp[1], score])
                    else:
                        valid_kps_view.append([np.nan, np.nan, score])
                
                # Calculate 2D hip center for tracking
                hip_2d = calculate_2d_com(valid_kps_view, hip_indices)
                
                person_data = {
                    'keypoints': np.array(valid_kps_view),
                    'hip_2d': hip_2d,
                    'confidence': detection_confidence
                }
                
                port_detections.append((detection_confidence, person_data))
            
            # Sort by confidence (highest first)
            port_detections.sort(key=lambda x: x[0], reverse=True)
            view_detections[port] = port_detections
            
            #debugging 
            #print(f"Port {port}: {len(port_detections)} detections, confidences: {[f'{conf:.2f}' for conf, _ in port_detections]}")

        # --- Generate 3D candidates using all possible combinations ---
        candidate_3d_persons = generate_3d_candidates(
            view_detections, port_to_cam_index, camera_params, projection_matrices, hip_indices
        )

        # --- Associate 3D candidates with existing tracks ---
        track_assignments, used_candidates = assign_3d_candidates_to_tracks(
            active_tracks, candidate_3d_persons, com_distance_threshold
        )
        
        # Update existing tracks
        for track_idx, candidate_idx in track_assignments.items():
            if candidate_idx is not None:
                candidate = candidate_3d_persons[candidate_idx]
                active_tracks[track_idx].update(candidate['keypoints_3d'], sync_index)
                
                # Store result
                person_id = active_tracks[track_idx].id
                if person_id not in all_results_by_person:
                    all_results_by_person[person_id] = []
                    
                kps_3d_list = [[np.nan]*3 if kp is None else kp.tolist() 
                               for kp in candidate['keypoints_3d']]
                all_results_by_person[person_id].append({
                    'sync_index': sync_index,
                    'person_id': person_id,
                    'keypoints_3d': kps_3d_list
                })
                
                #print(f"Updated track {person_id} with candidate quality {candidate['quality']:.3f}")
        
        # Create new tracks for high-quality unassigned candidates
        for candidate_idx, candidate in enumerate(candidate_3d_persons):
            if candidate_idx not in used_candidates and len(active_tracks) < max_persons:
                # Only create new tracks for high-quality candidates
                if candidate['quality'] > person_confidence:  # Minimum quality threshold
                    new_track = PersonTrack(next_track_id, candidate['keypoints_3d'], sync_index)
                    active_tracks.append(new_track)
                    
                    # Store first result for new track
                    if next_track_id not in all_results_by_person:
                        all_results_by_person[next_track_id] = []
                        
                    kps_3d_list = [[np.nan]*3 if kp is None else kp.tolist() 
                                   for kp in candidate['keypoints_3d']]
                    all_results_by_person[next_track_id].append({
                        'sync_index': sync_index,
                        'person_id': next_track_id,
                        'keypoints_3d': kps_3d_list
                    })
                    
                    print(f"Started tracking new person ID {next_track_id} at sync {sync_index} (quality: {candidate['quality']:.3f})")
                    next_track_id += 1
        
        # Update lost counters for unmatched tracks
        for i, track in enumerate(active_tracks):
            if i not in track_assignments or track_assignments[i] is None:
                track.increment_lost_counter()
                if not track.is_active:
                    print(f"Lost track of person {track.id} at sync {sync_index}")
        
        # Remove inactive tracks
        active_tracks = [t for t in active_tracks if t.is_active]

    # --- 8. Cleanup and Save Results ---
    print("\nReleasing video captures...")
    [cap.release() for cap in caps.values()]
    end_time = time.time()
    print(f"Processing finished in {end_time - start_time:.2f} seconds.")

    # Save results for each person
    base_output_path = output_path
    
    for person_id, results in all_results_by_person.items():
        if not results:
            continue
            
        print(f"\nSaving results for Person {person_id} ({len(results)} frames)...")
        
        # Define output filenames for this person
        person_csv_path = f"{base_output_path}_person{person_id}.csv"
        person_pickle_path = f"{base_output_path}_person{person_id}.pkl"
        
        # Save pickle
        try:
            with open(person_pickle_path, 'wb') as f_pkl:
                pickle.dump(results, f_pkl)
            print(f"Saved pickle: {person_pickle_path}")
        except Exception as e:
            print(f"Error saving pickle for person {person_id}: {e}")
        
        # Save CSV (same format as original)
        save_person_csv(results, person_csv_path, expected_num_kps=52)
    
    print(f"\nTotal persons tracked: {len(all_results_by_person)}")
    return all_results_by_person


# --- Helper Functions ---

def calculate_2d_com(keypoints_2d, hip_indices):
    """Calculate 2D center of mass from hip keypoints."""
    valid_hips = []
    for hip_idx in hip_indices:
        if hip_idx < len(keypoints_2d):
            hip = keypoints_2d[hip_idx][:2]  # Just x,y
            if not np.isnan(hip).any():
                valid_hips.append(hip)
    
    if len(valid_hips) >= 1:  # Even one hip is enough for 2D
        return np.mean(valid_hips, axis=0)
    return None


def generate_3d_candidates(view_detections, port_to_cam_index, camera_params, projection_matrices, hip_indices, consolidation_threshold=0.15):
    """
    Generate 3D person candidates by triangulating all possible combinations of detections across views.
    Returns candidates sorted by quality (confidence + triangulation consistency).
    """
    import itertools
    
    # Get all view ports that have detections
    active_ports = [port for port, detections in view_detections.items() if len(detections) > 0]
    
    if len(active_ports) < 2:
        print(f"Not enough views with detections ({len(active_ports)} < 2)")
        return []
    
    # print(f"Generating 3D candidates from {len(active_ports)} views: {active_ports}")
    
    candidates = []
    
    # Generate all combinations of 2 or more views
    for num_views in range(2, len(active_ports) + 1):
        for view_combination in itertools.combinations(active_ports, num_views):
            # Get all possible detection combinations across these views
            detection_indices = []
            for port in view_combination:
                num_detections = len(view_detections[port])
                detection_indices.append(list(range(num_detections)))
            
            # Generate all combinations of detection indices
            for detection_combo in itertools.product(*detection_indices):
                # Build triangulation input
                kps_to_triangulate = {}
                total_confidence = 0
                valid_views = 0
                
                for i, port in enumerate(view_combination):
                    detection_idx = detection_combo[i]
                    confidence, person_data = view_detections[port][detection_idx]
                    
                    if person_data['hip_2d'] is not None:  # Must have valid hip position
                        kps_to_triangulate[port] = person_data['keypoints']
                        total_confidence += confidence
                        valid_views += 1
                
                if valid_views >= 2:  # Need at least 2 valid views
                    # Triangulate this combination
                    keypoints_3d = triangulate_keypoints(
                        kps_to_triangulate, port_to_cam_index, 
                        camera_params, projection_matrices
                    )
                    
                    if keypoints_3d and any(kp is not None for kp in keypoints_3d):
                        # Calculate 3D hip position for tracking
                        hip_3d = calculate_3d_com_from_keypoints(keypoints_3d, hip_indices)
                        
                        if hip_3d is not None:
                            # Calculate triangulation quality score
                            avg_confidence = total_confidence / valid_views
                            triangulation_score = calculate_triangulation_quality(
                                kps_to_triangulate, keypoints_3d, projection_matrices, 
                                port_to_cam_index, hip_indices
                            )
                            
                            candidate_quality = avg_confidence * triangulation_score
                            
                            candidate = {
                                'keypoints_3d': keypoints_3d,
                                'hip_3d': hip_3d,
                                'quality': candidate_quality,
                                'avg_confidence': avg_confidence,
                                'triangulation_score': triangulation_score,
                                'views_used': view_combination,
                                'detection_indices': detection_combo
                            }
                            candidates.append(candidate)
    
    # Sort by quality (highest first)
    candidates.sort(key=lambda x: x['quality'], reverse=True)
    
    print(f"Generated {len(candidates)} raw 3D candidates")
    
    # Consolidate candidates that likely represent the same person
    consolidated_candidates = consolidate_duplicate_candidates(candidates, hip_distance_threshold=consolidation_threshold)
    
    print(f"After consolidation: {len(consolidated_candidates)} unique candidates")
    if consolidated_candidates:
        print(f"Best candidate quality: {consolidated_candidates[0]['quality']:.3f} (conf: {consolidated_candidates[0]['avg_confidence']:.2f}, tri: {consolidated_candidates[0]['triangulation_score']:.3f})")
    
    return consolidated_candidates


def calculate_triangulation_quality(kps_to_triangulate, keypoints_3d, projection_matrices, port_to_cam_index, hip_indices):
    """
    Calculate quality of triangulation by measuring reprojection error of hip keypoints.
    Returns score between 0 and 1 (1 = perfect, 0 = terrible).
    """
    # Focus on hip keypoints for quality assessment
    hip_errors = []
    
    for hip_idx in hip_indices:
        if hip_idx < len(keypoints_3d) and keypoints_3d[hip_idx] is not None:
            hip_3d = keypoints_3d[hip_idx]
            
            # Calculate reprojection error across all views
            for port, kps_2d in kps_to_triangulate.items():
                if hip_idx < len(kps_2d):
                    observed_2d = kps_2d[hip_idx][:2]
                    if not np.isnan(observed_2d).any():
                        # Project 3D back to 2D
                        cam_idx = port_to_cam_index[port]
                        P = projection_matrices[cam_idx]
                        projected_2d = project_3d_to_2d(hip_3d, P)
                        
                        if projected_2d is not None:
                            error = np.linalg.norm(observed_2d - projected_2d)
                            hip_errors.append(error)
    
    if not hip_errors:
        return 0.0
    
    # Convert pixel error to quality score (lower error = higher quality)
    avg_error = np.mean(hip_errors)
    max_acceptable_error = 50  # pixels
    quality = max(0, 1 - (avg_error / max_acceptable_error))
    
    return quality


def consolidate_duplicate_candidates(candidates, hip_distance_threshold):
    """
    Consolidate 3D candidates that likely represent the same physical person.
    
    TODO: This currently just picks the "best quality" candidate from each group,
    which throws away valuable triangulation data. A better approach would be:
    - Kalman filter to combine multiple estimates
    - Weighted average based on triangulation quality
    - Outlier detection to reject bad triangulations
    - Consider temporal consistency with previous frames
    
    Args:
        candidates (list): List of candidate dictionaries with 'hip_3d' and 'quality' keys
        hip_distance_threshold (float): Distance threshold in meters (fragile parameter!)
        
    Returns:
        list: Consolidated candidates (currently just picks best from each group)
    """
    if len(candidates) <= 1:
        return candidates
    
    consolidated = []
    used_indices = set()
    
    print(f"Consolidating {len(candidates)} candidates with threshold {hip_distance_threshold}m:")
    
    for i, candidate_a in enumerate(candidates):
        if i in used_indices:
            continue
            
        # Find all candidates within threshold of this one
        group = [i]
        hip_a = candidate_a['hip_3d']
        
        if hip_a is None:
            continue
            
        for j, candidate_b in enumerate(candidates[i+1:], start=i+1):
            if j in used_indices:
                continue
                
            hip_b = candidate_b['hip_3d']
            if hip_b is None:
                continue
                
            distance = np.linalg.norm(hip_a - hip_b)
            if distance <= hip_distance_threshold:
                group.append(j)
                print(f"  Grouping candidates {i} and {j} (hip distance: {distance:.3f}m)")
        
        # Mark all group members as used
        for idx in group:
            used_indices.add(idx)
        
        # TODO: Replace this naive "pick best" approach with proper sensor fusion
        # Current approach: Keep the highest quality candidate from this group
        if len(group) == 1:
            consolidated.append(candidates[i])
        else:
            best_candidate = max([candidates[idx] for idx in group], key=lambda x: x['quality'])
            consolidated.append(best_candidate)
            print(f"  Kept 'best' candidate from group of {len(group)} (quality: {best_candidate['quality']:.3f}) - TODO: Should fuse all estimates!")
    
    return consolidated


def group_detections_across_views(detected_persons_2d, view_results, 
                                 projection_matrices, port_to_cam_index, 
                                 camera_params, epipolar_threshold=30):
    """
    Group person detections across views using epipolar constraints.
    """
    person_groups = []
    ports = list(detected_persons_2d.keys())
    
    if len(ports) < 2:
        return person_groups
    
    # Build cost matrix for Hungarian algorithm
    all_detections = []
    detection_to_view = []
    
    for port in ports:
        for person_idx, com_2d in detected_persons_2d[port]:
            all_detections.append((port, person_idx, com_2d))
            detection_to_view.append(port)
    
    n_detections = len(all_detections)
    
    # For each pair of views, calculate epipolar distances
    epipolar_distances = {}
    
    for i in range(len(ports)):
        for j in range(i+1, len(ports)):
            port1, port2 = ports[i], ports[j]
            cam_idx1 = port_to_cam_index[port1]
            cam_idx2 = port_to_cam_index[port2]
            
            # Calculate fundamental matrix between views
            F = calculate_fundamental_matrix(
                projection_matrices[cam_idx1], 
                projection_matrices[cam_idx2]
            )
            
            # Calculate epipolar distances for all detection pairs
            for det1_idx, (p1, idx1, com1) in enumerate(all_detections):
                if p1 != port1:
                    continue
                    
                for det2_idx, (p2, idx2, com2) in enumerate(all_detections):
                    if p2 != port2:
                        continue
                    
                    # Calculate distance from com2 to epipolar line of com1
                    dist = point_to_epipolar_line_distance(com1, com2, F)
                    epipolar_distances[(det1_idx, det2_idx)] = dist
    
    # Use graph-based clustering or Hungarian algorithm to find consistent groups
    groups = cluster_with_epipolar_constraints(
        all_detections, epipolar_distances, epipolar_threshold
    )
    
    return groups

def assign_3d_candidates_to_tracks(active_tracks, candidates, distance_threshold):
    """
    Assign existing tracks to 3D candidates based on hip distance, preferring higher quality candidates.
    Returns (assignments dict, set of used candidate indices)
    """
    assignments = {}
    used_candidates = set()
    
    # For each active track, find the best matching candidate
    for track_idx, track in enumerate(active_tracks):
        track_com = track.get_com_3d()
        if track_com is None:
            assignments[track_idx] = None
            continue
        
        best_match_idx = None
        best_score = -1  # Combined distance + quality score
        
        for candidate_idx, candidate in enumerate(candidates):
            if candidate_idx in used_candidates:
                continue
                
            candidate_com = candidate['hip_3d']
            if candidate_com is None:
                continue
                
            # Calculate distance
            dist = np.linalg.norm(track_com - candidate_com)
            
            if dist < distance_threshold:
                # Score combines proximity and quality (closer + higher quality = better)
                proximity_score = max(0, 1 - (dist / distance_threshold))  # 1=very close, 0=at threshold
                combined_score = 0.5 * proximity_score + 0.5 * candidate['quality']
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match_idx = candidate_idx
        
        if best_match_idx is not None:
            assignments[track_idx] = best_match_idx
            used_candidates.add(best_match_idx)
            #print(f"Assigned track {track_idx} to candidate {best_match_idx} (score: {best_score:.3f})")
        else:
            assignments[track_idx] = None
    
    return assignments, used_candidates


def assign_tracks_to_detections(active_tracks, new_persons_3d, distance_threshold):
    """
    Legacy function - kept for compatibility.
    Assign existing tracks to new 3D detections based on COM distance.
    Returns dict: {track_index: detection_index or None}
    """
    assignments = {}
    used_detections = set()
    
    # Calculate COM for each new detection
    new_coms = []
    for person_3d in new_persons_3d:
        com = calculate_3d_com_from_keypoints(person_3d, hip_indices=(11, 12))
        new_coms.append(com)
    
    # Assign tracks to detections
    for track_idx, track in enumerate(active_tracks):
        track_com = track.get_com_3d()
        if track_com is None:
            assignments[track_idx] = None
            continue
        
        best_match_idx = None
        best_match_dist = float('inf')
        
        for det_idx, det_com in enumerate(new_coms):
            if det_com is None or det_idx in used_detections:
                continue
                
            dist = np.linalg.norm(track_com - det_com)
            if dist < distance_threshold and dist < best_match_dist:
                best_match_dist = dist
                best_match_idx = det_idx
        
        if best_match_idx is not None:
            assignments[track_idx] = best_match_idx
            used_detections.add(best_match_idx)
        else:
            assignments[track_idx] = None
    
    return assignments

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def calculate_fundamental_matrix(P1, P2):
    """
    Calculate the fundamental matrix F from two projection matrices.
    
    The fundamental matrix relates corresponding points x1 and x2 in two images:
    x2^T * F * x1 = 0
    
    Args:
        P1: 3x4 projection matrix for camera 1
        P2: 3x4 projection matrix for camera 2
    
    Returns:
        F: 3x3 fundamental matrix
    """
    # Method: F = [e2]_x * P2 * P1^+
    # where P1^+ is the pseudo-inverse of P1
    # and [e2]_x is the skew-symmetric matrix of the epipole e2
    
    # Calculate the camera center C1 (null space of P1)
    # P1 * C1 = 0, where C1 is in homogeneous coordinates
    U, S, Vt = np.linalg.svd(P1)
    C1 = Vt[-1, :]  # Last row of V^T (null space)
    C1 = C1 / C1[3]  # Normalize so last coordinate is 1
    
    # Project C1 into image 2 to get epipole e2
    e2 = P2 @ C1
    e2 = e2 / e2[2]  # Normalize
    
    # Create skew-symmetric matrix [e2]_x
    e2_cross = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    
    # Calculate F = [e2]_x * P2 * P1^+
    P1_pinv = np.linalg.pinv(P1)
    F = e2_cross @ P2 @ P1_pinv
    
    # Enforce rank-2 constraint (fundamental matrix should have rank 2)
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # Set smallest singular value to 0
    F = U @ np.diag(S) @ Vt
    
    # Normalize F
    F = F / np.linalg.norm(F)
    
    return F


def point_to_epipolar_line_distance(point1, point2, F):
    """
    Calculate the distance from point2 to the epipolar line of point1.
    
    The epipolar line l2 in image 2 corresponding to point1 in image 1 is:
    l2 = F * point1
    
    Args:
        point1: (x, y) coordinates in image 1
        point2: (x, y) coordinates in image 2
        F: 3x3 fundamental matrix
    
    Returns:
        distance: Perpendicular distance from point2 to epipolar line
    """
    # Convert points to homogeneous coordinates
    p1_homo = np.array([point1[0], point1[1], 1.0])
    p2_homo = np.array([point2[0], point2[1], 1.0])
    
    # Calculate epipolar line in image 2: l2 = F * p1
    l2 = F @ p1_homo
    
    # Distance from point to line ax + by + c = 0 is |ax + by + c| / sqrt(a² + b²)
    # For line l = [a, b, c] and point p = [x, y, 1]:
    # distance = |l^T * p| / sqrt(a² + b²)
    
    numerator = abs(np.dot(l2, p2_homo))
    denominator = np.sqrt(l2[0]**2 + l2[1]**2)
    
    if denominator < 1e-10:  # Avoid division by zero
        return float('inf')
    
    distance = numerator / denominator
    
    return distance


def cluster_with_epipolar_constraints(all_detections, epipolar_distances, 
                                    epipolar_threshold, min_views=2):
    """
    Cluster detections across views using epipolar constraints.
    
    This function groups detections that could belong to the same person based on
    epipolar geometry constraints. It uses graph-based clustering where edges
    exist between detections that satisfy the epipolar constraint.
    
    Args:
        all_detections: List of tuples (port, person_idx, com_2d)
        epipolar_distances: Dict mapping (det1_idx, det2_idx) -> distance
        epipolar_threshold: Maximum allowed epipolar distance
        min_views: Minimum number of views required for a valid group
    
    Returns:
        List of groups, where each group is a dict {port: person_idx}
    """
    n_detections = len(all_detections)
    
    if n_detections == 0:
        return []
    
    # Build adjacency matrix for detections that satisfy epipolar constraints
    # We'll use a sparse matrix for efficiency
    adjacency = np.zeros((n_detections, n_detections))
    
    # Add edges between detections from different views that satisfy epipolar constraint
    for (i, j), dist in epipolar_distances.items():
        if dist < epipolar_threshold:
            # Check that detections are from different cameras
            port_i = all_detections[i][0]
            port_j = all_detections[j][0]
            if port_i != port_j:
                adjacency[i, j] = 1
                adjacency[j, i] = 1
    
    # Find connected components in the graph
    # Each component represents a potential person seen across multiple views
    sparse_adj = csr_matrix(adjacency)
    n_components, labels = connected_components(sparse_adj, directed=False)
    
    # Build groups from connected components
    groups = []
    for component_id in range(n_components):
        component_indices = np.where(labels == component_id)[0]
        
        # Build a group from this component
        group = {}
        ports_in_group = set()
        
        for idx in component_indices:
            port, person_idx, _ = all_detections[idx]
            
            # Only add if we don't already have a detection from this port
            # (this handles cases where multiple detections might be in the same component)
            if port not in ports_in_group:
                group[port] = person_idx
                ports_in_group.add(port)
        
        # Only keep groups with minimum number of views
        if len(group) >= min_views:
            groups.append(group)
    
    # Handle remaining detections that might form valid groups
    # (This handles cases where a component might have multiple detections from same camera)
    if n_components > 0:
        groups = refine_groups_with_hungarian(groups, all_detections, 
                                            epipolar_distances, epipolar_threshold)
    
    return groups


def refine_groups_with_hungarian(initial_groups, all_detections, 
                                epipolar_distances, epipolar_threshold):
    """
    Refine initial groups using Hungarian algorithm to handle cases where
    multiple detections from the same camera might be in the same connected component.
    
    This can happen when epipolar lines are close together.
    """
    refined_groups = []
    
    for group in initial_groups:
        # Get all detections in this group's component
        group_detections = []
        detection_indices = []
        
        for i, (port, person_idx, com_2d) in enumerate(all_detections):
            if port in group and group[port] == person_idx:
                group_detections.append((port, person_idx, com_2d))
                detection_indices.append(i)
        
        # Check if we have multiple detections from any camera
        ports_count = {}
        for port, _, _ in group_detections:
            ports_count[port] = ports_count.get(port, 0) + 1
        
        if max(ports_count.values()) > 1:
            # Need to resolve conflicts using Hungarian algorithm
            # This is a more complex case - for now, just keep the original group
            refined_groups.append(group)
        else:
            refined_groups.append(group)
    
    return refined_groups


# Alternative implementation using Hungarian algorithm for better matching
def group_detections_across_views_hungarian(detected_persons_2d, view_results, 
                                           projection_matrices, port_to_cam_index,
                                           camera_params, epipolar_threshold=30):
    """
    Alternative implementation using Hungarian algorithm for optimal matching.
    This can be more robust when you have exactly 2 people to track.
    """
    ports = list(detected_persons_2d.keys())
    if len(ports) < 2:
        return []
    
    # For each pair of views, find optimal matching
    reference_port = ports[0]
    reference_detections = detected_persons_2d[reference_port]
    
    if not reference_detections:
        return []
    
    # Initialize groups with reference detections
    groups = [{reference_port: det[0]} for det in reference_detections]
    
    # Match with each other view
    for other_port in ports[1:]:
        other_detections = detected_persons_2d[other_port]
        if not other_detections:
            continue
        
        # Calculate fundamental matrix
        cam_idx1 = port_to_cam_index[reference_port]
        cam_idx2 = port_to_cam_index[other_port]
        F = calculate_fundamental_matrix(
            projection_matrices[cam_idx1],
            projection_matrices[cam_idx2]
        )
        
        # Build cost matrix
        n_ref = len(reference_detections)
        n_other = len(other_detections)
        cost_matrix = np.full((n_ref, n_other), 1000.0)  # High cost for non-matches
        
        for i, (ref_idx, ref_com) in enumerate(reference_detections):
            for j, (other_idx, other_com) in enumerate(other_detections):
                dist = point_to_epipolar_line_distance(ref_com, other_com, F)
                if dist < epipolar_threshold:
                    cost_matrix[i, j] = dist
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Update groups with matches
        for i, j in zip(row_indices, col_indices):
            if cost_matrix[i, j] < epipolar_threshold:
                # Find which group contains reference_detections[i]
                ref_person_idx = reference_detections[i][0]
                for group in groups:
                    if group.get(reference_port) == ref_person_idx:
                        other_person_idx = other_detections[j][0]
                        group[other_port] = other_person_idx
                        break
    
    # Filter groups that have minimum views
    final_groups = [g for g in groups if len(g) >= 2]
    
    return final_groups

def calculate_3d_com_from_keypoints(keypoints_3d, hip_indices):
    """Calculate 3D center of mass from keypoints."""
    if keypoints_3d is None:
        return None
        
    valid_hips = []
    for hip_idx in hip_indices:
        if hip_idx < len(keypoints_3d) and keypoints_3d[hip_idx] is not None:
            hip = keypoints_3d[hip_idx]
            if not np.isnan(hip).any():
                valid_hips.append(hip)
    
    if len(valid_hips) >= 1:
        return np.mean(valid_hips, axis=0)
    return None


def save_person_csv(results, csv_path, expected_num_kps):
    """Save results for one person to CSV."""
    csv_rows = []
    
    for result in results:
        row_data = {
            'sync_index': result['sync_index'],
            'person_id': result['person_id']
        }
        
        kps_list = result['keypoints_3d']
        
        # Pad/truncate to expected length
        if len(kps_list) != expected_num_kps:
            kps_list.extend([[np.nan, np.nan, np.nan]] * (expected_num_kps - len(kps_list)))
            kps_list = kps_list[:expected_num_kps]
        
        for kp_idx in range(expected_num_kps):
            marker_name = SynthPoseMarkers.markers.get(kp_idx, f"KP_{kp_idx}")
            coords = kps_list[kp_idx]
            
            if coords is None or not isinstance(coords, (list, tuple)) or len(coords) != 3:
                x, y, z = np.nan, np.nan, np.nan
            else:
                x, y, z = coords[0], coords[1], coords[2]
            
            row_data[f"{marker_name}_X"] = x
            row_data[f"{marker_name}_Y"] = y
            row_data[f"{marker_name}_Z"] = z
        
        csv_rows.append(row_data)
    
    if csv_rows:
        results_df = pd.DataFrame(csv_rows)
        results_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Saved CSV: {csv_path}")
# ---

def wrap_process_synced_mwc_frames(directory_name,model_dir=LOCAL_SP_DIR,
    detector_dir=LOCAL_DET_DIR, calib_type='mwc', skip_sync_indices=1, person_confidence=0.3,
    keypoint_confidence=0.1, device_name="auto"):
    """Wrapper to call the multi-person processing function with default parameters."""
    csv_path = os.path.join(directory_name, "frame_time_history.csv")
    calibration_path = os.path.join(directory_name, "config.toml")
    video_dir = directory_name
    output_path = os.path.join(directory_name)
    
    process_synced_mwc_frames(
        csv_path=csv_path, calibration_path=calibration_path, video_dir=video_dir,
        output_path=output_path, model_dir=model_dir, detector_dir=detector_dir,
        calib_type=calib_type, skip_sync_indices=skip_sync_indices,
        person_confidence=person_confidence, keypoint_confidence=keypoint_confidence,
        device_name=device_name)    

def wrap_process_synced_mwc_frames_multi_person(directory_name,model_dir=LOCAL_SP_DIR,
    detector_dir=LOCAL_DET_DIR, calib_type='mwc', skip_sync_indices=1, person_confidence=0.8,
    keypoint_confidence=0.1, device_name="auto",
    max_persons=2,  # Maximum number of persons to track
    com_distance_threshold=0.5,  # meters - minimum distance between COMs to be different people
    track_lost_patience=10,  # frames to wait before considering a track lost
    min_keypoints_for_com=2,  # minimum valid hip keypoints needed to compute COM
    hip_indices=(11, 12),  # COCO format: left hip, right hip
    ):
    """Wrapper to call the multi-person processing function with default parameters."""
    
    process_synced_mwc_frames_multi_person(
        csv_path=os.path.join(directory_name, "frame_time_history.csv"),
        calibration_path=os.path.join(directory_name, "config.toml"),
        video_dir=directory_name,
        output_path=os.path.join(directory_name),
        model_dir=model_dir,
        detector_dir=detector_dir,
        calib_type=calib_type,
        skip_sync_indices=skip_sync_indices,
        person_confidence=person_confidence,
        keypoint_confidence=keypoint_confidence,
        device_name=device_name,
        max_persons=max_persons,
        com_distance_threshold=com_distance_threshold)

def process_synced_mwc_frames(
    # ... (arguments remain the same) ...
    csv_path, calibration_path, video_dir, output_path, model_dir=LOCAL_SP_DIR,
    detector_dir=LOCAL_DET_DIR, calib_type='mwc', skip_sync_indices=1, person_confidence=0.3,
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

    # --- 6.5. Validate Sync Range vs Video Length ---
    all_sync_indices = sorted(frame_history_df['sync_index'].unique())
    sync_range = all_sync_indices[-1] - all_sync_indices[0] + 1  # +1 because both endpoints inclusive
    
    print(f"\n--- Sync Validation ---")
    print(f"Sync index range: {all_sync_indices[0]} to {all_sync_indices[-1]} ({len(all_sync_indices)} unique indices)")
    print(f"Expected sync range: {sync_range} frames")
    
    # Check if sync range matches video lengths (should be exact or very close)
    sync_matches_video = True
    for port in common_ports:
        video_len = video_lengths[port]
        if abs(sync_range - video_len) > 1:  # Allow only 1 frame difference
            print(f"WARNING: Port {port} video length ({video_len}) doesn't match sync range ({sync_range})")
            sync_matches_video = False
        else:
            print(f"Port {port}: sync range ({sync_range}) matches video length ({video_len}) ✓")
    
    if sync_matches_video:
        print("✓ Frame alignment validation PASSED - videos appear to share first and last frames with sync data")
    else:
        print("⚠ Frame alignment validation FAILED - cannot assume sync_index corresponds to video frame positions!")
        print("  This may indicate missing frames at start/end or sync timing issues.")

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
            plot_this_frame = False; debug_prefix = None # Or your debug conditions
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


def read_posetrack_csv(csv_path):
    """
    Read a posetrack CSV file and return a dictionary with nx3 numpy arrays 
    grouped by body part prefix (e.g., 'NOSE', 'LEFT_EYE', etc.).
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        dict: Dictionary with body part names as keys and nx3 numpy arrays as values
              where n is the number of frames and columns are [x, y, z]
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Get sync_index and person_id columns
        sync_indices = df['sync_index'].values
        person_ids = df['person_id'].values if 'person_id' in df.columns else np.zeros(len(df))
        
        # Find all body part columns (those ending with _X, _Y, _Z)
        all_columns = df.columns.tolist()
        body_part_names = set()
        
        for col in all_columns:
            if col.endswith('_X') or col.endswith('_Y') or col.endswith('_Z'):
                body_part = col[:-2]  # Remove _X, _Y, or _Z suffix
                body_part_names.add(body_part)
        
        # Create dictionary with body part data
        body_parts_data = {}
        
        for body_part in body_part_names:
            x_col = f"{body_part}_X"
            y_col = f"{body_part}_Y"
            z_col = f"{body_part}_Z"
            
            if x_col in df.columns and y_col in df.columns and z_col in df.columns:
                x_data = df[x_col].values
                y_data = df[y_col].values
                z_data = df[z_col].values
                
                # Stack into nx3 array
                body_parts_data[body_part] = np.column_stack([x_data, y_data, z_data])
        
        # Also include metadata and direct access to sync_index/person_id
        body_parts_data['_metadata'] = {
            'sync_indices': sync_indices,
            'person_ids': person_ids,
            'num_frames': len(df)
        }
        
        # Add sync_index and person_id as direct arrays if they exist
        if 'sync_index' in df.columns:
            body_parts_data['sync_index'] = sync_indices
        if 'person_id' in df.columns:
            body_parts_data['person_id'] = person_ids
        
        return body_parts_data
        
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return None


def show_multi_person_results(results_directory, output_plot_path=None):
    """
    Read all person CSV files from a directory and create a timeline plot 
    showing when each detected person was active.
    
    Args:
        results_directory (str): Directory containing *_person*.csv files
        output_plot_path (str, optional): Path to save the plot. If None, displays plot.
    """
    # Find all person CSV files
    person_files = glob.glob(os.path.join(results_directory, "*_person*.csv"))
    
    if not person_files:
        print(f"No person CSV files found in {results_directory}")
        return
    
    print(f"Found {len(person_files)} person files:")
    for f in person_files:
        print(f"  - {os.path.basename(f)}")
    
    # Read data from each person file
    person_data = {}
    person_files_data = {}
    
    for person_file in person_files:
        # Extract person ID from filename
        filename = os.path.basename(person_file)
        if '_person' in filename:
            try:
                person_id = int(filename.split('_person')[1].split('.')[0])
            except (ValueError, IndexError):
                person_id = filename  # fallback to filename if parsing fails
        else:
            person_id = filename
        
        # Read the CSV data
        data = read_posetrack_csv(person_file)
        if data is not None and '_metadata' in data:
            person_data[person_id] = data['_metadata']
            person_files_data[person_id] = data  # Store full data for animation
    
    if not person_data:
        print("No valid person data found")
        return
    
    # Create the timeline plot
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(person_data)))
    
    # Plot each person's activity
    for i, (person_id, metadata) in enumerate(person_data.items()):
        sync_indices = metadata['sync_indices']
        num_frames = metadata['num_frames']
        
        # Create y-position for this person
        y_pos = i + 1
        
        # Plot the active frames
        plt.scatter(sync_indices, [y_pos] * len(sync_indices), 
                   color=colors[i], alpha=0.7, s=20, label=f'Person {person_id}')
        
        # Add text annotation
        if len(sync_indices) > 0:
            plt.text(sync_indices[0], y_pos + 0.1, f'Person {person_id} ({num_frames} frames)', 
                    fontsize=10, ha='left')
    
    # Customize the plot
    plt.xlabel('Sync Index')
    plt.ylabel('Person ID')
    plt.title(f'Multi-Person Detection Timeline\nDirectory: {os.path.basename(results_directory)}')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set y-axis limits and labels
    if person_data:
        plt.ylim(0.5, len(person_data) + 0.5)
        plt.yticks(range(1, len(person_data) + 1), [f'Person {pid}' for pid in person_data.keys()])
    
    # Show statistics
    total_detections = sum(metadata['num_frames'] for metadata in person_data.values())
    print(f"\nDetection Summary:")
    print(f"  Total persons detected: {len(person_data)}")
    print(f"  Total frame detections: {total_detections}")
    
    for person_id, metadata in person_data.items():
        sync_indices = metadata['sync_indices']
        if len(sync_indices) > 0:
            sync_range = f"{sync_indices.min()}-{sync_indices.max()}"
            print(f"  Person {person_id}: {metadata['num_frames']} frames, sync range {sync_range}")
    
    plt.tight_layout()
    
    if output_plot_path:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_plot_path}")
    else:
        plt.show()
    
    return person_data, person_files_data


def animate_multi_person_results(results_directory, output_plot_path=None):
    """
    Create an interactive animation of multi-person 3D pose data with timeline and 3D visualization.
    
    Args:
        results_directory (str): Directory containing *_person*.csv files
        output_plot_path (str, optional): Path to save the plot. If None, displays plot.
    """
    # Get person data and loaded CSV data
    person_data, person_files_data = show_multi_person_results(results_directory, output_plot_path)
    
    if not person_data or not person_files_data:
        print("No person data available for animation")
        return
    
    # Define keypoints to animate (subset for clarity) - using actual SynthPoseMarkers names
    keypoints_to_animate = [
        'L_Ankle', 'R_Ankle',  # feet
        'L_Knee', 'R_Knee',    # knees  
        'L_Hip', 'R_Hip',      # hips
        'L_Shoulder', 'R_Shoulder',  # shoulders
        'L_Elbow', 'R_Elbow',  # elbows
        'L_Wrist', 'R_Wrist',  # wrists
        'L_Eye', 'R_Eye',      # eyes
        'Nose'                 # head
    ]
    
    # Get all unique sync indices across all persons
    all_sync_indices = set()
    for data in person_files_data.values():
        if 'sync_index' in data:
            all_sync_indices.update(data['sync_index'])
    
    if not all_sync_indices:
        print("No sync indices found in data")
        return
    
    all_sync_indices = sorted(list(all_sync_indices))
    min_sync = min(all_sync_indices)
    max_sync = max(all_sync_indices)
    
    print(f"Animation range: sync_index {min_sync} to {max_sync} ({len(all_sync_indices)} frames)")
    
    # Calculate rotation matrix and offset for the first person only
    rotation_matrix = np.eye(3)
    x0 = np.zeros(3)
    if person_files_data:
        # Find the person whose first sync_index is the smallest (i.e., detected first)
        min_first_sync = float('inf')
        first_person_id = None
        for pid, pdata in person_files_data.items():
            sync_indices = pdata.get('sync_index', [])
            if len(sync_indices) > 0 and sync_indices[0] < min_first_sync:
                min_first_sync = sync_indices[0]
                first_person_id = pid
        if first_person_id is None:
            # Fallback: just pick the first key if none found
            first_person_id = list(person_files_data.keys())[0]
        first_person_data = person_files_data[first_person_id]
        print(f"\nCalculating rotation matrix using Person {first_person_id}...")
        rotation_matrix, x0 = quick_rotation_matrix(first_person_data)
    
    # Create figure with subplots
    fig, (ax_timeline, ax_3d) = plt.subplots(2, 1, figsize=(16, 10), 
                                            gridspec_kw={'height_ratios': [1, 2]})
    
    # Make 3D subplot
    ax_3d.remove()
    ax_3d = fig.add_subplot(2, 1, 2, projection='3d')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(person_data)))
    
    # Plot timeline (same as show_multi_person_results)
    for i, (person_id, metadata) in enumerate(person_data.items()):
        sync_indices = metadata['sync_indices']
        y_pos = i + 1
        ax_timeline.scatter(sync_indices, [y_pos] * len(sync_indices), 
                          color=colors[i], alpha=0.7, s=20, label=f'Person {person_id}')
    
    ax_timeline.set_xlabel('Sync Index')
    ax_timeline.set_ylabel('Person ID')
    ax_timeline.set_title('Multi-Person Detection Timeline')
    ax_timeline.grid(True, alpha=0.3)
    ax_timeline.legend()
    
    # Add current frame indicator
    current_frame_line = ax_timeline.axvline(x=min_sync, color='red', linewidth=2, label='Current Frame')
    
    # 3D pose subplot
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Pose Visualization')
    
    # Set fixed axis limits
    ax_3d.set_xlim([-5, 5])
    ax_3d.set_ylim([-5, 5])
    ax_3d.set_zlim([-5, 5])
    
    # Initialize 3D scatter plots for each person with dummy data
    person_scatters = {}
    for i, person_id in enumerate(person_data.keys()):
        person_scatters[person_id] = ax_3d.scatter([0], [0], [0], 
                                                  color=colors[i], s=50, alpha=0.8, 
                                                  label=f'Person {person_id}')
    ax_3d.legend()
    
    # Animation state
    current_sync_idx = 0
    is_playing = False
    
    def update_frame(current_sync_index):
        """Update the 3D visualization for a given sync index."""
        # Update timeline indicator
        current_frame_line.set_xdata([current_sync_index, current_sync_index])
        
        # Count people displayed
        people_displayed = 0
        
        print(f"\n=== FRAME UPDATE: sync_index = {current_sync_index} ===")
        
        # Update 3D poses
        for i, (person_id, data) in enumerate(person_files_data.items()):
            if 'sync_index' not in data:
                continue
                
            # Find frame with matching sync index
            sync_mask = data['sync_index'] == current_sync_index
            
            if np.any(sync_mask):
                # Get keypoints for this frame
                x_coords, y_coords, z_coords = [], [], []
                
                # Track feet coordinates for debugging
                left_ankle_coord = None
                right_ankle_coord = None
                
                for keypoint_name in keypoints_to_animate:
                    if keypoint_name in data:
                        kp_data = data[keypoint_name][sync_mask]
                        if len(kp_data) > 0:
                            x, y, z = kp_data[0]  # First (and should be only) matching frame
                            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                                # Apply rotation matrix and translation
                                original_point = np.array([x, y, z])
                                rotated_point = rotation_matrix.T @ (original_point - x0)
                                
                                x_coords.append(rotated_point[0])
                                y_coords.append(rotated_point[1])
                                z_coords.append(rotated_point[2])
                                
                                # Capture feet coordinates for debugging (rotated coordinates)
                                if keypoint_name == 'L_Ankle':
                                    left_ankle_coord = tuple(rotated_point)
                                elif keypoint_name == 'R_Ankle':
                                    right_ankle_coord = tuple(rotated_point)
                
                # Update scatter plot data
                if x_coords:
                    people_displayed += 1
                    person_scatters[person_id]._offsets3d = (x_coords, y_coords, z_coords)
                    
                    # Debug info for this person
                    print(f"Person {person_id}: {len(x_coords)} keypoints")
                    if left_ankle_coord:
                        print(f"  Left Ankle:  ({left_ankle_coord[0]:.3f}, {left_ankle_coord[1]:.3f}, {left_ankle_coord[2]:.3f})")
                    if right_ankle_coord:
                        print(f"  Right Ankle: ({right_ankle_coord[0]:.3f}, {right_ankle_coord[1]:.3f}, {right_ankle_coord[2]:.3f})")
                else:
                    print(f"Person {person_id}: no valid keypoints")
                    person_scatters[person_id]._offsets3d = ([], [], [])
            else:
                # No data for this sync index
                person_scatters[person_id]._offsets3d = ([], [], [])
                print(f"Person {person_id}: not present at this sync_index")
        
        print(f"Total people displayed: {people_displayed}")
        print("=" * 50)
        
        # Redraw the plot
        fig.canvas.draw()
    
    def on_slider_change(val):
        """Handle slider changes."""
        nonlocal current_sync_idx, is_playing
        current_sync_idx = int(val)
        is_playing = False  # Stop auto-play when user interacts
        update_frame(current_sync_idx)  # val is now actual sync_index
    
    def on_play_button(event):
        """Handle play button clicks."""
        nonlocal is_playing, current_sync_idx
        is_playing = not is_playing
        if is_playing:
            play_button.label.set_text('Pause')
            # Simple loop for playback
            import threading
            def play_loop():
                sync_idx_iter = 0
                while is_playing:
                    if sync_idx_iter >= len(all_sync_indices):
                        sync_idx_iter = 0  # Loop back to start
                    current_sync_index = all_sync_indices[sync_idx_iter]
                    update_frame(current_sync_index)
                    slider.set_val(current_sync_index)
                    current_sync_idx = current_sync_index
                    sync_idx_iter += 1
                    plt.pause(0.1)  # 100ms delay
                play_button.label.set_text('Play')
            threading.Thread(target=play_loop, daemon=True).start()
        else:
            play_button.label.set_text('Play')
    
    # Create slider - use actual sync_index values, not frame indices
    slider_ax = plt.axes([0.1, 0.02, 0.6, 0.03])
    slider = Slider(slider_ax, 'Sync Index', min_sync, max_sync, 
                   valinit=min_sync, valfmt='%d', valstep=1)
    slider.on_changed(on_slider_change)
    
    # Create play button
    play_button_ax = plt.axes([0.75, 0.02, 0.1, 0.03])
    play_button = Button(play_button_ax, 'Play')
    play_button.on_clicked(on_play_button)
    
    # Initial frame
    update_frame(min_sync)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for controls
    
    if output_plot_path:
        # For static save, just save the first frame
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Static plot saved to: {output_plot_path}")
    else:
        plt.show()
    
    return fig


def project_poses_to_video(results_directory, port_number, output_video_name="detected_people.mp4"):
    """
    Project 3D pose data back onto a video from a specific camera port and save as new video.
    
    Args:
        results_directory (str): Directory containing *_person*.csv files and config.toml
        port_number (int): Camera port number (e.g., 0, 1, 2)
        output_video_name (str): Name of output video file
    """
    import cv2
    import os
    
    # Load person data
    person_data, person_files_data = show_multi_person_results(results_directory, output_plot_path=None)
    
    if not person_data or not person_files_data:
        print("No person data available for video projection")
        return
    
    # Load calibration
    calibration_path = os.path.join(results_directory, "config.toml")
    video_path = os.path.join(results_directory, f"port_{port_number}.mp4")
    
    if not os.path.exists(calibration_path):
        print(f"Calibration file not found: {calibration_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    print(f"Loading calibration from: {calibration_path}")
    camera_params = parse_calibration_mwc(calibration_path)
    if not camera_params:
        print("Failed to load camera calibration")
        return
    
    # Find camera parameters for this port using the parsed data
    target_cam_params = None
    target_cam_name = f"cam_{port_number}"
    
    for params in camera_params:
        # Check if this camera has the matching port number
        if params.get('port') == port_number:
            target_cam_params = params
            print(f"Found camera with explicit port {port_number}")
            break
        # Check if camera name matches expected pattern (cam_0, cam_1, etc.)
        elif params.get('id') == target_cam_name:
            target_cam_params = params
            print(f"Assuming {target_cam_name} == port_{port_number}")
            break
    
    if target_cam_params is None:
        print(f"Camera parameters for port {port_number} not found")
        print(f"Available cameras: {[p.get('id', 'unknown') for p in camera_params]}")
        print(f"Expected camera name: {target_cam_name}")
        return
    
    # Calculate projection matrix for this camera
    projection_matrices = calculate_projection_matrices([target_cam_params])
    if not projection_matrices:
        print("Failed to calculate projection matrix")
        return
    
    P = projection_matrices[0]
    
    # Load frame time history to map sync_index to frame numbers
    frame_history_path = os.path.join(results_directory, "frame_time_history.csv")
    if not os.path.exists(frame_history_path):
        print(f"Frame history not found: {frame_history_path}")
        return
    
    import pandas as pd
    frame_history_df = pd.read_csv(frame_history_path)
    
    print(f"Frame history loaded: {len(frame_history_df)} total rows")
    print(f"Ports in frame history: {sorted(frame_history_df['port'].unique())}")
    print(f"Sync index range: {frame_history_df['sync_index'].min()} to {frame_history_df['sync_index'].max()}")
    
    # Filter for this port
    frame_history_df = frame_history_df[frame_history_df['port'] == port_number]
    print(f"Rows for port {port_number}: {len(frame_history_df)}")
    
    if len(frame_history_df) == 0:
        print(f"No frame history data found for port {port_number}")
        return
    
    frame_history_df = frame_history_df.sort_values(by='frame_time')
    frame_history_df['derived_frame_index'] = frame_history_df.groupby('port')['frame_time'].rank(method='min').astype(int) - 1
    
    print(f"Port {port_number} sync index range: {frame_history_df['sync_index'].min()} to {frame_history_df['sync_index'].max()}")
    print(f"Port {port_number} derived frame range: {frame_history_df['derived_frame_index'].min()} to {frame_history_df['derived_frame_index'].max()}")
    print(f"First few sync->frame mappings:")
    for i, (_, row) in enumerate(frame_history_df.head(10).iterrows()):
        print(f"  sync_index {row['sync_index']} -> frame {row['derived_frame_index']}")
        
    # Also show person data sync ranges
    print(f"\nPerson data sync index ranges:")
    for person_id, data in person_files_data.items():
        if 'sync_index' in data:
            sync_range = f"{data['sync_index'].min()} to {data['sync_index'].max()}"
            print(f"  Person {person_id}: {sync_range} ({len(data['sync_index'])} frames)")
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video
    output_path = os.path.join(results_directory, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Color map for different people
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    print(f"Processing {total_frames} frames...")
    
    # Track projection statistics
    frames_with_poses = 0
    frames_processed = 0
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_processed += 1
        
        # Find corresponding sync_index for this frame
        matching_rows = frame_history_df[frame_history_df['derived_frame_index'] == frame_idx]
        if matching_rows.empty:
            # No sync data for this frame, just write original frame
            if frame_idx % 200 == 0:
                print(f"Frame {frame_idx}: No sync data available")
            out.write(frame)
            continue
        
        current_sync_index = matching_rows.iloc[0]['sync_index']
        
        # Debug sync mapping every 200 frames
        if frame_idx % 200 == 0:
            print(f"Frame {frame_idx}: sync_index = {current_sync_index}")
        
        # Track if any person is found in this frame
        people_in_frame = 0
        
        # Draw poses for each person
        for person_idx, (person_id, data) in enumerate(person_files_data.items()):
            if 'sync_index' not in data:
                continue
            
            # Find frame with matching sync index
            sync_mask = data['sync_index'] == current_sync_index
            if not np.any(sync_mask):
                continue
            
            # Found a person at this sync index!
            people_in_frame += 1
            if frame_idx % 200 == 0 or people_in_frame == 1:  # Log first person found or every 200 frames
                print(f"  -> Found Person {person_id} at sync_index {current_sync_index} (frame {frame_idx})")
            
            # Get color for this person
            color = colors[person_idx % len(colors)]
            
            # Project and draw keypoints
            keypoints_2d = []
            for keypoint_name in ['L_Ankle', 'R_Ankle', 'L_Knee', 'R_Knee', 'L_Hip', 'R_Hip',
                                'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
                                'L_Eye', 'R_Eye', 'Nose']:
                if keypoint_name in data:
                    kp_data = data[keypoint_name][sync_mask]
                    if len(kp_data) > 0:
                        x, y, z = kp_data[0]
                        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                            # Project 3D point to 2D
                            point_3d = np.array([x, y, z])
                            point_2d = project_3d_to_2d(point_3d, P)
                            if point_2d is not None:
                                keypoints_2d.append((int(point_2d[0]), int(point_2d[1])))
                            else:
                                keypoints_2d.append(None)
                        else:
                            keypoints_2d.append(None)
                    else:
                        keypoints_2d.append(None)
                else:
                    keypoints_2d.append(None)
            
            # Draw keypoints
            for kp in keypoints_2d:
                if kp is not None:
                    cv2.circle(frame, kp, 3, color, -1)
            
            # Draw skeleton connections
            connections = [
                # Legs
                (0, 2), (1, 3),  # ankles to knees
                (2, 4), (3, 5),  # knees to hips
                (4, 5),          # hips together
                # Torso
                (4, 6), (5, 7),  # hips to shoulders
                (6, 7),          # shoulders together
                # Arms
                (6, 8), (7, 9),  # shoulders to elbows
                (8, 10), (9, 11), # elbows to wrists
                # Head
                (12, 14), (13, 14), # eyes to nose
                (6, 14), (7, 14)    # shoulders to nose (approximate neck)
            ]
            
            for start_idx, end_idx in connections:
                if (start_idx < len(keypoints_2d) and end_idx < len(keypoints_2d) and 
                    keypoints_2d[start_idx] is not None and keypoints_2d[end_idx] is not None):
                    cv2.line(frame, keypoints_2d[start_idx], keypoints_2d[end_idx], color, 2)
            
            # Add person ID label
            if keypoints_2d[14] is not None:  # Use nose position for label
                label_pos = (keypoints_2d[14][0], keypoints_2d[14][1] - 20)
                cv2.putText(frame, f'Person {person_id}', label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Track frames with poses
        if people_in_frame > 0:
            frames_with_poses += 1
        
        # Add frame info
        info_text = f'Frame: {frame_idx}, Sync: {current_sync_index}, People: {people_in_frame}'
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_idx % 200 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames, {frames_with_poses} frames with poses so far")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n=== VIDEO PROJECTION SUMMARY ===")
    print(f"Total frames processed: {frames_processed}")
    print(f"Frames with pose data: {frames_with_poses}")
    print(f"Percentage with poses: {100*frames_with_poses/frames_processed:.1f}%")
    print(f"Output video saved: {output_path}")
    return output_path

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process synchronized multi-camera video for 3D pose estimation with tracking.")
    # (Add the new arguments if needed, or keep defaults)
    testfolder = "recording_balance_stage1_v2"
    parser.add_argument("--csv_path", default=f"test/caliscope/{testfolder}/frame_time_history.csv",help="Path...")
    parser.add_argument("--calibration_path",default=f"test/caliscope/{testfolder}/config.toml", help="Path...")
    parser.add_argument("--video_dir", default=f"test/caliscope/{testfolder}", help="Dir...")
    parser.add_argument("--output_path", default=f"output/caliscope/{testfolder}/output_3d_poses_tracked.csv",help="Path...")
    parser.add_argument("--model_dir", default=LOCAL_SP_DIR, help="Path...")
    parser.add_argument("--detector_dir", default=LOCAL_DET_DIR, help="Path...")
    parser.add_argument("--calib_type", default="mwc", choices=['mwc', 'fmc'], help="Type...")
    parser.add_argument("--skip", type=int, default=1, help="Skip N...")
    parser.add_argument("--person_conf", type=float, default=0.8, help="Conf...")
    parser.add_argument("--keypoint_conf", type=float, default=0.1, help="Conf...")
    parser.add_argument("--device", default="mps", choices=['auto', 'cpu', 'mps', 'cuda'], help="Device...")
    parser.add_argument("--track_max_dist", type=float, default=100.0, help="Max 2D pixel distance for head tracking match (default: 100).")
    parser.add_argument("--head_idx", type=int, default=0, help="Index of keypoint used for tracking (default: 0, Nose).")

    args = parser.parse_args()

    process_synced_mwc_frames(
        csv_path=args.csv_path, calibration_path=args.calibration_path, video_dir=args.video_dir,
        output_path=args.output_path, model_dir=args.model_dir, detector_dir=args.detector_dir,
        calib_type=args.calib_type, skip_sync_indices=args.skip, person_confidence=args.person_conf,
        keypoint_confidence=args.keypoint_conf, device_name=args.device,
        tracking_max_2d_dist=args.track_max_dist, head_kp_index=args.head_idx)
    
    # process_synced_mwc_frames_multi_person(
    #     csv_path=args.csv_path, calibration_path=args.calibration_path, video_dir=args.video_dir,
    #     output_path=args.output_path, model_dir=args.model_dir, detector_dir=args.detector_dir,
    #     calib_type=args.calib_type, skip_sync_indices=args.skip, person_confidence=args.person_conf,
    #     keypoint_confidence=args.keypoint_conf, device_name=args.device,
    #     # --- Tracking parameters ---
    #     max_persons=2,  # Maximum number of persons to track
    #     com_distance_threshold=0.5,  # meters - minimum distance between COMs to be different people
    #     track_lost_patience=10,  # frames to wait before considering a track lost
    #     min_keypoints_for_com=2,  # minimum valid hip keypoints needed to compute COM
    #     hip_indices=(11, 12),  # COCO format: left hip, right hip
    # )