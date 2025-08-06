# note: this file is meant to run as a script to test out new features. 
# 2025-07-17: batch-processing a synthpose run. 
import posetrack as pt 
import os
base_dir = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-25/recordings_carylh'
pt.batch_process_subfolders(base_dir = base_dir,person_confidence=0.1,
    keypoint_confidence=0.1, device_name="mps",verbose_debug=False, override_views_used=[(0,2)])

# base_dir = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-10/'
# which_trial = 'recording_linda_balance4_v3'

# base_dir = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-01-07/'
# which_trial = 'recording_ing_4stage1v1'                                                               

# video_file_path = os.path.join(base_dir, 'recordings', which_trial,'port_0.mp4')
# config_toml_path = os.path.join(base_dir,'config.toml')
# specific_file = 'output_3d_poses_tracked.csv_person0.csv'
# csv_3d_coordinates_path = os.path.join(base_dir, 'recordings', which_trial,'synthpose',specific_file)
# returnvals = pt.report_clothing(video_file_path, config_toml_path, csv_3d_coordinates_path, 
#                     movement_threshold=10.0, sampling_radius=3, max_samples=5)
