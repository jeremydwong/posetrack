#%%
# note: this file is meant to run as a script to test out 'detect_people' batch processing, handy when we want to find the people. 
# 2025-07-17: batch-processing a synthpose run. 

base_dir = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-25/recordings_test'
# OR base_dir = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-01-07/recordings'

#%%
# note: this file is meant to run as a script to test out new features. 
# 2025-07-17: batch-processing a synthpose run. 
import posetrack as pt 
import os
import pandas as pd
import matplotlib
# matplotlib.use('qtagg')#tqagg
# %matplotlib qtagg % interactive mode flag

#%% now loop through each subfolder, and project into a video using
# project_poses_to_video(results_directory, port_number, output_video_name="detected_people.mp4"):

subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
print(subfolders)


for subfolder in subfolders:
    print(f"Processing subfolder: {subfolder}")
    results_directory = os.path.join(subfolder, 'synthpose')
    if not os.path.exists(results_directory):
        print(f"Skipping {results_directory} as it does not exist.")
        continue
    
    # project the poses to video
    pt.project_poses_to_video(results_directory, port_number=0, output_video_name="detected_people.mp4")
    
    # optionally, you can also animate the results
    

# %%

# %%
