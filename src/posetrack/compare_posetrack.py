import pickle
import pandas as pd
import numpy as np
import posetrack as pt
# Loading different types of data
fname = "/Users/jeremy/Git/KeypointInference/posetrack/output/caliscope/recording_balance_stage1_v2/output_3d_poses_tracked.pkl"
with open(fname, 'rb') as file:
    data = pickle.load(file)
    
print(data)  # Your original data structure

fname = "/Users/jeremy/Git/KeypointInference/posetrack/output/caliscope/recording_balance_stage1_v2/output_3d_poses_tracked.csv"
with open(fname, 'rb') as file:
    data_new = pd.read_csv(file)

fname = "/Users/jeremy/Git/KeypointInference/posetrack/tests/caliscope/recording_balance_stage1_v2/POSE/xyz_POSE_labelled.csv"
with open(fname, 'rb') as file:
    data_orig = pd.read_csv(file)

fnamecoord = "/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-06-11/recordings/recording_3by1_v3/POSE/xyz_POSE_labelled.csv"
with open(fnamecoord, 'rb') as file:
    data_coord = pd.read_csv(file)

right_foot_index_xyz = np.stack([
    data_coord['right_foot_index_x'].values,
    data_coord['right_foot_index_y'].values,
    data_coord['right_foot_index_z'].values
], axis=1)

r,x0,g = pt.get_rotmat_x0(right_foot_index_xyz)

right_foot_orig = np.stack([
    data_orig['right_foot_index_x'].values,
    data_orig['right_foot_index_y'].values,
    data_orig['right_foot_index_z'].values
], axis=1)

left_foot_orig = np.stack([
    data_orig['left_foot_index_x'].values,
    data_orig['left_foot_index_y'].values,
    data_orig['left_foot_index_z'].values
], axis=1)


left_foot_new = np.stack([
    data_new['l_big_toe_X'].values,
    data_new['l_big_toe_Y'].values,
    data_new['l_big_toe_Z'].values
], axis=1)

right_foot_new = np.stack([
    data_new['r_big_toe_X'].values,
    data_new['r_big_toe_Y'].values,
    data_new['r_big_toe_Z'].values
], axis=1)

# rotate with R
left_foot_new_rotated = (left_foot_new.T@r).T
right_foot_new_rotated = (right_foot_new.T@r).T

left_foot_orig_rotated = (left_foot_orig.T@r).T
right_foot_orig_rotated = (right_foot_orig.T@r).T

plt.plot(right_foot_orig_rotated[:, 0:3], 'r-', label='Original Right Foot')
plt.plot(left_foot_orig_rotated[:, 0:3], 'b-', label='Original Left Foot')
plt.plot(right_foot_new_rotated[:, 0:3], 'r.-', label='New Right Foot')
plt.plot(left_foot_new_rotated[:, 0:3], 'b.-', label='New Left Foot')

plt.legend()
plt.show()