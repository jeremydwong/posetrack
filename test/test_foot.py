#%%
import pandas as pd
import libwalk as lw
import numpy as np

# set figures to be vector graphics
import matplotlib
from scipy.signal import filtfilt, butter
import sys
# matplotlib.use('Agg')
folder_name = "coord_3x1_3"
# filename = "/Users/jeremy/Git/ProjectKeypointInference/posetrack/output/caliscope/recording_3by1/output_3d_poses_tracked.csv"
filename = f"/Users/jeremy/Git/ProjectKeypointInference/posetrack/output/caliscope/{folder_name}/output_3d_poses_tracked.csv"
df = pd.read_csv(filename)
# df = df.drop(columns=["Unnamed: 0"])
index = df["sync_index"]

# the following is a 'magic command' to set the backend for matplotlib
# need to have ipympl and pyside6 installed.
# check if running interactively in a jupyter notebook or vs code before running:
matplotlib.use('qtagg')

r_big_toe = df[["r_big_toe_X", "r_big_toe_Y", "r_big_toe_Z"]]
l_big_toe = df[["l_big_toe_X", "l_big_toe_Y", "l_big_toe_Z"]]
l_calc    = df[["l_calc_X", "l_calc_Y", "l_calc_Z"]]
r_calc    = df[["r_calc_X", "r_calc_Y", "r_calc_Z"]]    

r_big_toe = r_big_toe.to_numpy().reshape(-1, 3)
l_big_toe = l_big_toe.to_numpy().reshape(-1, 3)
l_calc    = l_calc.to_numpy().reshape(-1, 3)
r_calc    = r_calc.to_numpy().reshape(-1, 3)
# %%
GET_R = True
if GET_R:
    # Get the rotation matrix and x0 for the right big toe
    R,x0,g = lw.get_rotmat_x0(r_big_toe)
    print("R:", R)
    print("x0:", x0)
    print("g:", g)
    # Get the rotation matrix and x0 for the left big toe
else: 
    #hardcoded from previous
    R = np.array([[-0.89778292, -0.02025153,  0.43997239],
        [-0.43428817,  0.20705714, -0.87665337],
        [-0.07334585, -0.97811922, -0.19468736]])
    x0  = np.array([ 1.75900585, -5.3849832,  -1.7747548 ])
    g   = np.array([3.13831784, 2.96597575])
    
# %%
L_toe = (R @ (l_big_toe.T)).T - x0
R_toe = (R @ (r_big_toe.T)).T - x0
L_calc = (R @ (l_calc.T)).T - x0
R_calc = (R @ (r_calc.T)).T - x0
# %%
# Plot the data, with a title
import matplotlib.pyplot as plt
f,ax = plt.subplots()
ax.plot(L_toe[:,0],L_toe[:,1], label="Left Toe")
ax.plot(R_toe[:,0],R_toe[:,1], label="Right Toe")
ax.plot(L_calc[:,0],L_calc[:,1], label="Left Calc")
ax.plot(R_calc[:,0],R_calc[:,1], label="Right Calc")
# set equal aspect ratio
ax.set_aspect('equal', adjustable='box')
# %%  
# Subtract the first index to start at 0
index = index - index.iloc[0]

# Plot the toes and heels as a function of the index
f, ax = plt.subplots()
ax.plot(index, L_toe[:, 1], 'b--', label="Left Toe (Y)")
ax.plot(index, R_toe[:, 1], 'r--', label="Right Toe (Y)")
ax.plot(index, L_calc[:, 1], 'b-', label="Left Heel (Y)")
ax.plot(index, R_calc[:, 1], 'r-', label="Right Heel (Y)")

# Add labels, legend, and title
ax.set_xlabel("Index")
ax.set_ylabel("Y Coordinate")
ax.legend()
ax.set_title("Toes and Heels Y-Coordinate vs Index")
plt.show()
# %%
filecompareto = f"/Users/jeremy/Git/ProjectKeypointInference/posetrack/test/caliscope/{folder_name}/xyz_POSE_labelled.csv"
df2 = pd.read_csv(filecompareto)
# df2 = df2.drop(columns=["Unnamed: 0"])
# right_heel and left_heel columns, lowercase x y z
# right_heel_X, right_heel_Y, right_heel_Z
r_heel_cs = df2[["right_heel_x", "right_heel_y", "right_heel_z"]]
l_heel_cs = df2[["left_heel_x", "left_heel_y", "left_heel_z"]]

lfi = df2[["left_foot_index_x", "left_foot_index_y", "left_foot_index_z"]]
rfi = df2[["right_foot_index_x", "right_foot_index_y", "right_foot_index_z"]]

r_heel_cs = r_heel_cs.to_numpy().reshape(-1, 3)
l_heel_cs = l_heel_cs.to_numpy().reshape(-1, 3)
lfi = lfi.to_numpy().reshape(-1, 3)
rfi = rfi.to_numpy().reshape(-1, 3)

# no toe columns. now rotate and translate
L_heel_cs = (R @ (l_heel_cs.T)).T - x0
R_heel_cs = (R @ (r_heel_cs.T)).T - x0
L_fi_cs = (R @ (lfi.T)).T - x0
R_fi_cs = (R @ (rfi.T)).T - x0
# %%

# %%
# Define the sampling rate
sr = 30
dt = 1 / sr
ind_fromend = 100

# Define a lowpass filter
def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Apply the lowpass filter to column 1 (Y-coordinate) of each dataset
cutoff_frequency = 5  # Hz
for i in range(3):
    L_heel_cs[:, i] = lowpass_filter(L_heel_cs[:, i], cutoff_frequency, sr)
    R_heel_cs[:, i] = lowpass_filter(R_heel_cs[:, i], cutoff_frequency, sr)
    L_calc[:, i] = lowpass_filter(L_calc[:, i], cutoff_frequency, sr)
    R_calc[:, i] = lowpass_filter(R_calc[:, i], cutoff_frequency, sr)
    L_fi_cs[:, i] = lowpass_filter(L_fi_cs[:, i], cutoff_frequency, sr)
    R_fi_cs[:, i] = lowpass_filter(R_fi_cs[:, i], cutoff_frequency, sr)

# overplot 
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
labels = ["X Coordinate", "Y Coordinate", "Z Coordinate"]

for i in range(3):
    ax = axes[i]
    lmin = np.min([L_heel_cs.shape[0],L_calc.shape[0]])
    l = L_heel_cs.shape[0]
    ax.plot(L_heel_cs[(l-lmin):-1, i], 'b--', label="LHeel CS")
    ax.plot(R_heel_cs[(l-lmin):-1, i], 'r--', label="RHeel CS")
    l = L_calc.shape[0]
    ax.plot(L_calc[(l-lmin):-1, i], 'b-', label="Left Heel SP")
    ax.plot(R_calc[(l-lmin):-1, i], 'r-', label="Right Heel SP")

    ax.set_ylabel(labels[i])
    ax.legend()

axes[-1].set_xlabel('Index')
plt.tight_layout()
plt.show()


# Compute the derivative for y-velocity with respect to time
L_heel_cs_vel = np.gradient(L_heel_cs[:, 1], dt)
R_heel_cs_vel = np.gradient(R_heel_cs[:, 1], dt)
L_calc_vel = np.gradient(L_calc[:, 1], dt)
R_calc_vel = np.gradient(R_calc[:, 1], dt)
L_fi_cs_vel = np.gradient(L_fi_cs[:, 1], dt)
R_fi_cs_vel = np.gradient(R_fi_cs[:, 1], dt)

#filter again
L_heel_cs_vel = lowpass_filter(L_heel_cs_vel, cutoff_frequency, sr)
R_heel_cs_vel = lowpass_filter(R_heel_cs_vel, cutoff_frequency, sr)
L_calc_vel = lowpass_filter(L_calc_vel, cutoff_frequency, sr)
R_calc_vel = lowpass_filter(R_calc_vel, cutoff_frequency, sr)
L_fi_cs_vel = lowpass_filter(L_fi_cs_vel, cutoff_frequency, sr)
R_fi_cs_vel = lowpass_filter(R_fi_cs_vel, cutoff_frequency, sr)

# Overplot the velocities
f, ax = plt.subplots()
# ax.plot(L_fi_cs_vel, 'b:', label="LFI CS Velocity (Y)")
# ax.plot(R_fi_cs_vel, 'r:', label="RFI CS Velocity (Y)")
l = L_heel_cs.shape[0]
ax.plot(L_heel_cs_vel[(l-lmin):-1], 'b--', label="LHeel CS Velocity (Y)")
ax.plot(R_heel_cs_vel[(l-lmin):-1], 'r--', label="RHeel CS Velocity (Y)")
l = L_calc.shape[0]
ax.plot(L_calc_vel[(l-lmin):-1], 'b-', label="LHeel SP Heel Velocity (Y)")
ax.plot(R_calc_vel[(l-lmin):-1], 'r-', label="RHeel SP Heel Velocity (Y)")

# Set xlim 120,240
# ax.set_xlim(120, 240)
ax.legend()
ax.set_xlabel('index')
ax.set_ylabel('Y Velocity (m/s)')
plt.show(block = True)
# %%
