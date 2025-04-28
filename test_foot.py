#%%
import pandas as pd
import libwalk as lw
import numpy as np
filename = "/Users/jeremy/Git/ProjectKeypointInference/posetrack/output/caliscope/recording_3by1/output_3d_poses_tracked.csv"
df = pd.read_csv(filename)
# df = df.drop(columns=["Unnamed: 0"])
index = df["sync_index"]

r_big_toe = df[["r_big_toe_X", "r_big_toe_Y", "r_big_toe_Z"]]
l_big_toe = df[["l_big_toe_X", "l_big_toe_Y", "l_big_toe_Z"]]
l_calc    = df[["l_calc_X", "l_calc_Y", "l_calc_Z"]]
r_calc    = df[["r_calc_X", "r_calc_Y", "r_calc_Z"]]    

r_big_toe = r_big_toe.to_numpy().reshape(-1, 3)
l_big_toe = l_big_toe.to_numpy().reshape(-1, 3)
l_calc    = l_calc.to_numpy().reshape(-1, 3)
r_calc    = r_calc.to_numpy().reshape(-1, 3)
# %%
GET_R = False
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
L_toe = (R @ (l_big_toe.T)).T
R_toe = (R @ (r_big_toe.T)).T
L_calc = (R @ (l_calc.T)).T
R_calc = (R @ (r_calc.T)).T
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
