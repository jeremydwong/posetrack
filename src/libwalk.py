#%libwalk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
def get_rotmat_x0(data_nx3,pm = None, xdir=1):
  
  # plot the data, with a title
  f,ax = plt.subplots()
  plt.ion()
  plt.plot(data_nx3)
  titletxt = f"Click 3 times to back-left, front-left,front-right:"
  plt.title(titletxt)
  
  # initialize the list of x-clicks we will collect
  ind_xclicks = []
  # Function to capture mouse clicks
  def onclick(event):
    ind_xclicks.append((event.xdata))
    # break if coordinates are 3
    if len(ind_xclicks) == 3:
      plt.gcf().canvas.mpl_disconnect(cid)
      plt.close()

  # Connect the click event to the function
  cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
  plt.draw()
  plt.show(block=True)

  # round the indices of the clicks
  ind_xclicks = [int(np.round(ind_xclicks[i])) for i in range(3)]
  
  # define vectors in 3D
  pt1 = data_nx3[ind_xclicks[0],:]
  pt2 = data_nx3[ind_xclicks[1],:]
  pt3 = data_nx3[ind_xclicks[2],:]
  
  # define which order. assume standard:
  if xdir == 1:
    v1 = pt3 - pt2 # positive right
  else:
    v1 = pt2 - pt3

  v2 = pt2 - pt1 # positive forward
  # normalize v1 and v2 to be unit length
  v1 = v1/np.linalg.norm(v1)
  v2 = v2/np.linalg.norm(v2)
  # remove any component of v2 that is in the direction of v1
  v2 = v2 - np.dot(v2,v1)*v1
  v2 = v2/np.linalg.norm(v2)
  # now take the cross product
  v3 = np.cross(v1,v2)

  # compute the rotation matrix to go from the data frame to that defined by v1,v2,v3
  R = np.array([v1,v2,v3])

  # test by first rotating data_nx3, then subtracting out pt1, then plotting
  data_nx3_rot = R @ data_nx3.T
  data_nx3_rot = data_nx3_rot.T
  x0           = data_nx3_rot[ind_xclicks[0],:]
  data_nx3_rot = data_nx3_rot - x0
  
  f,ax = plt.subplots()
  # make it a 3D figure
  ax = f.add_subplot(111, projection='3d')
  ax.plot(data_nx3_rot[ind_xclicks[0]:-1,0],data_nx3_rot[ind_xclicks[0]:-1,1],data_nx3_rot[ind_xclicks[0]:-1,2],'k.')
  
  # plot the rotated points with circles
  pt2_R = R @ (pt2 - pt1)
  pt3_R = R @ (pt3 - pt1)
  ax.plot(0,0,0,'bo')
  ax.plot(pt2_R[0],pt2_R[1],pt2_R[2],'ro')
  ax.plot(pt3_R[0],pt3_R[1],pt3_R[2],'go')
  
  ax.set_xlabel('X (m)')
  ax.set_ylabel('Y (m)')
  ax.set_zlabel('Z (m)')

  if pm is not None:
    ax.set_xlim3d([-pm,pm])
    ax.set_ylim3d([-pm,pm])
    ax.set_zlim3d([-pm,pm])

  #set el az to be 70, -90, a nice overhead view of the movement
  ax.view_init(elev=70,azim=-90)

  plt.show(block = True)
  # set g to be lengths of pt2-pt1 and pt3-pt1
  gy = np.linalg.norm(pt2 - pt1)
  gx = np.linalg.norm(pt3 - pt1)
  gs = np.array([gx,gy])
  return R,x0,gs
