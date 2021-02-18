# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, '../utils')

import numpy as np
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from copy import deepcopy
import cv2
from utils_geom import *
# from utils_plot import *

def save_depth(depth_image, path):
	depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
	cv2.imwrite(path, depth_colormap)

def save_trial(load_path='data/test/', save_path='data/test/', graspInd=0, numGraspPerTrial=5):

	start_time = time.time()

	if not os.path.exists(load_path + 'images/'):
		os.mkdir(load_path + 'images/')
	images_dir = load_path + 'images/'

	# Process othe info
	info = np.load(load_path + 'info.npz', allow_pickle=True, encoding='latin1')
	obj_config = info['obj_config']
	camera_params = info['camera_params']
	print(obj_config)	
	print(camera_params)

	# Process depth
	depth = np.load(load_path + 'depth.npz', allow_pickle=True, encoding='latin1')['depth']
	depth = ((0.7 - depth)/0.2).clip(min=0.0, max=1.0)
	# depth_sorted = np.sort(depth.flatten())[::-1]
	# max_val = 1
	# min_val = 1
	# for val in depth_sorted:
	# 	if val < (1 - 1e-4) and max_val == 1:
	# 		max_val = val
	# 		# break
	# 	if val < min_val:
	# 		min_val = val
	# if max_val > 0.10:
	# 	print(max_val)
	# plt.imshow(depth, cmap='Greys', interpolation='nearest')
	# plt.show()

	# Process ee pose
	ee_poses = np.load(load_path + 'ee.npz', allow_pickle=True, encoding='latin1')['ee_poses']

	# plt.imshow(depth, cmap='Greys', interpolation='nearest')

	for ind in range(numGraspPerTrial):

		eePos, eeQuat = ee_poses[ind]
		# print(eeQuat)

		# Account for small roll/pitch
		center = (eePos + quat2rot(eeQuat).reshape(3,3)@np.array([0,0,0.118]))
		height_diff = eePos[2] - center[2]
		eePos_new = center
		eePos_new[2] += 0.118
  
		#! Add offset for long finger
		eePos_new[2] += 0.025
		# print(eePos, eePos_new, height_diff)

		# Process other info
		# info = np.load(load_path + 'info.npz', allow_pickle=True, encoding='latin1')
		# obj_config = info['obj_config']
		# camera_params = info['camera_params']

		# Get rotation matrix
		eeRot = quat2rot(eeQuat)
		rot01 = array([[1., 0.,  0.],
					[0., -1., 0.],
					[0., 0., -1.]])
		eeRotOffset = rot01.T@eeRot
		eeRotFlip = eeRotOffset@array([[-1,0,0],
								[0,-1,0],
								[0,0,1]])

		# Extract axis, angle, only use angle as yaw angle, constrain angle between -np.pi/2 to np.pi/2 to avoid ambiguity
		axis, angle = rot2aa(eeRotOffset)
		if axis[2] > 0:
			axis *= -1
			angle *= -1
		angle += np.pi
		# if angle > np.pi/2:
		# 	angle -= np.pi
		# elif angle < -np.pi/2:
		# 	angle += np.pi

		print(eePos_new[2], angle)

	# if axis[2] > -0.9:
		# fig = plt.figure()
		# ax = plt.axes(projection='3d')
		# plot_frame(ax, [0,0,0], eeRot)
		# # plot_frame(ax, [0,0,0], eeRotOffset)
		# ax.set_xlabel('x')
		# ax.set_ylabel('y')
		# ax.set_zlabel('z')
		# set_aspect_equal_3d(ax)
		# plt.show()

		# pixel2xy_mat = np.load('data/pixel2xy.npz')['pixel2xy']  # 128x128x2
		# dist = np.linalg.norm(pixel2xy_mat-eePos[:2], axis=2)
		# pixel_ind_all = np.unravel_index(dist.argmin(), dist.shape)
		# z_end = array([-np.cos(angle)*5, np.sin(angle)*5])

		# dist_2 = np.linalg.norm(pixel2xy_mat-eePos_new[:2], axis=2)
		# pixel_ind_all_new = np.unravel_index(dist_2.argmin(), dist_2.shape)
		# z_end = array([-np.cos(angle)*5, np.sin(angle)*5])

		# # Plot pose in 2D
		# plt.imshow(depth, cmap='Greys', interpolation='nearest') 
		# plt.scatter(x=pixel_ind_all[1], y=pixel_ind_all[0], s=15, c='red')  # pixels use coordinates from left_top corner, and height as x; scatter uses coordinates from left_bottom corner, and normal x
		# plt.scatter(x=pixel_ind_all_new[1], y=pixel_ind_all_new[0], s=15, c='green')
		# plt.plot([pixel_ind_all_new[1], pixel_ind_all_new[1]+z_end[0]], [pixel_ind_all_new[0], pixel_ind_all_new[0]+z_end[1]], color='black')
		# # plt.savefig('grasp_z_interp.png')
		# plt.show()

		# Scaling
		eePos_new[0] -= 0.50
		# eePos[2] -= 0.20
		eePos_new[:2] *= 20
		# print(eePos_new)

		# sin/cos encoding for yaw, decode using atan2(y1,y2)
		yaw_enc = array([np.sin(angle), np.cos(angle)])

		# Save obs, concatenate ee/joint/obj state
		np.savez(save_path + str(graspInd) + '.npz', 
			depth=depth,
			eePos=eePos_new,
			# eeRot=eeRotOffset,
			# eeRotFlip=eeRotFlip,
			yaw=angle,
			yaw_enc=yaw_enc,
		)

		# Increment observation counter
		graspInd += 1
	
	# plt.show()

	return graspInd


if __name__ == '__main__':
	save_trial()
