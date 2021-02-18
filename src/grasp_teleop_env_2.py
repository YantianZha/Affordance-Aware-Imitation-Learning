# -*- coding: utf-8 -*-
import os
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '../utils')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pybullet_data
import pybullet as p
import time
import numpy as np
from numpy import array
# import spnav

from utils_geom import *
# from utils_pb import *
from src.utils_depth import *
from src.panda import Panda
from src.panda_env import pandaEnv


class GraspTeleopEnv():

	def __init__(self,
				#  depth_hz=5,
				#  other_hz=20,
				 ):

		# Initialize Panda arm environment
		self.pandaEnv = pandaEnv()
		# self.env.reset_env()

		# Same camera parameters for all trials
		self.camera_params = getCameraParametersGrasp()
		self.viewMat = self.camera_params['viewMatPanda']
		self.projMat = self.camera_params['projMatPanda']
		self.width = self.camera_params['imgW']
		self.height = self.camera_params['imgH']
		self.near = self.camera_params['near']
		self.far = self.camera_params['far']

		# Set up GUI
		# p.connect(p.GUI, options="--width=2600 --height=1800 --minGraphicsUpdateTimeMs=0 --mp4=\"grasp.mp4\" --mp4fps="+str(fps))
		p.connect(p.GUI, options="--width=2600 --height=1800")

		# distance, yaw, pitch
		p.resetDebugVisualizerCamera(0.8, 180, -45, [0.5, 0, 0])


	def reset(self):
		self.ee_poses = []
		self.pandaEnv.reset_env()


	def grasp_teleop(self, objX=0.55, objY=-0.05, objYaw=0, data_path='data/test/', objPath='../data/processed_objects/3DNet/100_Mug/100_Mug.urdf', numGraspPerTrial=5):

		# Object configuration
		objPos = np.array([objX,objY,0])
		objOrn = np.array([0,0,objYaw])
		self._obj_config = [objPos, objOrn, objPath]

		# Delete files in data_path
		lst = os.listdir(data_path)
		for f in lst:
			os.remove(data_path+f)

		# Load object
		if len(objOrn) == 3:  # input Euler angles
			objOrn = p.getQuaternionFromEuler(objOrn)
		self._objId = p.loadURDF(objPath, basePosition=objPos, baseOrientation=objOrn)
		p.changeDynamics(self._objId, -1, lateralFriction=self.pandaEnv._mu,spinningFriction=self.pandaEnv._sigma, frictionAnchor=1, mass=0.1)

		# Let the object settle
		for _ in range(100):
			p.stepSimulation()

		#* Simulate while using keyboard events
		#* Four directions: left: +x, down: +y; d(100): up, c(99): down, z(122): yaw left, x(120): yaw right
		#* b(98): grasp
		#* Run on 20Hz

		initial_ee_pos_before_depth = array([0.3, -0.5, 0.35])
		initial_ee_pos = array([0.5, 0.0, 0.35])
		initial_ee_orn = array([1.0, 0.0, 0.0, 0.0])

		target_ee_pos = array([0.0, 0.0, 0.0])
		target_ee_euler = array([0.0, 0.0, 0.0])

		# Time configuration
		useRealTime = 0
		p.setRealTimeSimulation(useRealTime)
		timeStep = 0
		real_hz = 5  # control hz running on real system
		action_timesteps = 240/real_hz  # convert to PyBullet

		# Scaling 3D mouse input separately for translation and rotation
		trans_scale = 1e-4
		rot_scale = 1e-3

		# Count how many grasps have been recorded (until numGraspPerTrial)
		numGraspRecorded = 0

		######################### Finish configuring #######################

		# Open 3D mouse socket
		# spnav.spnav_open()

		# Wait for start, move arm away from camera image to take depth
		textId = p.addUserDebugText('Press right button to start...', textPosition=[1.0,0,0.3], textColorRGB=[1,0,0], textSize=2, lifeTime=0)
		self.pandaEnv.reset_arm_joints_ik(initial_ee_pos_before_depth, initial_ee_orn)

		# while 1:
		# 	event = spnav.spnav_poll_event()
		# 	if isinstance(event, spnav.SpnavButtonEvent):  # button event
		# 		if event.bnum == 0 and event.press:  # right pressed (left if seen from logo)
		# 			break

		# Save depth
		# self.record_save_depth(data_path=data_path)

		# Move arm back to normal starting position
		self.pandaEnv.reset_arm_joints_ik(initial_ee_pos, initial_ee_orn)
		eePos, eeOrn = self.pandaEnv.get_ee()
		# plot_frame_pb(eePos, eeOrn)
		print("XX", eePos, p.getEulerFromQuaternion(eeOrn))

		while 1:
			self.pandaEnv.move_pos(absolute_pos=eePos, absolute_global_euler=p.getEulerFromQuaternion(eeOrn),
														 gripper_target_pos=0.04)
			# self.pandaEnv.move_pos(absolute_pos=eePos, absolute_global_quat=eeOrn,
			# 											 gripper_target_pos=0.04)
			continue

		# Run trial until quit by user
		while numGraspRecorded < numGraspPerTrial:
			new_test = 'Grasp: %d' % (numGraspRecorded+1) 
			textId = p.addUserDebugText(new_test, textPosition=[1.0,0,0.3], textColorRGB=[1,0,0], textSize=2, lifeTime=0, replaceItemUniqueId=textId)

			# Get and process mouse event
			event = spnav.spnav_poll_event()
			if event:
				if isinstance(event, spnav.SpnavButtonEvent):
					if event.bnum == 0 and event.press:  # right pressed
						self.record_ee()	# record ee
						numGraspRecorded += 1
					elif event.bnum == 1 and event.press:  # left pressed
						# self.reset_object()
						p.resetBasePositionAndOrientation(self._objId, posObj=objPos, ornObj=objOrn)
				elif isinstance(event, spnav.SpnavMotionEvent):
					x, y, z = event.translation   # x, y, z in spnav
					target_ee_pos = np.array([x,y,z], dtype=np.float)*trans_scale
					roll, pitch, yaw = event.rotation
					target_ee_euler = [-yaw*rot_scale, 0, 0]
			else:
				target_ee_pos = array([0.0, 0.0, 0.0])
				target_ee_euler = array([0.0, 0.0, 0.0])

			# Move by velocity control, get new timeStep
			timeStep, _ = self.pandaEnv.move_pos(
       									 relative_pos=target_ee_pos, 		
                                		 relative_global_euler=target_ee_euler,
                                      	 numSteps=action_timesteps,
                                         timeStep=timeStep)  # 240/48=5Hz

			# Clear mouse event buffer
			spnav.spnav_remove_events(spnav.SPNAV_EVENT_MOTION)

		# Save other info (object config, camera params)
		self.save_info(data_path=data_path)

		# Decide save or redo the trial
		save_trial = True
		while 1:
			event = spnav.spnav_poll_event()
			if isinstance(event, spnav.SpnavButtonEvent):  # button event
				if event.bnum == 0 and event.press:  # right pressed (left if seen from logo)
					break
				elif event.bnum == 1 and event.press:  # left pressed
					save_trial = False  # discard
					break

		# Close mouse socket
		spnav.spnav_close()

		if save_trial:
			self.save_ee(data_path=data_path)	# record ee
			print('Save this trial!')
		else:
			print('Discard this trial!')

		# Disconnect PB server
		time.sleep(2)
		# p.disconnect()

		return save_trial


	def record_save_depth(self, data_path):
		depth = self.get_depth().clip(max=1)
		# plt.imshow(depth, cmap='Greys', interpolation='nearest')
		# plt.show()
		np.savez(data_path + 'depth', depth=depth)


	def record_ee(self):
		eePos, eeOrn = self.env.get_ee()
		self.ee_poses += [(eePos, eeOrn)]


	def save_ee(self, data_path):
		np.savez(data_path + 'ee', ee_poses=self.ee_poses)


	def save_info(self, data_path):
		np.savez(data_path + 'info', 
           		obj_config=self._obj_config,
             	camera_params=self._camera_params)


	def check_success(self):
		hold_object = self.check_hold_object(self._objId)
		return hold_object


	def get_depth(self):
		img_arr = p.getCameraImage(width=self.width, 
							 	   height=self.height, 
								   viewMatrix=self.viewMat, 
								   projectionMatrix=self.projMat, 
								   flags=p.ER_NO_SEGMENTATION_MASK)
		depth = img_arr[3][192:192+128, 192:192+128]  # center 128x128 from 512x512
		depth = self.far*self.near/(self.far - (self.far - self.near)*depth)
		# if self._jitterDepth:
		# 	depth += np.random.normal(0, 0.1, size=depth.shape)
		# plt.imshow(depth, cmap='Greys', interpolation='nearest')
		# plt.show()
		return depth


	def get_pcl(self):
		img_arr = p.getCameraImage(width=self.width,
								   height=self.height,
								   viewMatrix=self.viewMat,
								   projectionMatrix=self.projMat,
								   flags=p.ER_NO_SEGMENTATION_MASK)
		pcl = pixelToWorld(img_arr[3], self._jitterDepth)
		return pcl
