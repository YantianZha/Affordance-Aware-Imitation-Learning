# -*- coding: utf-8 -*-
import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
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
from collections import OrderedDict, deque
from utils_geom import *
# from utils_pb import *
from src.utils_depth import *
from src.panda import Panda
from src.panda_env import pandaEnv, accurateIK, setMotors
from experiment_record_utils import ExperimentLogger
from time import sleep
from itertools import chain

SIMPLE_STATES_SIZE = 14
demo_path = '/data/datasets/SERL/Fetch/Grasp-v0'

class GraspTeleopEnv():

	def __init__(self,
				#  depth_hz=5,
				#  other_hz=20,
				 demo_path=None):

		# Initialize Panda arm environment
		self.pandaEnv = pandaEnv(long_finger=False)
		# self.env.reset_env()

		self._cam_random = 0.03
		self._history_max_len = 10

		self.camera_params = getCameraParametersGrasp()
		self.width = self.camera_params['imgW']
		self.height = self.camera_params['imgH']
		self.near = self.camera_params['near']
		self.far = self.camera_params['far']

		look = [0.4, 0., 0.9]
		distance = 0.9
		pitch = -70 + self._cam_random * np.random.uniform(-3, 3)
		yaw = -90 + self._cam_random * np.random.uniform(-3, 3)
		roll = -120
		self.viewMat = p.computeViewMatrixFromYawPitchRoll(
			look, distance, yaw, pitch, roll, 2)
		fov = 20. + self._cam_random * np.random.uniform(-2, 2)
		aspect = self.width / self.height
		near = 0.01
		far = 10
		self.projMat = p.computeProjectionMatrixFOV(
			fov, aspect, near, far)

		# Same camera parameters for all trials
		# self.viewMat = self.camera_params['viewMatPanda']
		# self.projMat = self.camera_params['projMatPanda']

		self.merge_simple_states_to_img = True
		self._history_max_len = 5
		self.demo_path = demo_path

		# Set up GUI
		# p.connect(p.GUI, options="--width=2600 --height=1800 --minGraphicsUpdateTimeMs=0 --mp4=\"grasp.mp4\" --mp4fps="+str(fps))
		p.connect(p.GUI, options="--width=2600 --height=1800")

		# distance, yaw, pitch
		p.resetDebugVisualizerCamera(0.8, 180, -45, [0.5, 0, 0])


	def reset(self):
		self.ee_poses = []
		self.success = False
		self._history_env_shot = deque(maxlen=self._history_max_len)
		self._grasps_history = deque(maxlen=self._history_max_len)
		self.pandaEnv.reset_env()

		# Load object
		objPos, objOrn, objPath = self._obj_config
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
		initial_ee_pos = array([0.3, 0.0, 0.4])
		initial_ee_orn = array([0, 0.7071068, 0, 0.7071068])

		target_ee_pos = array([0.0, 0.0, 0.0])
		target_ee_euler = array([0.0, 0.0, 0.0])

		# Time configuration
		useRealTime = 0
		p.setRealTimeSimulation(useRealTime)
		timeStep = 0
		real_hz = 5  # control hz running on real system
		action_timesteps = 240/real_hz  # convert to PyBullet

		# Wait for start, move arm away from camera image to take depth
		# textId = p.addUserDebugText('Press right button to start...', textPosition=[1.0,0,0.3], textColorRGB=[1,0,0], textSize=2, lifeTime=0)
		self.pandaEnv.reset_arm_joints_ik(initial_ee_pos_before_depth, initial_ee_orn)

		# Move arm back to normal starting position
		self.pandaEnv.reset_arm_joints_ik(initial_ee_pos, initial_ee_orn)

		self._get_simple_observation()
		return self.get_depth()

	def grasp_teleop(self, objX=0.55, objY=-0.05, objYaw=0, data_path='data/test/', objPath='../data/processed_objects/3DNet/100_Mug/100_Mug.urdf', numGraspPerTrial=5):
		# Object configuration
		objPos = np.array([objX, objY, 0.35])
		objOrn = np.array([0,0,objYaw])
		self._obj_config = [objPos, objOrn, objPath]

		# Delete files in data_path
		lst = os.listdir(data_path)
		for f in lst:
			os.remove(data_path+f)

		# Count how many grasps have been recorded (until numGraspPerTrial)
		numGraspRecorded = 0

		# eePos, eeOrn = self.pandaEnv.get_ee()
		# plot_frame_pb(eePos, eeOrn)

		experiment_recorder = ExperimentLogger(demo_path, "grasp", save_trajectories=True, mode='depth')
		experiment_recorder.redirect_output_to_logfile_as_well()

		# Run trial until quit by user
		# while numGraspRecorded < numGraspPerTrial:
		for i_episode in range(1):
			self.simple_states = OrderedDict()
			traj = []
			grasp = 0
			step = 0
			done = False
			back_step = 0
			# self.pandaEnv.reset_env()
			state = self.reset()
			self.save(step)

			def restore(n, step, traj_now):
				step -= n + 1
				new_traj = traj_now[:n + 1]
				self.restore(n)
				return step, new_traj

			while not done:
				step += 1
				while True:
					try:
						self._grasps_history.append(grasp)
						print("Please use mouse to set up a proper target pose of end effector:")
						action = self.mouse_control(grasp, back_step)
						print("BB: ", action)
						next_state, reward, done, _ = self.step(action)
						grasp = action[-1]

						while True:
							try:
								print(
									"Press Enter if you are happy with the next state, otherwise Press # steps that you want to go back (1, 2, or ... 10) and then Enter to repeat\n")
								back_step = input()
								assert back_step == "" or int(back_step) in range(1, 4), "wrong back_step"
								back_step = 0 if back_step == "" else int(back_step)
								break
							except AssertionError:
								continue

						good = True if back_step == 0 else False

						if not good:
							print("Restoring to the previous environment step..")
							if back_step == 1:
								self.restore(1)
								grasp = self._grasps_history[-1]

							else:
								step, traj = restore(back_step, step, traj)
								grasp = self._grasps_history[-back_step]
							continue

						break
					except ValueError:
						continue

				self.save(step)

				# print(state, action, reward, next_state, done)
				traj.append((state, action, reward, next_state, done))
				print("epi and step: ", numGraspRecorded, step, action, done, reward)
				# step += 1
				state = next_state

			for e in traj:
				state, action, reward, next_state, done = e
				experiment_recorder.add_transition(state, action, reward, next_state, done)

			self.demo_path = experiment_recorder.save_trajectories(SIMPLE_STATES_SIZE, fps=1, is_separated_file=True,
																						is_save_utility=False, show_predicates=True)

		return True #save_trial


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
								   flags=p.ER_NO_SEGMENTATION_MASK,
						 			 renderer=p.ER_BULLET_HARDWARE_OPENGL
		)

		depth = img_arr[3]
		depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
		depth = 255 * (depth - depth.min()) / (depth.max() - depth.min())

		# t = cv2.cvtColor(np.expand_dims(img_arr[3], -1), cv2.COLOR_GRAY2RGB)
		# cv2.imwrite('/home/yz/dd.png', t)
		# depth = img_arr[3]
		# tt = 255 * (depth - depth.min()) / (depth.max() - depth.min())
		# tt = cv2.cvtColor(np.expand_dims(tt, -1), cv2.COLOR_GRAY2RGB)
		# tt = np.expand_dims(depth, -1)
		# cv2.imwrite('/home/yz/ee.png', tt)

		# depth = img_arr[3][192:192+128, 192:192+128]  # center 128x128 from 512x512

		# depth = self.far*self.near/(self.far - (self.far - self.near)*depth)
		# depth = 255 * (depth - depth.min()) / (depth.max() - depth.min())

		# t = 255 * (depth - depth.min()) / (depth.max() - depth.min())
		# cv2.imwrite('/home/yz/dd.png', t)
		# plt.imshow(depth, cmap='gray', vmin=0, vmax=1)
		# plt.show()

		# if self._jitterDepth:
		# 	depth += np.random.normal(0, 0.1, size=depth.shape)
		# plt.imshow(depth, cmap='Greys', interpolation='nearest')
		# plt.show()

		obs = np.expand_dims(depth, axis=0)

		simple_states = self.simple_states.values()

		simple_states = list(chain(*simple_states))
		if self.merge_simple_states_to_img:
			H, W = np.shape(depth)
			source = np.zeros(H * W)
			source[:len(simple_states)] = simple_states
			source = np.reshape(source, [H, W])
			obs = np.concatenate((obs, [source]))
		return obs


	def get_pcl(self):
		img_arr = p.getCameraImage(width=self.width,
								   height=self.height,
								   viewMatrix=self.viewMat,
								   projectionMatrix=self.projMat,
								   flags=p.ER_NO_SEGMENTATION_MASK)
		pcl = pixelToWorld(img_arr[3], self._jitterDepth)
		return pcl


	def mouse_control(self, grasp, back_step=1):
		if back_step == 0:
			back_step = 1
		p.removeAllUserParameters()
	
		state = p.getLinkState(
			self.pandaEnv._pandaId, self.pandaEnv._panda.pandaEndEffectorLinkIndex)
		actualEndEffectorPos = state[4]
		actualEndEffectorOrn = p.getEulerFromQuaternion(state[5])
	
		print("SSAA: ", [actualEndEffectorPos, actualEndEffectorOrn])

		targetPosXId = p.addUserDebugParameter("targetPosX", 0., 1.5, actualEndEffectorPos[0])
		targetPosYId = p.addUserDebugParameter("targetPosY", -1, 1, actualEndEffectorPos[1])
		targetPosZId = p.addUserDebugParameter("targetPosZ", 0, 1.15, actualEndEffectorPos[2])
		targetOriRollId = p.addUserDebugParameter("targetOriRoll", -3.15, 3.15, actualEndEffectorOrn[0])
		targetOriPitchId = p.addUserDebugParameter("targetOriPitch", -3.15, 3.15, actualEndEffectorOrn[1])
		targetOriYawId = p.addUserDebugParameter("targetOriYaw", -3.15, 3.15, actualEndEffectorOrn[2])
		graspId = p.addUserDebugParameter("grasp", 0, 1, grasp)


		sleep(1.)
	
		try:
			while True:
				targetPosX = p.readUserDebugParameter(targetPosXId)
				targetPosY = p.readUserDebugParameter(targetPosYId)
				targetPosZ = p.readUserDebugParameter(targetPosZId)
				targetOriRoll = p.readUserDebugParameter(targetOriRollId)
				targetOriPitch = p.readUserDebugParameter(targetOriPitchId)
				targetOriYaw = p.readUserDebugParameter(targetOriYawId)
				grasp = p.readUserDebugParameter(graspId)

				targetPosition = [targetPosX, targetPosY, targetPosZ]
				targetOrientation = [targetOriRoll, targetOriPitch, targetOriYaw]

				fingleValue = 0.0 if grasp > 0.5 else 0.04
				self.pandaEnv.move_pos(absolute_pos=targetPosition, absolute_global_euler=targetOrientation, gripper_target_pos=fingleValue)

				# self.pandaEnv.move_pos(absolute_pos=targetPosition, absolute_global_quat=p.getQuaternionFromEuler(targetOrientation), gripper_target_pos=fingleValue)

				# jointPoses = accurateIK(self.pandaEnv._pandaId, self.pandaEnv._panda.pandaEndEffectorLinkIndex,
				# 															 targetPosition, p.getQuaternionFromEuler(targetOrientation),
				# 															 [-2.90, -1.76, -2.90, -3.07, -2.90, -0.02, -2.90], [2.90, 1.76,	2.90, -0.07, 2.90, 3.75, 2.90], [5.8, 3.5, 5.8, 3, 5.8, 3.8, 5.8], [0, -1.4, 0, -1.4, 0, 1.2, 0],
				# 															 useNullSpace=True)
				# setMotors(self.pandaEnv._pandaId, jointPoses)
				p.stepSimulation()

		except KeyboardInterrupt:
			pass
	
		# print("SSBB: ", [ALLtargetPositions['L'], ALLtargetOrientations['L'], ALLtargetPositions['R'], ALLtargetOrientations['R']])
	
		# Calculate pose offsets
		TransEE = np.array(targetPosition) - actualEndEffectorPos
		RotEE = np.array(targetOrientation) - actualEndEffectorOrn

		self.restore(back_step)
		print("AA: ", np.concatenate((TransEE, RotEE, [grasp])))
		return np.concatenate((TransEE, RotEE, [grasp]))
	
	
	def save(self, id):
		# p.removeState(self.stateId)
		# self.stateId = p.saveState()
		# self._history_env_shot.append(self.stateId)
	
		# id %= self._history_max_len
		# print("BB", id)
		# self._history_env_shot.append(id)

		self._history_env_shot.append("state_" + str(id) + ".bullet")

		print("Saving state_" + str(id) + ".bullet")
		p.saveBullet("state_" + str(id) + ".bullet")
	
	
	def restore(self, id):
		print("Taking " + self._history_env_shot[-id])
		p.restoreState(fileName=self._history_env_shot[-id])

	def step(self, action):
			print("action ", action)
			return self._step_continuous(action)

	def _step_continuous(self, action):
		"""Applies a continuous velocity-control action.

    Args:
      action: 5-vector parameterizing XYZ offset, vertical angle offset
      (radians), and grasp angle (radians).
    Returns:
      observation: Next observation.
      reward: Float of the per-step reward as a result of taking the action.
      done: Bool of whether or not the episode has ended.
      debug: Dictionary of extra information provided by environment.
    """
		# Perform commanded action.
		# self._env_step += 1
		fingleValue = 0.0 if action[6] > 0.5 else 0.04
		self.pandaEnv.move_pos(relative_pos=action[:3], relative_global_euler=action[3:6], gripper_target_pos=fingleValue, numSteps=1000)
		observation, state = self.observe()
		done = self._termination()
		reward = self._reward()
		print(self.simple_states)

		debug = {
			# 'grasp_success': self._grasp_success
		}
		# self.simple_states_prev = self.simple_states
		return observation, reward, done, debug


	def _termination(self):
		if self.simple_states['mug'][2] < self._obj_config[0][2]-0.2:
			print("A target fell down. Impossible to finish the task. Episode ends.")
			return True
		# If the mug is not hold up above 0.05m, return false to continue
		if self.simple_states['mug'][2] <= self._obj_config[0][2] + 0.05:  # 0.65:
			return False

		self.success = True
		return True

	def _reward(self):
		if self.success:
			return 10
		else:
			return 0

	def observe(self):
		s_obs = self._get_simple_observation()
		img_obs = self.get_depth()
		return img_obs, s_obs

	def _get_simple_observation(self):
		"""Observations for simplified observation space.

    Returns:
      Numpy array containing location and orientation of nearest block and
      location of end-effector.
    """
		# _ = self._get_image_observation()
		self.simple_states = OrderedDict()
		# self.simple_states['locs'] = np.array(self.Locs)

		states = []
		state = p.getLinkState(
			self.pandaEnv._pandaId, self.pandaEnv._panda.pandaEndEffectorLinkIndex)
		# end_effector_pos = np.array(state[0])
		# end_effector_ori = pybullet.getEulerFromQuaternion(np.array(state[1]))
		end_effector_pos = np.array(state[4])
		end_effector_ori = p.getEulerFromQuaternion(np.array(state[5]))
		states.extend(np.concatenate((end_effector_pos, end_effector_ori)))
		self.simple_states['ee'] = np.array(states)

		pos, ori = p.getBasePositionAndOrientation(self._objId)
		ori = p.getEulerFromQuaternion(ori)
		pos, ori = np.array(pos), np.array(ori)
		self.simple_states['mug'] = np.concatenate((pos, ori))

		# self.simple_states['fingers'] = np.array([p.getJointState(self.pandaEnv._panda.pandaRightFingerJointIndex), p.getJointState(self.pandaEnv._panda.pandaLeftFingerJointIndex)])

		self.simple_states['fingers'] = p.getJointStates(self.pandaEnv._panda.pandaId, [self.pandaEnv._panda.pandaLeftFingerJointIndex, self.pandaEnv._panda.pandaRightFingerJointIndex])
		self.simple_states['fingers'] = [s[0] for s in self.simple_states['fingers']]

	def traj_playback(self):
		# Load demo data
		for i in range(1):
			with open(self.demo_path[0]+str(i)+self.demo_path[1], "rb") as f:
				demos = pickle.load(f)

			# Play back
			for id, transition in enumerate(demos):
				transition = np.array(transition)
				action = transition[1]
				self.step(action)

		return