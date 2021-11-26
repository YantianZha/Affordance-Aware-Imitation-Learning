import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import cv2
import pybullet as p
import numpy as np
from numpy import array
import torch
import matplotlib.pyplot as plt
import ray
import itertools
from src.utils_depth import *
from src.nn_grasp_siamese_full import PolicyNet
from src.panda_env import pandaEnv
from src.utils_geom import quatMult, euler2quat, quat2euler


class GraspRolloutEnv():

	def __init__(self,
				encoder,
				actor,
				z_total_dim,
				num_cpus=10,
				checkPalmContact=True,
				useLongFinger=False,
			  resz=None,
				):
		self.resz = resz
		self.encoder = encoder

		# Policy for inference
		self.actor = actor
		self.z_total_dim = z_total_dim

		# Initialize Panda arm environment
		self.env = pandaEnv(long_finger=useLongFinger)
		self.checkPalmContact = checkPalmContact  # if contact -> fail
		self._cam_random = 0.03

		# Camera parameters
		camera_params = getCameraParametersGrasp()
		# self.viewMat = camera_params['viewMatPanda']
		# self.projMat = camera_params['projMatPanda']
		self.width = camera_params['imgW']
		self.height = camera_params['imgH']
		self.near = camera_params['near']
		self.far = camera_params['far']

		look = [0.45, 0., 1.1]
		distance = 1.1
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

		# Object ID in PyBullet
		self._objId = None

		# Number of cpus for Ray, no constraint if set to zero
		self.num_cpus = num_cpus


	def shutdown(self):
		p.disconnect()


	def reset(self):
		self.env.reset_env()

	def parallel(self, zs_all, objPos, objOrn, objPathInd, objPathList, figure_path=None):
		# print("objPathInd, objPathList ", len(objPathInd), len(objPathList), len(objOrn), len(objPos), len(zs_all))
		# print(objPos, objOrn)
		numTrials = zs_all.shape[0]
		if self.num_cpus == 0:
			ray.init()
		else:
			ray.init(num_cpus=self.num_cpus, num_gpus=0)
		success_list = ray.get([self.parallel_wrapper.remote(self, 
                                    zs=zs_all[trialInd], 
                                    objPos=objPos[trialInd], 
                                    objOrn=objOrn[trialInd], 
                                    objPath=objPathList[objPathInd[trialInd]],
																	  figure_path=figure_path)
                        			for trialInd in range(numTrials)])
		ray.shutdown()
		return success_list


	@ray.remote
	def parallel_wrapper(self, zs, objPos, objOrn, objPath, figure_path=None):
		return self.grasp(zs, objPos, objOrn, objPath, gui=False, figure_path=figure_path)


	def single(self, zs, objPos, objOrn, objPath, gui=True, save_figure=True, figure_path=None, mu=None, sigma=None):
		return self.grasp(zs, objPos, objOrn, objPath, gui, save_figure, figure_path, mu, sigma)


	def grasp(self, zs, objPos, objOrn, objPath, gui, save_figure=False, figure_path=None, mu=None, sigma=None):
		self._obj_config = [objPos]
		# Connect to an PB instance
		if gui:
			p.connect(p.GUI, options="--width=2600 --height=1800")
			p.resetDebugVisualizerCamera(0.8, 180, -45, [0.5, 0, 0])
		else:
			p.connect(p.DIRECT)

		objId = os.path.splitext(os.path.basename(objPath))[0]

		######################### Reset #######################
		self.reset()

		# Load object
		if len(objOrn) == 3:  # input Euler angles
			objOrn = p.getQuaternionFromEuler(objOrn)
		self._objId = p.loadURDF(objPath, 
                           		basePosition=objPos, 
                             	baseOrientation=objOrn)
		p.changeDynamics(self._objId, -1, 
                   		lateralFriction=self.env._mu,
                     	spinningFriction=self.env._sigma, 
                      	frictionAnchor=1, 
                       	mass=0.1)

		# Let the object settle
		for _ in range(20):
			p.stepSimulation()

		######################### Decision #######################

		initial_ee_pos = array([0.3, 0.0, 0.4])
		initial_ee_orn = array([0, 0.7071068, 0, 0.7071068])
		# Set arm to a pose away from camera image, keep fingers open
		self.env.reset_arm_joints_ik(initial_ee_pos,
                               		initial_ee_orn) #fingerPos=0.04)

		# Start the execution loop

		action = np.array([[0.0] * 7])
		action = torch.from_numpy(action).float()
		self.encoder.infering_beginning = True
		hidden_cell = []
		all_imgs = []
		all_affords = []
		depth = self.get_depth()

		for step in range(15):

			# Get observations
			# depth = ((0.7 - self.get_depth())/0.20).clip(max=1.0)
			if gui:
				plt.imshow(depth, cmap='Greys', interpolation='nearest')
				plt.show()
			depth, state, ori_img = self.observe()
			all_imgs.append(cv2.normalize(ori_img, None, 0, 255, cv2.NORM_MINMAX))

			depth_np = depth
			state = np.array(state)
			# if relative:
			state = np.concatenate((state[:6] - state[6:12], state[12:]))
			depth = torch.from_numpy(depth).float().unsqueeze(0)#.unsqueeze(0)
			state = torch.from_numpy(state).float().unsqueeze(0)

			inputs = [depth, state, action]

			# Get skill_z
			zs, featAff, hidden_cell = self.encoder(inputs, 0, 0, train=False, prev_hidden_cell_lstm=hidden_cell)

			afford_map = featAff[0,-1,:,:].detach().numpy()

			afford_map = cv2.resize(afford_map, (512, 512))
			afford_map = cv2.normalize(afford_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

			depth_np_map = cv2.normalize(ori_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
			afford_map = cv2.applyColorMap(afford_map, cv2.COLORMAP_JET)
			# print(afford_map.shape)

			depth_np_map = np.stack((depth_np_map,) * 3, axis=-1)

			# merge map and frame
			fin = cv2.addWeighted(afford_map, 0.5, depth_np_map, 0.5, 0)
			all_affords.append(fin)

			# Infer action
			action = self.actor(featAff, zs, state)#.squeeze(0).detach().numpy()
			pred = action.detach().numpy()[0]
			# print("pred ", pred)
			fingleValue = 0.0 if pred[6] > 0.5 else 0.04
			self.env.move_pos(relative_pos=pred[:3], relative_global_euler=pred[3:6], gripper_target_pos=fingleValue, numSteps=1000)

			# Check success
			done, success = self._termination()

			self.encoder.infering_beginning = False

			if done:
				# Close instance and return result
				p.disconnect()
				if success:
					figs = datetime.now().strftime("_%Y%m%d-%H%M%S")
					for i, im in enumerate(list(zip(all_imgs, all_affords))):
						cv2.imwrite(figure_path + figs + '_obj_' + objId + '_depth_' + str(i) + '.png', im[0])
						cv2.imwrite(figure_path + figs + '_obj_' + objId + '_affordance_' + str(i) + '.png', im[1])
					return 1
				else:
					return 0

		return 0


	def get_depth(self):
		img_arr = p.getCameraImage(width=self.width,
															 height=self.height,
															 viewMatrix=self.viewMat,
															 projectionMatrix=self.projMat,
															 flags=p.ER_NO_SEGMENTATION_MASK,
															 renderer=p.ER_BULLET_HARDWARE_OPENGL
															 )
		depth_ori = img_arr[3]
		depth = self.far * self.near / (self.far - (self.far - self.near) * depth_ori)
		depth = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
		# depth = ((0.9 - depth) / 0.2).clip(min=0.0, max=1.0)
		depth_rs = cv2.resize(depth, dsize=self.resz) if self.resz is not None else depth

		return depth_rs, depth_ori

	def _termination(self):
		if self.simple_states['mug'][2] < self._obj_config[0][2]-0.2:
			print("A target fell down. Impossible to finish the task. Episode ends.")
			return (True, False)

		# Check if the two mugs stands correctly
		rot_r, rot_p, rot_y = self.simple_states['mug'][3:]
		# roll and pitch angle should be around zeros
		if not np.allclose(a=[rot_r, rot_p],
											 b=[0.0, 0.0],
											 atol=1.39626):
			print("A target fell down. Impossible to finish the task. Episode ends.")
			return (True, False)

		# If the mug is not hold up above 0.05m, return false to continue
		if self.simple_states['mug'][2] <= self._obj_config[0][2] + 0.05:
			return (False, False)

		print("Success! ", self._obj_config[0][2])
		return (True, True)

	def observe(self):
		self._get_simple_observation()
		img_obs, img = self.get_depth()
		simple_states = self.simple_states.values()
		simple_states = list(itertools.chain(*simple_states))
		return img_obs, simple_states, img

	def _get_simple_observation(self):
		"""Observations for simplified observation space.

		Returns:
			Numpy array containing location and orientation of nearest block and
			location of end-effector.
		"""
		self.simple_states = OrderedDict()

		states = []
		state = p.getLinkState(
			self.env._pandaId, self.env._panda.pandaEndEffectorLinkIndex)

		end_effector_pos = np.array(state[4])
		end_effector_ori = p.getEulerFromQuaternion(np.array(state[5]))
		states.extend(np.concatenate((end_effector_pos, end_effector_ori)))
		self.simple_states['ee'] = np.array(states)

		pos, ori = p.getBasePositionAndOrientation(self._objId)
		ori = p.getEulerFromQuaternion(ori)
		pos, ori = np.array(pos), np.array(ori)
		self.simple_states['mug'] = np.concatenate((pos, ori))

		self.simple_states['fingers'] = p.getJointStates(self.env._panda.pandaId,
														 [self.env._panda.pandaLeftFingerJointIndex,
														  self.env._panda.pandaRightFingerJointIndex])
		self.simple_states['fingers'] = [s[0] for s in self.simple_states['fingers']]
