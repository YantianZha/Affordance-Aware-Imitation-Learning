import glob
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import warnings
import os
warnings.filterwarnings('ignore')


class train_dataset(Dataset):
	'''
	This class reads data of the raw image files reducing the RAM requirement;
	tradeoff is slow speed.
	'''    
	def __init__(self, trainTrialsList, dataFolderName, device):
		self.device = device
		self.trainTrialsList = trainTrialsList
		self.dataFolderName = dataFolderName
		self.len = len(trainTrialsList)
		self.resz = (192, 192)

		demo_files = glob.glob(dataFolderName + '/*/*/*/traj*.pickle')
		self.demos = []
		for file in demo_files:
			with open(file, "rb") as f:
				d = pickle.load(f, encoding="latin1")
				# demos.extend(d[0])
				self.demos.append(d[0])

		self.depths, self.states, self.actions = [], [], []
		for traj in self.demos:
			depth = [self.resize_img(tran[0][0]) for tran in traj]
			states = [self.extract_state_from_img(tran[0], 14, relative=True) for tran in traj]
			# eePos = [s[:3] for s in states]
			# yaw_enc = [np.array([np.sin(s[5]), np.cos(s[5])]) for s in states]
			actions = [tran[1] for tran in traj]

			self.depths.extend(depth)
			self.states.extend(states)
			self.actions.extend(actions)

	def __getitem__(self, index):
		graspInd = self.trainTrialsList[index]
		depth = np.array(self.depths[graspInd])
		state = np.array(self.states[graspInd])
		action = np.array(self.actions[graspInd])

		return (torch.from_numpy(depth).float().to(self.device),
				torch.from_numpy(state).float().to(self.device),
				torch.from_numpy(action).float().to(self.device),
				)

	def resize_img(self, img, resz=None):
		img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)  # Normalize the pixels to values in 0-1

		img = cv2.resize(img, dsize=self.resz) if self.resz is not None else img
		return img

	def resize_traj_obs(self, traj_data, H, W):
		traj_data = [[cv2.resize(trans[0], (H, W)), trans[1], cv2.resize(trans[2], (H, W))] for trans in traj_data]
		return traj_data

	def extract_state_from_img(self, img, num_states, c=-1, relative=False):
		num_dims = len(img.shape)
		assert num_dims == 3 or num_dims == 4, "input is not an image"
		state = img[:, c, :, :].flatten()[:num_states] if num_dims == 4 else img[c, :, :].flatten()[
																																				 :num_states]

		if relative:
			state = np.concatenate((state[:6] - state[6:12], state[12:]))

		return state

	def __len__(self):
		return self.len


class test_dataset(Dataset):
	'''
	This class reads data of the raw image files reducing the RAM requirement;
	tradeoff is slow speed.    
	'''    
	def __init__(self, testTrialsList, dataFolderName, device):
		self.device = device
		self.testTrialsList = testTrialsList
		self.dataFolderName = dataFolderName
		self.len = len(testTrialsList)
		self.resz = (192, 192)

		demo_files = glob.glob(dataFolderName + '/*/*/*/traj*.pickle')
		self.demos = []
		for file in demo_files:
			with open(file, "rb") as f:
				d = pickle.load(f, encoding="latin1")
				# demos.extend(d[0])
				self.demos.append(d[0])

		self.depths, self.states, self.actions = [], [], []
		for traj in self.demos:
			depth = [self.resize_img(tran[0][0]) for tran in traj]
			states = [self.extract_state_from_img(tran[0], 14, relative=True) for tran in traj]
			# eePos = [s[:3] for s in states]
			# yaw_enc = [np.array([np.sin(s[5]), np.cos(s[5])]) for s in states]
			actions = [tran[1] for tran in traj]

			self.depths.extend(depth)
			self.states.extend(states)
			self.actions.extend(actions)

	def __getitem__(self, index):
		graspInd = self.testTrialsList[index]
		depth = np.array(self.depths[graspInd])
		state = np.array(self.states[graspInd])
		action = np.array(self.actions[graspInd])

		return (torch.from_numpy(depth).float().to(self.device),
				torch.from_numpy(state).float().to(self.device),
				torch.from_numpy(action).float().to(self.device),
				)

	def __len__(self):
		return self.len

	def resize_img(self, img, resz=None):
		img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)  # Normalize the pixels to values in 0-1

		img = cv2.resize(img, dsize=self.resz) if self.resz is not None else img
		return img

	def resize_traj_obs(self, traj_data, H, W):
		traj_data = [[cv2.resize(trans[0], (H, W)), trans[1], cv2.resize(trans[2], (H, W))] for trans in traj_data]
		return traj_data

	def extract_state_from_img(self, img, num_states, c=-1, relative=False):
		num_dims = len(img.shape)
		assert num_dims == 3 or num_dims == 4, "input is not an image"
		state = img[:, c, :, :].flatten()[:num_states] if num_dims == 4 else img[c, :, :].flatten()[
																																				 :num_states]

		if relative:
			state = np.concatenate((state[:6] - state[6:12], state[12:]))

		return state
