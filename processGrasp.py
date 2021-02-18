# -*- coding: utf-8 -*-
import os
import sys
# sys.path.insert(0, '../utils')

import time
import numpy as np
from numpy import array
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

from src.process_grasp_func import save_trial


def main():

	# Number of trials to be processed
	numTrials = 65
	numGraspPerTrial = 5
	# numTrials = 1
	# numGraspPerTrial = 10
 
	# Paths to store data and load object
	data_path = '/home/allen/data/training/grasp_0526_long/'
	# data_path = '/home/allen/data/training/grasp_0527_single_old/'

	# Used to count and name grasps for all trials
	graspInd = 0

	# Process all trials
	for trialInd in range(numTrials):
		print('Trial', trialInd)
	 
		# Folder to save trial data
		load_path = data_path + str(trialInd) +'/'
		save_path = data_path

		graspInd = save_trial(load_path=load_path, 
							  save_path=save_path, 
							  graspInd=graspInd,
							  numGraspPerTrial=numGraspPerTrial)

	# Save meta info for the samples
	np.savez(data_path + 'meta.npz',
		numTrials=graspInd, 
	)

	print('Number of trials processed:', numTrials)
	print('Total number of grasps:', graspInd)


if __name__ == '__main__':
	main()