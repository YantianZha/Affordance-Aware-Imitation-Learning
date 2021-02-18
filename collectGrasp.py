# -*- coding: utf-8 -*-
import os
import sys
sys.dont_write_bytecode = True
# sys.path.insert(0, '../utils')

import time
import numpy as np
from numpy import array
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

from src.grasp_teleop_env import GraspTeleopEnv


def main():

	# Number of trials to be collected
	numTrials = 1
	numGraspPerTrial = 10

	# Paths to store data and load object
	data_path = '/home/allen/data/training/grasp_test/'
	# obj_path = '/home/allen/data/processed_objects/SNC_v4/03797390/'
	obj_path = '/home/allen/data/processed_objects/SNC_v4_mug/'

	# Object index
	# obj_list = [10,11,12,13,19,28,31,33,34,51,62,67,68,71,72,73,78,83,84,88,1,3,6,7,9,15,16,20,23,41,47,50,52,54,56,57,61,65,74,77]  # 10 + 10
	# obj_list = np.arange(1000,1060)
	# obj_ind = np.random.randint(low=0, high=len(obj_list), size=numTrials)
	obj_list = [1000]
	obj_ind = [0]*numTrials

	# Object initial x/y/yaw, randomized
	xy_range = 0.05
	obj_x = np.random.uniform(low=0.5-xy_range, high=0.5+xy_range, size=(numTrials, ))
 	obj_y = np.random.uniform(low=-xy_range, high=xy_range, size=(numTrials, ))
 	obj_yaw = np.random.uniform(low=-np.pi, high=np.pi, size=(numTrials, ))

	# Set up environment
	env = GraspTeleopEnv()

	# Run all trials
	trialInd = 0
	while trialInd < numTrials:

		# Reset environment
		env.reset()

		# Data path for the trial, make directory if not made
		trial_data_path = data_path + str(trialInd) + '/'
		if not os.path.isdir(trial_data_path):
			os.mkdir(trial_data_path)

		# Delete files in trial folder
		lst = os.listdir(trial_data_path)
		for f in lst:
			os.remove(trial_data_path+f)

		# Object path
		obj = obj_list[obj_ind[trialInd]]
 		# trial_obj_path = obj_path + str(obj) + '_decom_wt.urdf'
 		trial_obj_path = obj_path + str(obj) + '.urdf'
		if not os.path.isfile(trial_obj_path):
			print('Error in object path. Path provided:', trial_obj_path)
			continue
	
		# print((obj_x[trialInd], obj_y[trialInd]))
     
		save_trial = env.grasp_teleop(objX=obj_x[trialInd], 
                                	  objY=obj_y[trialInd], 
                                	  objYaw=obj_yaw[trialInd], 
                            		  data_path=trial_data_path, 
                                	  objPath=trial_obj_path, 
                                   	  numGraspPerTrial=numGraspPerTrial)
		# save_trial = grasp_teleop(objX=0.45, objY=0.05, objYaw=obj_yaw[trialInd], data_path=trial_data_path, obj_path=trial_obj_path, numGraspPerTrial=numGraspPerTrial)

		# If not save, redo the trial
		if save_trial:
			trialInd += 1
			print('Trial %d done' % trialInd)

	# Print meta info
	print('\n\n Number of trials collected: %d\n\n' % numTrials)

	return 1

if __name__ == '__main__':
	main()


# p.addUserDebugLine(hitPos, initialTargetPos, lineColorRGB=[1,0,0], lineWidth=10, lifeTime=5)
