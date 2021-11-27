#!/usr/bin/env python3

import os
import sys
import warnings
from collections import OrderedDict
from datetime import datetime

from config.base_config import cfg_from_file
from dataset import custom_dset
from dataset.dataloader.base_dset import BaseLoader

warnings.filterwarnings("ignore")

import torch
from torch import tensor
import torch.nn.functional as F
from torch.autograd import Variable

print("device: ", torch.cuda.current_device())
import time
import numpy as np
from numpy import array
import visdom
import json
import random
import matplotlib.pyplot as plt
import wandb

from src.nn_grasp_siamese_full import SiameseEncoder, SiamesePolicyNet
from src.grasp_rollout_env_a_1 import GraspRolloutEnv
from dataset.datasetABCGrasp import train_dataset, test_dataset

curr_dir = os.getcwd()

class InferGrasp_BC:

	def __init__(self, json_file_name, weightfile):
		# Configure from JSON file
		self.json_file_name = json_file_name
		with open(json_file_name + '.json') as json_file:
			self.json_data = json.load(json_file)
		config_dic, ent_dic, loss_dic, self.optim_dic = [value for key, value in self.json_data.items()]
		self.N = config_dic['N']
		self.num_cpus = config_dic['num_cpus']
		self.checkPalmContact = config_dic['checkPalmContact']
		self.useLongFinger = config_dic['useLongFinger']
		self.numTest = config_dic['numTest']
		numTrainTrials = config_dic['numTrainTrials']
		numTestTrials = config_dic['numTestTrials']
		z_conv_dim = ent_dic['z_conv_dim']
		z_mlp_dim = ent_dic['z_mlp_dim']
		self.z_total_dim = z_mlp_dim

		# Set up seeding
		self.seed = 0
		random.seed(self.seed)
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)

		# Use GPU for BC
		device = 'cpu'

		# device = torch.device("cuda:0")
		device = torch.device("cpu")

		# Set up networks, calculate number of params
		self.encoder = SiameseEncoder(out_cnn_dim=ent_dic['encoder_out_cnn_dim'],
																	dim_mlp_append=config_dic['actionDim'] + config_dic['stateDim'],
																	z_total_dim=self.z_total_dim,
																	img_size=192,
																	device=device).to(device)
		self.actor = SiamesePolicyNet(input_num_chann=1,
																	dim_mlp_append=0,
																	num_mlp_output=config_dic['actionDim'],
																	out_cnn_dim=ent_dic['actor_out_cnn_dim'],
																	z_conv_dim=z_conv_dim,
																	z_mlp_dim=z_mlp_dim).to(device)

		# The training used dataParallel so loading weights in a special way
		# https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
		# original saved file with DataParallel
		state_dict = torch.load(weightfile['encoder'])#, map_location=torch.device('cpu'))
		# create new OrderedDict that does not contain `module.`
		# new_state_dict = OrderedDict()
		# for k, v in state_dict.items():
		# 	name = k[7:]  # remove `module.`
		# 	new_state_dict[name] = v
		# # load params
		# self.encoder.load_state_dict(new_state_dict)

		self.encoder.load_state_dict(state_dict)
		self.encoder.eval()

		state_dict = torch.load(weightfile['actor'])#, map_location=torch.device('cpu'))
		self.actor.load_state_dict(state_dict)
		self.actor.eval()

		self.seen_obj_ind_list = [28, 2035, 2041, 2348, 2530, 2583, 2087, 2901, 180, 2445, 2037, 2041, 2044, 2141, 2559, 2583, 2077, 2059, 2032, 2050, 2530, 2583, 2035, 2036] #np.arange(1000,1000+60)
		self.unseen_obj_ind_list = [28, 2035, 2041, 2348, 2530, 2583, 2087, 2901, 180, 2445, 2037, 2041, 2044, 2141, 2559, 2583, 2077, 2059, 2032, 2050, 2530, 2583, 2035, 2036]#np.arange(1000-20,1000)
		self.xy_range = 0.0
		self.obj_folder = config_dic['obj_folder']

	def infer(self, path):
		# Initialize rollout env
		rollout_env = GraspRolloutEnv(
			encoder=self.encoder.to('cpu'),
			actor=self.actor.to('cpu'),
			z_total_dim=self.z_total_dim,
			num_cpus=self.num_cpus,
			checkPalmContact=self.checkPalmContact,
			useLongFinger=self.useLongFinger,
			resz=(192, 192))

		# Get seen object configuration
		objPos, objOrn, objPathInd, objPathList = self.get_object_config \
			(numTrials=self.numTest,
			 obj_ind_list=self.seen_obj_ind_list)
		zs_all = torch.normal(mean=0, std=1,
													size=(self.numTest, self.z_total_dim))

		# Run a trial with GUI, debug, save a latent interp figure
		zs_single = torch.normal(mean=0, std=1,
														 size=(1, self.z_total_dim))
		success = rollout_env.single(
			zs=zs_single,
			objPos=[0.65, -0.04, 0.32],
			objOrn=[0., 0., 0.8],
			objPath=random.choice(objPathList),  # self.obj_folder+'2559.urdf',
			gui=False,
			save_figure=True,
			figure_path=path + str(0) + '_z_interp')

	def get_object_config(self, numTrials, obj_ind_list):
		obj_x = np.random.uniform(low=0.65-self.xy_range,
								  high=0.65+self.xy_range,
								  size=(numTrials, 1))
		obj_y = np.random.uniform(low=-0.04-self.xy_range,
								  high=-0.04+self.xy_range,
								  size=(numTrials, 1))
		obj_yaw = 0.8 * np.ones((numTrials, 1))
		objPos = np.hstack((obj_x, obj_y, 0.32*np.ones((numTrials, 1))))
		objOrn = np.hstack((np.zeros((numTrials, 2)), obj_yaw))
		objPathInd = np.random.randint(low=0, high=len(obj_ind_list), size=numTrials)  # use random ini cond for BC
		objPathList = []
		for obj_ind in obj_ind_list:
			objPathList += [self.obj_folder + str(obj_ind) + '.urdf']

		return (objPos, objOrn, objPathInd, objPathList)

class TrainGrasp_BC:

	def __init__(self, json_file_name):

		# Configure from JSON file
		self.json_file_name = json_file_name
		with open(json_file_name+'.json') as json_file:
			self.json_data = json.load(json_file)
		config_dic, ent_dic, loss_dic, self.optim_dic = [value for key, value in self.json_data.items()]
		self.N = config_dic['N']
		self.num_cpus = config_dic['num_cpus']
		self.checkPalmContact = config_dic['checkPalmContact']
		self.useLongFinger = config_dic['useLongFinger']
		self.numTest = config_dic['numTest']
		numTrainTrials = config_dic['numTrainTrials']
		numTestTrials = config_dic['numTestTrials']
		z_conv_dim = ent_dic['z_conv_dim']
		z_mlp_dim = ent_dic['z_mlp_dim']
		self.z_total_dim = z_mlp_dim

		# Set up seeding
		self.seed = 0
		random.seed(self.seed)
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)

		# Use GPU for BC
		device = 'cuda:0'
		self.device = device
		self.tripleCriterion = torch.nn.MarginRankingLoss(margin=optim_dic["tripleLoss_margin"])
		# Sample trials
		trainTrialsList = np.arange(0, numTrainTrials)
		testTrialsList = np.arange(numTrainTrials, numTrainTrials+numTestTrials)
		numTrain = len(trainTrialsList)-len(trainTrialsList)%self.N
		numTest = len(testTrialsList)-len(testTrialsList)%self.N
		print('Num of train trials: ', numTrain)
		print('Num of test trials: ', numTest)

		# Config object index for success test trials
		self.obj_folder = config_dic['obj_folder']
		self.xy_range = 0.0
		self.seen_obj_ind_list = [28, 2035, 2041, 2348, 2530, 2583, 2087, 2901, 180, 2445, 2037, 2041, 2044, 2141, 2559, 2583, 2077, 2059, 2032, 2050, 2530, 2583, 2035, 2036] #np.arange(1000,1000+60)
		self.unseen_obj_ind_list = [28, 2035, 2041, 2348, 2530, 2583, 2087, 2901, 180, 2445, 2037, 2041, 2044, 2141, 2559, 2583, 2077, 2059, 2032, 2050, 2530, 2583, 2035, 2036]#np.arange(1000-20,1000)
		# Body-graspable mug IDs: [28, 2035, 2041, 2348, 2530, 2583, 2087, 2901]
		# Handle_left_right_sides-graspable mug IDs: [28, 180, 2445, 2037, 2041, 2044, 2141, 2559, 2583, 2077]
		# Handle_front_back_sides-graspable mug IDs: [28, 2059, 2032, 2050, 2530, 2583, 2087, 2035, 2036, 2077]

		train_triplets = []
		test_triplets = []

		# Create dataholder
		dset_obj = custom_dset.Custom()
		dset_obj.load(config_dic['trainFolderDir'])

		for i in range(2500):
			pos_anchor_img, pos_img, neg_img = dset_obj.getTriplet()
			train_triplets.append([pos_anchor_img, pos_img, neg_img])
		for i in range(self.numTest):
			pos_anchor_img, pos_img, neg_img = dset_obj.getTriplet(split='test')
			test_triplets.append([pos_anchor_img, pos_img, neg_img])

		loader = BaseLoader

		self.train_dataloader = torch.utils.data.DataLoader(
	  								loader(train_triplets, resz=(192, 192)),
			  						batch_size=self.N, 
									shuffle=True, 
									drop_last=True, 
						   			pin_memory=True, 
							  		num_workers=5)
		self.test_dataloader = torch.utils.data.DataLoader(
	  								loader(test_triplets, resz=(192, 192)),
									batch_size=self.N, 
		 							shuffle=False, 
									drop_last=True, 
					 				pin_memory=True, 
						 			num_workers=5)  # assume small test size, single batch

		# Set up networks, calculate number of params
		self.encoder = SiameseEncoder(out_cnn_dim=ent_dic['encoder_out_cnn_dim'],
								dim_mlp_append=config_dic['actionDim']+config_dic['stateDim'],
								z_total_dim=self.z_total_dim,
								img_size=192,
								device=device).to(device)
		self.actor = SiamesePolicyNet(input_num_chann=1,
								dim_mlp_append=config_dic['stateDim'],
								num_mlp_output=config_dic['actionDim'],
								out_cnn_dim=ent_dic['actor_out_cnn_dim'],
								z_conv_dim=z_conv_dim,
								z_mlp_dim=z_mlp_dim,
								img_size=192,).to(device)
		print('Num of actor parameters: %d' % sum(p.numel() for p in self.actor.parameters() if p.requires_grad))
		print('Num of encoder parameters: %d' % sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))

		# Set up optimizer
		self.optimizer = torch.optim.AdamW([
				{'params': self.actor.parameters(), 
	 			 'lr': optim_dic['actor_lr'], 
		 		 'weight_decay': optim_dic['actor_weight_decay']},
				{'params': self.encoder.parameters(), 
	 			 'lr': optim_dic['encoder_lr'], 
		 		 'weight_decay': optim_dic['encoder_weight_decay']}
				])
		if optim_dic['decayLR']['use']:
			self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
				optimizer, 
				milestones=optim_dic['decayLR']['milestones'], 
		  		gamma=optim_dic['decayLR']['gamma'])


	def get_object_config(self, numTrials, obj_ind_list):
		obj_x = np.random.uniform(low=0.65-self.xy_range,
								  high=0.65+self.xy_range,
								  size=(numTrials, 1))
		obj_y = np.random.uniform(low=-0.04-self.xy_range,
								  high=-0.04+self.xy_range,
								  size=(numTrials, 1))
		obj_yaw = 0.8*np.ones((numTrials, 1))
		objPos = np.hstack((obj_x, obj_y, 0.32*np.ones((numTrials, 1))))
		objOrn = np.hstack((np.zeros((numTrials, 2)), obj_yaw))
		objPathInd = np.random.randint(low=0, high=len(obj_ind_list), size=numTrials)  # use random ini cond for BC
		objPathList = []
		for obj_ind in obj_ind_list:
			objPathList += [self.obj_folder + str(obj_ind) + '.urdf']

		return (objPos, objOrn, objPathInd, objPathList)


	def forward(self, data_batch):

		# Set up loss functions
		mse = torch.nn.MSELoss(reduction="mean")
		l1 = torch.nn.L1Loss(reduction="mean")

		# Extract data from batch
		(traj1_depth, traj2_depth, traj3_depth, traj1_states, traj2_states, traj3_states, traj1_actions, traj2_actions, traj3_actions) = data_batch
		traj1_depth, traj2_depth, traj3_depth, traj1_states, traj2_states, traj3_states, traj1_actions, traj2_actions, traj3_actions = traj1_depth.to(self.device), traj2_depth.to(self.device), traj3_depth.to(self.device), traj1_states.to(self.device), traj2_states.to(self.device), traj3_states.to(self.device), traj1_actions.to(self.device), traj2_actions.to(self.device), traj3_actions.to(self.device)

		triple_loss_total = tensor(0.0, requires_grad=True).to(self.device)
		trans_l1_loss_total = tensor(0.0, requires_grad=True).to(self.device)
		trans_l2_loss_total = tensor(0.0, requires_grad=True).to(self.device)

		for trialInd in range(self.N):  # each trial in the batch is a trajectory
			anchor_trial = [traj1_depth[trialInd], traj1_states[trialInd], traj1_actions[trialInd]]
			pos_trial = [traj2_depth[trialInd], traj2_states[trialInd], traj2_actions[trialInd]]
			neg_trial = [traj3_depth[trialInd], traj3_states[trialInd], traj3_actions[trialInd]]

			# torch.onnx.export(self.encoder, args=(anchor_trial, pos_trial, neg_trial), f="encoder.onnx", verbose=True, input_names=["anchor", "pos", "neg"],
			# 									output_names=["siamese_emb", "conv_img_affordance_feature"])
			E1, E2, E3, A1, A2, A3, featAff1, featAff2, featAff3  = self.encoder(anchor_trial, pos_trial, neg_trial)
			# E: T Steps of Observation Embeddings for a trajectory;
			# A: A sequence of affordance embeddings for an interaction segment of a trajectory
			# featAff: T steps of predicted affordance cues for a trajectory

			A1, A2, A3 = torch.unsqueeze(A1[-1], 0), torch.unsqueeze(A2[-1], 0), torch.unsqueeze(A3[-1], 0)
			# We take the last-step embedding in the sequence of affordance embeddings which encodes the whole interaction segment

			dist_A1_A2 = F.pairwise_distance(A1, A2, 2)
			dist_A1_A3 = F.pairwise_distance(A1, A3, 2)

			dist_E1_A2 = F.pairwise_distance(E1, A2, 2)
			dist_E1_A3 = F.pairwise_distance(E1, A3, 2)

			target_1 = torch.FloatTensor(dist_A1_A2.size()).fill_(-1)
			target_2 = torch.FloatTensor(dist_E1_A2.size()).fill_(-1)

			target_1 = target_1.to(self.device)
			target_1 = Variable(target_1)
			target_2 = target_2.to(self.device)
			target_2 = Variable(target_2)

			# The following 3 lines forms a Coupled Triplet Loss (the Equation 4 in our paper)
			triplet_loss_1 = self.tripleCriterion(dist_A1_A2, dist_A1_A3, target_1)
			triplet_loss_2 = self.tripleCriterion(dist_E1_A2, dist_E1_A3, target_2)
			triplet_loss = torch.div((triplet_loss_1 + triplet_loss_2), 2)

			# torch.onnx.export(self.actor, args=(A1[:-1], z_skill_1[:-1], anchor_trial[1][:-1]), f="actor.onnx", verbose=True,
			# 									input_names=["convX_afford", "zs", "states"],
			# 									output_names=["actions"])
			# BC Losses
			pred_anchor = self.actor(featAff1[:-1], E1[:-1], anchor_trial[1][:-1])
			pred_pos = self.actor(featAff2[:-1], E2[:-1], pos_trial[1][:-1])
			pred_neg = self.actor(featAff3[:-1], E3[:-1], neg_trial[1][:-1])

			# Trans and Rot losses
			trans_l2_loss = mse(pred_anchor, anchor_trial[2][1:]) + mse(pred_pos, pos_trial[2][1:]) + mse(pred_neg, neg_trial[2][1:])
			trans_l1_loss = l1(pred_anchor, anchor_trial[2][1:]) + l1(pred_pos, pos_trial[2][1:]) + l1(pred_neg, neg_trial[2][1:])

			triple_loss_total += triplet_loss
			trans_l1_loss_total += trans_l1_loss
			trans_l2_loss_total += trans_l2_loss

		T = traj1_states.shape[1]
		triple_loss_total /= self.N
		trans_l1_loss_total /= self.N
		trans_l2_loss_total /= self.N

		# I found using the following instead of X_loss_total gives more meaningful affordance cues
		return triplet_loss, trans_l2_loss, trans_l1_loss


	def run(self, loss_dic, train=True):

		# To be divided by batch size
		epoch_loss = 0
		epoch_trans_loss = 0
		epoch_siamese_loss = 0
		num_batch = 0

		# Switch NN mode
		if train:
			self.encoder.train()
			self.actor.train()
			data_loader = self.train_dataloader
		else:
			self.encoder.eval()
			self.actor.eval()
			data_loader = self.test_dataloader

		# Run all batches
		for _, data_batch in enumerate(data_loader):

			# Forward pass to get loss
			siamese_loss, trans_l2_loss, trans_l1_loss = self.forward(data_batch)

			# Get training loss
			train_loss = loss_dic['trans_l2_loss_ratio']*trans_l2_loss + \
							loss_dic['triplet_loss_ratio']*siamese_loss + \
							trans_l1_loss

			if train:
				# zero gradients, perform a backward pass to get gradients
				self.optimizer.zero_grad()
				train_loss.backward()

				# Clip gradient if specified
				if loss_dic['gradientClip']['use']:
					torch.nn.utils.clip_grad_norm_(self.actor.parameters(), loss_dic['gradientClip']['thres'])

				# Update weights using gradient
				self.optimizer.step()

			# Store loss
			epoch_loss += train_loss.item()
			epoch_trans_loss += trans_l1_loss.item()
			epoch_siamese_loss += siamese_loss.item()
			num_batch += 1

		# Decay learning rate if specified
		if train and self.optim_dic['decayLR']['use']:
			self.scheduler.step()

		# Get batch average loss
		epoch_loss /= num_batch
		epoch_trans_loss /= num_batch
		epoch_siamese_loss /= num_batch

		return epoch_loss, epoch_trans_loss, epoch_siamese_loss


	def test_success(self, epoch, path):

		# Initialize rollout env
		rollout_env = GraspRolloutEnv(
							encoder=self.encoder.to('cpu'),
	  						actor=self.actor.to('cpu'), 
							z_total_dim=self.z_total_dim,
			  				num_cpus=self.num_cpus,
		 					checkPalmContact=self.checkPalmContact,
        					useLongFinger=self.useLongFinger,
							resz=(192, 192))

		s_result_path = result_path + 'seen_epoch_' + str(epoch) # Result path for evaluating seen objects
		us_result_path = result_path + 'unseen_epoch_' + str(epoch) # Result path for evaluating unseen objects

		# Get seen object configuration
		objPos, objOrn, objPathInd, objPathList = self.get_object_config \
  								(numTrials=self.numTest, 
           						obj_ind_list=self.seen_obj_ind_list)
		zs_all = torch.normal(mean=0, std=1, 
						   	  size=(self.numTest, self.z_total_dim))

		success_list = rollout_env.parallel(
	  						zs_all=zs_all,
							objPos=objPos,
							objOrn=objOrn,
				   			objPathInd=objPathInd, 
					  		objPathList=objPathList,
							figure_path=s_result_path)
		avg_success_seen = np.mean(array(success_list))

		# Get unseen object configuration
		objPos, objOrn, objPathInd, objPathList = self.get_object_config \
  								(numTrials=self.numTest, 
            					obj_ind_list=self.unseen_obj_ind_list)
		zs_all = torch.normal(mean=0, 
							  std=1, 
						   	  size=(self.numTest, self.z_total_dim))
		success_list = rollout_env.parallel(
	  						zs_all=zs_all,
							objPos=objPos,
							objOrn=objOrn,
				   			objPathInd=objPathInd, 
					  		objPathList=objPathList,
							figure_path=us_result_path)
		avg_success_unseen = np.mean(array(success_list))

		# Move model back to GPU for training
		self.encoder.to('cuda:0')
		self.actor.to('cuda:0')

		return avg_success_seen, avg_success_unseen


	def save_model(self, epoch, path):
		torch.save(self.encoder.state_dict(),
				 path + str(epoch) + '_encoder.pt')
		torch.save(self.actor.state_dict(), 
					path+str(epoch)+'_actor.pt')


if __name__ == '__main__':
	cfg_from_file("config/test.yaml")

	# Read JSON config
	json_file_name = curr_dir + '/' + sys.argv[1]
	with open(json_file_name+'.json') as json_file:
		json_data = json.load(json_file)
	config_dic, ent_dic, loss_dic, optim_dic = [value for key, value in json_data.items()]
	numEpochs = config_dic['numEpochs']

	# Create a new subfolder under result
	result_path = '/data/Yantian/affordance_IL/result/'+datetime.now().strftime("%Y%m%d-%H%M%S")+'/'
	if not os.path.exists(result_path):
		os.umask(0)
		os.makedirs(result_path)
		os.makedirs(result_path+'figure/')

	# Create a new subfolder under model
	model_path = '/data/Yantian/affordance_IL/model/'+datetime.now().strftime("%Y%m%d-%H%M%S")+'/'
	if not os.path.exists(model_path):
		os.umask(0)
		os.makedirs(model_path)

	# Inferer = InferGrasp_BC(json_file_name=json_file_name, weightfile={'encoder': 'model/home/yz/Research/Affordance-Discovery-Imitation/src/grasp_bc_13_a/5_encoder.pt', 'actor': 'model/home/yz/Research/Affordance-Discovery-Imitation/src/grasp_bc_13_a/5_actor.pt'})
	# Inferer.infer(result_path)

	# Initialize trianing env
	trainer = TrainGrasp_BC(json_file_name=json_file_name)

	if config_dic['wandb']:
		wandb.init(project="affordance-discovery-LfD")
		for hyper_params in [config_dic, ent_dic, loss_dic, optim_dic]:
			wandb.config.update(hyper_params)
		wandb.watch([trainer.actor, trainer.encoder], log="parameters")
	if config_dic['visdom']:
		vis = visdom.Visdom(env='grasp')
		trans_loss_window = vis.line(
			X=array([[0, 0]]),
			Y=array([[0, 0]]),
			opts=dict(xlabel='epoch', 
						ylabel='Loss', 
						title='Trans L1, '+json_file_name, 
						legend=['Train Loss', 'Test Loss']))
		rot_loss_window = vis.line(
			X=array([[0, 0]]),
			Y=array([[0, 0]]),
			opts=dict(xlabel='epoch', 
						ylabel='Loss', 
						title='Rot L2, '+json_file_name, 
						legend=['Train Loss', 'Test Loss']))
		accuracy_window = vis.line(
			X=array([[0, 0]]),
			Y=array([[0, 0]]),
			opts=dict(xlabel='epoch', 
						ylabel='Loss', 
						title='Test success rates, '+json_file_name, 
						legend=['Seen', 'Unseen']))

	# Training details to be recorded
	train_loss_list = []
	test_loss_list = []
	train_trans_loss_list = []
	test_trans_loss_list = []
	train_rot_loss_list = []
	test_siamese_loss_list = []
	test_seen_accuracy_list = []
	test_unseen_accuracy_list = []

	# Record best success rate on unseen model, to decide if save model
	best_unseen = 0

	# Train
	for epoch in range(numEpochs):
	
		epoch_start_time = time.time()

		# Run one pass of training
		epoch_loss, epoch_trans_loss, epoch_siamese_loss = trainer.run(loss_dic=loss_dic)
		train_loss_list += [epoch_loss]
		train_trans_loss_list += [epoch_trans_loss]
		train_rot_loss_list += [epoch_siamese_loss]
		print('Epoch: %d, loss: %f, Trans: %.4f, siamese: %.4f' % (epoch, epoch_loss, epoch_trans_loss, epoch_siamese_loss))
		if config_dic['wandb']:
			wandb.log(
				{
					"train_loss": epoch_loss,
					"train_trans_loss": epoch_trans_loss,
					"train_rot_loss": epoch_siamese_loss,
				}, step=epoch
			)

		# Test sample trials
		with torch.no_grad():
			if epoch % 10 == 0:# and epoch > 0:
				epoch_loss, epoch_trans_loss, epoch_siamese_loss = trainer.run(loss_dic=loss_dic, train=False)
				test_loss_list += [epoch_loss]
				test_trans_loss_list += [epoch_trans_loss]
				test_siamese_loss_list += [epoch_siamese_loss]
				print('Test, loss: %f, trans: %.4f, siamese: %.4f' % (epoch_loss, epoch_trans_loss, epoch_siamese_loss))
				if config_dic['wandb']:
					wandb.log(
						{
							"test_loss": epoch_loss,
							"test_trans_loss": epoch_trans_loss,
							"test_siamese_loss": epoch_siamese_loss,
						}, step=epoch
					)

		print('This epoch took: %.2f\n' % (time.time()-epoch_start_time))

		# Test success rate every 50 epochs
		with torch.no_grad():
			if (epoch % 10 == 0 or epoch == numEpochs-1) and epoch > 0:
				sim_start_time = time.time()
				avg_success_seen, avg_success_unseen = trainer.test_success(epoch=epoch,
					path=result_path+'figure/')

				print('Time took to sim:', time.time() - sim_start_time)
				print('Avg seen/unseen success rate:', avg_success_seen, avg_success_unseen)
				test_seen_accuracy_list += [avg_success_seen]
				test_unseen_accuracy_list += [avg_success_unseen]
				print("All seen success rates: ", test_seen_accuracy_list)
				print("All unseen success rates: ", test_unseen_accuracy_list)

				if config_dic['wandb']:
					wandb.log(
						{
							"test_seen_accuracy": avg_success_seen,
							"test_unseen_accuracy": avg_success_unseen,
						}, step=epoch
					)

				# Save model
				if avg_success_unseen > best_unseen-0.05:
					print('Saving model at epoch: ', epoch)
					trainer.save_model(epoch, model_path)
					best_unseen = avg_success_unseen

				if config_dic['visdom']:
					vis.line(X=array([[epoch, epoch]]),
							Y=np.array([[test_seen_accuracy_list[-1],
				  				 		 test_unseen_accuracy_list[-1]]]),
							win=accuracy_window,update='append')

		# Visualize
		if config_dic['visdom']:
			vis.line(X=array([[epoch, epoch]]),
					 Y=array([[train_trans_loss_list[-1], 
							   test_trans_loss_list[-1]]]),
					win=trans_loss_window,update='append')
			vis.line(X=array([[epoch, epoch]]),
					Y=np.array([[train_rot_loss_list[-1], 
				  				 test_siamese_loss_list[-1]]]),
					win=rot_loss_window,update='append')
