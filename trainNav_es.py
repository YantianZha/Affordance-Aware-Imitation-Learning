import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from numpy import array
import ray
import json
import time
import scipy
import random
import matplotlib.pyplot as plt

from src.nn_nav import CNN_nav, Decoder_nav
from src.nav_rollout_env import NavRolloutEnv
from src.pac_es import kl_inverse, compute_grad_ES


class TrainNav_PAC_ES:

	def __init__(self, json_file_name, result_path, model_path):

		# Extract JSON config
		self.json_file_name = json_file_name
		with open(json_file_name+'.json') as json_file:
			self.json_data = json.load(json_file)
		config_dic, pac_dic, nn_dic, optim_dic = \
  			[value for key, value in self.json_data.items()]

		self.delta = pac_dic['delta']
		self.delta_prime = pac_dic['delta_prime']
		self.delta_final = pac_dic['delta_final']
		self.numTrainEnvs = pac_dic['numTrainEnvs']
		self.numTestEnvs = pac_dic['numTestEnvs']
		self.L = pac_dic['L']
		self.include_reg = pac_dic['include_reg']

		dim_cnn_output = nn_dic['dim_cnn_output']
		dim_img_feat = 2*dim_cnn_output # combine RGB and depth
		self.z_dim = nn_dic['z_dim']

		self.actor_pr_path = config_dic['actor_pr_path']
		self.actor_pr_epoch = config_dic['actor_pr_epoch']
		self.numSteps = config_dic['numSteps']
		self.num_cpus = config_dic['num_cpus']
		self.num_gpus = config_dic['num_gpus']
		self.batch_size = config_dic['batch_size']
		self.saved_model_path = config_dic['saved_model_path']
		self.ES_method = config_dic['ES_method']
		self.use_reward = config_dic['use_reward']
		self.use_antithetic = config_dic['use_antithetic']
		self.collision_thres = config_dic['collision_thres']
		self.augment_z = config_dic['augment_z']
		self.num_epsilon = config_dic['num_epsilon']
		self.config_version = config_dic['config_version']
		self.max_rollout_steps = config_dic['max_rollout_steps']

		# Config for trials
		self.table_folder = '/home/ubuntu/SNC_furniture/04379243_v2/'
		self.chair_folder = '/home/ubuntu/SNC_furniture/03001627_v2/'

		self.mu_lr = optim_dic['mu_lr']
		self.logvar_lr = optim_dic['logvar_lr']
		self.decayLR = optim_dic['decayLR']

		# Set up seeding
		self.seed = 0
		random.seed(self.seed)
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		# torch.backends.cudnn.deterministic = True
		# torch.backends.cudnn.benchmark = False

		# Use CPU for ES for now
		device = 'cpu'

		# Load prior policy, freeze params
		self.CNN = CNN_nav(dim_cnn_output=dim_cnn_output, 
					 		img_size=200).to(device)
		self.decoder = Decoder_nav(dim_img_feat=dim_img_feat,
									z_dim=self.z_dim,
									dim_output=4).to(device)
		CNN_load_path = self.actor_pr_path+'bc_CNN_'+ \
      						str(self.actor_pr_epoch)+'.pt'
		decoder_load_path = self.actor_pr_path+'bc_dec_'+ \
      						str(self.actor_pr_epoch)+'.pt'
		self.CNN.load_state_dict(torch.load(CNN_load_path, map_location=device))
		self.decoder.load_state_dict(torch.load(decoder_load_path, map_location=device))
		for name, param in self.CNN.named_parameters():
			param.requires_grad = False
		for name, param in self.decoder.named_parameters():
			param.requires_grad = False
		self.CNN.eval()  # not needed, but anyway
		self.decoder.eval()

		# Set again later
		self.rollout_env = NavRolloutEnv(CNN=self.CNN,
									decoder=self.decoder,
									z_dim=self.z_dim,
									max_rollout_steps=self.max_rollout_steps,
									num_cpus=self.num_cpus,
									num_gpus=self.num_gpus,
									batch_size=self.batch_size,
									AWS=1,
									collision_thres=self.collision_thres,
									config_version=self.config_version)

		# Set prior distribution of parameters
		self.mu_pr = torch.zeros((self.z_dim))
		self.logvar_pr = torch.zeros((self.z_dim))

		# Initialize the posterior distribution
		self.mu_param = torch.tensor(self.mu_pr, requires_grad=True)
		self.logvar_param = torch.tensor(self.logvar_pr, requires_grad=True)

		# Get training and test envs, fixed
		self.trainEnvs = self.get_object_config(numTrials=self.numTrainEnvs, train=True)
		self.testEnvs = self.get_object_config(numTrials=self.numTestEnvs, train=False)

		# Recording: training details and results
		self.result_path = result_path
		self.model_path = model_path
		self.best_bound_data = (0, 0, 0, None, None, (self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()))  # emp, bound, step, mu, logvar, seed
		self.best_emp_data = (0, 0, 0, None, None, (self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()))  # empirical
		self.reward_his = []
		self.cost_env_his = []  # history for plotting, discrete
		self.reg_his = []
		self.kl_his = []
		self.lr_his = []  # learning rate


	def get_object_config(self, numTrials, train=True):
		if train:
			info = np.load('nav_train_envs.npz')
		else:
			info = np.load('nav_test_envs.npz')
		env_ind_all = info['env_ind_all']
		env_pose_all = info['env_pose_all']

		obj_poses_all = env_pose_all[:numTrials]
		obj_paths_all = [(self.table_folder+str(env_ind_all[ind][0])+'/'+ \
									str(env_ind_all[ind][0])+'.obj',
						self.chair_folder+str(env_ind_all[ind][1])+'/'+ \
									str(env_ind_all[ind][1])+'.obj') for ind in range(numTrials)]
		return (obj_poses_all, obj_paths_all)


	def train(self):

		# Resume saved model if specified
		if self.saved_model_path is not "":
			checkpoint = torch.load(self.saved_model_path)
			start_step = checkpoint['step']

			# Retrieve
			self.best_bound_data = checkpoint['best_bound_data']
			self.best_emp_data = checkpoint['best_emp_data']
			self.reward_his = checkpoint['reward_his']
			self.cost_env_his = checkpoint['cost_env_his']
			self.reg_his = checkpoint['reg_his']
			self.kl_his = checkpoint['kl_his']
			self.lr_his = checkpoint['lr_his']

			# Update params
			self.mu_param = checkpoint['mu']
			self.logvar_param = checkpoint['logvar']

			# Load envs
			self.trainEnvs = checkpoint['trainEnvs']
			self.testEnvs = checkpoint['testEnvs']

			# Update seed state
			self.seed, python_seed_state, np_seed_state, torch_seed_state = checkpoint['seed_data']
			random.setstate(python_seed_state)
			np.random.set_state(np_seed_state)
			torch.set_rng_state(torch_seed_state)
		else:
			start_step = -1  # start from beginning

		# Use Adam optimizer from Pytorch, load optim state if resume
		optimizer = torch.optim.Adam([
	  					{'params': self.mu_param, 'lr': self.mu_lr},
						{'params': self.logvar_param, 'lr': self.logvar_lr}])
		if self.decayLR['use']:
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
											mode='max', 
											factor=self.decayLR['gamma'], 
											patience=self.decayLR['patience'], 
											threshold=1e-3, 
											threshold_mode='rel')
		if self.saved_model_path is not "":
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

		# Determine how many policies for one env
		if self.use_antithetic:
			num_trial_per_env = 2*self.num_epsilon
		else:
			num_trial_per_env = self.num_epsilon

		# Extract env configs
		obj_poses_all, obj_paths_all = self.trainEnvs

		# Repeat env config for policies in one env
		obj_poses_all = np.tile(obj_poses_all, (num_trial_per_env,1,1))
		obj_paths_all = obj_paths_all*num_trial_per_env  # tile list

		# Run steps
		for step in range(self.numSteps):
			if step <= start_step:
				continue

			step_start_time = time.time()
			with torch.no_grad():  # speed up
				# Make a copy for the step
				mu_ps = self.mu_param.clone().detach()
				logvar_ps = self.logvar_param.clone().detach()
				mu_pr = self.mu_pr.clone()
				logvar_pr = self.mu_pr.clone()

				# Sample xi used for the step
				if self.augment_z:
					epsilons = torch.normal(mean=0., std=1., 
									size=(self.numTrainEnvs*self.num_epsilon, 
											self.max_rollout_steps,self.z_dim))
				else:
					epsilons = torch.normal(mean=0., std=1., 
						size=(self.numTrainEnvs*self.num_epsilon, self.z_dim))

				# Antithetic if asked
				if self.use_antithetic:
					epsilons = torch.cat((epsilons, -epsilons)) # antithetic

				# Reparameterize
				sigma_ps = (0.5*logvar_ps).exp()
				zs_all = mu_ps + sigma_ps*epsilons

				# Run trials without GUI
				success_arr = self.rollout_env.roll_parallel(zs_all=zs_all, 
													obj_poses_all=obj_poses_all,
													obj_paths_all=obj_paths_all)
				reward_arr = success_arr
				reward_avg = np.mean(reward_arr)
				emp_rate = np.mean(success_arr)  # use discrete success
				cost_env_continuous = torch.tensor(1-reward_arr).float()  # use continuous reward
				cost_env_discrete = torch.tensor(1-success_arr).float()  # use discrete success

				# Include PAC-Bayes reg in ES
				theta = zs_all
				kld, R = self.get_pac_bayes(self.numTrainEnvs, 
											self.delta, 
											logvar_ps, 
											logvar_pr, 
											mu_ps,
											mu_pr)
				reg = np.sqrt(R)

				# Sum over z_dim
				if self.augment_z:
					log_pt_pr = torch.sum(0.5*(logvar_pr-logvar_ps) + \
										(theta-mu_pr)**2/(2*logvar_pr.exp()) - \
										(theta-mu_ps)**2/(2*logvar_ps.exp()) \
											, dim=2)
				else:
					log_pt_pr = torch.sum(0.5*(logvar_pr-logvar_ps) + \
										(theta-mu_pr)**2/(2*logvar_pr.exp()) - \
										(theta-mu_ps)**2/(2*logvar_ps.exp()) \
											, dim=1)

				# Get cost, check if include PAC-Bayes cost, use continuous cost for ES
				if self.use_reward:
					cost_env = cost_env_continuous
				else:
					cost_env = cost_env_discrete

				# Flatten over steps within a rollout
				if self.augment_z:
					log_pt_pr = log_pt_pr.reshape(-1)
					cost_env = cost_env.repeat_interleave(self.max_rollout_steps)
					epsilons = epsilons.reshape(self.numTrainEnvs*self.max_rollout_steps, self.z_dim)

				# Include regularizer gradient in ES
				if self.include_reg:
					cost_es = cost_env + log_pt_pr/(4*self.numTrainEnvs*reg)
				else:
					cost_es = cost_env

				# Get epsilons from mu and zs
				grad_mu, grad_logvar = compute_grad_ES(
										cost_es-torch.mean(cost_es), 
										epsilons, 
					  					sigma_ps, 
						   				method=self.ES_method)

			# Print and record result
			reg = reg.item()
			cost_env = torch.mean(cost_env_discrete).item()  # use discrete
			bound = 1-cost_env-reg
			print("\n", step, "Emp:", emp_rate, "Reward:", reward_avg, "Env:", cost_env, "Reg:", reg, "Bound:", bound, "KL:", kld)
			print('mu:', self.mu_param.data)
			print('logvar:', self.logvar_param.data)
			print('Time: %s\n' % (time.time() - step_start_time))

			# Save mu and logvar if at best McAllester bound
			if bound > self.best_bound_data[1]:
				self.best_bound_data = (emp_rate, bound, step, mu_ps, logvar_ps, (self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()))
			if emp_rate > self.best_emp_data[0]:
				self.best_emp_data = (emp_rate, bound, step, mu_ps, logvar_ps, (self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()))

			# Save training details, cover at each step
			self.reward_his += [reward_avg]
			self.cost_env_his += [cost_env]
			self.reg_his += [reg]
			self.kl_his += [kld]
			self.lr_his += [optimizer.state_dict()['param_groups'][0]['lr']] # only lr for mu since for sigma would be the same
			torch.save({
				'training_his':(self.reward_his, self.cost_env_his, self.reg_his, self.kl_his, self.lr_his),
				'cur_data': (mu_ps, logvar_ps),
				'best_bound_data': self.best_bound_data,
				'best_emp_data': self.best_emp_data,
				'seed_data':(self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()),
				'actor_pr_path':self.actor_pr_path,
				'actor_pr_epoch':self.actor_pr_epoch,
				'json_data':self.json_data,
			}, self.result_path+'train_details')  # not saving optim_state, grad

			# Do not update params until after saving results
			self.mu_param.grad = grad_mu
			self.logvar_param.grad = grad_logvar
			optimizer.step()

			# Decay learning rate if specified
			if self.decayLR['use']:
				# scheduler.step()
				scheduler.step(emp_rate)
	
			# Save model every 5 epochs
			if step % 5 == 0 and step > 0:
				torch.save({
					'step': step,
					'mu': self.mu_param,
					"logvar": self.logvar_param,
					'optimizer_state_dict': optimizer.state_dict(),
					'reward_his': self.reward_his,
					"cost_env_his": self.cost_env_his,
					"reg_his": self.reg_his,
					"kl_his": self.kl_his,
					"lr_his": self.lr_his,
					'best_bound_data': self.best_bound_data,
					'best_emp_data': self.best_emp_data,
					"trainEnvs": self.trainEnvs,
					"testEnvs": self.testEnvs,
					"seed_data": (self.seed, random.getstate(), np.random.get_state(), torch.get_rng_state()),
					}, self.model_path+'model_'+str(step))


	def estimate_train_cost(self, mu_ps, logvar_ps):
		# Extract envs
		obj_poses_all, obj_paths_all = self.trainEnvs

		# Run training trials
		estimate_success_list = np.empty((0))
		for sample_ind in range(self.L):
			with torch.no_grad():  # speed up
				print('\nRunning sample %d out of %d...\n' % (sample_ind+1, self.L))

				# Sample new latent every time
				epsilons = torch.normal(mean=0., std=1., 
									size=(self.numTrainEnvs, 
										self.max_rollout_steps,
										self.z_dim))
				sigma_ps = (0.5*logvar_ps).exp()
				zs_all = mu_ps + sigma_ps*epsilons

				success_sample = self.rollout_env.roll_parallel(zs_all=zs_all, 
													obj_poses_all=obj_poses_all,
													obj_paths_all=obj_paths_all)
				estimate_success_list = np.concatenate((estimate_success_list,
										   				array(success_sample)))
		estimate_cost = np.mean(1-estimate_success_list)
		return estimate_cost


	def estimate_true_cost(self, mu_ps, logvar_ps):
		# Extract envs
		obj_poses_all, obj_paths_all = self.testEnvs

		# Config all test trials
		epsilons = torch.normal(mean=0., std=1., 
							size=(self.numTestEnvs, 
								self.max_rollout_steps,
								self.z_dim))
		sigma_ps = (0.5*logvar_ps).exp()
		zs_all = mu_ps + sigma_ps*epsilons

		# Run test trials and get estimated true cost
		with torch.no_grad():  # speed up
			success_list = self.rollout_env.roll_parallel(zs_all=zs_all, 
												obj_poses_all=obj_poses_all,
												obj_paths_all=obj_paths_all)
		estimate_cost = np.mean(1-array(success_list))
		return estimate_cost


	def compute_final_bound(self, best_data):

		# Retrive mu and logvar from best bound step, or best emp step
		step_used = best_data[2]
		mu_ps = best_data[3]
		logvar_ps = best_data[4]
		seed, python_seed_state, np_seed_state, torch_seed_state = best_data[5]
		mu_pr = self.mu_pr.detach()  # prior, checked all zeros
		logvar_pr = self.logvar_pr.detach()  # prior, checked all zeros

		# Reload seed state
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		random.setstate(python_seed_state)
		np.random.set_state(np_seed_state)
		torch.set_rng_state(torch_seed_state)

		# Get estimated true cost using test envs
		print('Estimating true cost...')
		true_estimate_cost = self.estimate_true_cost(mu_ps, logvar_ps)

		# Get estimated train cost using trian envs and L=100
		print('Estimating training cost (may take a while)...')
		train_estimate_start_time = time.time()
		train_estimate_cost = self.estimate_train_cost(mu_ps, logvar_ps)
		print('\n\n\nTime to run estimate training cost:', time.time()-train_estimate_start_time)

		# Get inverse bound
		_, R_final = self.get_pac_bayes(self.numTrainEnvs, 
										self.delta_final, 
										logvar_ps, 
										logvar_pr, 
										mu_ps, 
										mu_pr)
		cost_chernoff = kl_inverse(train_estimate_cost, 
							 	(1/self.L)*np.log(2/self.delta_prime))
		inv_bound = 1-kl_inverse(cost_chernoff, 2*R_final)

		# McAllester and Quadratic PAC Bound, use estimated training costs, L
		_, R = self.get_pac_bayes(self.numTrainEnvs,
								self.delta,
								logvar_ps,
								logvar_pr,
								mu_ps,
								mu_pr)
		maurer_bound = 1-train_estimate_cost-np.sqrt(R)
		quad_bound = 1-(np.sqrt(train_estimate_cost + R) + np.sqrt(R))**2

		return step_used, R, maurer_bound, quad_bound, inv_bound, train_estimate_cost, true_estimate_cost


	def get_pac_bayes(self, N, delta, logvar_ps, logvar_pr, mu_ps, mu_pr):
		kld = (-0.5*torch.sum(1 + logvar_ps-logvar_pr \
							-(mu_ps-mu_pr)**2/logvar_pr.exp() \
							-(logvar_ps-logvar_pr).exp())).item()  # as scalar
		R = (kld + np.log(2*np.sqrt(N)/delta))/(2*N)
		return kld, R  # as scalar, not tensor


if __name__ == '__main__':

	# Read JSON config
	json_file_name = sys.argv[1]

	# Create a new subfolder under result
	result_path = 'result/'+json_file_name+'/'
	if not os.path.exists(result_path):
		os.mkdir(result_path)
		os.mkdir(result_path+'figure/')

	# Create a new subfolder under model
	model_path = 'model/'+json_file_name+'/'
	if not os.path.exists(model_path):
		os.mkdir(model_path)

	# Initialize trianing env
	trainer = TrainNav_PAC_ES(
	 				json_file_name=json_file_name, 
		 			result_path=result_path,
	  				model_path=model_path)

	# Train
	trainer.train()

	# Get bounds using best bound step, save
	step_used, R, maurer_bound, quad_bound, inv_bound, train_estimate_cost, true_estimate_cost= trainer.compute_final_bound(trainer.best_bound_data)
	print('Using best bound, step', step_used)
	print('R:', R)
	print("Maurer Bound:", maurer_bound)
	print("Quadratic Bound:", quad_bound)
	print("KL-inv bound:", inv_bound)
	print("Train estimate:", 1-train_estimate_cost)
	print("True estimate:", 1-true_estimate_cost)
	print('\n')
	np.savez(result_path+'bounds_best_bound.npz',
		step=step_used,
		R=R,
		maurer_bound=maurer_bound,
		quad_bound=quad_bound,
		inv_bound=inv_bound,
		train_estimate_cost=train_estimate_cost,
		true_estimate_cost=true_estimate_cost,
		)

	# Get bounds using best empirical rate step, save
	step_used, R, maurer_bound, quad_bound, inv_bound, train_estimate_cost, true_estimate_cost= trainer.compute_final_bound(trainer.best_emp_data)
	print('Using best emp, step', step_used)
	print('R:', R)
	print("Maurer Bound:", maurer_bound)
	print("Quadratic Bound:", quad_bound)
	print("KL-inv bound:", inv_bound)
	print("Train estimate:", 1-train_estimate_cost)
	print("True estimate:", 1-true_estimate_cost)
	print('\n')
	np.savez(result_path+'bounds_best_emp.npz',
		step=step_used,
		R=R,
		maurer_bound=maurer_bound,
		quad_bound=quad_bound,
		inv_bound=inv_bound,
		train_estimate_cost=train_estimate_cost,
		true_estimate_cost=true_estimate_cost,
		)
