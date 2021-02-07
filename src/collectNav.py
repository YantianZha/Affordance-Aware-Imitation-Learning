import sys
sys.path.insert(0, '../utils')

from gibson2.envs.base_env import BaseEnv
import time
import numpy as np
from numpy import array
import pybullet as p
import cv2
import matplotlib.pyplot as plt

from src.utils_nav import get_key_pressed, quat2aa, make_dir
from utils_pb import plot_frame_pb
from utils_geom import euler2quat

tracking_camera = {
	'yaw': 270,
	'z_offset': 1,
	'distance': 1.5,
	'pitch': -20
}


class nav_env:
	
	def __init__(
			self, 
			config_file,
			start_pos,
			start_orn, 
			mode='gui', # or 'headless
			step_length=0.20,
			verbose=True):
		
		if mode == 'gui':
			mode = 'pb'+mode
		self.env = BaseEnv(config_file=config_file, mode=mode)
		self.renderer = self.env.simulator.renderer
		self.robot = self.env.robots[0]
		self.robot_id = self.robot.robot_ids[0]
		self.mesh_body_id = self.env.simulator.scene.mesh_body_id
		self.mode = mode
		self.verbose = verbose
		self.step_length = step_length
		self.renderer_num_inst = len(self.renderer.instances)

		# Robot
		self.start_pos = start_pos
		self.start_orn = start_orn

		# Thresholds
		self.target_thres = 0.8		
		self.collision_thres = -0.03	

  
	def get_robot_cam_rgb(self):
		rgb = self.env.simulator.renderer.render_robot_cameras(modes=('rgb'))[0]
		rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
		return rgb
	def get_robot_cam_depth(self):
		pc = self.env.simulator.renderer.render_robot_cameras(modes=('3d'))[0]
		depth = pc[:,:,2]
		# plt.imshow(depth)
		# plt.show()
		return depth
	def vis_robot_view(self, rgb=[], depth=[], sensors=['rgb','depth','norm_depth']):
		'''Visualize the robot's vision'''
		for sensor in sensors:
			if sensor=='rgb':
				cv2.imshow('RGB', rgb)
				cv2.waitKey(1)
			elif sensor=='depth':
				cv2.imshow('Depth', depth)
				cv2.waitKey(1)
			elif sensor=='norm_depth':
				norm_depth = cv2.normalize(
					depth, None, alpha=0, beta=1, 
					norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
				)
				cv2.imshow('Normalized Depth', norm_depth)
				cv2.waitKey(1)
			else:
				print("Warning: "+sensor+" is not a valid sensor display option")
	def robot_collision(self):
		collision_scene = list(p.getClosestPoints(
			bodyA=self.robot.robot_ids[0], 
			bodyB=self.mesh_body_id, 
			distance=self.collision_thres)  # small threshold
		)
		# check collision position on mesh, count as collision if higher than 5cm (filter out the floor, assume floor as z=0)
		collision_scene_count = len([col[6][2] for col in collision_scene if col[6][2] > 0.05])
		collision_shapenet = []
		for shapenet_obj_id in self.pb_obj_ids:
			collision_shapenet.extend(
				list(p.getClosestPoints(
				bodyA=self.robot.robot_ids[0], 
				bodyB=shapenet_obj_id, 
				distance=self.collision_thres))  # small threshold
			)
		# collision_shapenet_count = len([col for col in collision_shapenet if col[8] < 0]) # check if negative distance (penetration)
		collision_shapenet_count = len([col for col in collision_shapenet]) # check if negative distance (penetration)
		if collision_shapenet_count > 0 or collision_scene_count > 0:
			print([[col[8] for col in collision_shapenet]])
			return True
		else:
			return False


	def rollout_human(self, itr=0, rollout_length=100):
		prim_all, rgb_all, depth_all, pose_all = [], [], [], []
		save_trial = False
		for i in range(rollout_length):

			# Move the PyBullet camera with the robot
			pos = self.robot.get_position()
			pos[2] += tracking_camera['z_offset']
			dist = tracking_camera['distance']/self.robot.scale
			axis, angle = quat2aa(self.robot.get_orientation())	# compute robot's yaw angle to adjust the camera
			if axis[2]<0:
				angle = 2*np.pi - angle
			p.resetDebugVisualizerCamera(dist, tracking_camera['yaw']+angle*180/np.pi, tracking_camera['pitch'], array(pos))

			# Get camera views
			rgb = self.get_robot_cam_rgb()
			depth = self.get_robot_cam_depth()
			if self.mode[-3:] == 'gui':
				self.vis_robot_view(rgb, depth, sensors=['rgb','norm_depth'])
			if self.robot_collision():
				print("Collision!")
				break
	
			# Get key
			key_press_flag = False
			prim = 0  # stand still
			while not key_press_flag:
				key_pressed = get_key_pressed()
				if len(key_pressed)>0:
					if key_pressed[0]==65297:	# Up Arrow Key
						prim = 1
						self.robot.move_forward(forward=self.step_length)
					if key_pressed[0]==65298:	# Down Arrow Key
						prim = 2
						self.robot.move_backward(backward=self.step_length)
					if key_pressed[0]==65295:	# Left Arrow Key
						prim = 3
						self.robot.turn_left(delta=self.step_length)
					if key_pressed[0]==65296:	# Right Arrow Key
						prim = 4
						self.robot.turn_right(delta=self.step_length)
					key_press_flag = True
				else:
					self.robot.keep_still()

			# Check distance to target
			dist = np.sqrt((pos[0]+4.3)**2+(pos[1]-1.7)**2)

			# rgb = rgb[:,:,:3]
			# depth = (-depth/6.0).clip(min=0.0, max=1.0)
			# depth[depth==0] = 1  # fill infinity
			# plt.imshow(rgb)
			# plt.show()
			# plt.imshow(depth)
			# plt.show()

			# Save
			pos = self.robot.get_position()
			orn = self.robot.get_orientation()
			prim_all.append(prim)
			rgb_all.append(rgb)
			depth_all.append(depth)
			pose_all.append((pos, orn))
			if self.verbose:
				print("Frame: %d, Dist to target: %.3f" % (i, dist))
			
			# End if close enough
			if dist < self.target_thres:
				save_trial = True  # keep trial
				break
  
		return prim_all, rgb_all, depth_all, pose_all, save_trial


	def load_shapenet_obj(self, poses, path_list):
		shapenet_obj_ids = []
		pb_obj_ids = []
		for (i, (pose, path)) in enumerate(zip(poses, path_list)):
			pos = [pose[0], pose[1], 0.0]
			orn = euler2quat([pose[2],0,0])

			pb_obj_id = []
			visual_id = p.createVisualShape(
				p.GEOM_MESH,
				fileName=path,
				meshScale=1.,
			)
			collision_id = p.createCollisionShape(
				p.GEOM_MESH,
				fileName=path,
				meshScale=1.,
			)
			pb_obj_id = p.createMultiBody(
				baseVisualShapeIndex=visual_id,
				baseCollisionShapeIndex=collision_id,
				basePosition=pos,
				baseOrientation=orn,
			)
			pb_obj_ids.append(pb_obj_id)

			# Load object in iGibson renderer
			self.renderer.load_object(
				obj_path=path,
	 			transform_pos=pos,
				transform_orn=orn,
				load_texture=True
			)
			vis_obj_id = len(self.renderer.visual_objects) - 1
			shapenet_obj_ids.append(vis_obj_id)
			self.renderer.add_instance(object_id=shapenet_obj_ids[-1])

		# Target
		target_path = '/home/allen/data/processed_objects/YCB/003_cracker_box/google_16k/textured.obj'
		target_pos = [-4.3,1.7,0.9]
		target_orn = euler2quat([-np.pi/4,0,0])
		target_visual_id = p.createVisualShape(
			p.GEOM_MESH,
			fileName=target_path,
			meshScale=2,
		)
		self.pb_target_id = p.createMultiBody(
			baseVisualShapeIndex=target_visual_id,
			basePosition=target_pos,
			baseOrientation=target_orn,
		)
		self.renderer.load_object(
			obj_path=target_path,
			transform_pos=target_pos,
			transform_orn=target_orn,
			load_texture=True,
			scale=2,
		)
		self.renderer.add_instance(object_id=len(self.renderer.visual_objects)-1)

		# # Let object settle
		# p.setGravity(0,0,-9.81)
		# for _ in range(240):
		#     p.stepSimulation()

		return shapenet_obj_ids, pb_obj_ids


	def collect_data(self, obj_poses, obj_path_list, save_path, num_rollouts=1):

		# Make directory for saving data
		make_dir(save_path+'rgbd/')
		make_dir(save_path+'prims/')
		make_dir(save_path+'robot_traj/')

		rollout_ind = 0
		while rollout_ind < num_rollouts:
			print("\n")
			print("=====================")
			print("Starting New Trial...")
			print("=====================")
			time.sleep(1)

			# Load objects
			num_objects = len(obj_path_list)
			self.shapenet_obj_ids, self.pb_obj_ids = self.load_shapenet_obj(
				poses=obj_poses,
				path_list=obj_path_list,
			)

			# Reset robot
			p.resetBasePositionAndOrientation(self.robot_id, 
									 		  self.start_pos, 
											  self.start_orn)		

			# plot_frame_pb(pos=[-3,2,0.1])

			# Run demos
			prim_all, rgb_all, depth_all, pose_all, save_trial = self.rollout_human(itr=rollout_ind)

			# Clear objects
			for pb_obj_id in self.pb_obj_ids:
				p.removeBody(pb_obj_id)
			p.removeBody(self.pb_target_id)
			self.renderer.instances = self.renderer.instances[:2]
			self.renderer.textures = self.renderer.textures[:7]

			if save_trial:
				# Log
				np.save(save_path+'prims/'+str(rollout_ind)+'.npy', prim_all)
				np.save(save_path+'robot_traj/'+str(rollout_ind)+'.npy', pose_all)
				for (i, (rgb, depth)) in enumerate(zip(rgb_all, depth_all)):
					np.save(save_path+'rgbd/'+str(rollout_ind)+'_'+str(i)+'.npy', np.concatenate((rgb, depth[:,:,np.newaxis]), axis=2))

				# Count as one rollout
				rollout_ind += 1

		# p.disconnect()


if __name__ == '__main__':

	seed = 1000
	np.random.seed(seed)

	# Data path
	data_path = '/home/allen/data/training/nav_0720/'

	# Object path
	table_folder = '/home/allen/data/processed_objects/SNC_furniture/04379243_v2/'
	chair_folder = '/home/allen/data/processed_objects/SNC_furniture/03001627_v2/'

	# Config all
	numTrials = 100
	offset = 87
	# table_id_range = np.arange(140,184)  # was (150, 184)
	# chair_id_range = np.arange(140,162)  # was (150, 162)
	table_id_range = np.arange(200,293)
	chair_id_range = np.arange(180,266)
	pose_range = [array([[-3.0, -1.0, -np.pi/2]]), 
               	  array([[-1.0,  1.5,  np.pi/2]])]  # was -0.5 for x

	table_id_all = np.random.choice(a=table_id_range, size=numTrials, replace=True)
	chair_id_all = np.random.choice(a=chair_id_range, size=numTrials, replace=True)
	table_pose_all = np.random.uniform(low=pose_range[0], high=pose_range[1], size=(numTrials,3))
	chair_pose_all = np.random.uniform(low=pose_range[0], high=pose_range[1], size=(numTrials,3))

	# Initialize env
	nav = nav_env(config_file='config/fetch_p2p_nav_gui.yaml', 
				  verbose=True,
				  step_length=0.20,					
				  start_pos=[0.5, 0.0, 0.0],
				  start_orn=list(euler2quat([np.pi*7/8,0,0])))

	# Collect all demos
	for trialInd in range(numTrials):

		trial_path = data_path + str(trialInd+offset) + '/'
		make_dir(trial_path)

		# config for the trial
		obj_path_list = [table_folder+str(table_id_all[trialInd])+'/'+ \
      						str(table_id_all[trialInd])+'.obj',
            			chair_folder+str(chair_id_all[trialInd])+'/'+ \
                   			str(chair_id_all[trialInd])+'.obj']
		obj_poses = np.vstack((table_pose_all[trialInd], 
                         	   chair_pose_all[trialInd]))
		nav.collect_data(save_path=trial_path,
                   		 obj_poses=obj_poses,
				   		 obj_path_list=obj_path_list,
					  	 num_rollouts=2)

	p.disconnect()
