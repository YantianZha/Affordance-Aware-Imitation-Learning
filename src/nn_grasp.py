import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
from src.nn_func import SpatialSoftmax


class ConvEncoder(nn.Module):
	def __init__(self, in_channel, out_channel):
		super().__init__()

		self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
		self.bn = nn.BatchNorm2d(out_channel)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = F.relu(x)
		x, idx = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
		return x, idx


class ConvDecoder(nn.Module):
	def __init__(self, in_channel, out_channel):
		super().__init__()

		self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, padding=1, stride=2)
		self.bn = nn.BatchNorm2d(out_channel)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = F.relu(x)
		return x

class SiameseEncoder(nn.Module):
	def __init__(self,
				 input_num_chann=1,
				 dim_mlp_append=5,
				 out_cnn_dim=64,  # 32 features x 2 (x/y) = 64
				 z_total_dim=23,
				 img_size=128,
				 dim_lstm_hidden=512,
				 num_lstm_layer=1,
				 device='cpu',
				):

		super(SiameseEncoder, self).__init__()
		self.z_total_dim = z_total_dim
		self.num_lstm_layer = num_lstm_layer
		self.dim_lstm_hidden = dim_lstm_hidden
		self.device = device
		self.old_hidden_a = None
		self.old_hidden_b = None
		self.infering_beginning = False


		# CNN
		self.conv_1 = ConvEncoder(1, out_cnn_dim // 32)
		self.conv_2 = ConvEncoder(out_cnn_dim // 32, out_cnn_dim // 8)
		self.conv_3 = ConvEncoder(out_cnn_dim // 8, out_cnn_dim // 2)
		self.conv_4 = ConvEncoder(out_cnn_dim // 32+1, out_cnn_dim // 8)
		self.conv_5 = ConvEncoder(out_cnn_dim // 8, out_cnn_dim // 32)





		# self.conv_1 = nn.Sequential(  # Nx1x128x128
		# 	nn.Conv2d(in_channels=input_num_chann,
		# 						out_channels=out_cnn_dim // 4,
		# 						kernel_size=5, stride=1, padding=2),
		# 	nn.ReLU(),
		# )  # Nx16x128x128

		# self.conv_2 = nn.Sequential(
		# 	nn.Conv2d(in_channels=out_cnn_dim // 4,
		# 						out_channels=out_cnn_dim // 2,
		# 						kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(),
		# )  # Nx32x128x128

		# Affordance
		self.deconv_1 = ConvDecoder(out_cnn_dim // 2, out_cnn_dim // 8)
		# self.deconv_2 = ConvDecoder(out_cnn_dim // 8, 2)
		self.deconv_2 = nn.Sequential(  # Nx1x128x128
					nn.ConvTranspose2d(in_channels=out_cnn_dim // 8,
								out_channels=2,
								kernel_size=4, padding=1, stride=2),
			# nn.ReLU(),
		)  # Nx16x128x128

		# self.deconv_3 = ConvDecoder(out_cnn_dim // 8, 2)


		# MLP
		self.linear_1 = nn.Sequential(
			nn.Linear(dim_mlp_append,
								out_cnn_dim // 2,
								bias=True),
			nn.ReLU(),
		)

		self.linear_2 = nn.Sequential(
			nn.Linear(out_cnn_dim * 2,
								out_cnn_dim * 2,
								bias=True),
			nn.ReLU(),
		)

		# LSTM, input both img features, states, and actions, learn initial hidden states

		self.lstm = nn.LSTM(input_size=2304,
												hidden_size=self.dim_lstm_hidden,
												num_layers=self.num_lstm_layer,
												batch_first=True,
												bidirectional=False)

		# LSTM and latent output
		self.linear_lstm = nn.Sequential(
			nn.Linear(2304,
								self.dim_lstm_hidden,
								bias=True),
			nn.ReLU(),
		)
		# self.att_vec = Parameter(torch.randn((self.dim_lstm_hidden,
		# 																			1),
		# 																		 dtype=torch.float).to('cuda:0'), requires_grad=True)

		# Output action
		self.linear_out = nn.Linear(self.dim_lstm_hidden,
																z_total_dim,
																bias=True)

	def embeddingNet(self, img_seq, state_seq, prev_action_seq):
		img_seq = torch.unsqueeze(img_seq, 1)
		if img_seq.dim() == 3:
			img_seq = img_seq.unsqueeze(1)  # Bx1xHxW
		B = img_seq.shape[0]  # seq_len

		# CNN
		x1, idx1 = self.conv_1(img_seq)
		#print("AAA ", x1.shape)
		x, idx2 = self.conv_2(x1)
		#print("BBB ", x.shape)
		x, idx3 = self.conv_3(x)
		# print("CCC ", x.shape)

		# Concatenate state and prev_action -> y
		state_action_seq = torch.cat((state_seq, prev_action_seq), dim=1)
		# print("zzz ", state_action_seq.shape)

		# mlp(y)
		y = self.linear_1(state_action_seq)
		#print("DDD ", y.shape)

		# Tile to CNN feature size
		# y = torch.reshape(y, [1, 1, 64])
		# y = torch.unsqueeze(y, -1)
		# y = torch.unsqueeze(y, -1)

		# #print("EEE  ", y.shape)
		# y = repeat(y, 'b x i j -> b x (tilei i) (tilej j)', tilei=x.shape[-2], tilej=x.shape[-1]) # Yantian: change it
		y = y.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[-2], x.shape[-1])  # repeat for all pixels, Nx(z_conv_dim)x200x200

		#print("FFF  ", y.shape)

		# add(x, y)
		x_y = x + y # Yantian check if should just append y as extra channels to x, consistent with the way in PolicyNet

		# Deconv
		x_y = self.deconv_1(x_y)
		#print("GGG  ", x_y.shape)

		x_y = self.deconv_2(x_y)
		#print("HHH  ", x_y.shape)
		# x_y = self.deconv_3(x_y)
		# #print("III  ", x_y.shape)


		# Channelwise Softmax to get affordance map
		aff_map = F.softmax(x_y, 1) # Yantian: don't use relu before softmax layer
		#print("JJJ  ", aff_map.shape)

		aff_map = aff_map[:,0,:,:] # graspable affordance
		aff_map = torch.unsqueeze(aff_map, 1)
		#print("KKK  ", aff_map.shape)

		x_aff_map_0 = torch.cat((x1, aff_map), dim=1)
		#print("LLL  ", x_aff_map_0.shape)

		x_aff_map, id4 = self.conv_4(x_aff_map_0)
		x_aff_map, id5 = self.conv_5(x_aff_map)
		#print("MMM  ", x_aff_map.shape)

		x_aff_map = torch.reshape(x_aff_map, (x_aff_map.shape[0], x_aff_map.shape[1]*x_aff_map.shape[2]*x_aff_map.shape[3]))
		# x_aff_map_l = self.linear_lstm(x_aff_map)
		# #print("NNN  ", x_aff_map_l.shape)

		# Pass concatenated feature timeseries into LSTM to get features at each step.
		hidden_a = torch.randn(self.num_lstm_layer,
													 1,  # batch size
													 self.dim_lstm_hidden).float().to(self.device)
		hidden_b = torch.randn(self.num_lstm_layer,
													 1,
													 self.dim_lstm_hidden).float().to(self.device)

		lstm_in = torch.unsqueeze(x_aff_map, 0)
		x_aff_map_lstm, _ = self.lstm(lstm_in, (hidden_a,hidden_b)) # 1xBx(hidden_size)
		#print("NNN  ", x_aff_map_lstm.shape)
		E = self.linear_out(x_aff_map_lstm[0])
		#print("OOO  ", E.shape)

		# E = F.normalize(E, p=2, dim=1)
		# Yantian This normalization is for the current trajectory, not across multiple trajectories
		# But normaliza over dim=1 is normalize at each time step

		return E, x_aff_map_0

	def embeddingNetInfer(self, img_seq, state_seq, prev_action_seq, prev_lstm_states=None):
		img_seq = torch.unsqueeze(img_seq, 1)
		if img_seq.dim() == 3:
			img_seq = img_seq.unsqueeze(1)  # Bx1xHxW
		B = img_seq.shape[0]  # seq_len

		# CNN
		x1, idx1 = self.conv_1(img_seq)
		#print("AAA ", x1.shape)
		x, idx2 = self.conv_2(x1)
		#print("BBB ", x.shape)
		x, idx3 = self.conv_3(x)
		# print("CCC ", x.shape)

		# Concatenate state and prev_action -> y
		state_action_seq = torch.cat((state_seq, prev_action_seq), dim=1)
		# print("zzz ", state_action_seq.shape)

		# mlp(y)
		y = self.linear_1(state_action_seq)
		#print("DDD ", y.shape)

		# Tile to CNN feature size
		# y = torch.reshape(y, [1, 1, 64])
		# y = torch.unsqueeze(y, -1)
		# y = torch.unsqueeze(y, -1)

		# #print("EEE  ", y.shape)
		# y = repeat(y, 'b x i j -> b x (tilei i) (tilej j)', tilei=x.shape[-2], tilej=x.shape[-1]) # Yantian: change it
		y = y.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.shape[-2], x.shape[-1])  # repeat for all pixels, Nx(z_conv_dim)x200x200

		#print("FFF  ", y.shape)

		# add(x, y)
		x_y = x + y # Yantian check if should just append y as extra channels to x, consistent with the way in PolicyNet

		# Deconv
		x_y = self.deconv_1(x_y)
		#print("GGG  ", x_y.shape)

		x_y = self.deconv_2(x_y)
		#print("HHH  ", x_y.shape)
		# x_y = self.deconv_3(x_y)
		# #print("III  ", x_y.shape)


		# Channelwise Softmax to get affordance map
		aff_map = F.softmax(x_y, 1) # Yantian: don't use relu before softmax layer
		#print("JJJ  ", aff_map.shape)

		aff_map = aff_map[:,0,:,:] # graspable affordance
		aff_map = torch.unsqueeze(aff_map, 1)
		#print("KKK  ", aff_map.shape)

		x_aff_map_0 = torch.cat((x1, aff_map), dim=1)
		#print("LLL  ", x_aff_map_0.shape)

		x_aff_map, id4 = self.conv_4(x_aff_map_0)
		x_aff_map, id5 = self.conv_5(x_aff_map)
		#print("MMM  ", x_aff_map.shape)

		x_aff_map = torch.reshape(x_aff_map, (x_aff_map.shape[0], x_aff_map.shape[1]*x_aff_map.shape[2]*x_aff_map.shape[3]))
		# x_aff_map_l = self.linear_lstm(x_aff_map)
		# #print("NNN  ", x_aff_map_l.shape)

		# Pass concatenated feature timeseries into LSTM to get features at each step.
		if self.infering_beginning:
			hidden_a = torch.randn(self.num_lstm_layer,
														 1,  # batch size
														 self.dim_lstm_hidden).float().to('cpu')
			hidden_b = torch.randn(self.num_lstm_layer,
														 1,
														 self.dim_lstm_hidden).float().to('cpu')
		else:
			hidden_a = prev_lstm_states[0]
			hidden_b = prev_lstm_states[1]

		lstm_in = torch.unsqueeze(x_aff_map, 0)
		x_aff_map_lstm, hidden_cell = self.lstm(lstm_in, (hidden_a,hidden_b)) # 1xBx(hidden_size)

		#print("NNN  ", x_aff_map_lstm.shape)
		E = self.linear_out(x_aff_map_lstm[0])
		#print("OOO  ", E.shape)

		# E = F.normalize(E, p=2, dim=1)
		# Yantian This normalization is for the current trajectory, not across multiple trajectories
		# But normaliza over dim=1 is normalize at each time step

		# self.old_hidden_a = hidden_cell[0]
		# self.old_hidden_b = hidden_cell[1]

		return E, x_aff_map_0, hidden_cell


	def forward(self, anchor, pos, neg, train=True, prev_hidden_cell_lstm=None):
		if train == True:
			E1, A1 = self.embeddingNet(anchor[0], anchor[1], anchor[2])
			E2, A2 = self.embeddingNet(pos[0], pos[1], pos[2])
			E3, A3 = self.embeddingNet(neg[0], neg[1], neg[2])
			return E1, E2, E3, A1, A2, A3
		else:
			E1, A1, hidden_cell = self.embeddingNetInfer(anchor[0], anchor[1], anchor[2], prev_lstm_states=prev_hidden_cell_lstm)
			return E1, A1, hidden_cell


class Encoder(nn.Module):
	def __init__(self,
				 input_num_chann=1,
				 dim_mlp_append=5,
				 out_cnn_dim=64,  # 32 features x 2 (x/y) = 64
				 z_total_dim=23,
				 img_size=128,
				):

		super(Encoder, self).__init__()

		self.z_total_dim = z_total_dim

		# CNN
		self.conv_1 = nn.Sequential(  # Nx1x128x128
								nn.Conv2d(in_channels=input_num_chann,
				  						  out_channels=out_cnn_dim//4, 
				  						  kernel_size=5, stride=1, padding=2),
								nn.ReLU(),
								)    # Nx16x128x128

		self.conv_2 = nn.Sequential(
								nn.Conv2d(in_channels=out_cnn_dim//4, 
				  						  out_channels=out_cnn_dim//2, 
						  				  kernel_size=3, stride=1, padding=1),
								nn.ReLU(),
								)    # Nx32x128x128

		# Spatial softmax, output 64 (32 features x 2d pos)
		self.sm = SpatialSoftmax(height=img_size, 
						   		 width=img_size, 
							  	 channel=out_cnn_dim//2)

		# MLP
		self.linear_1 = nn.Sequential(
									nn.Linear(out_cnn_dim+dim_mlp_append,
				   							  out_cnn_dim*2,
											  bias=True),
									nn.ReLU(),
									)

		self.linear_2 = nn.Sequential(
									nn.Linear(out_cnn_dim*2, 
											  out_cnn_dim*2, 
										   	  bias=True),
									nn.ReLU(),
									)

		# Output action
		self.linear_out = nn.Linear(out_cnn_dim*2, 
									z_total_dim*2, 
									bias=True) 


	def forward(self, img, mlp_append):
		if img.dim() == 3:
			img = img.unsqueeze(1)  # Nx1xHxW

		# CNN
		x = self.conv_1(img)
		x = self.conv_2(x)

		# Spatial softmax
		x = self.sm(x)

		# MLP
		x = self.linear_1(torch.cat((x, mlp_append), dim=1))
		x = self.linear_2(x)
		out = self.linear_out(x)

		return out[:,:self.z_total_dim], out[:,self.z_total_dim:] # mu, var


class SiamesePolicyNet(nn.Module):
	def __init__(self,
							 input_num_chann=1,  # not counting z_conv
							 dim_mlp_append=0,  # not counting z_mlp
							 num_mlp_output=5,
							 out_cnn_dim=64,  # 32 features x 2 (x/y) = 64
							 z_conv_dim=4,
							 z_mlp_dim=4,
							 img_size=128,
							 ):

		super(SiamesePolicyNet, self).__init__()

		self.dim_mlp_append = dim_mlp_append
		self.num_mlp_output = num_mlp_output
		self.z_conv_dim = z_conv_dim
		self.z_mlp_dim = z_mlp_dim

		# CNN
		self.conv_1 = nn.Sequential(  # Nx1x128x128
			nn.Conv2d(in_channels=out_cnn_dim // 16 + 1, # + z_conv_dim,
								out_channels=out_cnn_dim // 2 + 8,
								kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(out_cnn_dim // 2 + 8),
			nn.ReLU(),
		)  # Nx16x128x128
		# self.conv_1 = ConvEncoder(out_cnn_dim // 16 + 1, out_cnn_dim // 2 + 8)
		self.conv_2 = ConvEncoder(2 * (out_cnn_dim // 2 + 8), out_cnn_dim // 2)
		self.conv_3 = ConvEncoder(out_cnn_dim // 2, out_cnn_dim // 8)


		# self.conv_2 = nn.Sequential(
		# 	nn.Conv2d(in_channels=out_cnn_dim // 16 + 1,
		# 						out_channels=out_cnn_dim // 4,
		# 						kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(),
		# )  # Nx32x128x128

		# Spatial softmax, output 64 (32 features x 2d pos)
		self.sm = SpatialSoftmax(height=img_size // 2,
														 width=img_size // 2,
														 channel=out_cnn_dim // 2)

		# MLP
		self.linear_1 = nn.Sequential(
			nn.Linear(4608,
								out_cnn_dim * 10,
								bias=True),
			nn.ReLU(),
		)

		self.linear_2 = nn.Sequential(
			nn.Linear(out_cnn_dim * 10,
								out_cnn_dim * 2,
								bias=True),
			nn.ReLU(),
		)

		# Output action
		self.linear_out = nn.Linear(out_cnn_dim * 2,
																num_mlp_output,
																bias=True)

	def forward(self, convX_afford, zs, states):
		zs_state = torch.cat((zs, states), dim=1)
		#print("PPP ", zs_state.shape)

		convX_afford = self.conv_1(convX_afford)
		#print("QQQ ", convX_afford.shape)

		N, _, H, W = convX_afford.shape

		# Attach latent to image
		# if self.z_conv_dim > 0:
		zs_conv = zs_state.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)  # repeat for all pixels, Nx(z_conv_dim)x200x200
		# #print("zs_conv ", zs_conv.shape)
		x = torch.cat((convX_afford, zs_conv), dim=1)  # along channel
		#print("RRR ", x.shape)

		x, x2 = self.conv_2(x)
		#print("SSS ", x.shape)

		x, x3 = self.conv_3(x)
		#print("TTT ", x.shape)

		x = torch.flatten(x, start_dim=1)
		#print("UUU ", x.shape)

		x = self.linear_1(x)
		x = self.linear_2(x)
		out = self.linear_out(x)


		# if convX_afford.dim() == 3:
		# 	img = convX_afford.unsqueeze(1)  # Nx1xHxW

		# CNN
		# x, id1 = self.conv_1(img)

		# #print("PPP ", x.shape, affordance.shape, zs.shape)
		# N, _, H, W = x.shape
		# # Attach latent to image
		# if self.z_conv_dim > 0:
		# 	# zs_conv = zs.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H,
		# 	# 																																		 W)  # repeat for all pixels, Nx(z_conv_dim)x200x200
		# 	# #print("zs_conv ", zs_conv.shape)
		# 	x = torch.cat((x, affordance), dim=1)  # along channel
		#
		# #print("QQQ ", x.shape)
		# x = self.conv_2(x)
		# #print("RRR ", x.shape)
		#
		# # Spatial softmax
		# x = self.sm(x)
		# #print("SSS ", x.shape)
		#
		#
		# # MLP, add latent as concat
		# if self.z_mlp_dim > 0:
		# 	x = torch.cat((x, zs[:, self.z_conv_dim:]), dim=1)
		# # if mlp_append is not None:
		# # 	x = torch.cat((x, mlp_append), dim=1)
		#
		# x = self.linear_1(x)
		# x = self.linear_2(x)
		# out = self.linear_out(x)

		return out


class PolicyNet(nn.Module):
	def __init__(self, 
				 input_num_chann=1, # not counting z_conv
				 dim_mlp_append=0, # not counting z_mlp
				 num_mlp_output=5,
				 out_cnn_dim=64,  # 32 features x 2 (x/y) = 64
				 z_conv_dim=4,
				 z_mlp_dim=4,
				 img_size=128,
				 ):

		super(PolicyNet, self).__init__()

		self.dim_mlp_append = dim_mlp_append
		self.num_mlp_output = num_mlp_output
		self.z_conv_dim = z_conv_dim
		self.z_mlp_dim = z_mlp_dim

		# CNN
		self.conv_1 = nn.Sequential(  # Nx1x128x128
								nn.Conv2d(in_channels=input_num_chann			+z_conv_dim,
				  						  out_channels=out_cnn_dim//4, 
				  						  kernel_size=5, stride=1, padding=2),
								nn.ReLU(),
								)    # Nx16x128x128

		self.conv_2 = nn.Sequential(
								nn.Conv2d(in_channels=out_cnn_dim//4, 
				  						  out_channels=out_cnn_dim//2, 
						  				  kernel_size=3, stride=1, padding=1),
								nn.ReLU(),
								)    # Nx32x128x128

		# Spatial softmax, output 64 (32 features x 2d pos)
		self.sm = SpatialSoftmax(height=img_size, 
                           		 width=img_size, 
                              	 channel=out_cnn_dim//2)

		# MLP
		self.linear_1 = nn.Sequential(
								nn.Linear(out_cnn_dim+dim_mlp_append+z_mlp_dim, 
				   						out_cnn_dim*2,
										bias=True),
								nn.ReLU(),
								)

		self.linear_2 = nn.Sequential(
									nn.Linear(out_cnn_dim*2, 
											  out_cnn_dim*2, 
										   	  bias=True),
									nn.ReLU(),
									)

		# Output action
		self.linear_out = nn.Linear(out_cnn_dim*2, 
									num_mlp_output, 
									bias=True) 


	def forward(self, img, zs, mlp_append=None):

		if img.dim() == 3:
			img = img.unsqueeze(1)  # Nx1xHxW
		N, _, H, W = img.shape

		# Attach latent to image
		if self.z_conv_dim > 0:
			zs_conv = zs[:,:self.z_conv_dim].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)  # repeat for all pixels, Nx(z_conv_dim)x200x200
			img = torch.cat((img, zs_conv), dim=1)  # along channel

		# CNN
		x = self.conv_1(img)
		x = self.conv_2(x)

		# Spatial softmax
		x = self.sm(x)

		# MLP, add latent as concat
		if self.z_mlp_dim > 0:
			x = torch.cat((x, zs[:,self.z_conv_dim:]), dim=1)
		if mlp_append is not None:
			x = torch.cat((x, mlp_append), dim=1)
   
		x = self.linear_1(x)
		x = self.linear_2(x)
		out = self.linear_out(x)

		return out

