import torch
import torch.nn as nn
import torch.autograd as ag
import random
import torch.nn.functional as F

basic_no_linear_func = nn.ReLU

from torch.autograd import Variable
from numbers import Number
import math

from .reg_helper import Reg_Helper

class MLP(nn.Module):
	def __init__(self, channels, noise_scale=1, beta=0.001, droprate=0, version=[1, 2, 3], bn=False, batch_s=32):
		super().__init__()

		self.beta = beta
		self.noise_scale = noise_scale

		self.channels = channels
		self.layers  = {}
		self.fnetwork = nn.Sequential()
		self.batch_s = batch_s

		for i in range(self.num_layers):
			if i < self.num_layers - 1:
				if bn:
					layer_list = [nn.Linear(channels[i], channels[i + 1]), nn.BatchNorm1d(channels[i+1]), basic_no_linear_func()]
				else:
					layer_list = [nn.Linear(channels[i], channels[i + 1]), basic_no_linear_func()]
				self.layers[i + 1] = nn.Sequential(*layer_list)

			else:
				self.layers[i + 1] = nn.Sequential(nn.Linear(channels[i], channels[i + 1]))
			self.fnetwork.add_module(f"flayer{i+1}", self.layers[i+1])

		self.inp_fake = None
		self.droprate = droprate

		self.reg_helper = Reg_Helper(self.noise_scale)

		self.init_modules()

	def init_modules(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		for i in range(self.num_layers):
			x = self.layers[i + 1](x)
		return x

	def forward_dropout(self, x):
		for i in range(self.num_layers - 1):
			x = self.layers[i + 1](x)
			x = F.dropout(x, p=self.droprate, training=self.training)

		x = self.layers[self.num_layers](x)
		return x

	def cal_info_complexity(self, device, dropout_tag=False):
		reg_loss = 0
		for ind in range(self.num_layers):
			reg_loss += self.reg_helper.reg_module(self.layers[ind + 1], [self.batch_s] + [self.channels[ind]], device, rate=1, do_abs=(ind != 0))
		reg_loss  += self.reg_helper.reg_module(self, [self.batch_s] + [self.channels[0]], device=device, rate=1)
		return reg_loss

	def cal_info_complexity_mic(self, device):
		reg_loss = 0
		for ind in range(self.num_layers):
			reg_loss += self.reg_helper.reg_module_mic(self.layers[ind + 1], [self.batch_s] + [self.channels[ind]], device, rate=1)
		reg_loss  += self.reg_helper.reg_module_mic(self, [self.batch_s] + [self.channels[0]], device=device, rate=1)
		return reg_loss

	# def cal_info_complexity_mic(self, device):
	# 	reg_loss = 0
	# 	for ind in range(self.num_layers):
	# 		# noi_tmp = torch.randn([batch_size]+ fake_shape).to(device)
	# 		noi_tmp = self.reg_helper.get_fak_inp([self.channels[ind]] + [self.batch_s], 0.1, device)
	# 		out_tmp = torch.matmul(self.layers[ind + 1][0].weight, noi_tmp)

	# 		noi_tmp = torch.mul(noi_tmp, noi_tmp)
	# 		out_tmp = torch.mul(out_tmp, out_tmp)

	# 		noi_tmp = torch.sqrt(noi_tmp.sum(0))
	# 		out_tmp = torch.sqrt(out_tmp.sum(0))

	# 		out_ = torch.div(out_tmp, noi_tmp)
	# 		out_ = out_ - out_.mean()
	# 		reg_loss += out_.mean()
	# 		reg_loss += torch.mul(out_, out_).mean()
	# 	return reg_loss

	# different way to train NN
	def get_loss_baseline(self, x, y, inforeg=False, device=None):
		loss_func = nn.CrossEntropyLoss()
		out = self.forward(x)
		bp_loss = loss_func(out, y)

		if inforeg == "eic":
			reg_loss = self.cal_info_complexity(device, dropout_tag=False)
			return out, {"total_loss": bp_loss + self.beta * reg_loss, "bp_loss": bp_loss, "reg_loss": reg_loss}
		elif inforeg == 'mic':
			reg_loss = self.cal_info_complexity_mic(device)
			return out, {"total_loss": bp_loss + self.beta * reg_loss, "bp_loss": bp_loss, "reg_loss": reg_loss}
		else:
			return out, {"total_loss": bp_loss, "bp_loss": bp_loss}

	def get_loss_dropout(self, x, y, inforeg=False, device=None):
		loss_func = nn.CrossEntropyLoss()
		out = self.forward_dropout(x)
		bp_loss = loss_func(out, y)

		if inforeg == "eic":
			reg_loss = self.cal_info_complexity(device, dropout_tag=True)
			return out, {"total_loss": bp_loss + self.beta * reg_loss, "bp_loss": bp_loss, "reg_loss": reg_loss}
		elif inforeg == 'mic':
			reg_loss = self.cal_info_complexity_mic(device)
			return out, {"total_loss": bp_loss + self.beta * reg_loss, "bp_loss": bp_loss, "reg_loss": reg_loss}
		else:
			return out, {"total_loss": bp_loss, "bp_loss": bp_loss}

	def get_loss_LS(self, x, y, inforeg=False, device=None, beta=0):
		loss_func = nn.CrossEntropyLoss()
		out = self.forward(x)
		bp_loss = loss_func(out, y)

		log_pro = F.log_softmax(out, dim=1)
		kl_div_loss_func = nn.KLDivLoss()
		class_num = 10.
		u = torch.ones_like(log_pro) / class_num
		ls_loss = kl_div_loss_func(log_pro, u)

		if inforeg == 'eic':
			reg_loss = self.cal_info_complexity(device, dropout_tag=False)
			return out, {"total_loss": bp_loss + self.beta * reg_loss + beta * ls_loss, 
						 "bp_loss": bp_loss, "inforeg_loss": reg_loss, "ls_loss": ls_loss}
		elif inforeg == 'mic':
			reg_loss = self.cal_info_complexity_mic(device)
			return out, {"total_loss": bp_loss + self.beta * reg_loss + beta * ls_loss, 
						 "bp_loss": bp_loss, "inforeg_loss": reg_loss, "ls_loss": ls_loss}
		else:
			return out, {"total_loss": bp_loss + beta * ls_loss, "bp_loss": bp_loss, "ls_loss": ls_loss}

	def get_loss_CP(self, x, y, inforeg=False, device=None, beta=0):
		loss_func = nn.CrossEntropyLoss()
		out = self.forward(x)
		bp_loss = loss_func(out, y)

		pro = F.softmax(out, dim=1)
		kl_div_loss_func = nn.KLDivLoss()
		class_num = 10.
		u = torch.ones_like(pro) / class_num
		u = torch.log(u)
		cp_loss = kl_div_loss_func(u, pro)

		if inforeg == 'eic':
			reg_loss = self.cal_info_complexity(device, dropout_tag=False)
			return out, {"total_loss": bp_loss + self.beta * reg_loss + beta * cp_loss, 
						 "bp_loss": bp_loss, "inforeg_loss": reg_loss, "cp_loss": cp_loss}
		elif inforeg == 'mic':
			reg_loss = self.cal_info_complexity_mic(device)
			return out, {"total_loss": bp_loss + self.beta * reg_loss + beta * cp_loss, 
						 "bp_loss": bp_loss, "inforeg_loss": reg_loss, "cp_loss": cp_loss}
		else:
			return out, {"total_loss": bp_loss + beta * cp_loss, "bp_loss": bp_loss, "cp_loss": cp_loss}

	# def get_loss_vib(self, x, y, inforeg=False, device=None, beta=0):
	@property
	def num_layers(self):
		return len(self.channels) - 1

class MLP_4_DIB(nn.Module):
	def __init__(self, channels, K=256, noise_scale=1, beta=0.001):
		super().__init__()
		self.K = K
		self.encode = MLP(channels + [2*self.K], noise_scale=noise_scale, beta=beta)
		self.decode = MLP([self.K, 10], noise_scale=noise_scale, beta=beta)

		self.beta = beta
		self.init_modules()

	def init_modules(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
				nn.init.constant_(m.bias, 0)

	def forward(self, x, num_sample=1, extra_info=False):
		if not extra_info:
			num_sample = 12

		if x.dim() > 2 : x = x.view(x.size(0),-1)

		statistics = self.encode(x)

		mu = statistics[:,:self.K]
		std = F.softplus(statistics[:,self.K:]-5,beta=1)

		encoding = self.reparametrize_n(mu,std,num_sample)
		logit = self.decode(encoding)

		if num_sample == 1 : pass
		elif num_sample > 1 : logit = F.softmax(logit, dim=2).mean(0)

		if extra_info:
			return (mu, std), logit
		else:
			return logit

	def get_loss_VIB(self, x, y, inforeg=False, device=None, beta=0):
		(mu, std), logit = self.forward(x, extra_info=True)

		class_loss = F.cross_entropy(logit, y).div(math.log(2))
		ib_loss = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum(1).mean().div(math.log(2))

		if inforeg == 'eic':
			reg_loss = self.encode.cal_info_complexity(device, dropout_tag=False) + self.decode.cal_info_complexity(device, dropout_tag=False)
			return logit, {"total_loss": class_loss + self.beta * reg_loss + beta * ib_loss, 
						 "bp_loss": class_loss, "inforeg_loss": reg_loss, "ib_loss": ib_loss}
		elif inforeg == 'mic':
			reg_loss = self.encode.cal_info_complexity_mic(device) + self.decode.cal_info_complexity_mic(device)
			return logit, {"total_loss": class_loss + self.beta * reg_loss + beta * ib_loss, 
						 "bp_loss": class_loss, "inforeg_loss": reg_loss, "ib_loss": ib_loss}
		else:
			return logit, {"total_loss": class_loss + beta * ib_loss, "bp_loss": class_loss, "ib_loss": ib_loss}

	def reparametrize_n(self, mu, std, n=1):
		# reference :
		# http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
		def expand(v):
			if isinstance(v, Number):
				return torch.Tensor([v]).expand(n, 1)
			else:
				return v.expand(n, *v.size())

		if n != 1 :
			mu = expand(mu)
			std = expand(std)

		device = std.device
		eps = Variable(std.data.new(std.size()).normal_().to(device))

		return mu + eps * std
