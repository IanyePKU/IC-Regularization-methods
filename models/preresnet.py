from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from .reg_helper import Reg_Helper
__all__ = ['PreResNet']

def conv3x3(in_planes, out_planes, stride=1):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0):
		super(BasicBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes)
		self.downsample = downsample
		self.stride = stride

		self.in_planes = inplanes
		self.out_planes = self.expansion * planes
		self.stride = stride
		self.dropout_rate = dropout_rate

	def forward(self, x):
		residual = x

		out = self.bn1(x)
		out = self.relu(out)
		out = self.conv1(out)

		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv2(out)

		if self.dropout_rate > 0:
			out = F.dropout(out, p=self.dropout_rate, inplace=False, training=self.training)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		return out

	def get_out_shape(self, in_shape):
		assert in_shape[1] == self.in_planes
		out_shape = list(in_shape)
		out_shape[1] = self.out_planes
		out_shape[2] = out_shape[2] // self.stride
		out_shape[3] = out_shape[3] // self.stride
		return out_shape


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(inplanes)
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.bn1(x)
		out = self.relu(out)
		out = self.conv1(out)

		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv2(out)

		out = self.bn3(out)
		out = self.relu(out)
		out = self.conv3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual

		return out


class PreResNet(nn.Module):

	def __init__(self, depth, num_classes=10, block_name='BasicBlock', noise_scale=1, beta=0.1, dropRate=0, batch_s=32):
		super(PreResNet, self).__init__()
		# Model type specifies number of layers for CIFAR-10 model
		if block_name.lower() == 'basicblock':
			assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
			n = (depth - 2) // 6
			block = BasicBlock
		elif block_name.lower() == 'bottleneck':
			assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
			n = (depth - 2) // 9
			block = Bottleneck
		else:
			raise ValueError('block_name shoule be Basicblock or Bottleneck')

		self.inplanes = 16
		self.noise_scale = noise_scale
		self.beta = beta
		self.dropout_rate = dropRate
		self.batch_s = batch_s

		self.inp_fake = None
		self.block1_in_planes = 16
		self.reg_helper = Reg_Helper(self.noise_scale)

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
							   bias=False)
		self.layer1 = self._make_layer(block, 16, n)
		self.layer2 = self._make_layer(block, 32, n, stride=2)
		self.layer3 = self._make_layer(block, 64, n, stride=2)
		self.bn = nn.BatchNorm2d(64 * block.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(8)
		self.fc = nn.Linear(64 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)

		x = self.layer1(x)  # 32x32
		x = self.layer2(x)  # 16x16
		x = self.layer3(x)  # 8x8
		x = self.bn(x)
		x = self.relu(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

	def mod_bn_layer(self, mod):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				if mod:
					m.track_running_stats = True
				else:
					m.track_running_stats = False

	def cal_info_complexity_Layerwise(self, x, device):
		self.mod_bn_layer(False)
		in_shape = list(x.shape)
		in_shape[0] = self.batch_s
		reg_loss = 0

		reg_loss += self.reg_helper.reg_module(lambda x: self.conv1(x), in_shape, device, rate=1)
		in_shape[1] = self.block1_in_planes

		for ind in range(len(self.layer1)):
			reg_loss += self.reg_helper.reg_module(self.layer1[ind], in_shape, device, rate=1)
			in_shape = self.layer1[ind].get_out_shape(in_shape)
		for ind in range(len(self.layer2)):
			reg_loss += self.reg_helper.reg_module(self.layer2[ind], in_shape, device, rate=1)
			in_shape = self.layer2[ind].get_out_shape(in_shape)
		for ind in range(len(self.layer3)):
			reg_loss += self.reg_helper.reg_module(self.layer3[ind], in_shape, device, rate=1)
			in_shape = self.layer3[ind].get_out_shape(in_shape)

		in_shape[2] = in_shape[2] // 8
		in_shape[3] = in_shape[3] // 8
		reg_loss += self.reg_helper.reg_module(self.linear, [in_shape[0], in_shape[1] * in_shape[2] * in_shape[3]], device, rate=1)

		reg_loss += self.reg_helper.reg_module(self, list(x.shape), device=device, rate=1)
		self.mod_bn_layer(True)

		return reg_loss

	def cal_info_complexity_Blockwise(self, device):
		self.mod_bn_layer(False)
		batch_s = self.batch_s
		fake_shape = [[batch_s, 3, 32, 32], [batch_s, 16, 32, 32], [batch_s, 32, 16, 16], [batch_s, 10]]
		
		def block3(x):
			x = self.layer3(x)
			x = self.bn(x)
			x = self.relu(x)
			x = self.avgpool(x)
			x = x.view(x.size(0), -1)
			x = self.fc(x)
			return x

		block_list = [lambda x: self.layer1(self.conv1(x)), self.layer2, block3]

		reg_loss = 0
		for i in range(3):
			reg_loss += self.reg_helper.reg_module(block_list[i], fake_shape[i], device=device, rate=0.25)
		reg_loss += self.reg_helper.reg_module(self, fake_shape[0], device=device, rate=0.25)
		self.mod_bn_layer(True)

		return reg_loss

	def cal_info_complexity_mic(self, device):
		self.mod_bn_layer(False)
		batch_s = self.batch_s
		fake_shape = [[batch_s, 3, 32, 32], [batch_s, 16, 32, 32], [batch_s, 32, 16, 16], [batch_s, 10]]
		
		def block3(x):
			x = self.layer3(x)
			x = self.bn(x)
			x = self.relu(x)
			x = self.avgpool(x)
			x = x.view(x.size(0), -1)
			x = self.fc(x)
			return x

		block_list = [lambda x: self.layer1(self.conv1(x)), self.layer2, block3]

		reg_loss = 0
		for i in range(3):
			reg_loss += self.reg_helper.reg_module_mic(block_list[i], fake_shape[i], device=device, rate=0.25)
		reg_loss += self.reg_helper.reg_module_mic(self, fake_shape[0], device=device, rate=0.25)
		self.mod_bn_layer(True)

		return reg_loss

	# def cal_info_complexity_mic(self, x, device):
	# 	self.mod_bn_layer(False)
	# 	in_shape = list(x.shape)
	# 	reg_loss = 0

	# 	def cal_mic_layer(mod, shape, fun_=None):
	# 		noi_tmp = self.reg_helper.get_fak_inp(shape, 0.1, device)
	# 		out_tmp = mod(noi_tmp)

	# 		noi_tmp = torch.mul(noi_tmp, noi_tmp)
	# 		out_tmp = torch.mul(out_tmp, out_tmp)

	# 		if fun_ == None:
	# 			noi_tmp = noi_tmp.sum(1).sum(1).sum(1)
	# 			out_tmp = out_tmp.sum(1).sum(1).sum(1)
	# 		else:
	# 			noi_tmp = fun_(noi_tmp)
	# 			out_tmp = fun_(out_tmp)

	# 		out_ = torch.sqrt(torch.div(out_tmp, noi_tmp))
	# 		out_ = out_ - out_.mean()
	# 		return torch.mul(out_, out_).mean()

	# 	in_shape[0] = self.batch_s
	# 	reg_loss = cal_mic_layer(self.conv1, in_shape)
	# 	in_shape[1] = self.block1_in_planes

	# 	for ind in range(len(self.layer1)):
	# 		reg_loss += cal_mic_layer(self.layer1[ind].conv1, in_shape)
	# 		in_shape = self.layer1[ind].get_out_shape(in_shape)
	# 		reg_loss += cal_mic_layer(self.layer1[ind].conv2, in_shape)
	# 	for ind in range(len(self.layer2)):
	# 		reg_loss += cal_mic_layer(self.layer2[ind].conv1, in_shape)
	# 		in_shape = self.layer2[ind].get_out_shape(in_shape)
	# 		reg_loss += cal_mic_layer(self.layer2[ind].conv2, in_shape)
	# 	for ind in range(len(self.layer3)):
	# 		reg_loss += cal_mic_layer(self.layer3[ind].conv1, in_shape)
	# 		in_shape = self.layer3[ind].get_out_shape(in_shape)
	# 		reg_loss += cal_mic_layer(self.layer3[ind].conv2, in_shape)

	# 	# print(in_shape)
	# 	# exit()
	# 	in_shape[2] = in_shape[2] // 8
	# 	in_shape[3] = in_shape[3] // 8
	# 	reg_loss += cal_mic_layer(lambda x: torch.matmul(self.fc.weight, x), [in_shape[1] * in_shape[2] * in_shape[3], in_shape[0]], 
	# 							  lambda x: x.sum(0))
	# 	self.mod_bn_layer(True)

	# 	return reg_loss

	def get_loss_baseline(self, x, y, inforeg=None, device=None):
		loss_func = nn.CrossEntropyLoss()
		out = self.forward(x)
		bp_loss = loss_func(out, y)

		if inforeg is None:
			return out, {"total_loss": bp_loss, "bp_loss": bp_loss}
		elif inforeg == "block":
			reg_loss = self.cal_info_complexity_Blockwise(device)
			return out, {"total_loss": bp_loss + self.beta * reg_loss, "bp_loss": bp_loss, "reg_loss": reg_loss}
		elif inforeg == "layer":
			reg_loss = self.cal_info_complexity_Layerwise(x, device)
			return out, {"total_loss": bp_loss + self.beta * reg_loss, "bp_loss": bp_loss, "reg_loss": reg_loss}
		elif inforeg == 'mic':
			reg_loss = self.cal_info_complexity_mic(device)
			return out, {"total_loss": bp_loss + self.beta * reg_loss, "bp_loss": bp_loss, "reg_loss": reg_loss}

	def get_loss_LS(self, x, y, inforeg=None, device=None, beta=0):
		loss_func = nn.CrossEntropyLoss()
		out = self.forward(x)
		bp_loss = loss_func(out, y)

		log_pro = F.log_softmax(out, dim=1)
		kl_div_loss_func = nn.KLDivLoss()
		class_num = 10.
		u = torch.ones_like(log_pro) / class_num
		ls_loss = kl_div_loss_func(log_pro, u)

		if inforeg is None:
			return out, {"total_loss": bp_loss + beta * ls_loss, "bp_loss": bp_loss, "ls_loss": ls_loss}
		elif inforeg == "block":
			reg_loss = self.cal_info_complexity_Blockwise(device)
			return out, {"total_loss": bp_loss + self.beta * reg_loss + beta * ls_loss, 
						 "bp_loss": bp_loss, "inforeg_loss": reg_loss, "ls_loss": ls_loss}
		elif inforeg == "layer":
			reg_loss = self.cal_info_complexity_Layerwise(x, device)
			return out, {"total_loss": bp_loss + self.beta * reg_loss + beta * ls_loss, 
						 "bp_loss": bp_loss, "inforeg_loss": reg_loss, "ls_loss": ls_loss}
		elif inforeg == 'mic':
			reg_loss = self.cal_info_complexity_mic(device)
			return out, {"total_loss": bp_loss + self.beta * reg_loss + beta * ls_loss, 
						 "bp_loss": bp_loss, "inforeg_loss": reg_loss, "ls_loss": ls_loss}


	def get_loss_CP(self, x, y, inforeg=None, device=None, beta=0):
		loss_func = nn.CrossEntropyLoss()
		out = self.forward(x)
		bp_loss = loss_func(out, y)

		pro = F.softmax(out, dim=1)
		kl_div_loss_func = nn.KLDivLoss()
		class_num = 10.
		u = torch.ones_like(pro) / class_num
		u = torch.log(u)
		cp_loss = kl_div_loss_func(u, pro)

		if inforeg is None:
			return out, {"total_loss": bp_loss + beta * cp_loss, "bp_loss": bp_loss, "cp_loss": cp_loss}
		elif inforeg == "block":
			reg_loss = self.cal_info_complexity_Blockwise(device)
			return out, {"total_loss": bp_loss + self.beta * reg_loss + beta * cp_loss, 
						 "bp_loss": bp_loss, "inforeg_loss": reg_loss, "cp_loss": cp_loss}
		elif inforeg == "layer":
			reg_loss = self.cal_info_complexity_Layerwise(x, device)
			return out, {"total_loss": bp_loss + self.beta * reg_loss + beta * cp_loss, 
						 "bp_loss": bp_loss, "inforeg_loss": reg_loss, "cp_loss": cp_loss}
		elif inforeg == "mic":
			reg_loss = self.cal_info_complexity_mic(device)
			return out, {"total_loss": bp_loss + self.beta * reg_loss + beta * cp_loss, 
						 "bp_loss": bp_loss, "inforeg_loss": reg_loss, "cp_loss": cp_loss}