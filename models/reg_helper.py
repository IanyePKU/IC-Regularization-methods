import random
import torch

class Reg_Helper():
	def __init__(self, noise_scale):
		self.noise_scale = noise_scale
		self.fak_inp = {}
		self.noises = {}
		# self.means = {}

	def get_fak_inp(self, shape, update_rate, device):
		shape_ind = tuple(shape)

		ran_num = random.random()
		if shape_ind not in self.fak_inp.keys() or ran_num < update_rate:
			batch_size = shape[0]
			fake_shape = shape[1:]
			m_noise_scale = (torch.rand(fake_shape).reshape([1] + fake_shape) * self.noise_scale).to(device)
			m_noise_scale = torch.cat([m_noise_scale for _ in range(batch_size)], 0)
			inp_fake = m_noise_scale.mul(torch.randn([batch_size]+ fake_shape).to(device))
			# inp_fake = torch.randn([batch_size]+ fake_shape).to(device)
			self.fak_inp[shape_ind] = inp_fake

		return self.fak_inp[shape_ind]

	def get_noises(self, shape, update_rate, device):
		shape_ind = tuple(shape)

		ran_num = random.random()
		if shape_ind not in self.noises.keys() or ran_num < update_rate:
			noise = (torch.randn(shape) * self.noise_scale).to(device)
			self.noises[shape_ind] = noise

		return self.noises[shape_ind]

	def reg_module(self, module, shape_in, device, rate=0.0, do_abs=False):
		reg_loss = 0
		if rate > 0:
			ran_num = random.random()
			if ran_num < rate:
				inp_fake = self.get_fak_inp(shape_in, 0.1, device)
				if do_abs:
					inp_fake = inp_fake.abs()
				out = module(inp_fake)
				# out_n = self.get_noises(out_.shape, 0.1, device)
				# out = out_ + out_n
				out = out - out.mean(0)
				reg_loss += torch.mul(out, out).mean()

		return reg_loss

	def reg_module_mic(self, module, shape_in, device, rate=0.0, do_abs=False):
		reg_loss = 0
		if rate > 0:
			ran_num = random.random()
			if ran_num < rate:
				inp_fake = self.get_fak_inp(shape_in, 0.1, device)
				if do_abs:
					inp_fake = inp_fake.abs()
				out = module(inp_fake)
				# out_n = self.get_noises(out_.shape, 0.1, device)
				# out = out_ + out_n

				out = torch.mul(out, out)
				inp_fake = torch.mul(inp_fake, inp_fake)
				while(len(list(out.shape)) > 1):
					out = out.sum(1)
				while(len(list(inp_fake.shape)) > 1):
					inp_fake = inp_fake.sum(1)

				out_ = torch.div(out, inp_fake)
				reg_loss += out_.mean()
				# out_mean = float(out_.mean())
				out_ = out_ - out_.mean()
				reg_loss += torch.mul(out_, out_).mean()
				while(reg_loss > 5):
					reg_loss /= 2
				# out = out - out.mean(0)
				# reg_loss += torch.mul(out, out).mean()
		return reg_loss