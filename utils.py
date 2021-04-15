import torch
import random
import numpy as np

# import torchvision.models as std_models
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim

import models

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def set_seed(seed=0):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def get_dataloader(cfg):
	root = cfg.root
	name_2_datagen = {"mnist": torchvision.datasets.MNIST, "kmnist": torchvision.datasets.KMNIST, 
					  "svhn": torchvision.datasets.SVHN, "cifar10":torchvision.datasets.CIFAR10, 
					  "cifar100": torchvision.datasets.CIFAR100}

	assert cfg.name in name_2_datagen.keys()
	dataset_gen = name_2_datagen[cfg.name]

	transform_train = transforms.ToTensor()
	transform_test = transforms.ToTensor()

	if cfg.name == 'cifar10':
		# RGB
		mean = np.array([0.4914, 0.4822, 0.4465])
		std = np.array([0.2470, 0.2435, 0.2616])
	elif cfg.name == 'cifar100':
		# RGB
		mean = np.array([0.5071, 0.4865, 0.4409])
		std = np.array([0.2673, 0.2564, 0.2762])
	elif cfg.name == 'mnist':
		mean = np.array([0.1307])
		std = np.array([0.3081])
	elif cfg.name == 'kmnist':
		mean = np.array([0.1904])
		std = np.array([0.3475])
	elif cfg.name == 'svhn':
		mean = np.array([0.5, 0.5, 0.5])
		std = np.array([0.5, 0.5, 0.5])
	else:
		raise ValueError()

	if cfg.name == 'mnist' or cfg.name == 'kmnist':
		transform_train = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
		])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
		])
	elif cfg.name == 'cifar10' or cfg.name == 'cifar100' or cfg.name == 'svhn':
		normalize = transforms.Normalize(mean=mean, std=std)
		transforms_train_list = [transforms.ToTensor(), normalize]
		if cfg.aug:
			transforms_train_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + transforms_train_list
		transform_train = transforms.Compose(transforms_train_list)
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			normalize,
		])

	if cfg.name != "svhn":
		train_dataset = dataset_gen(root, train=True, transform=transform_train, download=True)
		test_dataset = dataset_gen(root, train=False, transform=transform_test, download=True)
	else:
		train_dataset = dataset_gen(root, split='train', transform=transform_train, download=True)
		test_dataset = dataset_gen(root, split='test', transform=transform_test, download=True)

	train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size,
								  shuffle=True, num_workers=cfg.workers, pin_memory=True)
	test_dataloader = DataLoader(dataset=test_dataset, batch_size=100, 
								  shuffle=False, num_workers=cfg.test_workers, pin_memory=True)

	return {"train": train_dataloader, "test": test_dataloader}

def get_model(cfg, device):
	model_gen = models.__dict__[cfg.name]
	model = model_gen(**cfg.kwargs)

	return model.to(device)

def get_opt_materials(cfg, model):
	opt_gen = torch.optim.__dict__[cfg.name]
	opt = opt_gen(model.parameters(), **cfg.kwargs)
	
	if "schedule" in cfg.keys():
		schedule_gen = torch.optim.lr_scheduler.__dict__[cfg.schedule.name]
		schedule = schedule_gen(opt, **cfg.schedule.kwargs)
	else:
		schedule = None

	return opt, schedule

def get_lr_schedule(opt):
	pass

def get_opt(model, cfg):
	pass

def save_model(path, net, opt, schedule, epoch):
	checkpoint = {
		"net": net.state_dict(),
		'optimizer': opt.state_dict(),
		"epoch": epoch,
		'lr_schedule': schedule.state_dict()
	}
	torch.save(checkpoint, path)

def load_model(path, net, opt, schedule):
	checkpoint = torch.load(path)
	net.load_state_dict(checkpoint['net'])
	opt.load_state_dict(checkpoint['optimizer'])
	schedule.load_state_dict(checkpoint['lr_schedule'])

	start_epoch = checkpoint['epoch']
	return start_epoch