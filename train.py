import os
import sys

import torch
import torch.nn as nn
import torch.optim
from torchvision.utils import make_grid as mg
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter
from utils import *

import time
import datetime

out_file = None

def my_print(*args):
	print(*args)
	out_file.flush()
	if out_file is not None:
		print(*args, file=out_file)

def get_loss(model, images, labels, device, cfg):
	# mnist and cifar10
	if cfg.loss_type == 'basic':
		outputs, losses = model.get_loss_baseline(images, labels)
	elif cfg.loss_type == 'mic':
		outputs, losses = model.get_loss_baseline(images, labels, inforeg='mic', device=device)

	elif cfg.loss_type == 'LS_basic':
		outputs, losses = model.get_loss_LS(images, labels, beta=cfg.LS_beta)
	elif cfg.loss_type == 'LS_mic':
		outputs, losses = model.get_loss_LS(images, labels, inforeg='mic', device=device, beta=cfg.LS_beta)

	elif cfg.loss_type == 'CP_basic':
		outputs, losses = model.get_loss_CP(images, labels, beta=cfg.CP_beta)
	elif cfg.loss_type == 'CP_mic':
		outputs, losses = model.get_loss_CP(images, labels, inforeg='mic', device=device, beta=cfg.CP_beta)

	# elif cfg.loss_type == 'dropout_basic':
	# 	outputs, losses = model.get_loss_dropout(images, labels)
	# mnist
	if cfg.loss_type == 'inforeg':
		outputs, losses = model.get_loss_baseline(images, labels, inforeg='eic', device=device)

	elif cfg.loss_type == 'dropout':
		outputs, losses = model.get_loss_dropout(images, labels, device=device)
	elif cfg.loss_type == 'dropout_inforeg':
		outputs, losses = model.get_loss_dropout(images, labels, inforeg='eic', device=device)
	elif cfg.loss_type == 'dropout_mic':
		outputs, losses = model.get_loss_dropout(images, labels, inforeg='mic', device=device)

	elif cfg.loss_type == 'LS_inforeg':
		outputs, losses = model.get_loss_LS(images, labels, inforeg='eic', device=device, beta=cfg.LS_beta)

	elif cfg.loss_type == 'CP_inforeg':
		outputs, losses = model.get_loss_CP(images, labels, inforeg='eic', device=device, beta=cfg.CP_beta)

	elif cfg.loss_type == 'VIB_basic':
		outputs, losses = model.get_loss_VIB(images, labels, beta=cfg.VIB_beta)
	elif cfg.loss_type == 'VIB_inforeg':
		outputs, losses = model.get_loss_VIB(images, labels, inforeg='eic', device=device, beta=cfg.VIB_beta)
	elif cfg.loss_type == 'VIB_mic':
		outputs, losses = model.get_loss_VIB(images, labels, inforeg='mic', device=device, beta=cfg.VIB_beta)

	# cifar10
	elif cfg.loss_type == 'inforeg_block':
		outputs, losses = model.get_loss_baseline(images, labels, inforeg="block", device=device)
	elif cfg.loss_type == 'inforeg_layer':
		outputs, losses = model.get_loss_baseline(images, labels, inforeg="layer", device=device)
	elif cfg.loss_type == "LS_inforeg_block":
		outputs, losses = model.get_loss_LS(images, labels, inforeg="block", device=device, beta=cfg.LS_beta)
	elif cfg.loss_type == "CP_inforeg_block":
		outputs, losses = model.get_loss_CP(images, labels, inforeg="block", device=device, beta=cfg.CP_beta)

	return outputs, losses

def train(model, data_loader, device, cfg):
	my_print("start training!!!")
	my_print(model)

	if not os.path.exists(cfg.save_path):
		os.mkdir(cfg.save_path)

	opt, schedule = get_opt_materials(cfg.opt, model)
	exp_info = cfg.exp_info
	writer = SummaryWriter(f"log/{exp_info}")

	best_acc = 0
	loss_f = nn.CrossEntropyLoss()
	
	start_time = time.time()

	start_epoch = -1
	if hasattr(cfg, "resume"):
		start_epoch = load_model(cfg.resume, model, opt, schedule)

	for epoch in range(start_epoch + 1, cfg.epoch):
		train_epoch(model, data_loader["train"], opt, writer, epoch, device, cfg)
		val_result = validate(model, data_loader["test"], loss_f, writer, epoch, device, cfg)
		pre = val_result[0]
		if pre > best_acc:
			best_acc = pre
			if cfg.save_tag:
				# torch.save(model.state_dict(), f'./{cfg.save_path}/{exp_info}.pkl')
				save_model(f'./{cfg.save_path}/{exp_info}.pkl', model, opt, schedule, epoch)
			writer.add_scalar(f"best_acc", best_acc, global_step=0)
		if cfg.save_tag and epoch > 0 and epoch % cfg.save_epoch == 0:
			# torch.save(model.state_dict(), f'./{cfg.save_path}/{exp_info}_epoch{epoch}.pkl')
			save_model(f'./{cfg.save_path}/{exp_info}_epoch{epoch}.pkl', model, opt, schedule, epoch)

		if schedule is not None:
			schedule.step()

		my_print(f"lr: {opt.state_dict()['param_groups'][0]['lr']}, acc: {pre}, best_acc: {best_acc}," \
		+ f"ETA: {datetime.timedelta(seconds=time.time() - start_time)}/" \
		+ f"{datetime.timedelta(seconds=(cfg.epoch / (epoch + 1)) * (time.time() - start_time))}")

		if hasattr(cfg, 'analyze'):
			my_print(f"model weight: {val_result[2]}, activation val: {val_result[1]}")

	writer.close()

def train_epoch(model, train_loader, opt, writer, epoch, device, cfg):
	tot_loss_ave = AverageMeter()
	bp_loss_ave = AverageMeter()
	prec1_ave = AverageMeter()
	batch_time = AverageMeter()

	model.train()
	end_t = time.time()
	for step, (images, labels) in enumerate(train_loader):
		opt.zero_grad()
		images, labels = images.to(device), labels.to(device)
		if isinstance(model, models.__dict__["MLP"]):
			images = images.reshape(-1, images.shape[1] * images.shape[2] * images.shape[3])

		outputs, losses = get_loss(model, images, labels, device, cfg)

		tot_loss = losses["total_loss"]
		prec1_tmp = int((torch.argmax(outputs, 1) == labels.flatten()).int().sum()) / images.size(0)
		prec1_ave.update(prec1_tmp, images.size(0))
		tot_loss_ave.update(float(tot_loss), images.size(0))
		bp_loss_ave.update(float(losses["bp_loss"]), images.size(0))

		tot_loss.backward()
		opt.step()

		batch_time.update(time.time() - end_t)
		end_t = time.time()

		if step % cfg.log_step == 0:
			my_print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Tot_Loss {tot_loss.val:.4f} ({tot_loss.avg:.4f})\t'
				  'Bp_Loss {bp_loss.val:.4f} ({bp_loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					  epoch, step, len(train_loader), batch_time=batch_time,
					  tot_loss=tot_loss_ave, bp_loss=bp_loss_ave, top1=prec1_ave))

	writer.add_scalar(f"loss/train_total_loss", tot_loss_ave.avg, global_step=epoch)
	writer.add_scalar(f"loss/train_bp_loss", bp_loss_ave.avg, global_step=epoch)
	writer.add_scalar(f"acc/train_acc", prec1_ave.avg, global_step=epoch)

def validate(model, test_loader, criterion, writer, epoch, device, cfg):
	loss_ave = AverageMeter()
	prec1_ave = AverageMeter()
	batch_time = AverageMeter()

	# analysis
	if hasattr(cfg, 'analyze'):
		act_L1 = AverageMeter()
		model_w_L1 = model.get_model_L1()

	model.eval()
	end_t = time.time()
	for step, (images, labels) in enumerate(test_loader):
		images, labels = images.to(device), labels.to(device)
		if isinstance(model, models.__dict__["MLP"]):
			images = images.reshape(-1, images.shape[1] * images.shape[2] * images.shape[3])

		outputs = model(images)
		loss = criterion(outputs, labels)

		prec1_tmp = int((torch.argmax(outputs, 1) == labels.flatten()).int().sum()) / images.size(0)
		prec1_ave.update(prec1_tmp, images.size(0))
		loss_ave.update(float(loss), images.size(0))

		batch_time.update(time.time() - end_t)
		end_t = time.time()
		if step % cfg.log_step == 0:
			my_print('Test: [{0}/{1}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					  step, len(test_loader), batch_time=batch_time, loss=loss_ave,
					  top1=prec1_ave))

		if hasattr(cfg, 'analyze'):
			act_L1_tmp = model.get_act_L1(images, cfg.analyze)
			act_L1.update(act_L1_tmp, images.size(0))

	writer.add_scalar(f"loss/test_loss", loss_ave.avg, global_step=epoch)
	writer.add_scalar(f"acc/test_acc", prec1_ave.avg, global_step=epoch)

	if hasattr(cfg, 'analyze'):
		return [prec1_ave.avg, act_L1.avg, model_w_L1]
	return [prec1_ave.avg]

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser("regularization_args")
	parser.add_argument("--config", type=str, default="./experiments/mnist_baseline.yaml")
	parser.add_argument('--seed', type=int)
	parser.add_argument('--gpu', type=int)

	args = parser.parse_args()
	
	with open(args.config) as f:
		config = yaml.load(f)
	cfg = EasyDict(config)

	if args.seed is not None:
		cfg.seed = args.seed
	if args.gpu is not None:
		cfg.gpu = args.gpu

	if hasattr(cfg, "out_file"):
		cfg.out_file = cfg.out_file + f"_seed{cfg.seed}"
		out_file = open(cfg.out_file, "w")
		print(out_file)

	cfg.train.save_path += f"_seed{cfg.seed}"
	cfg.train.exp_info += f"_seed{cfg.seed}"
	my_print(cfg)
	set_seed(cfg.seed)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if "gpu" in cfg.keys():
		device = torch.device(f"cuda:{cfg.gpu}")
	data_loader = get_dataloader(cfg.dataset)
	model = get_model(cfg.model, device)
	train(model, data_loader, device, cfg.train)

	if hasattr(cfg, "out_file"):
		out_file.close()