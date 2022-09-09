import argparse
import torch, re
from byol_pytorch import byolinstance
from torchvision import models
import global_cache as gc
import multiprocessing
from post_process import *
import os
from util import COCO2014, save_checkpoint, wbin_pkl, get_all_filename
import torchvision.transforms as transforms



def par_option():
	parser = argparse.ArgumentParser(description='WILDCAT Training')
	parser.add_argument('data', metavar='DIR',
						help='path to dataset (e.g. data/')
	parser.add_argument('--image-size', '-i', default=224, type=int,
						metavar='N', help='image size (default: 224)')
	parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
						help='number of data loading workers (default: 4)')
	parser.add_argument('--epochs', default=100, type=int, metavar='N',
						help='number of total epochs to run')
	parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
						help='number of epochs to change learning rate')
	parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
						help='manual epoch number (useful on restarts)')
	parser.add_argument('-b', '--batch-size', default=8, type=int,
						metavar='N', help='mini-batch size (default: 256)')
	parser.add_argument('--lr', '--learning-rate', default=3e-2, type=float,
						metavar='LR', help='initial learning rate')
	parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
						metavar='LR', help='learning rate for pre-trained layers')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
						help='momentum')
	parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
						metavar='W', help='weight decay (default: 1e-4)')
	parser.add_argument('--print-freq', '-p', default=0, type=int,
						metavar='N', help='print frequency (default: 10)')
	parser.add_argument('--resume', default='', type=str, metavar='PATH',
						help='path to latest checkpoint (default: none)')
	parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
						help='evaluate model on validation set')
	parser.add_argument('--test_set_amount', type=int, default=1)
	parser.add_argument('--query_code_amount', type=int, default=496)
	parser.add_argument('--testset_pkl_path', type=str, default='./data/coco/coco_test_set.pkl')
	parser.add_argument("--query_pkl_path", type=str, default='./data/coco/coco_query_set.pkl')
	parser.add_argument("--hashcode_pool", type=str,
						default='./data/coco/coco_hashcode_pool.pkl')
	parser.add_argument("--hashcode_pool_limit", type=int, default=500)
	
	parser.add_argument('--DROPOUT_RATIO', type=float, default=0.1)
	parser.add_argument('--CLASSIFIER_CHANNEL', type=str, default=2048)
	parser.add_argument('--IMAGE_CHANNEL', type=int,
						default=2048)
	parser.add_argument("--accumulate_steps", type=int, default=0)

	parser.add_argument("--HASH_TASK", action='store_true')
	parser.add_argument("--HASH_BIT", type=int, default=64)
	parser.add_argument("--backbone", type=str, default='resnet101')
	parser.add_argument("--hiddenlayer", type=str, default='avgpool')
	
	return parser

def expand_greyscale(t):
	return t.expand(3, -1, -1)


def get_average(list_1):
	assert len(list_1) != 0, 'input list is empty'
	s = 0
	for item in list_1:
		s += item
	return 1.0 * s / len(list_1)


def main_coco():
	global args, use_gpu
	parser = par_option()
	args = parser.parse_args()
	num_classes = 80
	use_gpu = torch.cuda.is_available()
	single = True
	device = torch.device('cuda' if use_gpu else "cpu")
	device_ids = [int(i) for i in range(torch.cuda.device_count())]
	state = {'batch_size': args.batch_size,
			 'image_size': args.image_size,
			 'max_epochs': args.epochs,
			 'evaluate': args.evaluate,
			 'resume': args.resume,
			 'num_classes': num_classes,
			 'hidden_layer': args.hiddenlayer,
			 'backbone': args.backbone,
			 'use_gpu': use_gpu,
			 'hashcode_pool': args.hashcode_pool, }
	state['save_model_path'] = 'checkpoint/coco/'
	state['workers'] = args.workers
	state['lr'] = args.lr
	print('*****************The config parameters:*******************')
	for k, v in state.items():
		print("{0} = {1}".format(k, v))
	print("\n*****************config parameters print end*******************\n")
	
	train_dataset = COCO2014(args.data, phase='train', )
	val_dataset = COCO2014(args.data, phase='val', )
	state['transform'] = transforms.Compose([
		transforms.Resize(args.image_size),
		transforms.CenterCrop(args.image_size),
		transforms.ToTensor(),
		transforms.Lambda(expand_greyscale)
	])
	train_dataset.transform = state['transform']
	val_dataset.transform = state['transform']
	train_loader = torch.utils.data.DataLoader(train_dataset,
											   batch_size=args.batch_size, shuffle=False,
											   num_workers=multiprocessing.cpu_count())
	val_loader = torch.utils.data.DataLoader(val_dataset,
											 batch_size=args.batch_size, shuffle=False,
											 num_workers=multiprocessing.cpu_count())
	
	learner = byolinstance(backbone=args.backbone,
						   image_size=args.image_size,
						   hidden_layer=args.hiddenlayer,
						   projection_size=args.HASH_BIT,
						   projection_hidden_size=4096,
						   moving_average_decay=0.99,
						   ).to(device)
	
	opt = torch.optim.Adam(learner.parameters(), lr=args.lr)
	
	state['start_time_str'] = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	state['temp_dir'] = state['hashcode_pool'][:-4] + "_hb" + str(args.HASH_BIT) + '_' + \
						state['start_time_str'] + '_temp.pkl'
	state['destination'] = state['hashcode_pool'][:-4] + "_hb" + str(args.HASH_BIT) + '_' + \
						   state['start_time_str'] + '.pkl'
	
	epoch_record = 0
	loss_score = 0
	for epoch in range(args.epochs):
		is_better = False
		train_loader = tqdm(train_loader, desc=str("(" + str(epoch) + ')Training'))
		lower_loss, val_loss, train_loss = 100000.0, [], []
		learner.train()
		gc.training_flag = True
		for i, (input, target) in enumerate(train_loader):
			imgs = input[0]
			print('imgs = ', imgs.shape)
			img_paths = input[1]
			img_gt = target
			img_gt[img_gt == -1] = 0
			feature_var = torch.autograd.Variable(imgs).float().cuda() if use_gpu else \
				torch.autograd.Variable(imgs).float()
			loss = learner(feature_var)
			print('now loss = ', loss)
			train_loss.append(float(loss.item()))
			opt.zero_grad()
			loss.mean().backward()
			opt.step()
			if use_gpu and not single:
				learner.module.update_moving_average()
			else:
				learner.update_moving_average()
		
		mean_train_loss = get_average(train_loss)
		print('Epoch({epoch}):\ntrain loss: {loss}'.format(epoch=epoch, loss=mean_train_loss), end=';\t')
		
		torch.cuda.empty_cache()  
		val_loader = tqdm(val_loader, desc='Test')
		learner.eval()
		gc.training_flag = False
		for i, (input, target) in enumerate(val_loader):
			imgs = input[0]
			img_paths = input[1]
			img_gt = target
			img_gt[img_gt == -1] = 0
			with torch.no_grad():
				feature_var = torch.autograd.Variable(imgs).float().cuda() if use_gpu else \
					torch.autograd.Variable(imgs).float()
			try:
				loss, hashcode = learner(feature_var)
			except RuntimeError as exception:
				if "out of memory" in str(exception):
					print("GPU insufficient！")
					if hasattr(torch.cuda, "empty_cache"):
						torch.cuda.empty_cache()
				else:
					raise exception
			print('val now loss :', loss.item(), )
			print('hashcode :\n', hashcode)
			val_loss.append(float(loss.item()))
			dic_temp = {"img_name": img_paths,
						"target": target.cpu(),
						'output': hashcode.cpu()}
			wbin_pkl(state['temp_dir'], dic_temp)
		torch.cuda.empty_cache()
		
		mean_val_loss = get_average(val_loss)
		if mean_val_loss <= lower_loss:
			lower_loss = mean_val_loss
			is_better = True
		print('validation loss: {loss}'.format(epoch=epoch, loss=mean_val_loss))
		
		if is_better:
			nt = datetime.datetime.now().strftime('%Y%m%d%H%M')
			print("^_^ find a better score, this will rename temp by destination({0}) ^_^".format(nt))
			if os.path.exists(state['destination']):
				os.remove(state['destination'])
			if os.path.exists(state['temp_dir']):
				os.rename(state['temp_dir'], state['destination'])
			state['convergence_point'] = epoch
		else:
			nt = datetime.datetime.now().strftime('%Y%m%d%H%M')
			str1 = str(state['start_time_str']) + '.pkl'
			filename_pool = get_all_filename(state['hashcode_pool'])
			for item in filename_pool:
				if re.search(str1, str(item)) != None:
					flag = True
					break
			else:
				flag = False
			if flag:
				print("is_best is False, so delete the temp file({0})\n".format(nt))
				if os.path.exists(state['temp_dir']):
					os.remove(state['temp_dir'])
			else:
				os.rename(state['temp_dir'], state['destination'])
		
		loss_score, epoch_record = mean_val_loss, epoch
		torch.cuda.empty_cache()  
	
	save_checkpoint(
		{
			'save_model_path': state['save_model_path'],
			'filename_previous_best': None,
			'epoch': epoch_record + 1,
			'state_dict': learner.state_dict(),
			'best_score': loss_score,
		}, False
	)
	
	new_state = {
		'num_classes': 20,
		'hashcode_pool': args.hashcode_pool,
		'query_pkl_path': './data/coco/coco_query_set.pkl',
		'start_time': state['start_time_str'],
		'hash_bit': args.HASH_BIT,
		'hashcode_pool_limit': args.hashcode_pool_limit,
		'use_gpu': False,
	}
	obj = PostPro(new_state)
	obj.select_img()
	obj.test_final()


if __name__ == "__main__":
	start_time = datetime.datetime.now()
	print("\nSTART TIME:", start_time.strftime('%Y-%m-%d %H:%M:%S'), "\n")
	main_coco()
	end_time = datetime.datetime.now()
	print("\nENE TIME:", end_time.strftime('%Y-%m-%d %H:%M:%S'))
	use_time = (end_time - start_time).seconds
	m, s = divmod(use_time, 60)
	h, m = divmod(m, 60)
	d, h = divmod(h, 24)
	d = (end_time - start_time).days
	print("[Elapse time]:%02d-days:%02d-hours:%02d-minutes:%02d-seconds\n" % (
		d, h, m, s))
