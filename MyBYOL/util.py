import math, os
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
import torch.nn.functional as F
import datetime
from functools import wraps
import shutil, pickle, sys
import torch.utils.data as data
import json, csv
from PIL import Image

DEBUG_UTIL = False
mirflickr25k_categories = ['animals', 'baby', 'bird', 'car','clouds','dog',
                     'female', 'flower', 'food','indoor','lake','male',
                     'night','people','plant_life','portrait','river','sea',
                     'sky','structures', 'sunset', 'transport','tree','water',
                     ]

nuswide_categories = ['airport','animal','beach','bear','birds','boats','book','bridge','buildings',
					'cars','castle','cat','cityscape','clouds','computer','coral','cow','dancing',
					'dog','earthquake','elk','fire','fish','flags','flowers','food','fox','frost',
					'garden','glacier','grass','harbor','horses','house','lake','leaf','map','military',
					'moon','mountain','nighttime','ocean','person','plane','plants','police','protest',
					'railroad','rainbow','reflection','road','rocks','running','sand','sign','sky',
					'snow','soccer','sports','statue','street','sun','sunset','surf','swimmers','tattoo',
					'temple','tiger','tower','town','toy','train','tree','valley','vehicle','water',
					'waterfall','wedding','whales','window','zebra',
                     ]

def read_object_labels_csv(file, header=True):
	images = []
	num_categories = 0
	print('[dataset] read', file)
	with open(file, 'r') as f:
		reader = csv.reader(f)
		rownum = 0
		for row in reader:
			if header and rownum == 0:
				header = row
			else:
				if num_categories == 0:
					num_categories = len(row) - 1
				name = row[0]
				if '/' in name:
					name = name.split('/')[-1]
				labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
				labels = torch.from_numpy(labels)
				item = (name, labels)
				images.append(item)
			rownum += 1
	return images

def write_object_labels_csv(file, labeled_data):
	print('[dataset] write file %s' % file)
	with open(file, 'w') as csvfile:
		fieldnames = ['name']
		fieldnames.extend(nuswide_categories)
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		writer.writeheader()
		for (name, labels) in labeled_data.items():
			example = {'name': name}
			for i in range(len(nuswide_categories)):
				example[fieldnames[i + 1]] = int(labels[i])
			writer.writerow(example)

	csvfile.close()

def read_image_label(file):
	print('[dataset] read ' + file)
	data = []
	with open(file, 'r') as f:
		for line in f:
			tmp = line[:-1]
			data.append(tmp[-1])
	return data

def read_image_name(file):
	print('[dataset] read the image name' + file)
	namelist = []
	imgpathlist = []
	name_imgpath_dic = {}
	with open(file, 'r') as f:
		for line in f:
			tmp = line[:-1].split('\\')
			imgpath = '/'.join(tmp)
			name = tmp[-1].split('.')[0]

			namelist.append(name)
			imgpathlist.append(imgpath)
			name_imgpath_dic[str(name)] = imgpath
	return name_imgpath_dic, namelist, imgpathlist

def read_object_labels(root, dataset, set):
	path_labels = os.path.join(root,'data', 'Groundtruth','TrainTestLabels')
	path_imglist = os.path.join(root, 'data','ImageList')
	labeled_data = dict()
	num_classes = len(nuswide_categories)

	set = 'T' + (str.lower(set))[1:]

	imglist_file = os.path.join(path_imglist, set+'Imagelist.txt')
	name_img_dic, imgnamelist, imgpathlist = read_image_name(imglist_file)

	for i in range(num_classes):
		data={}
		file = os.path.join(path_labels, 'Labels_' + nuswide_categories[i] + '_' + set + '.txt')
		data_x = read_image_label(file)

		for x in range(len(data_x)):
			data[str(imgpathlist[x])] = data_x[x]

		if i == 0:
			for (name, label) in data.items():
				labels = np.zeros(num_classes)
				labels[i] = label
				labeled_data[name] = labels
		else:
			for (name, label) in data.items():
				labeled_data[name][i] = label

	return labeled_data

class Warp(object):
	def __init__(self, size, interpolation=Image.BILINEAR):
		
		self.size = int(size)
		self.interpolation = interpolation
	
	def __call__(self, img):
		
		return img.resize((self.size, self.size), self.interpolation)
	
	def __str__(self):

		return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
																								interpolation=self.interpolation)


class MultiScaleCrop(object):
	
	
	def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
		
		self.scales = scales if scales is not None else [1, .875, .75, .66]
		self.max_distort = max_distort
		self.fix_crop = fix_crop
		self.more_fix_crop = more_fix_crop
		self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
		self.interpolation = Image.BILINEAR
	
	def __call__(self, img):
		
		im_size = img.size
		crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
		crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
		ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
		return ret_img_group
	
	def _sample_crop_size(self, im_size):
		image_w, image_h = im_size[0], im_size[1]
		
		base_size = min(image_w, image_h)
		crop_sizes = [int(base_size * x) for x in self.scales]
		crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
		crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]
		
		pairs = []
		for i, h in enumerate(crop_h):
			for j, w in enumerate(crop_w):
				if abs(i - j) <= self.max_distort:
					pairs.append((w, h))
		
		crop_pair = random.choice(pairs)
		if not self.fix_crop:
			w_offset = random.randint(0, image_w - crop_pair[0])
			h_offset = random.randint(0, image_h - crop_pair[1])
		else:
			w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])
		
		return crop_pair[0], crop_pair[1], w_offset, h_offset
	
	def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
		offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
		return random.choice(offsets)
	
	@staticmethod
	def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
		
		w_step = (image_w - crop_w) // 4
		h_step = (image_h - crop_h) // 4
		
		ret = list()
		ret.append((0, 0))
		ret.append((4 * w_step, 0))
		ret.append((0, 4 * h_step))
		ret.append((4 * w_step, 4 * h_step))
		ret.append((2 * w_step, 2 * h_step))
		
		if more_fix_crop:
			ret.append((0, 2 * h_step))
			ret.append((4 * w_step, 2 * h_step))
			ret.append((2 * w_step, 4 * h_step))
			ret.append((2 * w_step, 0 * h_step))
			
			ret.append((1 * w_step, 1 * h_step))
			ret.append((3 * w_step, 1 * h_step))
			ret.append((1 * w_step, 3 * h_step))
			ret.append((3 * w_step, 3 * h_step))
		
		return ret
	
	def __str__(self):
		return self.__class__.__name__


def download_url(url, destination=None, progress_bar=True):
	
	def my_hook(t):
		last_b = [0]
		
		def inner(b=1, bsize=1, tsize=None):
			if tsize is not None:
				t.total = tsize
			if b > 0:
				t.update((b - last_b[0]) * bsize)
			last_b[0] = b
		
		return inner
	
	if progress_bar:
		with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
			filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
	else:
		filename, _ = urlretrieve(url, filename=destination)


class AveragePrecisionMeter(object):
	
	def __init__(self, difficult_examples=False):
		super(AveragePrecisionMeter, self).__init__()
		self.reset()
		self.difficult_examples = difficult_examples
	
	def reset(self):
		self.scores = torch.FloatTensor(torch.FloatStorage())
		self.targets = torch.LongTensor(torch.LongStorage())
	
	def add(self, output, target):
		if not torch.is_tensor(output):
			output = torch.from_numpy(output)
		if not torch.is_tensor(target):
			target = torch.from_numpy(target)
		
		if output.dim() == 1:
			output = output.view(-1, 1)
		else:
			assert output.dim() == 2, \
				'wrong output size (should be 1D or 2D with one column \
				per class)'
		if target.dim() == 1:
			target = target.view(-1, 1)
		else:
			assert target.dim() == 2, \
				'wrong target size (should be 1D or 2D with one column \
				per class)'
		if self.scores.numel() > 0:
			assert target.size(1) == self.targets.size(1), \
				'dimensions for output should match previously added examples.'
		
		if self.scores.storage().size() < self.scores.numel() + output.numel():
			new_size = math.ceil(self.scores.storage().size() * 1.5)
			self.scores.storage().resize_(int(new_size + output.numel()))
			self.targets.storage().resize_(int(new_size + output.numel()))
		
		offset = self.scores.size(0) if self.scores.dim() > 0 else 0
		self.scores.resize_(offset + output.size(0), output.size(1))
		self.targets.resize_(offset + target.size(0), target.size(1))
		self.scores.narrow(0, offset, output.size(0)).copy_(output)
		self.targets.narrow(0, offset, target.size(0)).copy_(target)
	
	def value(self):
		if self.scores.numel() == 0:
			return 0
		ap = torch.zeros(self.scores.size(1))
		rg = torch.arange(1, self.scores.size(0)).float()
		for k in range(self.scores.size(1)):  

			scores = self.scores[:, k]  
			targets = self.targets[:, k]  
			ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
		return ap
	
	@staticmethod
	def average_precision(output, target, difficult_examples=True):
		sorted, indices = torch.sort(output, dim=0, descending=True)
		pos_count = 0.
		total_count = 0.
		precision_at_i = 0.

		for i in indices:
			label = target[i]
			if difficult_examples and label == 0:
				continue
			if label == 1:
				pos_count += 1
			total_count += 1
			if label == 1:
				precision_at_i += pos_count / total_count
		precision_at_i /= pos_count
		return precision_at_i
	
	def overall(self):
		if self.scores.numel() == 0:
			return 0
		scores = self.scores.cpu().numpy()
		targets = self.targets.cpu().numpy()
		targets[targets == -1] = 0
		return self.evaluation(scores, targets)
	
	def overall_topk(self, k):
		targets = self.targets.cpu().numpy()
		targets[targets == -1] = 0
		n, c = self.scores.size()
		scores = np.zeros((n, c)) - 1
		index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
		tmp = self.scores.cpu().numpy()
		for i in range(n):
			for ind in index[i]:
				scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
		return self.evaluation(scores, targets)
	
	def evaluation(self, scores_, targets_):
		
		n, n_class = scores_.shape
		Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
		for k in range(n_class):
			scores = scores_[:, k]
			targets = targets_[:, k]
			targets[targets == -1] = 0
			Ng[k] = np.sum(targets == 1)
			Np[k] = np.sum(scores >= 0)
			Nc[k] = np.sum(targets * (scores >= 0))
		Np[Np == 0] = 1
		OP = np.sum(Nc) / np.sum(Np)
		OR = np.sum(Nc) / np.sum(Ng)
		OF1 = (2 * OP * OR) / (OP + OR)
		
		CP = np.sum(Nc / Np) / n_class
		CR = np.sum(Nc / Ng) / n_class
		CF1 = (2 * CP * CR) / (CP + CR)
		return OP, OR, OF1, CP, CR, CF1

	
	
def timecount(fn):
	@wraps(fn)
	def wrapper(*args, **kwargs):
		start_time = datetime.datetime.now()
		print("\nSTART TIME:", start_time.strftime('%Y-%m-%d %H:%M:%S'), "\n")  
		fn(*args, **kwargs)
		end_time = datetime.datetime.now()
		print("\nENE TIME:", end_time.strftime('%Y-%m-%d %H:%M:%S'))  
		use_time = (end_time - start_time).seconds
		m, s = divmod(use_time, 60)
		h, m = divmod(m, 60)
		d, h = divmod(h, 24)
		d = (end_time - start_time).days
		print("[Elapse time]:%02d-days:%02d-hours:%02d-minutes:%02d-seconds\n" % (
			d, h, m, s))  
	return wrapper
	
	
def save_checkpoint(state, is_best, filename='hash_checkpoint.pth.tar'):
		filename_ = filename
		if state['save_model_path'] is not None:
			filename = os.path.join(state['save_model_path'], filename_)
			if not os.path.exists(state['save_model_path']):
				os.makedirs(state['save_model_path'])
		print('save model {filename}\n'.format(filename=filename))
		torch.save(state, filename)
		if is_best:
			split_filename_ = filename_.split('.')
			filename_best = split_filename_[0] + '_best.' + split_filename_[1] + '.' + split_filename_[2]
			if state['save_model_path'] is not None:
				filename_best = os.path.join(state['save_model_path'], filename_best)
			shutil.copyfile(filename, filename_best)
			if state['save_model_path'] is not None:
				if state['filename_previous_best'] is not None:
					os.remove(state['filename_previous_best'])
				filename_best = os.path.join(state['save_model_path'],
				                             split_filename_[0] + 'best_{score:.4f}_{time}.pth.tar'. \
				                             format(score=state['best_score'],
				                                    time=datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
				shutil.copyfile(filename, filename_best)
				state['filename_previous_best'] = filename_best
				
				
def wbin_pkl( file_dir, content):
	with open(str(file_dir), 'ab') as fi:
		pickle.dump(content, fi)
		
def get_all_filename(hash_pool):
	tmp = hash_pool.split('/')[:-1]
	prefix = '/'.join(tmp) + '/'
	pool = []
	print(os.walk(prefix))
	for root, dirs, files in os.walk(prefix):
		pool = files
		break
	return pool


class COCO2014(data.Dataset):
	def __init__(self, root, transform=None, phase='train',
				 inp_name=None):  
		print("load {0} file\n".format(inp_name))
		self.root = root
		self.phase = phase
		self.img_list = []
		self.transform = transform
		self.get_anno()
		self.num_classes = len(self.cat2idx)
		
		if inp_name:
			with open(inp_name, 'rb') as f:
				self.inp = pickle.load(f)  
		self.inp_name = inp_name
	
	def get_anno(self):
		list_path = os.path.join(self.root, 'data', '{}_anno.json'.format(self.phase))
		self.img_list = json.load(open(list_path, 'r'))
		self.cat2idx = json.load(open(os.path.join(self.root, 'data', 'category.json'), 'r'))
	
	def __len__(self):
		return len(self.img_list)
	
	def __getitem__(self, index):
		item = self.img_list[index]
		return self.get(item)
	
	def get(self, item):
		filename = item['file_name']
		labels = sorted(item['labels'])
		img = Image.open(os.path.join(self.root, 'data', '{}2014'.format(self.phase), filename)).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		target = np.zeros(self.num_classes, np.float32) - 1
		target[labels] = 1
		if self.inp_name:
			return (img, filename, self.inp), target
		else:
			return (img, filename,), target


class MirFlickr25kPreProcessing(data.Dataset):
	def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
		print("load {0} file\n".format(inp_name))
		self.root = root
		self.set = set  
		self.transform = transform
		self.target_transform = target_transform
		
		path_csv = os.path.join(self.root, 'csv_files')  
		file_csv = os.path.join(path_csv, 'mirflickr25k_' + set + '.csv')
		if not os.path.exists(file_csv):
			if not os.path.exists(path_csv): 
				os.makedirs(path_csv)
		
		self.classes = mirflickr25k_categories
		self.images = read_object_labels_csv(file_csv)
		
		if inp_name:
			with open(inp_name, 'rb') as f:
				self.inp = pickle.load(f)
		self.inp_name = inp_name
		
		print('[dataset] MirFlickr25k classification set=%s number of classes=%d  number of images=%d' % (
			set, len(self.classes), len(self.images)))
	
	def __getitem__(self, index):
		
		path, target = self.images[index]
		img = Image.open(os.path.join(self.path_images, 'im' + path + '.jpg')).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		if self.inp_name:
			return (img, path, self.inp), target
		else:
			return (img, path,) , target
	
	def __len__(self):
		return len(self.images)
	
	def get_number_classes(self):
		return len(self.classes)

class NuswideClassification(data.Dataset):
	def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
		print("load {0} file\n".format(inp_name))
		self.root = root
		self.path_imglist = os.path.join(root, 'data','ImageList')
		self.path_traintestlabellist = os.path.join(root,'data','Groundtruth','TrainTestLabels')
		self.path_images = os.path.join(root, 'data', 'NUSWIDE', 'Flickr')
		self.set = set
		self.transform = transform
		self.target_transform = target_transform

		path_csv = os.path.join(self.root,'data', 'files')  
		file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')  

		if not os.path.exists(file_csv):
			if not os.path.exists(path_csv):  
				os.makedirs(path_csv)
			labeled_data = read_object_labels(self.root, '', self.set)
			write_object_labels_csv(file_csv, labeled_data)

		self.classes = nuswide_categories
		self.images = read_object_labels_csv(file_csv)

		if inp_name:
			with open(inp_name, 'rb') as f:
				self.inp = pickle.load(f)
		self.inp_name = inp_name

		print('[dataset] NUSWIDE classification set=%s number of classes=%d  number of images=%d' % (
			set, len(self.classes), len(self.images)))

	def __getitem__(self, index):

		path, target = self.images[index]
		print('path = ', path)
		img = Image.open(os.path.join(self.path_images, path)).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		if self.inp_name:
			return (img, path, self.inp), target
		else:
			return (img, path,), target
			

	def __len__(self):
	
		return len(self.images)

	def get_number_classes(self):
		return len(self.classes)