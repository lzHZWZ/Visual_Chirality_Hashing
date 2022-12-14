import csv, sys
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import util
from util import *

import torchvision.transforms as transforms

DEBUG_VOCPY = False

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
					 'bottle', 'bus', 'car', 'cat', 'chair',
					 'cow', 'diningtable', 'dog', 'horse',
					 'motorbike', 'person', 'pottedplant',
					 'sheep', 'sofa', 'train', 'tvmonitor']

urls = {
	'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
	'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
	'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
	'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}


def read_image_label(file):
	print('[dataset] read ' + file)
	data = dict()
	with open(file, 'r') as f:
		for line in f:
			tmp = line.split(' ')
			name = tmp[0]
			label = int(tmp[-1])
			data[name] = label
	return data


def read_object_labels(root, dataset, set):
	path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')  
	labeled_data = dict()
	num_classes = len(object_categories)
	
	for i in range(num_classes):
		file = os.path.join(path_labels, object_categories[i] + '_' + set + '.txt')
		data = read_image_label(file)
		
		if i == 0:
			for (name, label) in data.items():
				labels = np.zeros(num_classes)
				labels[i] = label
				labeled_data[name] = labels
		else:
			for (name, label) in data.items():
				labeled_data[name][i] = label
	
	return labeled_data


def write_object_labels_csv(file, labeled_data):
	print('[dataset] write file %s' % file)
	with open(file, 'w') as csvfile:
		fieldnames = ['name']
		fieldnames.extend(object_categories)
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		
		writer.writeheader()
		for (name, labels) in labeled_data.items():
			example = {'name': name}
			for i in range(len(object_categories)):
				example[fieldnames[i + 1]] = int(labels[i])
			writer.writerow(example)
	
	csvfile.close()


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
				labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
				labels = torch.from_numpy(labels)
				item = (name, labels)
				images.append(item)
			rownum += 1
	return images


def find_images_classification(root, dataset, set):
	path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
	images = []
	file = os.path.join(path_labels, set + '.txt')
	with open(file, 'r') as f:
		for line in f:
			images.append(line)
	return images


def download_voc2007(root):
	path_devkit = os.path.join(root, 'VOCdevkit')
	path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
	tmpdir = os.path.join(root, 'tmp')
	
	if not os.path.exists(root):
		os.makedirs(root)
	
	if not os.path.exists(path_devkit):
		
		if not os.path.exists(tmpdir):
			os.makedirs(tmpdir)
		
		parts = urlparse(urls['devkit'])
		filename = os.path.basename(parts.path)
		cached_file = os.path.join(tmpdir, filename)
		
		if not os.path.exists(cached_file):
			print('Downloading: "{}" to {}\n'.format(urls['devkit'], cached_file))
			util.download_url(urls['devkit'], cached_file)
		
		print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
		cwd = os.getcwd()
		tar = tarfile.open(cached_file, "r")
		os.chdir(root)
		tar.extractall()
		tar.close()
		os.chdir(cwd)
		print('[dataset] Done!')
	
	if not os.path.exists(path_images):
		
		parts = urlparse(urls['trainval_2007'])
		filename = os.path.basename(parts.path)
		cached_file = os.path.join(tmpdir, filename)
		
		if not os.path.exists(cached_file):
			print('Downloading: "{}" to {}\n'.format(urls['trainval_2007'], cached_file))
			util.download_url(urls['trainval_2007'], cached_file)
		
		print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
		cwd = os.getcwd()
		tar = tarfile.open(cached_file, "r")
		os.chdir(root)
		tar.extractall()
		tar.close()
		os.chdir(cwd)
		print('[dataset] Done!')
	
	test_anno = os.path.join(path_devkit, 'VOC2007/ImageSets/Main/aeroplane_test.txt')
	if not os.path.exists(test_anno):
		
		parts = urlparse(urls['test_images_2007'])
		filename = os.path.basename(parts.path)
		cached_file = os.path.join(tmpdir, filename)
		
		if not os.path.exists(cached_file):
			print('Downloading: "{}" to {}\n'.format(urls['test_images_2007'], cached_file))
			util.download_url(urls['test_images_2007'], cached_file)
		
		print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
		cwd = os.getcwd()
		tar = tarfile.open(cached_file, "r")
		os.chdir(root)
		tar.extractall()
		tar.close()
		os.chdir(cwd)
		print('[dataset] Done!')
	
	test_image = os.path.join(path_devkit, 'VOC2007/JPEGImages/000001.jpg')
	if not os.path.exists(test_image):
		
		parts = urlparse(urls['test_anno_2007'])
		filename = os.path.basename(parts.path)
		cached_file = os.path.join(tmpdir, filename)
		
		if not os.path.exists(cached_file):
			print('Downloading: "{}" to {}\n'.format(urls['test_anno_2007'], cached_file))
			util.download_url(urls['test_anno_2007'], cached_file)
		
		print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
		cwd = os.getcwd()
		tar = tarfile.open(cached_file, "r")
		os.chdir(root)
		tar.extractall()
		tar.close()
		os.chdir(cwd)
		print('[dataset] Done!')


class Voc2007Classification(data.Dataset):
	def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
		self.root = root
		self.path_devkit = os.path.join(root, 'VOCdevkit')
		self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
		self.set = set
		self.transform = transform
		self.target_transform = target_transform
		path_csv = os.path.join(self.root, 'files', 'VOC2007')  
		file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')  
		
		if not os.path.exists(file_csv):
			if not os.path.exists(path_csv):  
				os.makedirs(path_csv)
			labeled_data = read_object_labels(self.root, 'VOC2007', self.set)
			write_object_labels_csv(file_csv, labeled_data)
		
		self.classes = object_categories
		self.images = read_object_labels_csv(file_csv)
		self.inp, self.inp_name = None, None
		if inp_name is not None:
			with open(inp_name, 'rb') as f:
				self.inp = pickle.load(f)
			self.inp_name = inp_name
		
		print('[dataset] VOC 2007 classification set=%s number of classes=%d  number of images=%d' % (
			set, len(self.classes), len(self.images)))
	
	def __getitem__(self, index):
		
		path, target = self.images[index]
		path = path.zfill(6)
		img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		
		if self.inp:
			return (img, path, self.inp), target
		else:
			return (img, path), target
	
	def __len__(self):
		return len(self.images)
	
	def get_number_classes(self):
		return len(self.classes)


if __name__ == "__main__":
	state = {}
	state['image_size'] = 448
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
	train_dataset = Voc2007Classification('../Amatrix_ML/data/voc', 'trainval',)
	state['train_transform'] = transforms.Compose([
		MultiScaleCrop(state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
	])
	train_dataset.transform = state['train_transform']
	train_loader = torch.utils.data.DataLoader(train_dataset,
											   batch_size=1, shuffle=False,
											   num_workers=4)
	for i, (input, target) in enumerate(train_loader):
		print("img = ", input[0], input[0].shape)
		print('img_name = ', input[1][0])
		print("target = ", target)
		sys.exit()
