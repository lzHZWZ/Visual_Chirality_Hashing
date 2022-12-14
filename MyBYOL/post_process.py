import pickle, torch, os, sys, math, datetime
from tqdm import tqdm, tgrange
import numpy as np

REPEAT_CALC = False

class PostPro(object):
	def __init__(self, state={}):
		super(PostPro, self).__init__()
		self.state = state
		self.all_final_item = 0
		
		if self._state('use_gpu') is None:
			self.state['use_gpu'] = torch.cuda.is_available()

		if self._state('prefix_path') is None:
			tmp = self.state['hashcode_pool'].split('/')[:-1]
			self.state['prefix_path'] = '/'.join(tmp) + '/'

		if self._state('final_hashcode_pool') is None:
			self.state['final_hashcode_pool'] = self.state['prefix_path'] + \
			                                    "final_" + self.state['hashcode_pool'].split('/')[-1]
			
		if self._state('before_fc_destination') is None:
			self.state['before_fc_destination'] = ''
		
		if self._state('hashcode_pool_image_name_list') is None:
			self.state['hashcode_pool_image_name_list'] = []
		
		if self._state('final_hashcode_pool_image_name_list') is None:
			self.state['final_hashcode_pool_image_name_list'] = []
		
		if self._state('query_pool_image_name_list') is None:
			self.state['query_pool_image_name_list'] = []
		
		if self._state('threshold') is None:
			self.state['threshold'] = 0.09
		
		if self._state('redius') is None:
			self.state['redius'] = 2
		
		if self._state('hashcode_pool_limit') is None:
			self.state['hashcode_pool_limit'] = 500

		if self._state('query_pool_limit') is None:
			self.state['query_pool_limit'] = 100
		
		self.init_variables()
	
	def _state(self, name):
		if name in self.state:
			return self.state[name]
	
	@staticmethod
	def calc_mean_var(nlist):
		N = float(len(nlist))
		narray = np.array(nlist)
		sum1 = float(narray.sum())
		narray2 = narray * narray
		sum2 = float(narray2.sum())
		mean = sum1 / N
		var = sum2 / N - mean ** 2
		
		return mean, var
	
	def _deleterow(self, ts, idx):

		ts = ts[torch.arange(ts.size(0)) != idx]
		print(ts)
	
	def reset(self):
		if self.state['use_gpu']:
			self.output = torch.IntTensor(torch.IntStorage()).cuda()
			self.targets = torch.IntTensor(torch.IntStorage()).cuda()
		else:
			self.output = torch.IntTensor(torch.IntStorage())
			self.targets = torch.IntTensor(torch.IntStorage())
			
		if self.state['use_gpu']:
			self.bf_output = torch.FloatTensor(torch.FloatStorage()).cuda()
			self.bf_targets = torch.IntTensor(torch.IntStorage()).cuda()
		else:
			self.bf_output = torch.FloatTensor(torch.FloatStorage())
			self.bf_targets = torch.IntTensor(torch.IntStorage())
		self.all_item = 0
		self.all_final_item = 0
		self.mAP1_list = []
		self.mAP2_list = []
		self.P_list = []
		self.R_list = []
	
	def init_variables(self):
		self.state['query_pool_image_name_list'] = []
		self.state['final_hashcode_pool_image_name_list'] = []
		self.state['hashcode_pool_image_name_list'] = []
		if type(self.state['start_time'])==type('123'):
			self.state['hashcode_pool'] = os.path.splitext(self.state['hashcode_pool'])[0] + "_hb" + str(
				self.state['hash_bit']) + '_' + self.state['start_time'] + '.pkl'
			self.state['query_pool'] = os.path.splitext(self.state['query_pkl_path'])[0] + "_hb" + str(
				self.state['hash_bit']) + '_' + self.state['start_time'] + '.pkl'
		else:
			self.state['hashcode_pool'] = os.path.splitext(self.state['hashcode_pool'])[0] + "_hb" + str(
				self.state['hash_bit']) + '_' + self.state['start_time'].strftime('%Y%m%d%H%M%S') + '.pkl'
			self.state['query_pool'] = os.path.splitext(self.state['query_pkl_path'])[0] + "_hb" + str(
				self.state['hash_bit']) + '_' + self.state['start_time'].strftime('%Y%m%d%H%M%S') + '.pkl'
		self.state['final_hashcode_pool'] = self.state['prefix_path'] + \
		                                    "final_" + self.state['hashcode_pool'].split('/')[-1]
		self.state['before_fc_destination'] = self.state['hashcode_pool'][:-4] + 'before_fc.pkl'
	
	def _gpu_cpu(self, input):
		if torch.is_tensor(input):
			if self.state['use_gpu']:
				input = input.float()
				return input.cuda()
		return input.float()
	
	def _wbin_pkl(self, file_dir, content):
		with open(str(file_dir), 'ab') as fi:
			pickle.dump(content, fi)
	
	def stack_from_src(self, ):
		print('search the file named {0}\n'.format(self.state['hashcode_pool']))
		if os.path.exists(self.state['hashcode_pool']):
			print("The {0} file exists...\n".format(self.state['hashcode_pool']))
			u = 0
			with open(self.state['hashcode_pool'], 'rb') as f:
				while True:
					u += 1
					try:
						data = pickle.load(f)
						name = data['img_name']
						target_gt = data['target'].cpu().int()
						output = data['output'].cpu().int()
						self.state['hashcode_pool_image_name_list'].extend(name)
						if self.state['use_gpu']:
							self.output = torch.cat((self.output, output.reshape(output.size(0), -1).cuda()), 0)
							self.targets = torch.cat((self.targets, target_gt.reshape(target_gt.size(0), -1).cuda()), 0)
						else:
							self.output = torch.cat((self.output, output.reshape(output.size(0), -1)), 0)
							self.targets = torch.cat((self.targets, target_gt.reshape(target_gt.size(0), -1)), 0)
						self.all_item += len(name)
					except EOFError:
						break
						
		else:
			print("Cannot find the {0} file ! Please check...\n The process aborting... ... ! \n".
			      format(self.state['hashcode_pool']))
			sys.exit()
	
	def createSeveralMat(self, query_num):
		query_code = self.output[query_num, :]
		query_label = self.targets[query_num, :]
		
		tmp = (torch.mul(query_code, self.output) + 1) / 2.0
		dist_mat = self.state['hash_bit'] - torch.sum(tmp, 1)

		tmp = torch.matmul(self.targets.float(), torch.transpose(self.targets.float(), 0, 1))
		bak = tmp[query_num, :].reshape(1, -1)
		all_gt_match_num = torch.nonzero(bak).size(0)
		upper_tri = torch.triu(tmp, diagonal=1)
		upper_tri += torch.transpose(upper_tri, 0, 1)

		mAP = 0
		seq = 0
		shot = 0
		redius_pool = {}
		interset_pool = {}
		for i in range(dist_mat.size(0)):
			if i != query_num:
				if dist_mat[i] <= int(self.state['redius']):
					if int(dist_mat[i]) not in redius_pool:
						redius_pool[int(dist_mat[i])] = {}
					redius_pool[int(dist_mat[i])][int(i)] = int(upper_tri[query_num][i])
		all_keys = []
		for k, v in redius_pool.items():
			all_keys.append(k)
			v = sorted(v.items(), key=lambda item: item[1], reverse=True)
			redius_pool[k] = v
		all_keys.sort()
		
		num_within_redius = 0
		
		for ele in all_keys:
			vl = redius_pool[int(ele)]
			num_within_redius += len(vl)
			for i in range(len(vl)):
				seq += 1
				t_e = vl[i][1]
				if t_e:
					shot += 1
					mAP += shot / seq
		
		mAP1, mAP2 = mAP, mAP
		P = shot / num_within_redius if num_within_redius else None
		R = shot / all_gt_match_num if all_gt_match_num else None
		mAP1 = (mAP / shot) if shot else None
		mAP2 = (mAP / seq) if seq else None
		print("shot={0},all_gt_match_num={1}".format(shot, all_gt_match_num))
		print("mAP/shot vlaue = {0},\nmAP/seq value = {1},\nPrecision = {2},\nRecall = {3}". \
		      format(mAP1, mAP2, P, R))

		if mAP1:
			self.mAP1_list.append(mAP1)
		if mAP2:
			self.mAP2_list.append(mAP2)
			self.P_list.append(P)
			self.R_list.append(R)
		print()
		
		if mAP2 != None and mAP1 !=None :
			final_dic = {
				'gt': self.targets[query_num, :].char(),
				'out': self.output[query_num, :].char(),
			}
			self.state['final_hashcode_pool_image_name_list'].append(
				(self.state['hashcode_pool_image_name_list'][query_num], mAP1, mAP2, P, R, final_dic)
			)

	
	def create_final_pkl(self):
		print('create_final_pkl says:\n')
		
		self.state['final_hashcode_pool_image_name_list'].sort(key=lambda x: x[1], reverse=True)
		self.state['query_pool_image_name_list'].sort(key=lambda x: x[1], reverse=True)

		print("Selected final pool capacity={0}\n"\
		      .format(len(self.state['final_hashcode_pool_image_name_list'])))
		self.state['final_hashcode_pool_image_name_list'] = \
			self.state['final_hashcode_pool_image_name_list'][:self.state['hashcode_pool_limit']]
		self.state['query_pool_image_name_list'] = \
			self.state['query_pool_image_name_list'][:self.state['query_pool_limit']]
		print("Real final pool capacity={0}\n" \
		      .format(len(self.state['final_hashcode_pool_image_name_list'])))
		
		for i in range(len(self.state['final_hashcode_pool_image_name_list'])):
			content = self.state['final_hashcode_pool_image_name_list'][i]
			tmp_dic = {'id': content[0],
			           'target_01': content[-1]['gt'].cpu().char(),
			           'output': content[-1]['out'].cpu().char(),
			           }
			self._wbin_pkl(self.state['final_hashcode_pool'], tmp_dic)
		print("final_pool content output over...\n\n")
	
	def final_stack_tensor(self, ):
		print("final_stack_tensor says:\n")
		print("processing the file named {0}".format(self.state['final_hashcode_pool']))
		if os.path.exists(self.state['final_hashcode_pool']):
			u = 0
			with open(self.state['final_hashcode_pool'], 'rb') as f:
				while True:
					try:
						data = pickle.load(f)
						name = data['id']
						target_01 = data['target_01'].cpu().int()
						output = data['output'].cpu().int()
						if self.state['use_gpu']:
							self.output = torch.cat((self.output, output.reshape(1, -1).cuda()), 0)
							self.targets = torch.cat((self.targets, target_01.reshape(1, -1).cuda()), 0)
						else:
							self.output = torch.cat((self.output, output.reshape(1, -1)), 0)
							self.targets = torch.cat((self.targets, target_01.reshape(1, -1)), 0)
					except EOFError:
						break
					u += 1
			self.all_final_item = u
			print("final item count = ", self.all_final_item)
		else:
			print("cannot find the file named {0}\n Processing aborting... ...\n".
			      format(self.state['final_hashcode_pool']))
			sys.exit()

	def final_calc(self, query_num):
		query_code = self.output[query_num, :]
		query_label = self.targets[query_num, :]

		tmp = (torch.mul(query_code, self.output) + 1) / 2
		dist_mat = self.state['hash_bit'] - torch.sum(tmp, 1)

		tmp = torch.matmul(self.targets.float(), torch.transpose(self.targets.float(), 0, 1))
		bak = tmp[query_num, :].reshape(1, -1)
		all_gt_match_num = torch.nonzero(bak).size(0)
		upper_tri = torch.triu(tmp, diagonal=1)
		upper_tri += torch.transpose(upper_tri, 0, 1)
		
		mAP = 0
		seq = 0
		shot = 0
		redius_pool = {}
		interset_pool = {}
		for i in range(dist_mat.size(0)):
			if i != query_num:
				if dist_mat[i] <= int(self.state['redius']):
					if int(dist_mat[i]) not in redius_pool:
						redius_pool[int(dist_mat[i])] = {}
					redius_pool[int(dist_mat[i])][int(i)] = int(upper_tri[query_num][i])
		all_keys = []
		for k, v in redius_pool.items():
			all_keys.append(k)
			v = sorted(v.items(), key=lambda item: item[1], reverse=True)
			redius_pool[k] = v
		all_keys.sort()
		
		num_within_redius = 0
		
		for ele in all_keys:
			vl = redius_pool[int(ele)]
			num_within_redius += len(vl)
			for i in range(len(vl)):
				seq += 1
				t_e = vl[i][1]
				if t_e:
					shot += 1
					mAP += shot / seq

		P = shot / num_within_redius if num_within_redius else None
		R = shot / all_gt_match_num if all_gt_match_num else None
		mAP1 = (mAP / shot) if shot else None
		mAP2 = (mAP / seq) if seq else None
		print("shot={0},all_gt_match_num={1}".format(shot, all_gt_match_num))
		print("mAP/shot vlaue = {0},\nmAP/seq value = {1},\nPrecision = {2},\nRecall = {3}". \
		      format(mAP1, mAP2, P, R))
		if mAP1:
			self.mAP1_list.append(mAP1)
		if mAP2:
			self.mAP2_list.append(mAP2)
			self.P_list.append(P)
			self.R_list.append(R)
		print()
	
	def read_format(self):
		print('self.state["before_fc_destination"] = ',self.state['before_fc_destination'])
		if os.path.exists(self.state['before_fc_destination']):
			u = 0
			with open(self.state['before_fc_destination'], 'rb') as f:
				while True:
					u += 1
					try:
						data = pickle.load(f)
						name = data['img_name']
						target_gt = data['target'].cpu().int()
						output = data['output'].cpu().float()

						if self.state['use_gpu']:
							self.bf_output = torch.cat((self.bf_output, output.reshape(output.size(0), -1).cuda()), 0)
							self.bf_targets = torch.cat((self.bf_targets, target_gt.reshape(target_gt.size(0), -1).cuda()), 0)
						else:
							self.bf_output = torch.cat((self.bf_output, output.reshape(output.size(0), -1)), 0)
							self.bf_targets = torch.cat((self.bf_targets, target_gt.reshape(target_gt.size(0), -1)), 0)
						self.all_item += len(name)
					except EOFError:
						break
			print('self.bf_output shape={0}, \ncontent=\n{1}'.format(self.bf_output.shape, self.bf_output))
			print('self.bf_target shape={0}, \ncontent=\n{1}'.format(self.bf_targets.shape, self.bf_targets))
	
	def test_final(self):
		print('Start test final hash pool...')
		print()
		self.reset()
		self.final_stack_tensor()
		if self.all_final_item:
			iteration = tqdm(range(self.all_final_item), desc='FinalTest')
			for i in iteration:
				self.final_calc(i)
			print()
			print("amount:{0}\n".format(self.all_final_item))
			print("*" * 10, 'Final overall mean mAP1={0}'.format(self.calc_mean_var(self.mAP1_list)[0]), '*' * 10, '\n')
			print("*" * 10, 'Final overall mean mAP2={0}'.format(self.calc_mean_var(self.mAP2_list)[0]), '*' * 10, '\n')
			print("*" * 10, 'Final overall mean P={0}'.format(self.calc_mean_var(self.P_list)[0]), '*' * 10, '\n')
			print("*" * 10, 'Final overall mean R={0}'.format(self.calc_mean_var(self.R_list)[0]), '*' * 10, '\n')
		else:
			print('Final pool have no value')
	
	def select_img(self, ):
		print('Start select hash code...')
		print()
		self.reset()
		self.stack_from_src()
		if os.path.exists(self.state['final_hashcode_pool']):
			os.remove(self.state['final_hashcode_pool'])
		if os.path.exists(self.state['query_pool']):
			os.remove(self.state['query_pool'])
		iteration = tqdm(range(self.all_item), desc='OriginTest')
		for i in iteration:
			self.createSeveralMat(i)
		print()
		print("overall amount:{0}\n".format(self.all_item))
		print("*" * 10, 'Original overall mean mAP1={0}'.format(self.calc_mean_var(self.mAP1_list)[0]), '*' * 10, '\n')
		print("*" * 10, 'Original overall mean mAP2={0}'.format(self.calc_mean_var(self.mAP2_list)[0]), '*' * 10, '\n')
		print("*" * 10, 'Original overall mean P={0}'.format(self.calc_mean_var(self.P_list)[0]), '*' * 10, '\n')
		print("*" * 10, 'Original overall mean R={0}'.format(self.calc_mean_var(self.R_list)[0]), '*' * 10, '\n')
		self.create_final_pkl()
	
	@property
	def display(self, ):
		for k, v in self.state.items():
			print("{0}={1}".format(k, v))
		print("all_final_item={0}".format(self.all_final_item))
		print("all_item={0}".format(self.all_item))

	def read_bf_info(self):
		self.reset()
		self.read_format()

def rbin_pkl(file_dir):
	with open(str(file_dir), 'rb') as fi:
		content = pickle.load(fi, encoding='bytes')
	return content

if __name__ == "__main__":
	state_voc = {'num_classes': 20,
	             'hashcode_pool': './data/voc/voc_hashcode_pool.pkl',
	             'query_pkl_path': './data/voc/voc_query_set.pkl',
	             'start_time': str(201912232152),
	             'hash_bit': 64,
	             'hashcode_pool_limit': 3000,
	             }
	state_mirflickr25k = {'num_classes': 24,
	                      'hashcode_pool': './data/mirflickr25k/mirflickr25k_hashcode_pool.pkl',
	                      'query_pkl_path': './data/mirflickr25k/mirflickr25k_query_set.pkl',
	                      'start_time': str(201912251332),
	                      'hash_bit': 64,
	                      'hashcode_pool_limit': 6000,
	                      }
	state_voc1 = {'num_classes': 20,
	             'hashcode_pool': './data/voc/voc_hashcode_pool.pkl',
	             'query_pkl_path': './data/voc/voc_query_set.pkl',
	             'start_time': str(202001032023),
	             'hash_bit': 128,
	             'hashcode_pool_limit': 3000,
	             }
	state_coco1 = {'num_classes': 80,
	              'hashcode_pool': './data/coco/coco_hashcode_pool.pkl',
	              'query_pkl_path': './data/coco/coco_query_set.pkl',
	              'start_time': str(202001081640),
	              'hash_bit': 16,
	              'hashcode_pool_limit': 25000,
	              }
	state_voc2 = {'num_classes': 20,
	               'hashcode_pool': './voc_hashcode_pool.pkl',
	               'query_pkl_path': './data/voc/voc_query_set.pkl',
	               'start_time': str(20210616105243),
	               'hash_bit': 64,
	               'hashcode_pool_limit': 500,
				  'use_gpu':False,
	               }
	obj = PostPro(state_voc2)
	obj.select_img()
	obj.test_final()
