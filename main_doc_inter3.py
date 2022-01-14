'''
    This is an experiment which uses coarse-grained documents interaction([qs, ql])
'''
import random
import torch
import pickle
import numpy as np
from torch.functional import Tensor
from model.Model_doc_inter3 import Contextual
import Constants
import torch
import torch.nn as nn
from metric import AP
import os
from tqdm import tqdm
import argparse
from dataset import *
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default=1, type=int, help='0-check whether the code has bugs, 1-run the code on small dataset, 2-run the code on the whole dataset')
parser.add_argument('--cudaid', default=0, type=int, help='the id of the cuda')
parser.add_argument('--seed', default=100, type=int, help='')
parser.add_argument('--batch_size', default=64, type=int, help='')
parser.add_argument('--max_querylen', default=25, type=int, help='the maximum length of query')
parser.add_argument('--max_doclen', default=25, type=int, help='the maximum length of document')
parser.add_argument('--max_qdlen', default=50, type=int, help='the maximum length of the concat of query and document')
parser.add_argument('--max_hislen', default=50, type=int, help='the maximum length of the long history')
parser.add_argument('--max_sessionlen', default=20, type=int, help='the maximum length of the session')
parser.add_argument('--max_doc_list_train', default=5, type=int, help='the maximum size of the document set of train data') #change
parser.add_argument('--max_doc_list_test', default=50, type=int, help='the maximum size of the document set of test data') #change
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epochs', default=10, type=int, help='')	# change
parser.add_argument('--in_path', default='../../data/ValidQLSample_HF', type=str, help='the path of data') # change
parser.add_argument('--test_score_path', default='result/test_score.txt', type=str, help='the path of the test score')
parser.add_argument('--result_path', default='result/results.txt', type=str, help='the path of the results')
parser.add_argument('--log_path', default='./log/', type=str, help='The path to save log.')
parser.add_argument('--model_path', default='model/trained_model.pth', type=str, help='the path of the trained model')
args = parser.parse_args()

args.log_path += os.path.split(__file__)[-1].split(".")[0] + '.log'
logging.basicConfig(filename=args.log_path, level=logging.DEBUG, 
					format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')  
					# debug level - avoid printing log information during training
logger = logging.getLogger(__name__)
logger.info(args)


filenames = sorted(os.listdir(args.in_path))
if args.mode == 0:
	filenames = filenames[:200]
elif args.mode == 1:
	filenames = filenames[:30000]

vocab = pickle.load(open('../../data/vocab.dict', 'rb'))
mistake = pickle.load(open('../../data/mistake.dict', 'rb'))
#device_ids = [0, 1]
# args.batch_size = 64
# args.max_querylen = 20
# args.max_doclen = 30
# args.max_qdlen = 50
# args.max_hislen = 50
# args.max_sessionlen = 20
# args.max_doc_list = 50

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def sen2qid(sen):
	idx = []
	for word in sen.split():
		if word in vocab:
			idx.append(vocab[word])
		else:
			idx.append(vocab['<unk>'])
	idx = idx[:args.max_querylen]
	padding = [0] * (args.max_querylen - len(idx))
	idx = idx + padding
	return	idx

def sen2did(sen):
	idx = []
	for word in sen.split():
		if word in vocab:
			idx.append(vocab[word])
		else:
			idx.append(vocab['<unk>'])
	idx = idx[:args.max_doclen]
	padding = [0] * (args.max_doclen - len(idx))
	idx = idx + padding
	return	idx

def sen2id(sen):
	idx = []
	for word in sen.split():
		if word in vocab:
			idx.append(vocab[word])
		else:
			idx.append(vocab['<unk>'])
	idx = idx[:args.max_qdlen]
	padding = [0] * (args.max_qdlen - len(idx))
	idx = idx + padding
	return idx

def divide_dataset(filename):  # 检查用户的实验数据中有几个session，滤掉少于一定session的用户
	session_sum = 0
	query_sum = 0
	last_queryid = 0
	last_sessionid = 0
	with open(os.path.join(args.in_path, filename)) as fhand:
		for line in fhand:
			try:
				line, features = line.strip().split('###')
			except:
				line = line.strip()
			user, sessionid, querytime, query, url, title, sat, urlrank = line.strip().split('\t') 
			queryid = sessionid + querytime + query

			if querytime < '2006-04-03 00:00:00':
				if queryid != last_queryid:
					last_queryid = queryid
				query_sum += 1
			elif querytime < '2006-05-16 00:00:00':
				if query_sum < 2:
					return False
				if sessionid != last_sessionid:
					session_sum += 1
					assert(last_queryid != queryid)
					last_sessionid = sessionid
				if queryid != last_queryid:
					last_queryid = queryid
			else:
				if sessionid != last_sessionid:  # 这里不区分valid 和 test
					session_sum += 1
					assert(last_queryid != queryid)
					last_sessionid = sessionid
				if queryid != last_queryid:
					last_queryid = queryid

		if session_sum < 3:
			return False
	return True

def cal_delta(targets):
	n_targets = len(targets)
	deltas = np.zeros((n_targets, n_targets))
	total_num_rel = 0
	total_metric = 0.0
	for i in range(n_targets):
		if targets[i] == '1':
			total_num_rel += 1
			total_metric += total_num_rel / (i + 1.0)
	metric = (total_metric / total_num_rel) if total_num_rel > 0 else 0.0
	num_rel_i = 0
	for i in range(n_targets): # start from here
		if targets[i] == '1':
			num_rel_i += 1
			num_rel_j = num_rel_i
			sub = num_rel_i / (i + 1.0)
			for j in range(i+1, n_targets):
				if targets[j] == '1':
					num_rel_j += 1
					sub += 1 / (j + 1.0)
				else:
					add = (num_rel_j / (j + 1.0))
					new_total_metric = total_metric + add - sub
					new_metric = new_total_metric / total_num_rel
					deltas[i, j] = new_metric - metric 
		else:
			num_rel_j = num_rel_i
			add = (num_rel_i + 1) / (i + 1.0)
			for j in range(i + 1, n_targets):
				if targets[j] == '1':
					sub = (num_rel_j + 1) / (j + 1.0)
					new_total_metric = total_metric + add - sub
					new_metric = new_total_metric / total_num_rel
					deltas[i, j] = new_metric - metric
					num_rel_j += 1
					add += 1 / (j + 1.0)
	return deltas

def prepare_train_data(sat_list, feature_list, doc_list, qids, long_qdids, short_qdids):
	if len(doc_list) > 5:
		return
	short_qdids_his = short_qdids[-args.max_sessionlen:]
	long_qdids_his = long_qdids[-args.max_hislen:]
	init_short_qdids = np.zeros((args.max_sessionlen,args.max_qdlen))
	init_long_qdids = np.zeros((args.max_hislen,args.max_qdlen))
	# ??? don't understand all the code below
	init_shortpos = np.zeros(args.max_sessionlen+1)
	init_shortpos[-1] = args.max_sessionlen+1
	init_longpos = np.zeros(args.max_hislen+1)
	init_longpos[-1] = args.max_hislen+1

	for i in range(args.max_sessionlen):
		init_short_qdids[i][0]=1
	for i in range(len(short_qdids_his)):
		init_short_qdids[i] = short_qdids_his[i]
		init_shortpos[i] = i+1
	for i in range(args.max_hislen):
		init_long_qdids[i][0]=1
	for i in range(len(long_qdids_his)):
		init_long_qdids[i] = long_qdids_his[i]
		init_longpos[i] = i+1
	## update
	doc_list = doc_list[:args.max_doc_list_train]
	init_docids = np.zeros((args.max_doc_list_train, args.max_doclen))
	init_docpos = np.zeros(args.max_doc_list_train + 2)
	init_docpos[-2] = args.max_doc_list_train + 1
	init_docpos[-1] = args.max_doc_list_train + 2

	for i in range(args.max_doc_list_train):
		init_docids[i][0]=1
	for i in range(len(doc_list)):
		init_docids[i] = doc_list[i]
		init_docpos[i] = i+1
	
	delta = cal_delta(sat_list)
	n_targets = len(sat_list)
	for i in range(n_targets):
		for j in range(i+1, n_targets):
			if delta[i, j]>0:
				rel_doc = doc_list[j]
				rel_features = feature_list[j]
				irr_doc = doc_list[i]
				irr_features = feature_list[i]
				lbd = delta[i, j]
			elif delta[i, j]<0:
				rel_doc = doc_list[i]
				rel_features = feature_list[i]
				irr_doc = doc_list[j]
				irr_features = feature_list[j]
				lbd = -delta[i, j]
			else:
				continue
			if True:
				short_qdids_train.append(init_short_qdids) #[args.max_sessionlen, args.max_qdlen]
				shortpos_train.append(init_shortpos)
				long_qdids_train.append(init_long_qdids)
				longpos_train.append(init_longpos)
				querys_train.append(qids)
				docs1_train.append(rel_doc)
				docs2_train.append(irr_doc)
				label_train.append(0)
				features1_train.append(rel_features)
				features2_train.append(irr_features)
				delta_train.append(lbd)

				## update
				docs_train.append(init_docids)
				docs_pos_train.append(init_docpos)
				if delta[i, j]>0:
					doc1_order_train.append(j)
					doc2_order_train.append(i)
				elif delta[i, j]<0:
					doc1_order_train.append(i)
					doc2_order_train.append(j)

def prepare_test_data(short_qdids, long_qdids, qids, feature_list, line_list, doc_list):
	if len(doc_list) != 50:
		return
	for docid in range(len(doc_list)):
		short_qdids_his = short_qdids[-args.max_sessionlen:]
		long_qdids_his = long_qdids[-args.max_hislen:]
		init_short_qdids = np.zeros((args.max_sessionlen,args.max_qdlen))
		init_long_qdids = np.zeros((args.max_hislen,args.max_qdlen))
		init_shortpos = np.zeros(args.max_sessionlen+1)
		init_shortpos[-1] = args.max_sessionlen+1
		init_longpos = np.zeros(args.max_hislen+1)
		init_longpos[-1] = args.max_hislen+1

		# init_sessionpos = np.zeros(args.max_sessionlen+1)
		# init_sessionpos[-1] = args.max_sessionlen+1
		for i in range(args.max_sessionlen):
			init_short_qdids[i][0]=1
		for i in range(len(short_qdids_his)):
			init_short_qdids[i] = short_qdids_his[i]
			init_shortpos[i] = i+1
		for i in range(args.max_hislen):
			init_long_qdids[i][0]=1
		for i in range(len(long_qdids_his)):
			init_long_qdids[i] = long_qdids_his[i]
			init_longpos[i] = i+1
		short_qdids_test.append(init_short_qdids)
		shortpos_test.append(init_shortpos)
		long_qdids_test.append(init_long_qdids)
		longpos_test.append(init_longpos)
		querys_test.append(qids)
		docs_test.append(doc_list[docid])
		features_test.append(feature_list[docid])
		lines_test.append(line_list[docid].strip('\n'))
		doc_order_test.append(docid)
		
		init_docids = np.zeros((args.max_doc_list_test, args.max_doclen))
		init_docpos = np.zeros(args.max_doc_list_test + 2)
		init_docpos[-2] = args.max_doc_list_test + 1
		init_docpos[-1] = args.max_doc_list_test + 2
		
		for i in range(args.max_doc_list_test):
			init_docids[i][0]=1
		for i in range(len(doc_list)):
			init_docids[i] = doc_list[i]
			init_docpos[i] = i+1
		docs_list_test.append(init_docids)
		docs_pos_test.append(init_docpos)

def predata():
	x_train = []
	filenum=0
	key=0
	filetqdm = tqdm(filenames, desc='loading data')
	for filename in filetqdm:
		if not divide_dataset(filename): # 判断该用户的log是否符合要求
			continue
		filenum += 1
		last_queryid = 0
		last_sessionid = 0
		last_qids = 0
		queryid = 0
		sessionid = 0
		key = 0
		satcount = 0
		intent = ''
		doc_list = []
		sat_list = []
		feature_list = []
		long_qdids = []
		short_qdids = []
		line_list = []
		last_querytime = ''

		fhand = open(os.path.join(args.in_path, filename))
		for line in fhand:
			try:
				line, features = line.strip().split('###') # 
				features = [float(item) for item in features.split('\t')]
				features = features[:14]+features[26:] # ???
				if np.isnan(np.array(features)).sum():
					continue
			except:
				line = line.strip()
				features = [0]*98
			user, sessionid, querytime, query, url, title, sat, urlrank = line.strip().split('\t')
			#ser, sessionid, querytime, query, url, title, sat = line.strip().split('\t')
			queryid = sessionid + querytime + query
			qids = sen2qid(query)
			dids = sen2did(title)			

			if queryid != last_queryid:
				if last_querytime >= '2006-04-03 00:00:00' and last_querytime <= '2006-05-16 00:00:00' and key == 1: # training data
					prepare_train_data(sat_list, feature_list, doc_list, last_qids, long_qdids, short_qdids)
					key = 0
				elif last_querytime > '2006-05-16 00:00:00':
					prepare_test_data(short_qdids, long_qdids, last_qids, feature_list, line_list, doc_list)
				if intent != '':	# ??? what about the last query
					qdids = sen2id(intent)
					short_qdids.append(qdids)
				if sessionid != last_sessionid:
					if len(short_qdids) != 0:
						long_qdids.extend(short_qdids)
					short_qdids = []
					last_sessionid = sessionid
				intent = query
				doc_list = []
				sat_list = []
				feature_list = []
				line_list = []
				last_queryid = queryid
				last_querytime = querytime
				last_qids = qids
			doc_list.append(dids)
			sat_list.append(sat)
			feature_list.append(features)
			line_list.append(line)
			if int(sat) == 1:
				intent += ' ' + title
				key = 1
		# put the last query in the file into the train/test dataset
		if last_querytime >= '2006-04-03 00:00:00' and last_querytime <= '2006-05-16 00:00:00' and key == 1: # training data
			prepare_train_data(sat_list, feature_list, doc_list, last_qids, long_qdids, short_qdids)
			key = 0
		elif last_querytime > '2006-05-16 00:00:00':
			prepare_test_data(short_qdids, long_qdids, last_qids, feature_list, line_list, doc_list)
			
def train(train_loader):
	tqdm_train = tqdm(train_loader, desc='Training (epoch #{})'.format(epoch + 1), ncols=100)
	postfix = {'Average loss': 0}
	avg_loss = 0
	model.train()
	for batch_idx, (query, docs1, docs2, label, next_q, features1, features2, delta, long_qdids, longpos, short_qdids, shortpos, docs, docspos, doc1_order, doc2_order) in enumerate(tqdm_train):
		optimizer.zero_grad()

		query = query.cuda(args.cudaid)
		docs1 =docs1.cuda(args.cudaid)
		docs2 = docs2.cuda(args.cudaid)
		label = label.cuda(args.cudaid)
		next_q = next_q.cuda(args.cudaid)
		features1 = features1.cuda(args.cudaid)
		features2 = features2.cuda(args.cudaid)
		long_qdids = long_qdids.cuda(args.cudaid)
		longpos = longpos.cuda(args.cudaid)
		short_qdids = short_qdids.cuda(args.cudaid)
		shortpos = shortpos.cuda(args.cudaid)

		docs = docs.cuda(args.cudaid)
		docspos = docspos.cuda(args.cudaid)

		score, pre, p_score = model(query, docs1, docs2, features1, features2, long_qdids, longpos, short_qdids, shortpos, docs, docspos, doc1_order, doc2_order)
		# score, pre, p_score = model(query, docs1, docs2, features1, features2, long_qdids, longpos, short_qdids, shortpos)
		correct = torch.max(pre,1)[1].eq(label).cpu().sum()
		# print(score.shape)
		loss = criterion(p_score, label)# + loss_2/2.0
		loss.backward()
		optimizer.step()
		avg_loss += loss.item()
		postfix['Average loss'] = format(avg_loss / (batch_idx + 1), '.6f')
		tqdm_train.set_postfix(postfix)
		# print(loss)
	# torch.save(model.state_dict(),'model.lwh')

def score(test_loader, load=False):
	tqdm_test = tqdm(test_loader, desc='Testing (epoch #{})'.format(epoch + 1), ncols=100)
	postfix = {'Average loss': 0}
	avg_loss = 0
	# if load == True:
	# 	model.load_state_dict(torch.load('model.lwh'))
	model.eval()
	test_loss = 0
	correct = 0
	f = open(args.test_score_path,'w')
	for batch_idx, (query, docs, features, long_qdids, longpos, short_qdids, shortpos, lines, docslist, docspos, doc_order) in enumerate(tqdm_test):

		query = query.cuda(args.cudaid)
		docs =docs.cuda(args.cudaid)
		features = features.cuda(args.cudaid)
		long_qdids = long_qdids.cuda(args.cudaid)
		longpos = longpos.cuda(args.cudaid)
		short_qdids = short_qdids.cuda(args.cudaid)
		shortpos = shortpos.cuda(args.cudaid)
		docslist = docslist.cuda(args.cudaid)
		docspos = docspos.cuda(args.cudaid)

		score, pre, p_score = model(query, docs, docs, features, features, long_qdids, longpos, short_qdids, shortpos, docslist, docspos, doc_order, doc_order)
		# score, pre, p_score = model(query, docs, docs, features, features, long_qdids, longpos, short_qdids, shortpos)
		# print(score.shape)
		# print(score)
		for line, sc in zip(lines, score):
			f.write(line+'\t'+str(float(sc[0]))+'\n')

q_train = []
querys_train = []
docs1_train = []
docs2_train = []
title1_train = []
title2_train = []
label_train = []
delta_train = []
features1_train = []
features2_train = []
long_qdids_train = []
longpos_train = []
short_qdids_train = []
shortpos_train = []
docs_train = []
docs_pos_train = []
doc1_order_train = []
doc2_order_train = []

querys_test = []
docs_test = []
features_test = []
lines_test = []
long_qdids_test = []
longpos_test = []
short_qdids_test = []
shortpos_test = []
docs_list_test = []
docs_pos_test = []
doc_order_test = []

set_seed(args.seed)
model = Contextual(args = args, max_querylen=args.max_querylen, max_qdlen=args.max_qdlen, max_hislen=args.max_hislen, max_sessionlen=args.max_sessionlen,
				batch_size=args.batch_size, d_word_vec=100,
				n_layers=1, n_head=6, d_k=50, d_v=50,
				d_model=100, d_inner=256, dropout=0.1)
#model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.cuda(args.cudaid)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

if __name__ == '__main__':
	predata()

	train_loader = torch.utils.data.DataLoader(
		Dataset_train(
			querys=querys_train, docs1=docs1_train, docs2=docs2_train, label=label_train, next_q=[1],
			features1=features1_train, features2=features2_train, delta=delta_train, long_qdids=long_qdids_train, longpos=longpos_train,
			short_qdids=short_qdids_train, shortpos=shortpos_train, docslist = docs_train, docspos = docs_pos_train, doc1_order=doc1_order_train, doc2_order=doc2_order_train),
		batch_size=args.batch_size,
		collate_fn=collate_fn_train)

	score_loader = torch.utils.data.DataLoader(
		Dataset_score(
			querys=querys_test, docs=docs_test, features=features_test, long_qdids=long_qdids_test, longpos=longpos_test,
			short_qdids=short_qdids_test, shortpos=shortpos_test, lines=lines_test, docslist = docs_list_test, docspos = docs_pos_test, doc_order=doc_order_test),
		batch_size=args.batch_size,
		collate_fn=collate_fn_score)
	evaluation = AP()
	# model.load_state_dict(torch.load('model.zyj'))
	# with open('test_score.txt', 'r') as f:
	# 	evaluation.evaluate(f)
	#score(score_loader)
	#clear the result file
	with open(args.result_path,'a+',encoding='utf-8') as test:
		test.truncate(0)

	for epoch in range(args.epochs):
		train(train_loader)
		score(score_loader, False)
		with open(args.test_score_path, 'r') as f:
			evaluation.evaluate(f, args.result_path, epoch + 1, logger)

	
		