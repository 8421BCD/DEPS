import torch


def collate_fn_train(insts):
	''' Pad the instance to the max seq length in batch '''
	querys, docs1, docs2, label, next_q, features1, features2, delta, long_qdids, longpos, short_qdids, shortpos, docslist, docspos, doc1_order, doc2_order = zip(*insts)

	querys = torch.LongTensor(querys)
	docs1 = torch.LongTensor(docs1)
	docs2 = torch.LongTensor(docs2)
	label = torch.LongTensor(label)
	next_q = torch.LongTensor(next_q)
	features1 = torch.FloatTensor(features1)
	features2 = torch.FloatTensor(features2)
	delta = torch.FloatTensor(delta)
	long_qdids = torch.LongTensor(long_qdids)
	longpos = torch.LongTensor(longpos)
	short_qdids = torch.LongTensor(short_qdids)
	shortpos = torch.LongTensor(shortpos)
	
	docslist = torch.LongTensor(docslist)
	docspos = torch.LongTensor(docspos)

	return querys, docs1, docs2, label, next_q, features1, features2, delta, long_qdids, longpos, short_qdids, shortpos, docslist, docspos, doc1_order, doc2_order

class Dataset_train(torch.utils.data.Dataset):
	def __init__(
		self, querys, docs1, docs2, label, next_q, features1, features2, delta, long_qdids, longpos, short_qdids, shortpos, docslist, docspos, doc1_order, doc2_order):
		self.querys = querys
		self.docs1 = docs1
		self.docs2 = docs2
		self.label = label
		self.next_q = next_q
		self.features1 = features1
		self.features2 = features2
		self.delta = delta
		self.long_qdids = long_qdids
		self.longpos = longpos
		self.short_qdids = short_qdids
		self.shortpos = shortpos
		self.docslist = docslist
		self.docspos = docspos
		self.doc1_order = doc1_order
		self.doc2_order = doc2_order
		
	def __len__(self):
		return len(self.querys)

	def __getitem__(self, idx):
		querys = self.querys[idx]
		docs1 = self.docs1[idx]
		docs2 = self.docs2[idx]
		label = self.label[idx]
		next_q = self.next_q[0]
		features1 = self.features1[idx]
		features2 = self.features2[idx]
		delta = self.delta[idx]
		long_qdids = self.long_qdids[idx]
		longpos = self.longpos[idx]
		short_qdids = self.short_qdids[idx]
		shortpos = self.shortpos[idx]
		docslist = self.docslist[idx] #update
		docspos = self.docspos[idx]
		doc1_order = self.doc1_order[idx]
		doc2_order = self.doc2_order[idx]

		return querys, docs1, docs2, label, next_q, features1, features2, delta, long_qdids, longpos, short_qdids, shortpos, docslist, docspos, doc1_order, doc2_order

def collate_fn_score(insts):
	''' Pad the instance to the max seq length in batch '''
	querys, docs, features, long_qdids, longpos, short_qdids, shortpos, lines, docslist, docspos, doc_order = zip(*insts)

	querys = torch.LongTensor(querys)
	docs = torch.LongTensor(docs)
	features = torch.FloatTensor(features)
	long_qdids = torch.LongTensor(long_qdids)
	longpos = torch.LongTensor(longpos)
	short_qdids = torch.LongTensor(short_qdids)
	shortpos = torch.LongTensor(shortpos)

	docslist = torch.LongTensor(docslist)
	docspos = torch.LongTensor(docspos)

	return querys, docs, features, long_qdids, longpos, short_qdids, shortpos, lines, docslist, docspos, doc_order

class Dataset_score(torch.utils.data.Dataset):
	def __init__(
		self, querys, docs, features, long_qdids, longpos, short_qdids, shortpos, lines, docslist, docspos, doc_order):
		self.querys = querys
		self.docs = docs
		self.features = features
		self.long_qdids = long_qdids
		self.longpos = longpos
		self.short_qdids = short_qdids
		self.shortpos = shortpos
		self.lines = lines
		self.docslist = docslist
		self.docspos = docspos
		self.doc_order = doc_order

	def __len__(self):
		return len(self.querys)

	def __getitem__(self, idx):
		querys = self.querys[idx]
		docs = self.docs[idx]
		features = self.features[idx]
		long_qdids = self.long_qdids[idx]
		longpos = self.longpos[idx]
		short_qdids = self.short_qdids[idx]
		shortpos = self.shortpos[idx]
		lines = self.lines[idx]
		docslist = self.docslist[idx] # update
		docspos = self.docspos[idx]
		doc_order = self.doc_order[idx]

		return querys, docs, features, long_qdids, longpos, short_qdids, shortpos, lines, docslist, docspos, doc_order

