from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
import torch

tokenizer = BertTokenizer.from_pretrained('/home/yujia_zhou/pytorch/transformer/data/bert-base-uncased-vocab.txt')
bert = BertModel.from_pretrained('/home/yujia_zhou/pytorch/transformer/data/bert-base-uncased/')

s1 = 'dallas swat photos'
s2 = 'a talker'
tokens1 = tokenizer.tokenize(s1)
tokens2 = tokenizer.tokenize(s2)
print(s1)
print(s2)
print(tokens1)
print(tokens2)

# ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
# print(ids.shape)

# result, *_ = bert(ids, output_all_encoded_layers=True)

# print(len(result[0][0]))