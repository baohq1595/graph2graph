import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import pickle

import sys
sys.path.append('D:\\workspace\\python\\graph2graph')

from fast_jtnn import *
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--test', required=False, default='k')
parser.add_argument('--vocab', required=False, default='j')
parser.add_argument('--model', required=False, default='diff_vae/models/logp04/model-iter-5')

parser.add_argument('--hidden_size', type=int, default=330)
parser.add_argument('--rand_size', type=int, default=8)
parser.add_argument('--depthT', type=int, default=6)
parser.add_argument('--depthG', type=int, default=3)
parser.add_argument('--share_embedding', action='store_true')
parser.add_argument('--use_molatt', action='store_true')

parser.add_argument('--num_decode', type=int, default=20)
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()
  
vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

# state_dict = torch.load(args.model)
# print(state_dict.keys())
# exit()
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     if k != 'embedding.weight':
#         name = k # remove `module.`
#         new_state_dict[name] = v
    # else:
    #     new_state_dict['encoder.embedding.weight'] = v



model = DiffVAE(vocab, args).cuda()
# model.load_state_dict(new_state_dict)
model.load_state_dict(torch.load(args.model))

with open(args.test) as f:
    data = [line.split()[0] for line in f]

data = [MolTree(s) for s in data]
batches = [data[i : i + 1] for i in range(0, len(data))]
dataset = MolTreeDataset(batches, vocab, assm=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])

torch.manual_seed(args.seed)
for batch in loader:
    mol_batch = batch[0]
    x_tree_vecs, _, x_mol_vecs = model.encode(batch[1], batch[2])
    assert x_tree_vecs.size(0) == x_mol_vecs.size(0)

    for k in range(args.num_decode):
        z_tree_vecs, z_mol_vecs = model.fuse_noise(x_tree_vecs, x_mol_vecs)
        smiles = mol_batch[0].smiles
        new_smiles = model.decode(z_tree_vecs[0].unsqueeze(0), z_mol_vecs[0].unsqueeze(0))
        print(smiles, new_smiles)
