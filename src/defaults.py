import torch
import os
PRECISION = 2
OPTIMIZER = torch.optim.Adam
BATCH_SIZE = 256
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3

# ROOT_DIR = None    # set a path (directory) where experiments should be saved
ROOT_DIR ='/home/rahul/smbpo_k/Safe-MBPO/results/one_c_sa/seed1237'
#ROOT_DIR = '/tiger/u/gwthomas/data/smbpo'
