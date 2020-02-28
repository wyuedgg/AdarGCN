import torch
import numpy as np

def get_batch_adj(batch_num, cleanSize, dirtySize, noisySize):
    
    adj = torch.ones((batch_num, cleanSize+dirtySize+noisySize, cleanSize+dirtySize+noisySize)).float()
    adj[:, :cleanSize, cleanSize: cleanSize+dirtySize].fill_(0)
    adj[:, cleanSize: cleanSize+dirtySize, :cleanSize].fill_(0)
    return adj