import torch
import torch.nn as nn
import numpy as np
import os
import json
import argparse

from models import Encoder, GNN_1, GNN_2, GNN_3, GNN_4
from utils import get_allPaths, get_noisy_dataloader, get_batch_adj

parser=argparse.ArgumentParser()
parser.add_argument('--episode',type=int,default=50)
parser.add_argument('--test_round',type=int,default=20)
parser.add_argument('--batch_num',type=int,default=32)
parser.add_argument('--clean_num',type=int,default=50)
parser.add_argument('--noisy_num',type=int,default=1200)
parser.add_argument('--cleanSize',type=int,default=10)
parser.add_argument('--dirtySize',type=int,default=10)
parser.add_argument('--noisySize',type=int,default=30)
parser.add_argument('--GNNType',type=int,default=1)
parser.add_argument('--feaSize',type=int,default=128)

parser.add_argument('--learning_rate', type = float, default = 0.001)
parser.add_argument('--gpu',type=int, default=0)
parser.add_argument('--data_dir',type=str, default='./webImages')
parser.add_argument('--encoder_dir',type=str, default='./savedEncoders')
args = parser.parse_args()

torch.manual_seed(724)
if args.gpu >= 0 :
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(724)

encoder = Encoder()
if args.gpu >= 0 :
    encoder = encoder.cuda()
print("Loading encoder...")
encoder.load_state_dict(torch.load(args.encoder_dir + str(args.clean_num) + '_best_encoder.pth')['encoder'])       
encoder.eval()

GNN_chosed = [GNN_1, GNN_2, GNN_3, GNN_4][args.GNNType]

all_paths = get_allPaths(args.data_dir, 'noisy_{}'.format(args.noisy_num), args.clean_num)

ans = {}
for class_id in all_paths['class_ids']:
    gat = GNN_chosed()
    if args.gpu >= 0:
        gat = gat.cuda()
    optimizer = torch.optim.Adam(gat.parameters(), lr=args.learning_rate)
    gat.train()

    images = []
    episode = 0
    end_flag = False
    while True:
        if end_flag:
            break
        noisy_dataloader = get_noisy_dataloader(all_paths, class_id, args, is_training=True)
        for one_images, _ in noisy_dataloader:
            images.append(one_images)  
            if len(images) < args.batch_num:
                continue     

            images = torch.stack(images, dim=0)
            adjs = get_batch_adj(batch_num=args.batch_num, cleanSize=args.cleanSize, dirtySize=args.dirtySize, noisySize=args.noisySize) 
            if args.gpu >= 0 :
                images = images.cuda()
                adjs = adjs.cuda()                  

            with torch.no_grad():
                feas = encoder(images.view(-1, 3, 84, 84)).detach()
            feas = feas.view(args.batch_num, args.cleanSize+args.dirtySize+args.noisySize, args.feaSize)
            
            optimizer.zero_grad()    
            logits = gat(feas, adjs)
            
            loss_clean = - torch.log(logits[:, :args.cleanSize] + 1e-9).mean()
            loss_dirty = - torch.log(1 - logits[:, args.cleanSize: args.cleanSize+args.dirtySize] + 1e-9).mean()
            loss = loss_clean + loss_dirty
            loss.backward() 
            optimizer.step()
            images = [] 
            episode += 1
            print(episode, loss.item())
            if episode >= args.episode:
                end_flag = True
                break
    
    ans[class_id] = {img_name.split('/')[-1]: [] for img_name in all_paths['noisy'][class_id]}
    for _ in range(args.test_round):
        noisy_dataloader = get_noisy_dataloader(all_paths, class_id, args, is_training=True)
        loader_iter = noisy_dataloader.__iter__()
        images = []
        image_roots = []
        for _ in range(int(len(all_paths['noisy'][class_id]) / args.noisySize)):
            one_images, one_image_roots = loader_iter.next()
            images.append(one_images)
            image_roots.append(one_image_roots)     

        images = torch.stack(images, dim=0)
        adjs = get_batch_adj(batch_num=int(len(all_paths['noisy'][class_id]) / args.noisySize), cleanSize=args.cleanSize, dirtySize=args.dirtySize, noisySize=args.noisySize) 
        if args.gpu >= 0:
            images = images.cuda()
            adjs = adjs.cuda()              

        with torch.no_grad():
            feas = encoder(images.view(-1, 3, 84, 84)).detach()
            feas = feas.view(-1, args.cleanSize+args.dirtySize+args.noisySize, args.feaSize)  
            logits = gat(feas, adjs)    
        for b_i, b_roots in enumerate(image_roots):
            for p_i, one_root in enumerate(b_roots[args.cleanSize+args.dirtySize:]):
                this_score = logits[b_i, args.cleanSize+args.dirtySize+p_i].item()
                ans[class_id][one_root.split('/')[-1]].append(this_score)      
    
    ans[class_id] = {img_name: np.mean(ans[class_id][img_name]) for img_name in ans[class_id]}
    num_choose = 0
    for img_name in ans[class_id]:
        if ans[class_id][img_name] >= 0.5:
            num_choose += 1

    print("class_id: ", num_choose)

with open('GNN{}_clean{}_noisy{}.json'.format(args.GNNType, args.clean_num, args.noisy_num), 'w') as f:
    f.write(json.dumps(ans))
 