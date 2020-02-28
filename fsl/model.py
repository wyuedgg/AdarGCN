from torchtools import *
from collections import OrderedDict
import math
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, userelu=True, momentum=0.1, affine=True, track_running_stats=True):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False))

        if tt.arg.normtype == 'batch':
            self.layers.add_module('Norm', nn.BatchNorm2d(out_planes, momentum=momentum, affine=affine, track_running_stats=track_running_stats))
        elif tt.arg.normtype == 'instance':
            self.layers.add_module('Norm', nn.InstanceNorm2d(out_planes))

        if userelu:
            self.layers.add_module('ReLU', nn.ReLU(inplace=True))

        self.layers.add_module(
            'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.layers(x)
        return out

class ConvNet(nn.Module):
    def __init__(self, opt, momentum=0.1, affine=True, track_running_stats=True):
        super(ConvNet, self).__init__()
        self.in_planes  = opt['in_planes']
        self.out_planes = opt['out_planes']
        self.num_stages = opt['num_stages']
        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert(type(self.out_planes)==list and len(self.out_planes)==self.num_stages)

        num_planes = [self.in_planes,] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else True

        conv_blocks = []
        for i in range(self.num_stages):
            if i == (self.num_stages-1):
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1], userelu=userelu))
            else:
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i+1]))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.size(0),-1)
        return out

# encoder for imagenet dataset
class EmbeddingImagenet(nn.Module):
    def __init__(self, emb_size):
        super(EmbeddingImagenet, self).__init__()
        # set size
        self.hidden = 64
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                              out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        output_data = self.conv_4(self.conv_3(self.conv_2(self.conv_1(input_data))))
        return self.layer_last(output_data.view(output_data.size(0), -1))



class Gconv(nn.Module):
    def __init__(self, d_in, d_out):
        super(Gconv, self).__init__()
        #print('YES------------------------------------------------')
        self.d_in = d_in
        self.d_out = d_out
        self.fc = nn.Linear(d_in, d_out)
        self.bn = nn.BatchNorm1d(d_out)

    def forward(self, A, nodes):
        # A: [batch_size, N, N]
        # nodes: [batch_size, N, d_in]
        nodes_tmp = torch.bmm(A, nodes) # [batch_size, N, d_in]
        nodes = self.fc(nodes_tmp) # [batch_size, N, d_out]
        nodes = self.bn(torch.transpose(nodes, 1, 2))
        nodes = torch.transpose(nodes, 1, 2)
        nodes = F.leaky_relu(nodes)

        return nodes


class Acompute(nn.Module):
    def __init__(self, d_in, d_hidden, ratio=[2, 2, 1, 1]):
        super(Acompute, self).__init__()
        self.conv1 = nn.Conv2d(d_in, d_hidden*ratio[0], 1, stride=1)
        self.bn1 = nn.BatchNorm2d(d_hidden*ratio[0])
        self.conv2 = nn.Conv2d(d_hidden*ratio[0], d_hidden*ratio[1], 1, stride=1)
        self.bn2 = nn.BatchNorm2d(d_hidden*ratio[1])
        self.conv3 = nn.Conv2d(d_hidden*ratio[1], d_hidden*ratio[2], 1, stride=1)
        self.bn3 = nn.BatchNorm2d(d_hidden*ratio[2])
        self.conv4 = nn.Conv2d(d_hidden*ratio[2], d_hidden*ratio[3], 1, stride=1)
        self.bn4 = nn.BatchNorm2d(d_hidden*ratio[3])
        self.conv_last = nn.Conv2d(d_hidden*ratio[3], 1, 1, stride=1)
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, nodes, A_identity):
        A1 = nodes.unsqueeze(2) # [batch_size, N, 1, dim_fea]
        A2 = torch.transpose(A1, 1, 2) # [batch_size, 1, N, dim_fea]
        A = torch.abs(A1 - A2) # [batch_size, N, N, dim_fea]
        A = torch.transpose(A, 1, 3) # [batch_size, dim_fea, N, N]

        A = self.conv1(A)
        A = self.bn1(A)
        A = F.leaky_relu(A)
        # [batch_size, d_hidden*ratio[0], N, N]

        A = self.conv2(A)
        A = self.bn2(A)
        A = F.leaky_relu(A)
        # [batch_size, d_hidden*ratio[1], N, N]

        A = self.conv3(A)
        A = self.bn3(A)
        A = F.leaky_relu(A)
        # [batch_size, d_hidden*ratio[2], N, N]

        A = self.conv4(A)
        A = self.bn4(A)
        A = F.leaky_relu(A)
        # [batch_size, d_hidden*ratio[3], N, N]

        A = self.conv_last(A) # [batch_size, 1, N, N]
        A = torch.transpose(A, 1, 3).squeeze(3) # [batch_size, N, N]

        # activate
        A = torch.sigmoid(A)
        A = A * (1 - A_identity) # set the diagonal elements to be 0

        return A


class basic_block(nn.Module):
    def __init__(self, d_in, d_out=64, n_branch=3):
        super(basic_block, self).__init__()
        self.hidden = 96
        self.n_branch = n_branch
        self.d_out = d_out

        
        for gconv_num in range(self.n_branch): # for each branch

            if gconv_num == 0:
                continue

            for i in range(gconv_num): # for each gconv layer
                in_dim = 0
                if i == 0:
                    in_dim = d_in
                else:
                    in_dim = self.d_out
                module_g = Gconv(in_dim, self.d_out)
                self.add_module('branch{}_g{}'.format(gconv_num, i), module_g)

        self.outputdim = d_in + (n_branch - 1)*d_out
        self.adj = Acompute(self.outputdim, self.hidden, ratio=[2, 2, 1, 1])

        self.trans_fc = nn.Linear(d_in, self.outputdim)
        self.bn = nn.BatchNorm1d(self.outputdim)

    def A_normalize(self, A, A_identity):
        # A: [batch_size, N, N]
        A = A + A_identity
        D = torch.sum(A, 2) # [batch_size, N]
        D = torch.sqrt(D)
        tmp = torch.empty_like(A).cuda()
        for i in range(A.size(0)):
            tmp[i, :, :] = torch.diag(1/D[i, :])
        A_norm = torch.bmm(torch.bmm(tmp, A), tmp)

        return A_norm

    def forward(self, x, A_norm):
        A_identity = torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).cuda()

        original = x.clone() # [batch_size, N, d_in]
        
        x_tmp_list = []
        for gconv_num in range(self.n_branch):
            # dense branch
            if gconv_num == 0:
                x_tmp_list.append(original)
                continue
            # incept branch
            x_tmp = x.clone()
            for i in range(gconv_num): # for each gconv layer

                x_tmp = self._modules['branch{}_g{}'.format(gconv_num, i)](A_norm, x_tmp)
                if i == gconv_num-1:
                    x_tmp_list.append(x_tmp)

        x = torch.cat(x_tmp_list, 2) # [batch_size, N, d_out]

        A = self.adj(x, A_identity)
        A = A * (1 - A_identity)
        A_norm = self.A_normalize(A, A_identity)

        # res branch
        original = self.trans_fc(original)
        original = self.bn(torch.transpose(original, 1, 2))
        original = torch.transpose(original, 1, 2)
        x = F.leaky_relu(x + original)

        return (A, A_norm, x)


class basic_block_mod(nn.Module):
    def __init__(self, d_in, d_out=128, n_branch=3):
        super(basic_block_mod, self).__init__()
        self.hidden = 96
        self.n_branch = n_branch
        self.d_out = d_out

        
        for gconv_num in range(self.n_branch): # for each branch

            if gconv_num == 0:
                weight = nn.Linear(d_in, 1)
                self.add_module('weight{}'.format(gconv_num), weight)
                continue

            for i in range(gconv_num): # for each gconv layer
                in_dim = 0
                if i == 0:
                    in_dim = d_in
                else:
                    in_dim = self.d_out
                module_g = Gconv(in_dim, self.d_out)
                self.add_module('branch{}_g{}'.format(gconv_num, i), module_g)
            weight = nn.Linear(self.d_out, 1)
            self.add_module('weight{}'.format(gconv_num), weight)


        self.outputdim = d_in + (n_branch - 1)*d_out
        self.adj = Acompute(self.outputdim, self.hidden, ratio=[2, 2, 1, 1])


        self.trans_fc = nn.Linear(d_in, self.outputdim)
        self.bn = nn.BatchNorm1d(self.outputdim)

    def A_normalize(self, A, A_identity):
        # A: [batch_size, N, N]
        A = A + A_identity
        D = torch.sum(A, 2) # [batch_size, N]
        D = torch.sqrt(D)
        tmp = torch.empty_like(A).cuda()
        for i in range(A.size(0)):
            tmp[i, :, :] = torch.diag(1/D[i, :])
        A_norm = torch.bmm(torch.bmm(tmp, A), tmp)

        return A_norm

    def forward(self, x, A_norm):
        A_identity = torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).cuda()

        original = x.clone() # [batch_size, N, d_in]
        
        x_tmp_list = []
        w_list = []
        for gconv_num in range(self.n_branch):
            # dense branch
            if gconv_num == 0:
                w = F.sigmoid(self._modules['weight{}'.format(gconv_num)](original))
                w_list.append(w)
                x_tmp_list.append(w * original)
                #x_tmp_list.append(original)
                continue
            # incept branch
            x_tmp = x.clone()
            for i in range(gconv_num): # for each gconv layer

                x_tmp = self._modules['branch{}_g{}'.format(gconv_num, i)](A_norm, x_tmp)
                if i == gconv_num-1:
                    w = F.sigmoid(self._modules['weight{}'.format(gconv_num)](x_tmp))
                    w_list.append(w)
                    x_tmp_list.append(w * x_tmp)
                    #x_tmp_list.append(x_tmp)

        W = torch.cat(w_list, 2) # [batch_size, N, 3]
        #print(W.size())
        x = torch.cat(x_tmp_list, 2) # [batch_size, N, d_out]

        A = self.adj(x, A_identity)
        A = A * (1 - A_identity)
        A_norm = self.A_normalize(A, A_identity)

        # res branch
        original = self.trans_fc(original)
        original = self.bn(torch.transpose(original, 1, 2))
        original = torch.transpose(original, 1, 2)
        x = F.leaky_relu(x + original)

        return (A, A_norm, W, x)



class GNN_resnext_new(nn.Module):
    def __init__(self, d_in, n_branch = 3):
        super(GNN_resnext_new, self).__init__()
        self.block1 = basic_block(d_in, n_branch=n_branch)
        self.block2 = basic_block(self.block1.outputdim, n_branch=n_branch)
        self.block3 = basic_block(self.block2.outputdim, n_branch=n_branch)
        self.d_in = d_in
        self.d_out = 64

    def A_normalize(self, A, A_identity):
        # A: [batch_size, N, N]
        A = A + A_identity
        D = torch.sum(A, 2) # [batch_size, N]
        D = torch.sqrt(D)
        tmp = torch.empty_like(A).cuda()
        for i in range(A.size(0)):
            tmp[i, :, :] = torch.diag(1/D[i, :])
        A_norm = torch.bmm(torch.bmm(tmp, A), tmp)

        return A_norm

    def forward(self, nodes, A):
        A_identity = torch.eye(nodes.size(1)).unsqueeze(0).repeat(nodes.size(0), 1, 1).cuda()
        A = A * (1 - A_identity)
        A_norm = self.A_normalize(A, A_identity)
        A_list = []


        (A, A_norm, nodes) = self.block1(nodes, A_norm)
        A_list.append(A)

        (A, A_norm, nodes) = self.block2(nodes, A_norm)
        A_list.append(A)

        (A, A_norm, nodes) = self.block3(nodes, A_norm)
        A_list.append(A)

        return A_list


class GNN_resnext_mod(nn.Module):
    def __init__(self, d_in, n_branch = 3):
        super(GNN_resnext_mod, self).__init__()
        self.block1 = basic_block_mod(d_in, n_branch=n_branch)
        self.block2 = basic_block_mod(self.block1.outputdim, n_branch=n_branch)
        self.block3 = basic_block_mod(self.block2.outputdim, n_branch=n_branch)
        self.d_in = d_in
        self.d_out = 64

    def A_normalize(self, A, A_identity):
        # A: [batch_size, N, N]
        A = A + A_identity
        D = torch.sum(A, 2) # [batch_size, N]
        D = torch.sqrt(D)
        tmp = torch.empty_like(A).cuda()
        for i in range(A.size(0)):
            tmp[i, :, :] = torch.diag(1/D[i, :])
        A_norm = torch.bmm(torch.bmm(tmp, A), tmp)

        return A_norm

    def forward(self, nodes, A):
        A_identity = torch.eye(nodes.size(1)).unsqueeze(0).repeat(nodes.size(0), 1, 1).cuda()
        A = A * (1 - A_identity)
        A_norm = self.A_normalize(A, A_identity)


        A_list = []
        W_list = []

        (A, A_norm, W, nodes) = self.block1(nodes, A_norm)
        A_list.append(A)
        W_list.append(W)

        (A, A_norm, W, nodes) = self.block2(nodes, A_norm)
        A_list.append(A)
        W_list.append(W)

        (A, A_norm, W, nodes) = self.block3(nodes, A_norm)
        A_list.append(A)
        W_list.append(W)

        return A_list, W_list



def GNN(d_in, n_layers, n_branch = 3):
    
    return GNN_resnext_new(d_in, n_branch = n_branch)
    


