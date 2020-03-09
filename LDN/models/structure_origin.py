import torch
import torch.nn as nn
import torch.nn.functional as F


class Gconv(nn.Module):
    def __init__(self, d_in, d_out):
        super(Gconv, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.fc = nn.Linear(d_in, d_out)
        self.bn = nn.BatchNorm1d(d_out)

    def forward(self, A, nodes):
        # A: [batch_size, N, N]
        # nodes: [batch_size, N, d_in]
        nodes_tmp = torch.bmm(A, nodes) # [batch_size, N, d_in]
        nodes_tmp = self.fc(nodes_tmp) # [batch_size, N, d_out]
        nodes_tmp = self.bn(torch.transpose(nodes_tmp, 1, 2))
        nodes_tmp = torch.transpose(nodes_tmp, 1, 2)

        return nodes_tmp


class Acompute(nn.Module):
    def __init__(self, d_in):
        super(Acompute, self).__init__()
        self.conv1 = nn.Conv2d(d_in, int(d_in/2), 1, stride=1)
        self.bn1 = nn.BatchNorm2d(int(d_in/2))
        self.conv2 = nn.Conv2d(int(d_in/2), 1, 1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, nodes, A_init):
        A1 = nodes.unsqueeze(2) # [batch_size, N, 1, dim_fea]
        A2 = torch.transpose(A1, 1, 2) # [batch_size, 1, N, dim_fea]
        A = torch.abs(A1 - A2) # [batch_size, N, N, dim_fea]
        A = torch.transpose(A, 1, 3) # [batch_size, dim_fea, N, N]

        A = self.conv1(A)
        A = self.bn1(A)
        A = F.leaky_relu(A)

        A = self.conv2(A)
        A = torch.transpose(A, 1, 3).squeeze(3) # [batch_size, N, N]

        # activate
        zero_vec = -9e15*torch.ones_like(A)
        A = torch.where(A_init > 0, A, zero_vec)
        A = self.softmax(A)

        return A

class basic_block(nn.Module):
    def __init__(self, d_in, d_out):
        super(basic_block, self).__init__()
        self.module_g = Gconv(d_in, d_out)
        self.adj = Acompute(d_in)

    def forward(self, x, A_init):
        A_norm = self.adj(x, A_init)
        x_tmp = self.module_g(A_norm, x)

        return x_tmp


class GNN_resnext_new(nn.Module):
    def __init__(self):
        super(GNN_resnext_new, self).__init__()
        self.block1 = basic_block(128, 64)
        self.block2 = basic_block(64, 32)
        self.block3 = basic_block(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, nodes, A_init):
           
        feas = self.block1(nodes, A_init)
        feas = F.leaky_relu(feas)
        feas = self.block2(feas, A_init)  
        feas = F.leaky_relu(feas)          
        feas = self.block3(feas, A_init)
        feas = self.sigmoid(feas)

        return feas.squeeze(-1)