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
        nodes = self.fc(nodes_tmp) # [batch_size, N, d_out]
        nodes = self.bn(torch.transpose(nodes, 1, 2))
        nodes = torch.transpose(nodes, 1, 2)
        nodes = F.leaky_relu(nodes)

        return nodes


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
    def __init__(self, d_in, d_hid, d_out):
        super(basic_block, self).__init__()
        self.n_branch = 2
        
        for gconv_num in range(self.n_branch): # for each branch

            if gconv_num == 0:
                continue

            for i in range(gconv_num): # for each gconv layer
                in_dim = 0
                if i == 0:
                    in_dim = d_in
                else:
                    in_dim = d_hid
                module_g = Gconv(in_dim, d_hid)
                self.add_module('branch{}_g{}'.format(gconv_num, i), module_g)


        self.outputdim = d_in + (self.n_branch - 1)*d_hid
        self.adj = Acompute(d_in)
        self.trans_fc = nn.Linear(self.outputdim, d_out)

    def forward(self, x, A_init):
        A_norm = self.adj(x, A_init)
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

        x = torch.cat(x_tmp_list, 2) # [batch_size, N, output_dim]
        x = self.trans_fc(x)

        return x


class GNN_resnext_new(nn.Module):
    def __init__(self):
        super(GNN_resnext_new, self).__init__()
        self.block1 = basic_block(128, 64, 64)
        self.block2 = basic_block(64, 32, 32)
        self.block3 = basic_block(32, 16, 1)
        self.trans_fc1 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.trans_fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, nodes, A_init):
    
        skip_feas = nodes.clone()
        feas = self.block1(nodes, A_init)
        skip_feas = self.trans_fc1(skip_feas)
        skip_feas = self.bn1(torch.transpose(skip_feas, 1, 2))
        skip_feas = torch.transpose(skip_feas, 1, 2)
        feas = F.leaky_relu(skip_feas+feas)

        skip_feas = feas.clone()
        feas = self.block2(feas, A_init)
        skip_feas = self.trans_fc2(skip_feas)
        skip_feas = self.bn2(torch.transpose(skip_feas, 1, 2))
        skip_feas = torch.transpose(skip_feas, 1, 2)
        feas = F.leaky_relu(skip_feas+feas)        
        
        feas = self.block3(feas, A_init)
        feas = self.sigmoid(feas)

        return feas.squeeze(-1)
