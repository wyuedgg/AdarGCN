from torchtools import *
from data import MiniImagenetLoader, CUBLoader
from model import GNN, ConvNet, EmbeddingImagenet
import shutil
import os
import random


class ModelTrainer(object):
    def __init__(self,
                 enc_module,
                 gnn_module,
                 data_loader,
                 query_num):
        self.train_Query_num = query_num[0]
        self.test_Query_num = query_num[1]
        # set encoder and gnn
        self.enc_module = enc_module.to(tt.arg.device)
        self.gnn_module = gnn_module.to(tt.arg.device)

        if tt.arg.num_gpus > 1:
            print('Construct multi-gpu model ...')
            self.enc_module = nn.DataParallel(self.enc_module, device_ids=list(range(tt.arg.num_gpus)), dim=0)
            self.gnn_module = nn.DataParallel(self.gnn_module, device_ids=list(range(tt.arg.num_gpus)), dim=0)
            print('done!\n')

        # get data loader
        self.data_loader = data_loader
        self.global_step = 0

        self.edge_loss = nn.BCELoss(reduction='mean')
        self.node_loss = nn.CrossEntropyLoss(reduction='none')
        self.val_acc = 0

    def smoothing(self, A_gt): # ground truth full edge
        # label smoothing
        A_gt = A_gt * (1 - tt.arg.eps) + (1 - A_gt) * tt.arg.eps / (tt.arg.num_ways - 1)
        A_identity = torch.eye(A_gt.size(2)).unsqueeze(0).repeat(A_gt.size(0), 1, 1)
        A_identity = A_identity.cuda()
        A_gt = A_gt * (1 - A_identity)

        return A_gt


    def eval(self, partition='test', log_flag=True):
        best_acc = 0
        # set edge mask (to distinguish support and query edges)
        num_supports = tt.arg.num_ways_test * tt.arg.num_shots_test
        num_queries = tt.arg.num_ways_test * self.test_Query_num
        num_samples = num_supports + num_queries
        
        query_edge_losses = []
        query_edge_accrs = []
        query_node_accrs = []

        with torch.no_grad():
            # for each iteration
            for iter in range(tt.arg.test_iteration//tt.arg.test_batch_size):
                # load task data list
                [support_data,
                 support_label,
                 query_data,
                 query_label] = self.data_loader[partition].get_task_batch(num_tasks=tt.arg.test_batch_size,
                                                                           num_ways=tt.arg.num_ways_test,
                                                                           num_shots=tt.arg.num_shots_test,
                                                                           num_queries=self.test_Query_num,
                                                                           seed=iter+tt.arg.seed)

                # set as single data
                full_data = torch.cat([support_data, query_data], 1)
                full_label = torch.cat([support_label, query_label], 1)
                full_edge = self.label2edge(full_label)  # <num_tasks, num_samples, num_samples>

                # set init edge
                init_edge = full_edge.clone()
                init_edge[:, num_supports:, :] = 0.5 # edge on query samples is unknown
                init_edge[:, :, num_supports:] = 0.5
                for i in range(num_queries):
                    init_edge[:, num_supports + i, num_supports + i] = 1.0

                # for semi-supervised setting,
                for c in range(tt.arg.num_ways_test):
                    init_edge[:, ((c+1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c+1) * tt.arg.num_shots_test, :num_supports] = 0.5
                    init_edge[:, :num_supports, ((c+1) * tt.arg.num_shots_test - tt.arg.num_unlabeled):(c+1) * tt.arg.num_shots_test] = 0.5

                # set as eval mode
                self.enc_module.eval()
                self.gnn_module.eval()

                # (1) encode data
                full_data = [self.enc_module(data.squeeze(1)) for data in full_data.chunk(full_data.size(1), dim=1)]
                full_data = torch.stack(full_data, dim=1)

                # (2) predict edge logit (consider only the last layer logit, num_tasks x num_samples x num_samples)
                full_logit = torch.zeros(tt.arg.test_batch_size, num_samples, num_samples).to(tt.arg.device)

                
                support_data = full_data[:, :num_supports] # batch_size x num_support x featdim
                query_data = full_data[:, num_supports:] # batch_size x num_query x featdim
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1, 1) # batch_size x num_queries x num_support x featdim
                support_data_tiled = support_data_tiled.view(tt.arg.test_batch_size * num_queries, num_supports, -1) # (batch_size x num_queries) x num_support x featdim
                query_data_reshaped = query_data.contiguous().view(tt.arg.test_batch_size * num_queries, -1).unsqueeze(1) # (batch_size x num_queries) x 1 x featdim
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped], 1) # (batch_size x num_queries) x (num_support + 1) x featdim

                input_edge_feat = torch.zeros(tt.arg.test_batch_size, num_supports + 1, num_supports + 1).to(tt.arg.device)  # batch_size x (num_support + 1) x (num_support + 1)
                input_edge_feat[:, :num_supports, :num_supports] = init_edge[:, :num_supports, :num_supports]  # batch_size x (num_support + 1) x (num_support + 1)
                input_edge_feat = input_edge_feat.repeat(num_queries, 1, 1)  # (batch_size x num_queries) x (num_support + 1) x (num_support + 1)

                # input_node_feat: (batch_size x num_queries) x (num_support + 1) x featdim
                # input_edge_feat: (batch_size x num_queries) x (num_support + 1) x (num_support + 1)
                logit = self.gnn_module(input_node_feat, input_edge_feat)[-1]
                # logit: (batch_size x num_queries) x (num_support + 1) x (num_support + 1)

                # logit : batch_size x num_queries x (num_support + 1) x (num_support + 1)
                logit = logit.view(tt.arg.test_batch_size, num_queries, num_supports + 1, num_supports + 1)

                # logit --> full_logit (batch_size x num_samples x num_samples)
                full_logit[:, :num_supports, :num_supports] = logit[:, :, :num_supports, :num_supports].mean(1)
                full_logit[:, :num_supports, num_supports:] = logit[:, :, :num_supports, -1].transpose(1, 2)
                full_logit[:, num_supports:, :num_supports] = logit[:, :, -1, :num_supports]

                edge_identity = torch.eye(num_samples).unsqueeze(0).repeat(tt.arg.test_batch_size, 1, 1).cuda()

                # (4) compute loss
                full_edge_loss = self.edge_loss(full_logit * (1-edge_identity), self.smoothing(full_edge))
                query_edge_loss = full_edge_loss

                # compute node accuracy (num_tasks x num_quries x num_ways)
                query_node_pred = torch.bmm(full_logit[:, num_supports:, :num_supports], self.one_hot_encode(tt.arg.num_ways_test, support_label.long())) # (num_tasks x num_quries x num_supports) * (num_tasks x num_supports x num_ways)
                query_node_accr = torch.eq(torch.max(query_node_pred, -1)[1], query_label.long()).float().mean()

                query_edge_losses += [query_edge_loss.item()]
                query_node_accrs += [query_node_accr.item()]

        # logging
        if log_flag:
            tt.log('---------------------------')
            tt.log_scalar('{}/edge_loss'.format(partition), np.array(query_edge_losses).mean(), self.global_step)
            tt.log_scalar('{}/node_accr'.format(partition), np.array(query_node_accrs).mean(), self.global_step)

            tt.log('evaluation: total_count=%d, accuracy: mean=%.2f%%, std=%.2f%%, ci95=%.2f%%' %
                   (iter,
                    np.array(query_node_accrs).mean() * 100,
                    np.array(query_node_accrs).std() * 100,
                    1.96 * np.array(query_node_accrs).std() / np.sqrt(float(len(np.array(query_node_accrs)))) * 100))
            tt.log('---------------------------')

        return np.array(query_node_accrs).mean()


    def label2edge(self, label): # <num_tasks, num_samples>
        # get size
        num_samples = label.size(1)

        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)

        # compute edge  # <num_tasks, num_samples, num_samples>
        edge = torch.eq(label_i, label_j).float().to(tt.arg.device)

        return edge  # <num_tasks, num_samples, num_samples>

    def hit(self, logit, label):
        # logit: <bsz, num_support + num_query, num_support + num_query>
        pred = logit.lt(0.5).long()
        hit = torch.eq(pred, label).float()
        return hit

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes)[class_idx].to(tt.arg.device)


if __name__ == '__main__':

    print(tt.arg.test_model)
    list1 = tt.arg.test_model.split("_")
    param = {}
    for i in range(len(list1)):
        param[list1[i].split("-", 1)[0]] = list1[i].split("-", 1)[1]
    tt.arg.dataset = param['D']
    tt.arg.num_ways = int(param['N'])
    tt.arg.num_shots = int(param['K'])
    tt.arg.num_unlabeled = int(param['U'])
    tt.arg.num_layers = int(param['L'])
    tt.arg.meta_batch_size = int(param['B'])
    tt.arg.transductive = False if param['T'] == 'False' else True


    ####################
    tt.arg.device = 'cuda:0' if tt.arg.device is None else tt.arg.device
    # replace dataset_root with your own
    tt.arg.dataset_root = './datasets'
    tt.arg.dataset = 'mini' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.train_query_num = 1 
    tt.arg.test_query_num = 15
    tt.arg.seed = 248 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_layers = 3 if tt.arg.num_layers is None else tt.arg.num_layers
    tt.arg.transductive = False
    tt.arg.num_gpus = 2 if tt.arg.num_gpus is None else tt.arg.num_gpus

    tt.arg.num_ways_train = tt.arg.num_ways
    tt.arg.num_ways_test = tt.arg.num_ways

    tt.arg.num_shots_train = tt.arg.num_shots
    tt.arg.num_shots_test = tt.arg.num_shots

    # model parameter related
    tt.arg.num_edge_features = 96
    tt.arg.num_node_features = 96
    tt.arg.emb_size = 128
    
    # train, test parameters
    tt.arg.test_iteration = 600
    tt.arg.test_batch_size = 30
    tt.arg.eps = 0.05

    #set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    tt.arg.log_dir_user = tt.arg.log_dir if tt.arg.log_dir_user is None else tt.arg.log_dir_user
    tt.arg.log_dir = tt.arg.log_dir_user

    enc_module = EmbeddingImagenet(emb_size=tt.arg.emb_size)
    gnn_module = GNN(d_in = enc_module.emb_size)

    if tt.arg.dataset == 'mini':
        test_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='test')
    elif tt.arg.dataset == 'cub':
        test_loader = CUBLoader(root=tt.arg.dataset_root, partition='test')
    else:
        print('Unknown dataset!')

    data_loader = {'test': test_loader}

    # create trainer
    # create trainer
    tester = ModelTrainer(enc_module=enc_module,
                           gnn_module=gnn_module,
                           data_loader=data_loader,
                           query_num=[tt.arg.train_query_num, tt.arg.test_query_num])

    checkpoint = torch.load('asset/checkpoints/{}/'.format(tt.arg.test_model) + 'model_best.pth.tar')


    tester.enc_module.load_state_dict(checkpoint['enc_module_state_dict'])
    print("load pre-trained enc_nn done!")

    # initialize gnn pre-trained
    print(checkpoint['gnn_module_state_dict'].keys())
    tester.gnn_module.load_state_dict(checkpoint['gnn_module_state_dict'])
    print("load pre-trained egnn done!")

    tester.val_acc = checkpoint['val_acc']
    tester.global_step = checkpoint['iteration']

    print('Best model\'s val acc: ' + str(tester.val_acc))
    print('Best models\'s global step: ' + str(tester.global_step))

    tester.eval(partition='test')
