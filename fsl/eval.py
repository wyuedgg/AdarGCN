from torchtools import *
from data import MiniImagenetLoader, CUBLoader
from model import GraphNetwork, GNN, ConvNet, EmbeddingImagenet
#import resnet
import shutil
import os
import random
from train import ModelTrainer

if __name__ == '__main__':

    tt.arg.test_model = 'D-mini_N-5_K-5_U-0_L-4_B-40_T-False_SEED-222_GNN-dense_EXP-general' if tt.arg.test_model is None else tt.arg.test_model

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
    tt.arg.gnn_option = param['GNN']
    tt.arg.exp_mode = param['EXP']



    ####################
    tt.arg.device = 'cuda:0' if tt.arg.device is None else tt.arg.device
    # replace dataset_root with your own
    tt.arg.dataset_root = './datasets'
    tt.arg.dataset = 'mini' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 5 if tt.arg.num_shots is None else tt.arg.num_shots
    tt.arg.num_unlabeled = 0 if tt.arg.num_unlabeled is None else tt.arg.num_unlabeled
    tt.arg.train_query_num = 1 if tt.arg.train_query_num is None else tt.arg.train_query_num
    tt.arg.test_query_num = 1

    tt.arg.num_layers = 3 if tt.arg.num_layers is None else tt.arg.num_layers
    tt.arg.meta_batch_size = 40 if tt.arg.meta_batch_size is None else tt.arg.meta_batch_size
    tt.arg.transductive = False if tt.arg.transductive is None else tt.arg.transductive
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 2 if tt.arg.num_gpus is None else tt.arg.num_gpus

    tt.arg.num_ways_train = tt.arg.num_ways
    tt.arg.num_ways_test = tt.arg.num_ways

    tt.arg.num_shots_train = tt.arg.num_shots
    tt.arg.num_shots_test = tt.arg.num_shots

    tt.arg.train_transductive = tt.arg.transductive
    tt.arg.test_transductive = tt.arg.transductive

    # model parameter related
    
    tt.arg.num_edge_features = 96
    tt.arg.num_node_features = 96
    tt.arg.emb_size = 128
    tt.arg.eps = 0.1
    

    # train, test parameters
    tt.arg.train_iteration = 100000 if tt.arg.dataset == 'mini' else 200000
    tt.arg.test_iteration = 10000
    tt.arg.test_interval = 5000
    tt.arg.test_batch_size = 100
    tt.arg.log_step = 1000

    tt.arg.lr = 5e-4
    tt.arg.grad_clip = 5
    tt.arg.weight_decay = 1e-6
    tt.arg.dec_lr = 15000 if tt.arg.dataset == 'mini' else 30000
    tt.arg.dropout = 0.1 if tt.arg.dataset == 'mini' else 0.0

    #set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    #enc_module = resnet.get_model(model_name='resnet18', pretrained=True)
    enc_module = EmbeddingImagenet(emb_size=tt.arg.emb_size)
    

    # set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # to check
    exp_name = 'D-{}'.format(tt.arg.dataset)
    exp_name += '_N-{}_K-{}_U-{}'.format(tt.arg.num_ways, tt.arg.num_shots, tt.arg.num_unlabeled)
    exp_name += '_L-{}_B-{}'.format(tt.arg.num_layers, tt.arg.meta_batch_size)
    exp_name += '_T-{}'.format(tt.arg.transductive)
    exp_name += '_SEED-{}'.format(222)
    exp_name += '_GNN-{}'.format(tt.arg.gnn_option)
    exp_name += '_EXP-{}'.format(tt.arg.exp_mode)


    if not exp_name == tt.arg.test_model:
        print(exp_name)
        print(tt.arg.test_model)
        print('Test model and input arguments are mismatched!')
        AssertionError()

    gnn_module = GNN(d_in = enc_module.emb_size, n_layers = tt.arg.num_layers, opt = tt.arg.gnn_option)
    

    if tt.arg.dataset == 'mini':
        test_loader = MiniImagenetLoader(root=tt.arg.dataset_root, partition='test')
    elif tt.arg.dataset == 'cub':
        test_loader = CUBLoader(root=tt.arg.dataset_root, partition='test')
    else:
        print('Unknown dataset!')


    data_loader = {'test': test_loader}

    # create trainer
    tester = ModelTrainer(enc_module=enc_module,
                           gnn_module=gnn_module,
                           data_loader=data_loader,
                           query_num=[tt.arg.train_query_num, tt.arg.test_query_num])


    checkpoint = torch.load('asset/checkpoints/{}/'.format(exp_name) + 'model_best.pth.tar')
    #checkpoint = torch.load('./trained_models/{}/'.format(exp_name) + 'model_best.pth.tar')


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





