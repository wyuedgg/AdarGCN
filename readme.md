# AdarGCN: Adaptive-Aggregation GCN for Few-Shot Learning
Jianhong Zhang, Manli Zhang, Zhiwu Lu, Tao Xiang, Ji-Rong Wen.

The code repository for "[AdarGCN: Adaptive-Aggregation GCN for Few-Shot Learning](https://arxiv.org/abs/2002.12641)" in PyTorch.

## Abstract
Existing few-shot learning (FSL) methods assume that there exist sufficient training samples from source classes for knowledge transfer to target classes with few training samples. However, this assumption is often invalid, especially when it comes to fine-grained recognition. In this work, we define a new FSL setting termed few-shot few-shot learning (FSFSL), under which both the source and target classes have limited training samples. To overcome the source class data scarcity problem, a natural option is to crawl images from the web with class names as search keywords. However, the crawled images are inevitably corrupted by large amount of noise (irrelevant images) and thus may harm the performance. To address this problem, we propose a graph convolutional network (GCN)-based label denoising (LDN) method to remove the irrelevant images. Further, with the cleaned web images as well as the original clean training images, we propose a GCN-based FSL method. For both the LDN and FSL tasks, a novel adaptive aggregation GCN (AdarGCN) model is proposed, which differs from existing GCN models in that adaptive aggregation is performed based on a multi-head multilevel aggregation module. With AdarGCN, how much and how far information carried by each graph node is propagated in the graph structure can be determined automatically, therefore alleviating the effects of both noisy and outlying training samples. Extensive experiments demonstrate the superior performance of our AdarGCN under both the new FSFSL and the conventional FSL settings.

## Citation
If you find it useful, please consider citing our work using the bibtex:

    @article{Zhang2020AdarGCN-fsl,
    author    = {Jianhong Zhang and Manli Zhang and Zhiwu Lu and Tao Xiang and Ji{-}Rong Wen},
    title     = {AdarGCN: Adaptive-Aggregation GCN for Few-Shot Learning},
    journal   = {CoRR},
    volume    = {abs/2002.12641},
    year      = {2020},
    archivePrefix = {arXiv},
    eprint    = {2002.12641}
    }

## Environment
* Python 3.6
* PyTorch 1.0.1

## Get Started 
### Data Preparation
Folder 'FSL/datasets' should contain the raw images of 2 FSL benchmark datasets (i.e. miniImageNet and CUB). You can download the original images of miniImageNet from [ImageNet](http://image-net.org/image/ILSVRC2015/ILSVRC2015_CLS-LOC.tar.gz) and [CUB](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz).

### Model Training and Test
1. Train AdarGCN on miniImageNet under the 5-way 5-shot setting.
    python train.py --lr 3e-4 --meta_batch_size 32 --num_ways 5 --num_shots 5 --dataset mini
2. Train AdarGCN on CUB under the 5-way 5-shot setting.
    python train.py --lr 3e-4 --meta_batch_size 32 --num_ways 5 --num_shots 5 --dataset CUB
3. Evaluate AdarGCN under the 5-way 5-shot setting.
    python eval.py --test_model D-mini_N-5_K-5_U-0_L-3_B-32_T-False_SEED-222

Please don't forget to check other arguments before running the code.

## Acknowledgment
We thank following repos providing helpful components/functions in our work.
1. Edge-labeling Graph Neural Network for Few-shot Learning https://github.com/khy0809/fewshot-egnn
