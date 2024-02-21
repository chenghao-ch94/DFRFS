# Disentangled Feature Representation for Few-Shot Image Classification

The code repository for "Disentangled Feature Representation for Few-Shot Image Classification" [[paper]](https://ieeexplore.ieee.org/abstract/document/10045748) (Accepted by IEEE TNNLS 2023) in PyTorch. If you use any content of this repo for your work, please cite the following bib entry:

    @article{cheng2023disentangled,
        title={Disentangled feature representation for few-shot image classification},
        author={Cheng, Hao and Wang, Yufei and Li, Haoliang and Kot, Alex C and Wen, Bihan},
        journal={IEEE Transactions on Neural Networks and Learning Systems},
        year={2023},
        publisher={IEEE}
    }

## Abstract

Learning the generalizable feature representation is critical for few-shot image classification. While recent works exploited task-specific feature embedding using meta-tasks for few-shot learning, they are limited in many challenging tasks as being distracted by the excursive features such as the background, domain, and style of the image samples. In this work, we propose a novel Disentangled Feature Representation framework, dubbed DFR, for few-shot learning applications. DFR can adaptively decouple the discriminative features that are modeled by the classification branch, from the class-irrelevant component of the variation branch. In general, most of the popular deep few-shot learning methods can be plugged in as the classification branch, thus DFR can boost their performance on various few-shot tasks. Furthermore, we propose a novel FS-DomainNet dataset based on DomainNet, for benchmarking the few-shot domain generalization tasks. We conducted extensive experiments to evaluate the proposed DFR on general, fine-grained, and cross-domain few-shot classification, as well as few-shot domain generalization, using the corresponding four benchmarks, i.e., mini-ImageNet, tiered-ImageNet, CUB, as well as the proposed FS-DomainNet. Thanks to the effective feature disentangling, the DFR-based few-shot classifiers achieved state-of-the-art results on all datasets.


## Prerequisites

The following packages are required to run the scripts:

- [PyTorch-1.7.1 and torchvision](https://pytorch.org)

- Package [tensorboardX](https://github.com/lanpa/tensorboardX)

- Dataset: please download the dataset and put images (split) into the folder ''data/[name of the dataset]/images (split)''

- Pre-trained weights: please download the [pre-trained weights](https://drive.google.com/file/d/1HYceUi21WTB9o690Rc1uevy0cNievVmb/view?usp=sharing) of the encoder if needed.

## Dataset

Please download the following dataset to "./data/[name of the dataset]" folder and change the data path in the 'dataloader' folder before running the code.

### MiniImageNet Dataset

The MiniImageNet dataset is a subset of the ImageNet that includes a total number of 100 classes and 600 examples per class. We follow the [previous setup](https://github.com/twitter/meta-learning-lstm), and use 64 classes as SEEN categories, 16 and 20 as two sets of UNSEEN categories for model validation and evaluation, respectively.

### CUB Dataset
[Caltech-UCSD Birds (CUB) 200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) is initially designed for fine-grained classification. It contains in total 11,788 images of birds over 200 species. On CUB, we randomly sampled 100 species as SEEN classes, and another two 50 species are used as two UNSEEN sets. We crop all images with given bounding boxes before training. Data Split is available on GoogleDrive [here]([https://drive.google.com/file/d/1H3PsLXR6sJK6dKTIOpP3qznsypAQ4Ix6/view?usp=sharing](https://drive.google.com/file/d/1fXflwCYcr9VXu66eMASb9eUy2EHBP58g/view?usp=sharing)).


### TieredImageNet Dataset
[TieredImageNet](https://github.com/renmengye/few-shot-ssl-public) is a large-scale dataset  with more categories, which contains 351, 97, and 160 categories for model training, validation, and evaluation, respectively. The dataset can also be downloaded from [here](https://github.com/kjunelee/MetaOptNet).

### FS-DomainNet Dataset

FS-DomainNet is reorganized from [DomainNet dataset](http://ai.bu.edu/M3SDA/#dataset) with only 299 sub-classes with more than 20 samples (minimum requirements for 5-shot setting with 15 query samples) in each domain (Sketch, Quickdraw, Real, Painting, Clipart, Infograph), and use 191, 47, and 61 for model training, validation and evaluation, respectively. 

Data Split is available on BaiduYun [here](https://pan.baidu.com/s/1AL3EcAbUXDsEu4VQ2_AIWA) with the extra code "vrfx", and GoogleDrive [here](https://drive.google.com/file/d/1H3PsLXR6sJK6dKTIOpP3qznsypAQ4Ix6/view?usp=sharing). You can find the split file of each domain in "./data/domainnet/split".


The dataset directory should look like this:
```bash
├── data
    ├── mini-imagenet
        ├── split
            ├── train.csv
            ├── test.csv
            ├── val.csv
        ├── images
    ├── tiered-imagenet
        ├── test_images.npz
        ├── test_labels.pkl   
        ├── train_images.npz  
        ├── train_labels.pkl
        ├── val_images.npz
        ├── val_labels.pkl
    ├── cub
        ├── split
            ├── train.csv
            ├── test.csv
            ├── val.csv
        ├── images
    ├── cubraw
        ├── split
            ├── train.csv
            ├── test.csv
            ├── val.csv       
        ├── images
    ├── domainnet
        ├── split
            ├── xxx_train.txt
            ├── xxx_test.txt
            ├── xxx_val.txt
            ... xxx={clipart, infograph, painting, quickdraw, real, sketch}
        ├── images
            ├── sketch
            ├── quickdraw
            ├── real
            ├── painting
            ├── clipart
            ├── infograph
```


## Model Training and Evaluation
Please use **train_fsl.py** and follow the instructions below.


## Training scripts for DFR (Here we take DFR+FEAT as an example)

For example, to train the 1-shot/5-shot 5-way DFR with FEAT model with ResNet-12 backbone on MiniImageNet:

    $ python train_fsl.py --gpu 0 --way 15 --lr 0.0005 --step_size 10 --max_epoch 60 --eval_way 5 --eval_shot 1

    $ python train_fsl.py --gpu 0 --way 10 --lr 0.0005 --step_size 10 --max_epoch 60 --eval_way 5 --eval_shot 5

to train the 1-shot/5-shot 5-way DFR with FEAT model with ResNet-12 backbone on TieredImageNet:

    $ python train_fsl.py --gpu 0 --way 15 --step_size 10 --max_epoch 100 --lr 0.0005 --dataset TieredImageNet --init_weights ./initialization/tieredimagenet/Res12-pre.pth --eval_way 5 --eval_shot 1

    $ python train_fsl.py --gpu 0 --step_size 20 --gamma 0.5 --dataset TieredImageNet --init_weights ./initialization/tieredimagenet/Res12-pre.pth --eval_way 5 --eval_shot 5

to train the 1-shot/5-shot 5-way DFR with FEAT model with ResNet-12 backbone on CUB-200-2011:

Pretraining:

    $ python pretrain.py --lr 0.001 --batch_size 256 --max_epoch 600 --backbone_class Res12 --dataset CUB --schedule 200 300 400 500 550 580 --gamma 0.1

Meta-Training:

    $ python train_fsl.py --gpu 0 --way 20 --shot 1 --step_size 20 --max_epoch 80 --balance 0.1 --lr 0.0005 --gamma 0.5 --dataset CUB --init_weights ./initialization/cub/max_acc_dist.pth --eval_way 5 --eval_shot 1

    $ python train_fsl.py --gpu 0 --max_epoch 80 --step_size 20 --gamma 0.5 --dataset CUB --init_weights ./initialization/cub/max_acc_dist.pth --eval_way 5 --eval_shot 5 

to train the 1-shot/5-shot 5-way DFR with FEAT model with ResNet-12 backbone (Pretrained on Tiered-ImageNet Dataset) on FS-DomainNet:

Use ''tid'' to select the target domain. (from 0 to 5)

    $ python train_fsl.py --gpu 0 --way 5 --lr 0.001 --step_size 20 --max_epoch 120 --eval_shot 1 --gamma 0.5 --init_weights ./initialization/tieredimagenet/Res12-pre.pth --dataset Domain_FS --eval_way 5 --eval_shot 1 --tid 0 

    $ python train_fsl.py --gpu 0 --way 5 --lr 0.001 --step_size 20 --max_epoch 120 --eval_shot 1 --gamma 0.5 --init_weights ./initialization/tieredimagenet/Res12-pre.pth --dataset Domain_FS --eval_way 5 --eval_shot 5 --tid 0 

## Acknowledgment

Our code builds upon the following code publicly available, and thank the following repos for providing helpful components/functions in our work.

- [ProtoNet](https://github.com/cyvius96/prototypical-network-pytorch)

- [MatchingNet](https://github.com/gitabcworld/MatchingNetworks)

- [FRN](https://github.com/Tsingularity/FRN)

- [FEAT](https://github.com/Sha-Lab/FEAT)

- [DeepEMD](https://github.com/icoz69/DeepEMD)
