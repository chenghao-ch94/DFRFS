import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler
from model.dataloader.samplers_set2 import CategoriesSampler_Dom, CategoriesSampler_SetA
from model.models.protonet import ProtoNet
from model.models.matchnet import MatchNet
from model.models.feat import FEAT

class MultiGPUDataloader:
    def __init__(self, dataloader, num_device):
        self.dataloader = dataloader
        self.num_device = num_device

    def __len__(self):
        return len(self.dataloader) // self.num_device

    def __iter__(self):
        data_iter = iter(self.dataloader)
        done = False

        while not done:
            try:
                output_batch = ([], [])
                for _ in range(self.num_device):
                    batch = next(data_iter)
                    for i, v in enumerate(batch):
                        output_batch[i].append(v[None])
                
                yield ( torch.cat(_, dim=0) for _ in output_batch )
            except StopIteration:
                done = True
        return

def get_dataloader(args):
    if args.dataset == 'MiniImageNet':
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'Domain_FS':
        from model.dataloader.domain_fs import Domain_FS as Dataset 
    elif args.dataset == 'CUB_RAW':
        from model.dataloader.cub import CUB_RAW as Dataset 
    else:
        raise ValueError('Non-supported Dataset.')

    num_device = torch.cuda.device_count()
    num_episodes = args.episodes_per_epoch*num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers=args.num_workers*num_device if args.multi_gpu else args.num_workers
 
    if args.dataset == 'Domain_FS':

        ind = [i for i in range(6)]
        ind.remove(args.tid)
        d_id = ind

        trainset = Dataset(d_id, 'train', args, augment=args.augment)
        args.num_class = trainset.num_class
        train_sampler = CategoriesSampler_Dom(trainset.label,
                                        num_episodes,
                                        max(args.way, args.num_classes),
                                        args.shot + args.query,
                                        trainset.domain,
                                        d_id)

        train_loader = DataLoader(dataset=trainset,
                                    num_workers=num_workers,
                                    batch_sampler=train_sampler,
                                    pin_memory=True)
        
        valset = Dataset(args.tid, 'test', args)
        val_sampler = CategoriesSampler_Dom(valset.label,
                                args.num_eval_episodes,
                                args.eval_way, args.eval_shot + args.eval_query,
                                valset.domain,
                                args.tid)
        val_loader = DataLoader(dataset=valset,
                                batch_sampler=val_sampler,
                                num_workers=args.num_workers,
                                pin_memory=True)
        
        ### Evaluation Setting A: Support set from five source domains during testing phase
        # inda = [i for i in range(6)]
        # testset = Dataset(inda, 'test', args)
        # test_sampler = CategoriesSampler_SetA(testset.label,
        #                         600, # args.num_eval_episodes,
        #                         args.eval_way, args.eval_shot + args.eval_query,
        #                         testset.domain,
        #                         args.tid, d_id)

        ### Evaluation Setting B: Support set only from one target domain during testing phase
        testset = Dataset(args.tid, 'test', args)
        test_sampler = CategoriesSampler_Dom(testset.label,
                                600,
                                args.eval_way, args.eval_shot + args.eval_query,
                                testset.domain,
                                args.tid)
        
        test_loader = DataLoader(dataset=testset,
                                batch_sampler=test_sampler,
                                num_workers=args.num_workers,
                                pin_memory=True)  
    
    else:
        trainset = Dataset('train', args, augment=args.augment)
        args.num_class = trainset.num_class
        train_sampler = CategoriesSampler(trainset.label,
                                        num_episodes,
                                        max(args.way, args.num_classes),
                                        args.shot + args.query)

        train_loader = DataLoader(dataset=trainset,
                                    num_workers=num_workers,
                                    batch_sampler=train_sampler,
                                    pin_memory=True)

        valset = Dataset('val', args)
        val_sampler = CategoriesSampler(valset.label,
                                args.num_eval_episodes,
                                args.eval_way, args.eval_shot + args.eval_query)
        val_loader = DataLoader(dataset=valset,
                                batch_sampler=val_sampler,
                                num_workers=args.num_workers,
                                pin_memory=True)

        testset = Dataset('test', args)
        test_sampler = CategoriesSampler(testset.label,
                                600,
                                args.eval_way, args.eval_shot + args.eval_query)
        test_loader = DataLoader(dataset=testset,
                                batch_sampler=test_sampler,
                                num_workers=args.num_workers,
                                pin_memory=True)   

    return train_loader, val_loader, test_loader

def prepare_model(args):
    model = eval(args.model_class)(args)

    # load pre-trained model (no FC weights)
    if args.init_weights is not None:
        model_dict = model.state_dict()        
        pretrained_dict = torch.load(args.init_weights)['params']
        if args.backbone_class == 'ConvNet':
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, dim=0)
        para_model = model.to(device)
    else:
        para_model = model.to(device)

    return model, para_model

def prepare_optimizer(model, args):
    top_para = [v for k,v in model.named_parameters() if 'encoder' not in k]       
    # as in the literature, we use ADAM for ConvNet and SGD for other backbones
    if args.backbone_class == 'ConvNet':
        optimizer = optim.Adam(
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            # weight_decay=args.weight_decay, do not use weight_decay here
        )                
    else:
        optimizer = optim.SGD(
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            momentum=args.mom,
            nesterov=True,
            weight_decay=args.weight_decay
        )        

    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(args.step_size),
                            gamma=args.gamma
                        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=[int(_) for _ in args.step_size.split(',')],
                            gamma=args.gamma,
                        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            args.max_epoch,
                            eta_min=0   # a tuning parameter
                        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler