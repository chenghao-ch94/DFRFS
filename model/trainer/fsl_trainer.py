import time
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer #, prepare_optimizer_w,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm
import torchvision.utils as vutils


class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.para_model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label_s = torch.arange(args.way, dtype=torch.int16).repeat(args.shot)
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label_s = label_s.type(torch.LongTensor)
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label_s = label_s.cuda()
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label_s, label, label_aux

    def train(self):
        args = self.args
        self.para_model.train()
        if self.args.fix_BN:
            self.para_model.encoder.eval()
        
        writer = SummaryWriter(osp.join(args.save_path,'tf'))
        label_s, label, label_aux = self.prepare_label()

        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.para_model.train()
            if self.args.fix_BN:
                self.para_model.encoder.eval()
            
            tl1 = Averager()
            ta = Averager()

            start_tm = time.time()
            for batch in self.train_loader:
                self.train_step += 1

                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
               
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                logits, reg_logits, bce_loss, loss_rec, loss_rec_self, loss_tran, loss_tran2 = self.para_model(data, label_s, label, False, epoch)
                if reg_logits is not None: # For FEAT-based
                    loss = F.cross_entropy(logits, label)
                    loss_reg = F.cross_entropy(reg_logits, label_aux)
                    total_loss = loss + 0.01*bce_loss + 0.1*loss_rec + 0.1*loss_rec_self + self.args.balance*loss_reg  + loss_tran + loss_tran2
        
                else: # For others
                    loss = F.cross_entropy(logits, label)
                    total_loss = loss + 0.01*bce_loss + 0.1*loss_rec + 0.1*loss_rec_self + loss_tran + loss_tran2


                ### For original FEAT and ProtoNet

                # logits, reg_logits =  self.para_model(data, None, None, False)
                # if reg_logits is not None: # For FEAT-based
                #     loss = F.cross_entropy(logits, label)
                #     loss_reg = F.cross_entropy(reg_logits, label_aux)
                #     total_loss = loss + self.args.balance*loss_reg
        
                # else: # For others
                #     loss = F.cross_entropy(logits, label)
                #     total_loss = loss           

                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                writer.add_scalar('data/loss_cls', float(loss.item()), epoch)
                writer.add_scalar('data/acc', float(acc), epoch)

                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)    

                # refresh start_tm
                start_tm = time.time()

            self.lr_scheduler.step()
            self.try_evaluate(epoch)

            # torch.cuda.synchronize()

            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        writer.close()
        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

        print('best epoch {}, best tval acc={:.4f} + {:.4f}'.format(
                self.trlog['max_tacc_epoch'],
                self.trlog['max_tacc'],
                self.trlog['max_tacc_interval']))

        
        self.trlog['test_acc'] = self.trlog['max_tacc']
        self.trlog['test_acc_interval'] = self.trlog['max_tacc_interval']


    def evaluate(self, data_loader, epoch):
        # restore model args
        args = self.args
        
        # evaluation mode
        self.para_model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label_tr = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_shot)
        label_tr = label_tr.type(torch.LongTensor)
        if torch.cuda.is_available():
            label_tr = label_tr.cuda()                
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()

        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        print('best epoch {}, best tval acc={:.4f} + {:.4f}'.format(
                self.trlog['max_tacc_epoch'],
                self.trlog['max_tacc'],
                self.trlog['max_tacc_interval']))
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.para_model(data, label_tr, None)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc

                
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        # train mode
        self.para_model.train()
        if self.args.fix_BN:
            self.para_model.encoder.eval()

        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.para_model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.para_model.eval()
        record = np.zeros((600, 2)) # loss and acc
        label_tr = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_shot)
        label_tr = label_tr.type(torch.LongTensor)
        if torch.cuda.is_available():
            label_tr = label_tr.cuda()   
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        label_aux = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_shot + args.eval_query)
        label_aux = label_aux.type(torch.LongTensor)
        if torch.cuda.is_available():
            label_aux = label_aux.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))

        print(args.save_path)
        writer = SummaryWriter(osp.join(args.save_path,'tf'))      

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.para_model(data, label_tr, label)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc

        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))
        
        writer.close()
        return vl, va, vap


    def final_record(self):
        # save the best performance in a txt file
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))           