import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import math
import random
import numpy as np

from model.models import FewShotModel

from torch.autograd import Function
class _ReverseGrad(Function):

    @staticmethod
    def forward(ctx, input, grad_scaling):
        ctx.grad_scaling = grad_scaling
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_scaling = ctx.grad_scaling
        return -grad_scaling * grad_output, None

reverse_grad = _ReverseGrad.apply

class ReverseGrad(nn.Module):
    """Gradient reversal layer.

    It acts as an identity layer in the forward,
    but reverses the sign of the gradient in
    the backward.
    """

    def forward(self, x, grad_scaling=1.):
        assert grad_scaling >= 0, \
            'grad_scaling must be non-negative, ' \
            'but got {}'.format(grad_scaling)
        return reverse_grad(x, grad_scaling)

class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=0, groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Conv_block_noac(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=0, groups=1):
        super(Conv_block_noac, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)

class ProtoNet2(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1)) #nn.AvgPool2d(5, stride=1) #nn.AdaptiveAvgPool2d((1, 1))
        self.criterionL1 = nn.L1Loss()
        # self.criterionL2 = nn.MSELoss()

        self.reverse_layer = ReverseGrad()

        self.hdim = 640
        self.enc_sim = nn.Sequential(
            Conv_block(self.hdim , self.hdim , kernel=(3,3), padding=(1,1)),
            Conv_block(self.hdim , self.hdim , kernel=(3,3), padding=(1,1)),
            Conv_block(self.hdim , self.hdim , kernel=(3,3), padding=(1,1)),
        )
        # self.rev_rate = 0.1
        self.lambda_ = 0
        self.decoder_fc =nn.Linear(self.hdim *2, self.hdim *5*5)
        self.decoder =  nn.Sequential(
                Conv_block(self.hdim , self.hdim , kernel=(3,3), padding=(1,1)),
                Conv_block(self.hdim , self.hdim , kernel=(3,3), padding=(1,1)),
                Conv_block(self.hdim , self.hdim , kernel=(3,3), padding=(1,1)),
        )   


    def set_lambda(self, para):
        self.lambda_ = 2./(1+math.exp(-10.*para))-1
        return self.lambda_

    def decode(self, fea, code):
        fea_cat = self.decoder_fc(torch.cat([fea,code],dim=1)).view(-1, self.hdim, 5, 5)
        fea_rec = self.decoder(fea_cat)
        return fea_rec

    def _forward(self, support_idx, query_idx, ori=None, s_lab=None, q_lab=None, para=0):

        instance_embs = self.avgpool1(ori)
        # instance_embs = nn.MaxPool2d(5)(ori)
        instance_embs = instance_embs.view(instance_embs.size(0), -1)
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))

        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        #variation branch
        mm = self.enc_sim(ori)
        var_all = self.avgpool1(mm).squeeze()  #M*640
        # var_all = nn.MaxPool2d(5)(self.enc_sim(ori)).squeeze()
        
        #gradient reverse
        lam = self.set_lambda(para)
        var_all_r = self.reverse_layer(var_all, lam)

        # NK = s_lab.shape[0]
        if self.training:
            n_cls = self.args.way
        else:
            n_cls = self.args.eval_way
        numm = int(s_lab.shape[0]/n_cls) + int(q_lab.shape[0]/n_cls)

        ## Reconstruction 
        # reconstruct all samples with class-code
        x_rec = self.decode(var_all, proto.squeeze().repeat(numm, 1))
        loss_rec = self.criterionL1(x_rec, ori)
        # loss_rec = self.criterionL2(x_rec, ori.detach())

        # reconstruct all samples with sample-code
        x_rec_all = self.decode(var_all, instance_embs)
        loss_rec2 = self.criterionL1(x_rec_all, ori)
        # loss_rec2 = self.criterionL2(x_rec_all, ori.detach())

        ## Translation
        # translate all samples with class-code
        # index = [i for i in range(self.args.way)]
        index = [i for i in range(n_cls)]
        random.shuffle(index)
        trans_lab = torch.arange(n_cls, dtype=torch.int16)[index].repeat(numm).type(torch.LongTensor).cuda()
        x_trans = self.decode(var_all, proto[:,index].squeeze().repeat(numm, 1))
        x_trans = self.avgpool1(x_trans).squeeze()
        # x_trans = nn.MaxPool2d(5)(x_trans).squeeze()

        # translate all samples with sample-code
        lab_all = torch.cat([s_lab,q_lab],dim=0)
        index2 = [i for i in range(lab_all.shape[0])]
        random.shuffle(index2)
        trans_lab2 = lab_all[index2]
        x_trans2 = self.decode(var_all, instance_embs[index2])
        x_trans2 = self.avgpool1(x_trans2).squeeze()
        # x_trans2 = nn.MaxPool2d(5)(x_trans2).squeeze()

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto2 = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
            proto2 = proto2.contiguous().view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
            logits = - torch.sum((proto2 - query) ** 2, 2) / self.args.temperature

            proto3 = proto.unsqueeze(1).expand(num_batch, x_trans.shape[0], num_proto, emb_dim).contiguous()
            proto3 = proto3.view(num_batch*x_trans.shape[0], num_proto, emb_dim) # (Nbatch x Nq, Nk, d)      
            logits_trans = - torch.sum((proto3 - x_trans.unsqueeze(1)) ** 2, 2) / self.args.temperature2
            loss_trans = F.cross_entropy(logits_trans, trans_lab.cuda())

            proto4 = proto.unsqueeze(1).expand(num_batch, instance_embs.shape[0], num_proto, emb_dim).contiguous()
            proto4 = proto4.view(num_batch*instance_embs.shape[0], num_proto, emb_dim) # (Nbatch x Nq, Nk, d)      
            logits_bce = - torch.sum((proto4 - var_all_r.unsqueeze(1)) ** 2, 2) / self.args.temperature2
            bce_loss = F.cross_entropy(logits_bce, lab_all.cuda())

            proto5 = proto.unsqueeze(1).expand(num_batch, lab_all.shape[0], num_proto, emb_dim).contiguous()
            proto5 = proto5.view(num_batch*lab_all.shape[0], num_proto, emb_dim) # (Nbatch x Nq, Nk, d)      
            logits_trans2 = - torch.sum((proto5 - x_trans2.unsqueeze(1)) ** 2, 2) / self.args.temperature2
            loss_trans2 = F.cross_entropy(logits_trans2, trans_lab2.cuda())

        else: # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

        if self.training:
            return logits, None, bce_loss, loss_rec, loss_rec2, loss_trans, loss_trans2
        else:

            return logits, []
