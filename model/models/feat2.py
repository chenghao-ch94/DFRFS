import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.models import FewShotModel
import math
import random

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

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output

class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)

class Conv_block_noac(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=0, groups=1):
        super(Conv_block_noac, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x

class FEAT2(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = 640
        elif args.backbone_class == 'Res18':
            hdim = 512
        else:
            raise ValueError('')
        
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)  
        
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.criterionL1 = nn.L1Loss()
        #self.criterionL2 = nn.MSELoss()

        self.reverse_layer = ReverseGrad()
        self.lambda_ = 0

        self.hdim = hdim
        self.enc_sim = nn.Sequential(
            Conv_block(hdim, hdim, kernel=(3,3), padding=(1,1)),
            Conv_block(hdim, hdim, kernel=(3,3), padding=(1,1)),
            Conv_block(hdim, hdim, kernel=(3,3), padding=(1,1)),
        )

        self.map_dim = 5
        self.decoder_fc = nn.Linear(hdim*2, hdim*self.map_dim*self.map_dim)
        self.decoder =  nn.Sequential(
                Conv_block(hdim, hdim, kernel=(3,3), padding=(1,1)),
                Conv_block(hdim, hdim, kernel=(3,3), padding=(1,1)),
                Conv_block(hdim, hdim, kernel=(3,3), padding=(1,1)),
        )

    def set_lambda(self, para):
        self.lambda_ = 2./(1+math.exp(-10.*para))-1
        return self.lambda_

    def decode(self, fea, code):
        fea_cat = self.decoder_fc(torch.cat([fea,code],dim=1)).view(-1, self.hdim, self.map_dim, self.map_dim)
        fea_rec = self.decoder(fea_cat)
        return fea_rec

    def _forward(self, support_idx, query_idx, ori=None, s_lab=None, q_lab=None, para=0): # not using q_lab for testing

        instance_embs = self.avgpool1(ori)
        instance_embs = instance_embs.view(instance_embs.size(0), -1)

        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))
    
        # get mean of the support
        proto = support.mean(dim=1) # Ntask x N x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        # self_fea = instance_embs
        var_all = self.avgpool1(self.enc_sim(ori)).squeeze()  #M*640
        
        #gradient reverse
        lam = self.set_lambda(para)
        var_all_r = self.reverse_layer(var_all, lam)

        proto = self.slf_attn(proto, proto, proto)
        
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto2 = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto2 = proto2.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
            logits = - torch.sum((proto2 - query) ** 2, 2) / self.args.temperature

        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)
    

        # for regularization
        if self.training:
            aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
            # apply the transformation over the Aug Task
            aux_emb = self.slf_attn(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
            aux_center = torch.mean(aux_emb, 2) # T x N x d
            
            # NK = s_lab.shape[0]
            numm = int(s_lab.shape[0]/self.args.way) + int(q_lab.shape[0]/self.args.way)

            ## Reconstruction 
            # reconstruct all samples with class-code
            x_rec = self.decode(var_all, aux_center.squeeze().repeat(numm, 1))
            loss_rec = self.criterionL1(x_rec, ori)
            # loss_rec = self.criterionL2(x_rec, ori.detach())

            # reconstruct all samples with sample-code
            x_rec_all = self.decode(var_all, instance_embs)
            loss_rec2 = self.criterionL1(x_rec_all, ori)
            # loss_rec2 = self.criterionL2(x_rec_all, ori.detach())

            ## Translation
            # translate all samples with class-code
            index = [i for i in range(self.args.way)]
            random.shuffle(index)
            trans_lab = torch.arange(self.args.way, dtype=torch.int16)[index].repeat(numm).type(torch.LongTensor).cuda()
            x_trans = self.decode(var_all, aux_center[:,index].squeeze().repeat(numm, 1))
            x_trans = self.avgpool1(x_trans).squeeze()

            # translate all samples with sample-code
            lab_all = torch.cat([s_lab,q_lab],dim=0)
            index2 = [i for i in range(lab_all.shape[0])]
            random.shuffle(index2)
            trans_lab2 = lab_all[index2]
            x_trans2 = self.decode(var_all, instance_embs[index2])
            x_trans2 = self.avgpool1(x_trans2).squeeze()


            if self.args.use_euclidean:
                aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
                aux_center2 = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
                aux_center2 = aux_center2.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
    
                logits_reg = - torch.sum((aux_center2 - aux_task) ** 2, 2) / self.args.temperature2

                proto3 = aux_center.unsqueeze(1).expand(num_batch, x_trans.shape[0], num_proto, emb_dim).contiguous()
                proto3 = proto3.view(num_batch*x_trans.shape[0], num_proto, emb_dim) # (Nbatch x Nq, Nk, d)      
                logits_trans = - torch.sum((proto3 - x_trans.unsqueeze(1)) ** 2, 2) / self.args.temperature2
                loss_trans = F.cross_entropy(logits_trans, trans_lab.cuda())

                proto4 = aux_center.unsqueeze(1).expand(num_batch, instance_embs.shape[0], num_proto, emb_dim).contiguous()
                proto4 = proto4.view(num_batch*instance_embs.shape[0], num_proto, emb_dim) # (Nbatch x Nq, Nk, d)      
                logits_bce = - torch.sum((proto4 - var_all_r.unsqueeze(1)) ** 2, 2) / self.args.temperature2
                bce_loss = F.cross_entropy(logits_bce, lab_all.cuda())

                proto5 = aux_center.unsqueeze(1).expand(num_batch, lab_all.shape[0], num_proto, emb_dim).contiguous()
                proto5 = proto5.view(num_batch*lab_all.shape[0], num_proto, emb_dim) # (Nbatch x Nq, Nk, d)      
                logits_trans2 = - torch.sum((proto5 - x_trans2.unsqueeze(1)) ** 2, 2) / self.args.temperature2
                loss_trans2 = F.cross_entropy(logits_trans2, trans_lab2.cuda())

            else:
                aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
                aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
    
                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)            

            return logits, logits_reg, bce_loss, loss_rec, loss_rec2, loss_trans, loss_trans2      
        else:
            return logits, []