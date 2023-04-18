import torch
import torch.nn as nn
import numpy as np

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet() 
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        else:
            raise ValueError('')

    def split_instances(self, data):
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))

    def forward(self, x, lab_s, lab, get_feature=False, epo=0):
        if get_feature:
            # get feature with the provided embeddings
            _, emd_c = self.encoder(x)
            return emd_c
        else:
            # feature extraction
            x = x.squeeze(0)
            emd_v, _ = self.encoder(x)
            support_idx, query_idx = self.split_instances(x)

        if self.training:
            
            logits, logits_reg, bce_loss, loss_rec, loss_rec2, loss_trans, loss_trans2 = self._forward(support_idx, query_idx, emd_v, lab_s, lab, epo)
            return logits, logits_reg, bce_loss, loss_rec, loss_rec2, loss_trans, loss_trans2
        
            ### For original FEAT and ProtoNet
            # logits, logits_reg = self._forward(instance_embs, support_idx, query_idx)
            # return logits, logits_reg
        else:
            # logits, _ = self._forward(instance_embs, support_idx, query_idx, emd_v, lab_s, lab)
            logits, _ = self._forward(support_idx, query_idx, emd_v, lab_s, None)
            return logits
        
            ### For original FEAT and ProtoNet
            # logits = self._forward(instance_embs, support_idx, query_idx)
            # return logits
        

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')