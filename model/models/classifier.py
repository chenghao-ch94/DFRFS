import torch
import torch.nn as nn
from model.utils import euclidean_metric
import torch.nn.functional as F

class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            hdim = 64
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

        self.fc = nn.Linear(hdim, args.num_class)

    def forward(self, data, labels=None, is_emb = False):
        _, out = self.encoder(data)
        if not is_emb:
            out = self.fc(out)
        return out

    def forward_proto(self, data_shot, data_query, way = None):
        if way is None:
            way = self.args.num_class

        _, proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        _, query = self.encoder(data_query.contiguous())
        
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        return logits_dist, logits_sim