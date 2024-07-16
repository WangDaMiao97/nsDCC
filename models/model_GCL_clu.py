import torch
import torch.nn as nn
import torch.nn.functional as F
from .Clu_Head import Clu_Head
from .Feat_Extra import Feat_Extra


class GCL_clu(nn.Module):
    def __init__(self, in_channel, out_channel = None, fea_dropout = 0.2,
                 clu_cfg = None, clu_dropout = 0.2, clu_batch_norm = False,
                 last_activation="softmax"):
        super(GCL_clu, self).__init__()

        # in_channel: The number of features of the input matrix
        self.Feat = Feat_Extra(in_channel=in_channel, out_channel=out_channel, dropout_ratio=fea_dropout)
        self.cluHead = Clu_Head(cfg=clu_cfg, drop_out=clu_dropout, last_activation=last_activation, batch_norm=clu_batch_norm)
        self.A = None
        for m in self.modules():
            self.weights_init(m)
        self.to(torch.device('cuda'))

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, seq_a, epoch=0):
        h_a = self.Feat(seq_a)
        h_a_norm = nn.functional.normalize(h_a, dim=1)

        pred_a = self.cluHead(h_a, epoch=epoch)

        return h_a_norm, pred_a

    def embed(self,  seq_a):

        h_a = self.Feat(seq_a)
        h_a_norm = nn.functional.normalize(h_a, dim=1)
        pred_a = self.cluHead(h_a)

        return h_a_norm.detach(), pred_a.detach()

