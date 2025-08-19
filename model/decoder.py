import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import *

__all__ = [
    "MLP",
]

#####################################################################################
#  Common Decoder networks: MLP
#####################################################################################


class MLP(nn.Module):
    """
    Args:
        dim_in: input instance dimension.
        dim_emb: instance embedding dimension.
        num_cls: the number of class to predict.
        pooling: the type of MIL pooling, one of 'mean', 'max', and 'attention', default by attention pooling.
    """
    def __init__(self, dim_in=512, dim_emb=512, num_cls=2, num_feat_proj_layers=-1, drop_rate=0.25, 
        pred_head='default', **kwargs):
        super().__init__()
        assert pred_head in ['default']

        if num_feat_proj_layers < 0:
            self.feat_proj = nn.Identity()
            print("[MLP] no any feature projection layer.")
        else:
            self.feat_proj = create_mlp(
                in_dim=dim_in,
                hid_dims=[dim_emb] * (num_feat_proj_layers - 1),
                dropout=drop_rate,
                out_dim=dim_emb,
                end_with_fc=False
            )
            print("[MLP] use a layer for feature projection.")
        
        self.pred_head = nn.Linear(dim_emb, num_cls)

    def forward(self, x, **kwargs):
        """
        x: initial bag features, with shape B x C
           where B = 1 for batch size, and C is feature dimension.
        """
        x = self.feat_proj(x)
        logit = self.pred_head(x) # B x num_cls

        return logit
