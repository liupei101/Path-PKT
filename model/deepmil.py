import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import *


EPS = 1e-6
__all__ = [
    "DeepMIL", "DSMIL", "TransMIL", "DeepAttnMISL", "PatchGCN"
]


#####################################################################################
#  Common deep MIL networks: Max-pooling, Mean-pooling, ABMIL, DSMIL, and TransMIL 
#####################################################################################


class DeepMIL(nn.Module):
    """
    Deep Multiple Instance Learning for Bag-level Task.
    It is adapted from PANTHER (Song et al., CVPR, 2024):
        https://github.com/mahmoodlab/PANTHER/blob/main/src/mil_models/model_abmil.py

    Args:
        dim_in: input instance dimension.
        dim_emb: instance embedding dimension.
        num_cls: the number of class to predict.
        pooling: the type of MIL pooling, one of 'mean', 'max', and 'attention', default by attention pooling.
    """
    def __init__(self, dim_in=768, dim_emb=512, num_cls=2, dim_attn=384, num_feat_proj_layers=1, drop_rate=0.25, 
        pooling='attention', pred_head='default', **kwargs):
        super().__init__()
        assert pooling in ['mean', 'max', 'attention', 'gated_attention']
        assert pred_head in ['default']

        self.feat_proj = create_mlp(
            in_dim=dim_in,
            hid_dims=[dim_emb] * (num_feat_proj_layers - 1),
            dropout=drop_rate,
            out_dim=dim_emb,
            end_with_fc=False
        )
        
        if pooling == 'gated_attention':
            self.attention_net = Attn_Net_Gated(L=dim_emb, D=dim_attn, dropout=drop_rate)
        elif pooling == 'attention':
            self.attention_net = Attn_Net(L=dim_emb, D=dim_attn, dropout=drop_rate)
        else:
            self.attention_net = None

        self.agg_method = pooling
        
        self.pred_head = nn.Linear(dim_emb, num_cls)

    def forward_attention_pooling(self, X, attn_mask=None):
        # X is B x K x C (K is the number of instances)
        # num_head = 1 for ABMIL
        A = self.attention_net(X)  # B x K x num_head 
        A = torch.transpose(A, -2, -1)  # B x num_head x K

        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        A = F.softmax(A, dim=-1)  # softmax over K (the last dim)
        M = torch.bmm(A, X).squeeze(dim=1) # B x num_head x C --> B x C

        return M, A.squeeze(dim=1) # B x C, B x K

    def forward_slide_representation(self, X):
        assert X.shape[0] == 1
        X = self.feat_proj(X)
        
        # global pooling: B x K x C -> B x C
        if 'attention' in self.agg_method:
            out_feat, attn = self.forward_attention_pooling(X)
            return out_feat, attn
        elif self.agg_method == 'mean':
            out_feat = torch.mean(X, dim=1)
            return out_feat
        elif self.agg_method == 'max':
            out_feat, _ = torch.max(X, dim=1)
            return out_feat
        else:
            raise NotImplementedError("Not Implemented!")

    def forward(self, X, ret_with_attn=False, ret_bag_feat=False):
        """
        X: initial bag features, with shape B x K x C
           where B = 1 for batch size, K is the instance size of this bag, and C is feature dimension.
        """
        slide_results = self.forward_slide_representation(X)
        if isinstance(slide_results, tuple):
            bag_feat, attn = slide_results
        else:
            bag_feat = slide_results
        
        logit = self.pred_head(bag_feat) # B x num_cls

        if ret_bag_feat:
            return logit, bag_feat.detach() # B x num_cls, B x dim_feat

        if ret_with_attn:
            return logit, attn.detach() # B x num_cls, B x K
        
        return logit

################################################
# TransMIL, Shao et al., NeurIPS, 2021.
################################################
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            out, attn = self.attn(self.norm(x), return_attn=True)
            x = x + out
            return x, attn.detach()
        else:
            x = x + self.attn(self.norm(x))
            return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, dim_in=512, dim_emb=256, num_cls=2, **kwargs):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=dim_emb)
        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_emb), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_emb))
        self.num_cls = num_cls
        self.layer1 = TransLayer(dim=dim_emb)
        self.layer2 = TransLayer(dim=dim_emb)
        self.norm = nn.LayerNorm(dim_emb)
        self._fc2 = nn.Linear(dim_emb, self.num_cls)

    def forward(self, X, **kwargs):

        assert X.shape[0] == 1 # [1, n, 512], single bag
        
        h = self._fc1(X) # [B, n, dim_emb]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, dim_emb]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1) # token: 1 + H + add_length
        n1 = h.shape[1] # n1 = 1 + H + add_length

        #---->Translayer x1
        h = self.layer1(h) # [B, N, dim_emb]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) # [B, N, dim_emb]
        
        #---->Translayer x2
        if 'ret_with_attn' in kwargs and kwargs['ret_with_attn']:
            h, attn = self.layer2(h, return_attn=True) # [B, N, dim_emb]
            # attn shape = [1, n_heads, n2, n2], where n2 = padding + n1
            if add_length == 0:
                attn = attn[:, :, -n1, (-n1+1):]
            else:
                attn = attn[:, :, -n1, (-n1+1):(-n1+1+H)]
            attn = attn.mean(1).detach()
            assert attn.shape[1] == H
        else:
            h = self.layer2(h) # [B, N, dim_emb]
            attn = None

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, num_cls]

        if attn is not None:
            return logits, attn

        return logits

################################################
#    DeepAttnMISL (Yao et al., MedIA, 2020)
################################################


class DeepAttnMISL(nn.Module):
    """
        Adapted from the official implementation: 
        - DeepAttnMISL/blob/master/DeepAttnMISL_model.py
    """
    def __init__(self, dim_in=512, dim_emb=256, num_cls=1, num_clusters=8, dropout=0.25, **kwargs):
        super().__init__()
        print("[setup] got irrelevant kwargs:", kwargs)
        self.dim_emb = dim_emb
        self.num_clusters = num_clusters
        self.phis = nn.Sequential(*[nn.Conv2d(dim_in, dim_emb, 1), nn.ReLU()]) # It's equivalent to FC + ReLU
        self.pool1d = nn.AdaptiveAvgPool1d(1)    
        
        # attention pooling layer for clusters
        self.attention_net = nn.Sequential(*[
            nn.Linear(dim_emb, dim_emb), nn.ReLU(), nn.Dropout(dropout),
            Gated_Attention_Pooling(dim_emb, dim_emb, dropout=dropout)
        ])
        # output layer
        self.output_layer = nn.Linear(in_features=dim_emb, out_features=num_cls)

        print("[setup] initialized a DeepAttnMISL model.")

    def forward(self, X, cluster_id, *args):
        if cluster_id is not None:
            cluster_id = cluster_id.detach().cpu().numpy()
        X = X.squeeze(0) # assert batch_size = 1
        # FC Cluster layers + Pooling
        h_cluster = []
        for i in range(self.num_clusters):
            x_cluster_i = X[cluster_id==i].T.unsqueeze(0).unsqueeze(2) # [N, d] -> [1, d, 1, N]
            h_cluster_i = self.phis(x_cluster_i) # [1, d, 1, N] -> [1, d', 1, N]
            if h_cluster_i.shape[-1] == 0: # no any instance in this cluster
                h_cluster_i = torch.zeros((1, self.dim_emb, 1, 1), device=X.device)
            h_cluster.append(self.pool1d(h_cluster_i.squeeze(2)).squeeze(2))
        h_cluster = torch.stack(h_cluster, dim=1).squeeze(0) # [num_clusters, d']
        H, A = self.attention_net(h_cluster) # [1, d'], [1, num_clusters]
        out = self.output_layer(H)
        return out

################################################
#    PatchGCN (Chen et al., MICCAI, 2021)
################################################


class PatchGCN(nn.Module):
    """
        Adapted from the official implementation: 
        - https://github.com/mahmoodlab/Patch-GCN/blob/master/models/model_graph_mil.py#L116
    """
    def __init__(self, dim_in=512, dim_emb=256, num_cls=4, num_layers:int=3, edge_agg:str='spatial', dropout:float=0.25, **kwargs):
        super().__init__()
        from torch_geometric.nn import GENConv, DeepGCNLayer
        self.edge_agg = edge_agg
        self.num_layers = num_layers
        self.fc = nn.Sequential(*[nn.Linear(dim_in, dim_emb), nn.ReLU(), nn.Dropout(dropout)])
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            conv = GENConv(dim_emb, dim_emb, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(dim_emb, elementwise_affine=True)
            act = nn.ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=(i+1)%3)
            self.layers.append(layer)
        dim_sum = dim_emb * (1 + self.num_layers)
        self.path_phi = nn.Sequential(*[nn.Linear(dim_sum, dim_emb), nn.ReLU(), nn.Dropout(dropout)])
        # attention pooling layer for graph nodes
        self.path_attention_head = Gated_Attention_Pooling(dim_emb, dim_emb, dropout=dropout)
        # output layer
        self.output_layer = nn.Linear(in_features=dim_emb, out_features=num_cls)

        print("[setup] initialized a PatchGCN model.")

    def forward(self, x_path, *args):
        data = x_path
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent
        edge_attr = None
        x = self.fc(data.x)
        x_ = x 
        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)
        h_path = x_ # [N, dim_sum], dim_sum = dim_emb * (1 + num_layers)
        h_path = self.path_phi(h_path) 
        H, A = self.path_attention_head(h_path) # [1, d'], [1, N]
        out = self.output_layer(H)
        return out

################################################
# DSMIL, li et al., CVPR, 2021.
################################################

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats, **kwargs):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
    
    def forward(self, x, **kwargs):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, hid_size, output_class, dropout_v=0.0): # K, L, N
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, hid_size)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, hid_size)
        )
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=hid_size)
        
    def forward(self, feats, c, **kwargs): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 


class DSMIL(nn.Module):
    def __init__(self, dim_in=1024, dim_emb=256, num_cls=2, dim_attn=256, num_feat_proj_layers=0, drop_rate=0.25, **kwargs):
        super(DSMIL, self).__init__()
        if use_feat_proj:
            self.feat_proj = create_mlp(
                in_dim=dim_in,
                hid_dims=[dim_emb] * (num_feat_proj_layers - 1),
                dropout=drop_rate,
                out_dim=dim_emb,
                end_with_fc=False
            )
        else:
            self.feat_proj = None
        self.i_classifier = FCLayer(in_size=dim_in, out_size=num_cls)
        self.b_classifier = BClassifier(dim_in, dim_emb, num_cls, dropout_v=drop_rate)
        
    def forward(self, X, **kwargs):
        assert X.shape[0] == 1
        if self.feat_proj is not None:
            # ensure that feat_proj has been adapted 
            # for the input with shape of [B, N, C]
            X = self.feat_proj(X)
        X = X.squeeze(0) # to [N, C] for input to i and b classifier
        feats, classes = self.i_classifier(X)
        prediction_bag, A, B = self.b_classifier(feats, classes) # bag = [1, C], A = [N, C]
        
        max_prediction, _ = torch.max(classes, 0)
        logits = 0.5 * (prediction_bag + max_prediction) # logits = [1, C]

        if 'ret_with_attn' in kwargs and kwargs['ret_with_attn']:
            # average over class heads
            attn = A.detach()
            attn = attn.mean(dim=1).unsqueeze(0)
            return logits, attn
        
        return logits

