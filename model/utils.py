from typing import List, Optional
import torch
import torch.nn as nn
import math

from utils.func import parse_str_dims
from .deepmil import DeepMIL, DSMIL, TransMIL, DeepAttnMISL, PatchGCN
from .deepmil_tfl import DeepMIL_TFL_MFF, DeepMIL_TFL_MoE
from .decoder import MLP

##########################################
# Functions for loading models 
##########################################
def load_model(arch:str, **kws):
    if arch == 'DeepMIL':
        assert 'network' in kws, "Please specify a network for a DeepMIL arch."
        network = kws['network']
        if network == 'ABMIL':
            return Deep_ABMIL(**kws)
        elif network == 'MaxMIL':
            return Deep_MaxMIL(**kws)
        elif network == 'MeanMIL':
            return Deep_MeanMIL(**kws)
        elif network == 'DSMIL':
            return Deep_DSMIL(**kws)
        elif network == 'TransMIL':
            return Deep_TransMIL(**kws)
        elif network == 'DeepAttnMISL':
            return Deep_AttnMISL(**kws)
        elif network == 'PatchGCN':
            return Deep_PatchGCN(**kws)
        else:
            pass
    elif arch == 'Decoder':
        assert 'network' in kws, "Please specify a network for a Decoder arch."
        network = kws['network']
        if network == 'MLP':
            return Decoder_MLP(**kws)
        else:
            pass
    elif arch == 'MFFTFL':
        assert 'network' in kws, "Please specify a network for a MFFTFL arch."
        network = kws['network']
        if network == 'MFF':
            return Deep_MFF_TFL(**kws)
        else:
            pass
    elif arch == 'MoETFL':
        assert 'network' in kws, "Please specify a network for a MoETFL arch."
        network = kws['network']
        if network == 'ABMIL-MoE':
            return Deep_MoE_TFL(**kws)
        else:
            pass
    else:
        raise NotImplementedError("Backbone {} cannot be recognized".format(arch))

def Deep_MaxMIL(**kws):
    assert 'pooling' in kws and kws['pooling'] == 'max'
    model = DeepMIL(**kws)
    return model

def Deep_MeanMIL(**kws):
    assert 'pooling' in kws and kws['pooling'] == 'mean'
    model = DeepMIL(**kws)
    return model

def Deep_ABMIL(**kws):
    assert 'pooling' in kws and kws['pooling'] in ['attention', 'gated_attention']
    model = DeepMIL(**kws)
    return model

def Deep_DSMIL(**kws):
    model = DSMIL(**kws)
    return model

def Deep_TransMIL(**kws):
    model = TransMIL(**kws)
    return model

def Deep_PatchGCN(**kws):
    model = PatchGCN(**kws)
    return model

def Deep_AttnMISL(**kws):
    model = DeepAttnMISL(**kws)
    return model

def Decoder_MLP(**kws):
    model = MLP(**kws)
    return model

def Deep_MFF_TFL(**kws):
    assert 'mff' in kws
    model = DeepMIL_TFL_MFF(**kws)
    return model

def Deep_MoE_TFL(**kws):
    assert 'expert_size' in kws and 'expert_network' in kws
    model = DeepMIL_TFL_MoE(**kws)
    return model

##########################################
# Model weight initialization functions
##########################################
@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

@torch.no_grad()
def general_init_weight(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm1d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.BatchNorm2d):
        init_pytorch_defaults(m, version='041')
    elif isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)

def init_pytorch_defaults(m, version='041'):
    '''
    copied from AMDIM repo: https://github.com/Philip-Bachman/amdim-public/
    note from me: haven't checked systematically if this improves results
    '''
    if version == '041':
        # print('init.pt041: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            for k in m.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == '100':
        # print('init.pt100: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.Conv2d):
            n = m.in_channels
            init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.affine:
                m.weight.data.uniform_()
                m.bias.data.zero_()
        else:
            assert False
    elif version == 'custom':
        # print('init.custom: {0:s}'.format(str(m.weight.data.size())))
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        else:
            assert False
    else:
        assert False

