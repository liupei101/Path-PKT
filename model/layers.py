import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_mlp(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.,
               out_dim=None, end_with_fc=True, bias=True):

    layers = []
    if len(hid_dims) < 0:
        mlp = nn.Identity()
    elif len(hid_dims) >= 0:
        if len(hid_dims) > 0:
            for hid_dim in hid_dims:
                layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if not end_with_fc:
            layers.append(act)
        mlp = nn.Sequential(*layers)
    return mlp


class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super().__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, n_classes)]

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x)  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super().__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout)]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid(),
                            nn.Dropout(dropout)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A


class ModGRUCell(nn.Module):
    """Modified gated recurrent unit (GRU) cell"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size * 3))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 3))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size * 3))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size * 3))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x, h_t_minus_1):
        idx = self.hidden_size * 2
        gates_ih = torch.mm(x, self.weight_ih) + self.bias_ih
        gates_hh = torch.mm(h_t_minus_1, self.weight_hh[:, :idx]) + self.bias_hh[:idx]
        resetgate_i, updategate_i, output_i = gates_ih.chunk(3, dim=1)
        resetgate_h, updategate_h = gates_hh.chunk(2, dim=1)
        r = torch.sigmoid(resetgate_i + resetgate_h)
        z = torch.sigmoid(updategate_i + updategate_h)
        h_tilde = output_i + (torch.mm((r * h_t_minus_1), self.weight_hh[:, idx:]) + self.bias_hh[idx:])
        h = (1 - z) * h_t_minus_1 + z * h_tilde
        return h


class ModGRUCell_UGate(nn.Module):
    """Modified gated recurrent unit (GRU) cell only with update gate"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size * 2))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size * 2))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x, h_t_minus_1):
        gates_ih = torch.mm(x, self.weight_ih) + self.bias_ih
        updategate_i, output_i = gates_ih.chunk(2, dim=1)
        updategate_h = torch.mm(h_t_minus_1, self.weight_hh) + self.bias_hh
        z = torch.sigmoid(updategate_i + updategate_h)
        h = (1 - z) * h_t_minus_1 + z * output_i
        return h


class ModGRUCell_RGate(nn.Module):
    """Modified gated recurrent unit (GRU) cell only with reset gate"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, hidden_size * 2))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 2))
        self.bias_ih = nn.Parameter(torch.Tensor(hidden_size * 2))
        self.bias_hh = nn.Parameter(torch.Tensor(hidden_size * 2))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x, h_t_minus_1):
        idx = self.hidden_size
        gates_ih = torch.mm(x, self.weight_ih) + self.bias_ih
        resetgate_i, output_i = gates_ih.chunk(2, dim=1)
        resetgate_h = torch.mm(h_t_minus_1, self.weight_hh[:, :idx]) + self.bias_hh[:idx]
        r = torch.sigmoid(resetgate_i + resetgate_h)
        h_tilde = output_i + (torch.mm((r * h_t_minus_1), self.weight_hh[:, idx:]) + self.bias_hh[idx:])
        return h_tilde


class ModGRU(nn.Module):
    """Multi-layer GRUCell"""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, gate_type='R+U'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        if 'R' in gate_type and 'U' in gate_type:
            GRUCell = ModGRUCell
        elif 'R' in gate_type:
            GRUCell = ModGRUCell_RGate
        elif 'U' in gate_type:
            GRUCell = ModGRUCell_UGate
        else:
            pass

        layers = [GRUCell(input_size, hidden_size)]
        for i in range(num_layers - 1):
            layers += [GRUCell(hidden_size, hidden_size)]
        self.net = nn.Sequential(*layers)
        print(f"[ModGRU] gate_type = {gate_type}.")

    def forward(self, x, init_state=None):
        # Input and output size: (seq_length, batch_size, input_size)
        # State size: (num_layers, batch_size, hidden_size)
        if self.batch_first:
            x = x.transpose(0, 1)

        self.h = torch.zeros(x.size(0), self.num_layers, x.size(1), self.hidden_size).to(x.device)
        if init_state is not None:
            self.h[0, :] = init_state

        inputs = x
        for i, cell in enumerate(self.net):  # Layers
            h_t = self.h[0, i].clone()
            for t in range(x.size(0)):  # Sequences
                h_t = cell(inputs[t], h_t)
                self.h[t, i] = h_t
            inputs = self.h[:, i].clone()

        if self.batch_first:
            return self.h[:, -1].transpose(0, 1), self.h[-1]

        return self.h[:, -1], self.h[-1]