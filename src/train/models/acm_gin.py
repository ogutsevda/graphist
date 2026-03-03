"""
(c) Adaptation of the code from https://github.com/SitaoLuan/ACM-GNN
"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import Union
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptPairTensor, OptTensor, Size
from torch_geometric.utils import scatter

from .utils import create_activation


class ACM_GIN(MessagePassing):
    def __init__(
        self,
        nn_lowpass: torch.nn.Module,
        nn_highpass: torch.nn.Module,
        nn_fullpass: torch.nn.Module,
        nn_lowpass_proj: torch.nn.Module,
        nn_highpass_proj: torch.nn.Module,
        nn_fullpass_proj: torch.nn.Module,
        nn_mix: torch.nn.Module,
        T: float = 3.0,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.nn_lowpass = nn_lowpass
        self.nn_highpass = nn_highpass
        self.nn_fullpass = nn_fullpass
        self.nn_lowpass_proj = nn_lowpass_proj
        self.nn_highpass_proj = nn_highpass_proj
        self.nn_fullpass_proj = nn_fullpass_proj
        self.nn_mix = nn_mix
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.T = T
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn_lowpass)
        reset(self.nn_highpass)
        reset(self.nn_fullpass)
        reset(self.nn_lowpass_proj)
        reset(self.nn_highpass_proj)
        reset(self.nn_fullpass_proj)
        reset(self.nn_mix)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        deg = scatter(edge_weight, edge_index[1], 0, out.size(0), reduce="sum")
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)
        out = deg_inv.view(-1, 1) * out

        x_r = x[1]
        if x_r is not None:
            out_lowpass = (x_r + out) / 2.0
            out_highpass = (x_r - out) / 2.0

        # compute embeddings for each filter
        out_lowpass = self.nn_lowpass(out_lowpass)
        out_highpass = self.nn_highpass(out_highpass)
        out_fullpass = self.nn_fullpass(x_r)
        # compute importance weights per filter
        alpha_lowpass = self.sigmoid(self.nn_lowpass_proj(out_lowpass))
        alpha_highpass = self.sigmoid(self.nn_highpass_proj(out_highpass))
        alpha_fullpass = self.sigmoid(self.nn_fullpass_proj(out_fullpass))
        alpha_cat = torch.concat([alpha_lowpass, alpha_highpass, alpha_fullpass], dim=1)
        alpha_cat = self.softmax(self.nn_mix(alpha_cat / self.T))

        out = alpha_cat[:, 0].view(-1, 1) * out_lowpass
        out = out + alpha_cat[:, 1].view(-1, 1) * out_highpass
        out = out + alpha_cat[:, 2].view(-1, 1) * out_fullpass

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"


class ACM_GIN_model(nn.Module):
    """ """

    def __init__(
        self, in_dim, out_dim, num_layers, hidden_dim, batchnorm, activation="relu"
    ):
        super(ACM_GIN_model, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.gnn_batchnorm = batchnorm
        self.out_dim = out_dim

        self.ACM_convs = nn.ModuleList()
        self.nns_lowpass = nn.ModuleList()
        self.nns_highpass = nn.ModuleList()
        self.nns_fullpass = nn.ModuleList()
        self.nns_lowpass_proj = nn.ModuleList()
        self.nns_highpass_proj = nn.ModuleList()
        self.nns_fullpass_proj = nn.ModuleList()
        self.nns_mix = nn.ModuleList()

        self.activation = create_activation(activation)

        for i in range(self.num_layers):
            # projection modules to compute importance weights
            for channel_proj_module in [
                self.nns_lowpass_proj,
                self.nns_highpass_proj,
                self.nns_fullpass_proj,
            ]:
                if i == self.num_layers - 1:
                    channel_proj_module.append(nn.Linear(self.out_dim, 1))
                else:
                    channel_proj_module.append(nn.Linear(self.hidden_dim, 1))
            # weights mixing module as attention mechanism
            self.nns_mix.append(nn.Linear(3, 3))

            # GIN embedding scheme per channel
            if i == 0:
                local_input_dim = in_dim
            else:
                local_input_dim = self.hidden_dim

            if i == self.num_layers - 1:
                local_out_dim = self.out_dim
            else:
                local_out_dim = self.hidden_dim

            for channel_module in [
                self.nns_lowpass,
                self.nns_highpass,
                self.nns_fullpass,
            ]:
                if self.gnn_batchnorm:
                    sequential = nn.Sequential(
                        nn.Linear(local_input_dim, self.hidden_dim),
                        nn.BatchNorm1d(self.hidden_dim),
                        self.activation,
                        nn.Linear(self.hidden_dim, local_out_dim),
                        nn.BatchNorm1d(local_out_dim),
                        self.activation,
                    )
                else:
                    sequential = nn.Sequential(
                        nn.Linear(local_input_dim, self.hidden_dim),
                        self.activation,
                        nn.Linear(self.hidden_dim, local_out_dim),
                        self.activation,
                    )

                channel_module.append(sequential)

            self.ACM_convs.append(
                ACM_GIN(
                    nn_lowpass=self.nns_lowpass[i],
                    nn_highpass=self.nns_highpass[i],
                    nn_fullpass=self.nns_fullpass[i],
                    nn_lowpass_proj=self.nns_lowpass_proj[i],
                    nn_highpass_proj=self.nns_highpass_proj[i],
                    nn_fullpass_proj=self.nns_fullpass_proj[i],
                    nn_mix=self.nns_mix[i],
                )
            )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, x, edge_index, edge_attr, return_hidden=False):
        outs = []
        for i in range(self.num_layers):
            x = self.ACM_convs[i](x=x, edge_index=edge_index, edge_weight=edge_attr)
            outs.append(x)
        if return_hidden:
            return x, outs
        else:
            return x


if __name__ == "__main__":
    acm_gin = ACM_GIN_model(46, 46, 2, 256, True)
    print(sum(p.numel() for p in acm_gin.parameters() if p.requires_grad))
    print("")
