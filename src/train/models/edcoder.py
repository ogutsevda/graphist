"""
(c) Adaptation of the code from https://github.com/THUDM/GraphMAE
"""

from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from torch_geometric.utils import add_self_loops

from .acm_gin import ACM_GIN_model


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()

    return loss


def setup_module(
    m_type,
    in_dim,
    out_dim,
    num_hidden,
    num_layers,
    activation,
    batchnorm,
) -> nn.Module:

    if m_type == "acm_gin":
        mod = ACM_GIN_model(
            int(in_dim),
            int(out_dim),
            num_layers,
            int(num_hidden),
            batchnorm,
            activation=activation,
        )
    else:
        raise NotImplementedError

    return mod


class PreModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_in_dim: int,
        num_hidden: int,
        num_layers: int,
        nhead: int,
        nhead_out: int,
        activation: str,
        feat_drop: float,
        attn_drop: float,
        negative_slope: float,
        residual: bool,
        norm: Optional[str],
        mask_rate: float = 0.3,
        encoder_type: str = "gat",
        decoder_type: str = "gat",
        loss_fn: str = "sce",
        drop_edge_rate: float = 0.0,
        replace_rate: float = 0.1,
        alpha_l: float = 2,
        concat_hidden: bool = False,
        batchnorm=False,
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0

        enc_num_hidden = num_hidden
        enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden

        # Build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            in_dim=in_dim,
            out_dim=enc_num_hidden,
            num_hidden=enc_num_hidden,
            num_layers=num_layers,
            activation=activation,
            batchnorm=batchnorm,
        )

        # Build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            in_dim=dec_in_dim,
            out_dim=in_dim,
            num_hidden=dec_num_hidden,
            num_layers=1,
            activation=activation,
            batchnorm=batchnorm,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(
                dec_in_dim * num_layers, dec_in_dim, bias=False
            )
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # Setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, x, mask_rate=0.3, virtual_node_index=None):
        num_nodes = x.shape[0]
        all_indices = torch.arange(num_nodes, device=x.device)

        # Remove virtual node index from masking candidates
        if virtual_node_index is not None:
            all_indices = all_indices[~torch.isin(all_indices, virtual_node_index)]

        perm = all_indices[torch.randperm(len(all_indices), device=x.device)]

        # random masking
        num_mask_nodes = int(mask_rate * len(perm))
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[
                perm_mask[: int(self._mask_token_rate * num_mask_nodes)]
            ]
            noise_nodes = mask_nodes[
                perm_mask[-int(self._replace_rate * num_mask_nodes) :]
            ]
            noise_to_be_chosen = torch.randperm(len(perm), device=x.device)[
                :num_noise_nodes
            ]
            noise_to_be_chosen = all_indices[noise_to_be_chosen]

            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def forward(self, batch):
        # ---- attribute reconstruction ----
        x, edge_index, edge_attr, virtual_node_index, batch = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            getattr(batch, "virtual_node_index", None),
            batch.batch,
        )
        loss = self.mask_attr_prediction(
            x, edge_index, edge_attr, batch, virtual_node_index
        )
        return loss

    def mask_attr_prediction(self, x, edge_index, edge_attr, batch, virtual_node_index):

        use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
            x,
            self._mask_rate,
            virtual_node_index,
        )

        if self._drop_edge_rate > 0:
            use_edge_index, masked_edges = dropout_edge(
                edge_index, self._drop_edge_rate
            )
            use_edge_attr = edge_attr[masked_edges]
            use_edge_index, use_edge_attr = add_self_loops(
                use_edge_index, use_edge_attr, fill_value="min"
            )
        else:
            use_edge_index = edge_index
            use_edge_attr = edge_attr

        enc_rep, all_hidden = self.encoder(
            use_x, use_edge_index, use_edge_attr, return_hidden=True
        )
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "linear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(rep, use_edge_index, use_edge_attr)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)

        return loss

    def embed(self, x, edge_index, edge_attr, batch):
        if self._concat_hidden:
            enc_rep, all_hidden = self.encoder(
                x, edge_index, edge_attr, return_hidden=True
            )
            enc_rep = torch.cat(all_hidden, dim=1)
        else:
            enc_rep = self.encoder(x, edge_index, edge_attr)
        rep = self.encoder_to_decoder(enc_rep)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
