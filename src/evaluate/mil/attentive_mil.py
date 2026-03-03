"""
(c) Adaptation of the code from https://github.com/bifold-pathomics/xMIL and
the supplementary material from https://openreview.net/forum?id=5dHQyEcYDgA
"""

import torch.nn as nn

from torch import cat
from typing import Sequence

from mil.utils import Sum


class AttentiveClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Sequence[int] = (),
        hidden_activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.dropout = dropout
        self.attentive_function = Sum()
        self.model = self.build_model()

    def build_model(self):
        nodes_by_layer = [self.input_dim] + list(self.hidden_dim) + [self.output_dim]
        layers = []
        iterable = enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:]))
        for i, (nodes_in, nodes_out) in iterable:
            layer = nn.Linear(in_features=nodes_in, out_features=nodes_out)
            layers.append(layer)
            if i < len(self.hidden_dim):
                layers.append(self.hidden_activation)
                if i == len(self.hidden_dim) - 1 and self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
        model = nn.Sequential(*layers)
        return model

    def forward(self, features, attention, bag_sizes):
        aggregated_features_list = []
        start = 0
        for idx, size in enumerate(bag_sizes.tolist()):
            end = start + size
            bag_features = features[start:end]
            bag_attention = attention[idx].unsqueeze(-1)
            attended_features = bag_features * bag_attention
            aggregated = self.attentive_function.pool(
                attended_features, dim=0, keepdim=True
            )
            aggregated_features_list.append(aggregated)
            start = end
        aggregated_features = cat(aggregated_features_list, dim=0)
        bag_logits = self.model(aggregated_features)
        return {"bag_logits": bag_logits}
