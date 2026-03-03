import torch
import torch.nn as nn
from typing import Sequence


# Helper classes
class StableSoftmax(nn.Module):
    def __init__(self, dim=0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return nn.LogSoftmax(dim=self.dim)(inputs).exp()


class Sum:
    def pool(self, features, **kwargs) -> int:
        return torch.sum(features, **kwargs)


class Mean:
    def pool(self, features, **kwargs):
        return torch.mean(features, **kwargs)


# Default MIL Graph
class DefaultMILGraph(nn.Module):
    def __init__(
        self,
        pointer: nn.Module,
        classifier: nn.Module,
    ):
        super().__init__()
        self.pointer = pointer
        self.classifier = classifier
        self.num_classes = classifier.output_dim

    def forward(self, features, bag_sizes):
        attention = self.pointer(features, bag_sizes)
        classifier_out_dict = self.classifier(features, attention, bag_sizes)
        bag_logits = classifier_out_dict["bag_logits"]
        ins_logits = classifier_out_dict.get("ins_logits", None)
        out = {}
        out["bag_logits"] = bag_logits
        if ins_logits is not None:
            out["ins_logits"] = ins_logits
        out["attention"] = attention
        return out


# Default Attention Module
class DefaultAttentionModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Sequence[int] = (),
        hidden_activation: nn.Module = nn.Tanh(),
        output_activation: nn.Module = StableSoftmax(dim=1),
    ):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.model = self.build_model()

    def build_model(self):
        nodes_by_layer = [self.input_dim] + list(self.hidden_dim) + [1]
        layers = []
        iterable = enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:]))
        for i, (nodes_in, nodes_out) in iterable:
            layer = nn.Linear(in_features=nodes_in, out_features=nodes_out, bias=True)
            layers.append(layer)
            if i < len(self.hidden_dim):
                layers.append(self.hidden_activation)
        model = nn.Sequential(*layers)
        return model

    def forward(self, features, bag_sizes):
        out = self.model(features)
        attentions = []
        start = 0
        for size in bag_sizes.tolist():
            bag_out = out[start : start + size]
            bag_attention = self.output_activation(bag_out.view(1, -1))
            attentions.append(bag_attention)
            start += size

        try:
            attention = torch.cat(attentions, dim=0)
        except RuntimeError:
            attention = [attn.squeeze(0) for attn in attentions]
        return attention
