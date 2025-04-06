import torch.nn as nn
from torchvision import models


class SimpleNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_layers: list = [
            512,
            256,
            128,
        ],
        dropout: float = 0.5,
        activation: str = "relu",
    ):
        super(SimpleNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.activation = activation

        layers = []
        in_features = input_size

        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            in_features = out_features

        layers.append(nn.Linear(in_features, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
