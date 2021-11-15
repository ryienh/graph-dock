import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from utils import config

"""
Graph convolutional network for graph regression task
"""


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(
            dataset.num_node_features, config("model.hidden_layer_size_1")
        )
        self.conv2 = GCNConv(
            config("model.hidden_layer_size_1"), config("model.hidden_layer_size_2")
        )
        self.conv3 = GCNConv(config("model.hidden_layer_size_2"), 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
