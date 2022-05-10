import torch
from torch_geometric.nn import (
    GINConv,
    PNAConv,
    AttentiveFP,
    FiLMConv,
    GATv2Conv,
    global_mean_pool,
    SAGPooling,
)

from filmv2Conv import FiLMv2Conv


"""
Graph convolutional network for graph regression task
    Partially adapted from: https://web.stanford.edu/class/cs224w/index.html

For questions or comments, contact rhosseini@anl.gov
"""


class AttentiveFPREG(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout,
        num_conv_layers,
        num_out_channels,
        edge_dim,
        num_timesteps,
    ):

        super(AttentiveFPREG, self).__init__()

        # call attentive model
        self.attentive_fp = AttentiveFP(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=num_out_channels,
            edge_dim=edge_dim,
            num_layers=num_conv_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
        # fully connected layers
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(num_out_channels, 1),
        )

    def forward(self, data):
        x, edge_index, batch, edge_attr = (
            data.x,
            data.edge_index,
            data.batch,
            data.edge_attr,
        )
        edge_attr = torch.ones((edge_index.shape[1], 1)).cuda()
        if data.num_node_features == 0:
            print("Warning: No node features detected.")
            x = torch.ones(data.num_nodes, 1)

        # call model
        x = self.attentive_fp(x, edge_index, edge_attr, batch)

        # MLP
        x = self.post_mp(x)

        return x


class FiLMReg(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_conv_layers, v2=True):

        super(FiLMReg, self).__init__()

        self.dropout = dropout
        self.num_layers = num_conv_layers

        self.usev2 = v2

        self.convs = torch.nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
            self.lns.append(
                torch.nn.LayerNorm(hidden_dim)
            )  # one less lns than conv bc no lns after final conv

        self.conv_dropout = torch.nn.Dropout(p=self.dropout)
        self.ReLU = torch.nn.ReLU()

        # post-message-passing
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
        )

    def build_conv_model(self, input_dim, hidden_dim):
        if self.usev2 is False:
            return FiLMConv(in_channels=input_dim, out_channels=hidden_dim)
        else:
            return FiLMv2Conv(in_channels=input_dim, out_channels=hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.to(torch.float32)
        edge_index = edge_index.to(torch.long)
        if data.num_node_features == 0:
            print("Warning: No node features detected.")
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            # emb = x
            x = self.ReLU(x)
            x = self.conv_dropout(x)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = global_mean_pool(x, batch)
        x = self.post_mp(x)

        return x


class NovelReg(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_conv_layers, heads):

        super(NovelReg, self).__init__()

        self.dropout = dropout
        self.num_layers = num_conv_layers

        assert self.num_layers % 2 == 0
        idx = 0

        self.convs = torch.nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim, heads, idx))
        idx += 1
        self.lns = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim, heads, idx))
            self.lns.append(
                torch.nn.LayerNorm(hidden_dim)
            )  # one less lns than conv bc no lns after final conv
            idx += idx

        self.conv_dropout = torch.nn.Dropout(p=self.dropout)
        self.ReLU = torch.nn.ReLU()

        # post-message-passing
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
        )

    def build_conv_model(self, input_dim, hidden_dim, heads, idx):
        if idx % 2 == 0:
            return GATv2Conv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False,
            )
        else:
            return FiLMConv(in_channels=input_dim, out_channels=hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.to(torch.float32)
        edge_index = edge_index.to(torch.long)
        if data.num_node_features == 0:
            print("Warning: No node features detected.")
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            # emb = x
            x = self.ReLU(x)
            x = self.conv_dropout(x)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = global_mean_pool(x, batch)
        x = self.post_mp(x)

        return x


class GATREG(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_conv_layers, heads):

        super(GATREG, self).__init__()

        self.dropout = dropout
        self.num_layers = num_conv_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim, heads))
        self.lns = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim, heads))
            self.lns.append(
                torch.nn.LayerNorm(hidden_dim)
            )  # one less lns than conv bc no lns after final conv

        self.conv_dropout = torch.nn.Dropout(p=self.dropout)
        self.ReLU = torch.nn.ReLU()

        # post-message-passing
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
        )

    def build_conv_model(self, input_dim, hidden_dim, heads):
        return GATv2Conv(
            in_channels=input_dim, out_channels=hidden_dim, heads=heads, concat=False
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.to(torch.float32)
        edge_index = edge_index.to(torch.long)
        if data.num_node_features == 0:
            print("Warning: No node features detected.")
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            # emb = x
            x = self.ReLU(x)
            x = self.conv_dropout(x)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = global_mean_pool(x, batch)
        x = self.post_mp(x)

        return x


class PNAREG(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_conv_layers, deg):

        super(PNAREG, self).__init__()

        self.dropout = dropout
        self.num_layers = num_conv_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim, deg))
        self.lns = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim, deg))
            self.lns.append(
                torch.nn.LayerNorm(hidden_dim)
            )  # one less lns than conv bc no lns after final conv

        self.conv_dropout = torch.nn.Dropout(p=self.dropout)
        self.ReLU = torch.nn.ReLU()

        # post-message-passing
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
        )

    def build_conv_model(self, input_dim, hidden_dim, deg):
        return PNAConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            aggregators=["mean", "max", "min", "std"],
            scalers=["identity", "amplification", "attenuation"],
            deg=deg,
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            print("Warning: No node features detected.")
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            # emb = x
            x = self.ReLU(x)
            x = self.conv_dropout(x)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        # pooling
        x = global_mean_pool(x, batch)

        # MLP
        x = self.post_mp(x)

        return x


class GINREG(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_conv_layers):

        super(GINREG, self).__init__()

        self.dropout = dropout
        self.num_layers = num_conv_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
            self.lns.append(
                torch.nn.LayerNorm(hidden_dim)
            )  # one less lns than conv bc no lns after final conv

        self.conv_dropout = torch.nn.Dropout(p=self.dropout)
        self.ReLU = torch.nn.ReLU()

        # post-message-passing
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
        )

    def build_conv_model(self, input_dim, hidden_dim):
        return GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.to(torch.float32)
        edge_index = edge_index.to(torch.long)
        if data.num_node_features == 0:
            print("Warning: No node features detected.")
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            # emb = x
            x = self.ReLU(x)
            x = self.conv_dropout(x)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = global_mean_pool(x, batch)
        x = self.post_mp(x)

        return x
