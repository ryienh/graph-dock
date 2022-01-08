import torch
from torch_geometric.nn import GINConv, PNAConv, global_mean_pool


"""
Graph convolutional network for graph regression task
    Partially adapted from: https://web.stanford.edu/class/cs224w/index.html

For questions or comments, contact rhosseini@anl.gov
"""


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
            torch.nn.Linear(hidden_dim, int(hidden_dim / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim / 2), 1),
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

    def loss(self, pred, label):
        return torch.nn.functional.mse_loss(pred, label)


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
            torch.nn.Linear(hidden_dim, int(hidden_dim / 2)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden_dim / 2), 1),
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

    def loss(self, pred, label):
        return torch.nn.functional.mse_loss(pred, label)
