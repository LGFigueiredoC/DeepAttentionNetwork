from torch_geometric import nn
import torch

class GAT_Policy (nn.Module):
    def __init__ (self, node_dim, hidden_dim=128):
        super().__init__()

        self.encoding = nn.Sequential (
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
        )

        self.conv1 = nn.GATConv(hidden_dim, hidden_dim, heads=8)
        self.conv2 = nn.GATConv(hidden_dim, hidden_dim, heads=8)


        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLu(),
            nn.Linear(hidden_dim, 1)
        )

        def forward(self, data):
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

            x = self.encoding(x)

            x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = torch.relu(x)

            x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = torch.relu(x)

            return self.policy(x)


