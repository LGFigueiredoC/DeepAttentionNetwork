from torch_geometric.nn import GATConv
from torch import nn
import torch

class GAT_Policy (nn.Module):
    def __init__ (self, node_dim, edge_attr, hidden_dim=128):
        super().__init__()
        self.encoding = nn.Sequential (
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
        )
        heads = 8
        self.conv1 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_attr, heads=heads)
        self.conv2 = GATConv(hidden_dim*heads, hidden_dim, edge_dim=edge_attr, heads=heads, concat=False)


        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            #nn.Softmax(dim=1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.encoding(x)

        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = torch.relu(x)

        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = torch.relu(x)

        return self.policy(x)


