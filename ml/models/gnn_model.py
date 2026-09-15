import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# Ensure repository root is in sys.path for direct script execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.graph.graph_data import load_and_convert_trade_graph


class TradeGNN(nn.Module):
    """
    GraphSAGE-based Graph Neural Network for learning country node embeddings
    from global trade network topology and economic features.
    """

    def __init__(self, in_channels: int = 7, hidden_channels: int = 16, out_channels: int = 16):
        super(TradeGNN, self).__init__()

        # Layer 1: Aggregates 1-hop neighbor features (direct trading partners)
        self.conv1 = SAGEConv(in_channels, hidden_channels)

        # Layer 2: Aggregates 2-hop neighbor features (extended trade network)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TradeGNN.

        Parameters:
            x (Tensor): Node feature tensor of shape [num_nodes, in_channels].
            edge_index (Tensor): Graph connectivity tensor of shape [2, num_edges].

        Returns:
            Tensor: Learned node embeddings of shape [num_nodes, out_channels].
        """
        # 1-hop neighborhood convolution + ReLU non-linearity
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # 2-hop neighborhood convolution
        x = self.conv2(x, edge_index)

        return x


if __name__ == "__main__":
    input_filepath = "ml/data/processed_trade_edges_2023.csv"
    print(f"Loading trade graph data from '{input_filepath}'...\n")

    # Load PyG graph data
    pyg_data, node_to_idx, idx_to_node = load_and_convert_trade_graph(input_filepath)

    in_dim = pyg_data.x.shape[1]  # 7 features
    hidden_dim = 16
    out_dim = 16

    # Instantiate TradeGNN model
    model = TradeGNN(
        in_channels=in_dim,
        hidden_channels=hidden_dim,
        out_channels=out_dim
    )

    print("=" * 60)
    print("INITIALIZED TRADE GNN (GRAPHSAGE) MODEL")
    print("=" * 60)
    print(f"Model Structure:\n{model}")
    print("=" * 60)

    # Perform forward pass
    model.eval()
    with torch.no_grad():
        out_embeddings = model(pyg_data.x, pyg_data.edge_index)

    print("\nFORWARD PASS OUTPUT VERIFICATION:")
    print(f"Input Node Features shape (x):         {tuple(pyg_data.x.shape)}")
    print(f"Graph Edge Index shape (edge_index):   {tuple(pyg_data.edge_index.shape)}")
    print(f"Output Node Embedding shape:           {tuple(out_embeddings.shape)}")
    print("=" * 60)

    # Verification check
    expected_shape = (pyg_data.num_nodes, out_dim)
    assert tuple(out_embeddings.shape) == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {tuple(out_embeddings.shape)}"
    )

    print(f"\nSUCCESS: Output shape matches expected {expected_shape}.")
