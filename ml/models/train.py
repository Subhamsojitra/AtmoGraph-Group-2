import os
import sys
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling

# Ensure repository root is in sys.path for direct script execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.graph.graph_data import load_and_convert_trade_graph
from ml.models.gnn_model import TradeGNN


def compute_link_logits(embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Compute link score logits using dot-product similarity between node embeddings.

    Parameters:
        embeddings (Tensor): Node embedding tensor of shape [num_nodes, dim].
        edge_index (Tensor): Pairwise node indices of shape [2, num_edges].

    Returns:
        Tensor: Logits score vector of shape [num_edges].
    """
    src_indices = edge_index[0]
    dst_indices = edge_index[1]
    src_embeddings = embeddings[src_indices]
    dst_embeddings = embeddings[dst_indices]
    return (src_embeddings * dst_embeddings).sum(dim=-1)


def train_trade_gnn(
    input_filepath: str = "ml/data/processed_trade_edges_2023.csv",
    epochs: int = 100,
    lr: float = 0.01,
    model_save_path: str = "ml/models/trade_gnn.pt",
    embeddings_save_path: str = "ml/models/node_embeddings.pt",
):
    """
    Train TradeGNN using self-supervised link prediction on trade network data.
    """
    # Set seed for reproducible results
    torch.manual_seed(42)

    # 1. Load PyG Graph Data from graph_data.py pipeline
    pyg_data, node_to_idx, idx_to_node = load_and_convert_trade_graph(input_filepath)
    num_nodes = pyg_data.num_nodes
    pos_edge_index = pyg_data.edge_index
    num_pos_edges = pos_edge_index.size(1)

    # Normalize node feature matrix for numerical stability (large trade USD values)
    x_mean = pyg_data.x.mean(dim=0, keepdim=True)
    x_std = pyg_data.x.std(dim=0, keepdim=True) + 1e-6
    x_norm = (pyg_data.x - x_mean) / x_std

    print("=" * 60)
    print("STARTING TRADE GNN TRAINING PIPELINE")
    print("=" * 60)
    print(f"Graph Nodes:             {num_nodes}")
    print(f"Positive Trade Edges:    {num_pos_edges}")
    print(f"Input Node Features:     {pyg_data.x.shape[1]}")
    print(f"Embedding Output Dim:    16")
    print(f"Hyperparameters:         epochs={epochs}, lr={lr}")
    print("=" * 60)

    # 2. Instantiate Model, Loss Function, and Optimizer
    in_dim = pyg_data.x.shape[1]
    model = TradeGNN(in_channels=in_dim, hidden_channels=16, out_channels=16)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3. Model Training Loop
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Forward pass to derive node embeddings
        embeddings = model(x_norm, pos_edge_index)

        # Sample negative edges (non-existing trade pairs)
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_pos_edges,
        )

        # Calculate positive and negative link scores via dot-product
        pos_logits = compute_link_logits(embeddings, pos_edge_index)
        neg_logits = compute_link_logits(embeddings, neg_edge_index)

        # Combine logits and target binary labels
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        pos_labels = torch.ones(pos_logits.size(0), dtype=torch.float32)
        neg_labels = torch.zeros(neg_logits.size(0), dtype=torch.float32)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        # Calculate loss and backpropagate gradients
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Log training loss progress
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | BCE Loss: {loss.item():.6f}")

    print("=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    # 4. Generate Final 16-dimensional Node Embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings = model(x_norm, pos_edge_index)

    # Construct country name -> embedding tensor dictionary
    country_embeddings = {
        country: final_embeddings[idx] for country, idx in node_to_idx.items()
    }

    print("\nFinal Embedding Summary:")
    print(f"Total Country Embeddings: {len(country_embeddings)}")
    sample_country = list(country_embeddings.keys())[0]
    sample_tensor = country_embeddings[sample_country]
    print(f"Sample Country Embedding ('{sample_country}'): shape {tuple(sample_tensor.shape)}")

    # 5. Save Artifacts to Disk
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Save model weights state_dict
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved TradeGNN model weights -> {model_save_path}")

    # Save country node embeddings dictionary
    torch.save(
        {
            "embeddings": country_embeddings,
            "node_to_idx": node_to_idx,
            "idx_to_node": idx_to_node,
            "raw_embedding_matrix": final_embeddings,
        },
        embeddings_save_path,
    )
    print(f"Saved node embeddings tensor dictionary -> {embeddings_save_path}")
    print("=" * 60)
    print("Note: This pipeline performs self-supervised GNN link prediction training.")
    print("It does not directly model dynamic shock propagation or ripple effects.")
    print("=" * 60)


if __name__ == "__main__":
    train_trade_gnn()
