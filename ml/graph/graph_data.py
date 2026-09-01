import os
import sys
import torch
import networkx as nx
import pandas as pd
from torch_geometric.data import Data

# Ensure repository root is in sys.path for direct script execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.graph.graph_builder import build_trade_graph
from ml.graph.features import extract_node_features


def convert_graph_to_pyg_data(G: nx.DiGraph, df_features: pd.DataFrame):
    """
    Convert a NetworkX trade graph and node features DataFrame into a
    PyTorch Geometric Data object.

    Parameters:
        G (nx.DiGraph): NetworkX directed trade graph.
        df_features (pd.DataFrame): Node features extracted from features.py.

    Returns:
        tuple: (pyg_data, node_to_idx, idx_to_node)
            - pyg_data (Data): PyG Data object with x, edge_index, edge_attr.
            - node_to_idx (dict): Mapping from country name to integer index.
            - idx_to_node (dict): Mapping from integer index to country name.
    """
    # 1. Create node mapping preserving deterministic node order
    nodes = list(G.nodes())
    node_to_idx = {country: idx for idx, country in enumerate(nodes)}
    idx_to_node = {idx: country for country, idx in node_to_idx.items()}

    # 2. Reorder feature DataFrame to match node_to_idx mapping
    feature_cols = [
        "in_degree",
        "out_degree",
        "trading_partners",
        "weighted_in_degree_usd",
        "weighted_out_degree_usd",
        "total_trade_volume_usd",
        "degree_centrality",
    ]

    # Map country column to index and sort
    df_ordered = df_features.copy()
    df_ordered["node_idx"] = df_ordered["country"].map(node_to_idx)
    df_ordered = df_ordered.sort_values(by="node_idx").reset_index(drop=True)

    # Convert node feature matrix to PyTorch FloatTensor
    x_matrix = df_ordered[feature_cols].values
    x = torch.tensor(x_matrix, dtype=torch.float32)

    # 3. Construct edge_index and edge_attr tensors
    sources = []
    targets = []
    edge_attributes = []

    for src, dst, data in G.edges(data=True):
        sources.append(node_to_idx[src])
        targets.append(node_to_idx[dst])

        # Primary edge weight attribute (trade_value_usd / edge_weight)
        weight = float(data.get("edge_weight", data.get("trade_value_usd", 0.0)))
        log_val = float(data.get("log_trade_value", 0.0))
        edge_attributes.append([weight, log_val])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float32)

    # 4. Wrap into PyG Data object
    pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return pyg_data, node_to_idx, idx_to_node


def load_and_convert_trade_graph(filepath: str):
    """
    Pipeline wrapper to load trade edge CSV, build NetworkX graph,
    calculate node features, and return PyG Data object + mappings.
    """
    G = build_trade_graph(filepath)
    df_features = extract_node_features(G)
    pyg_data, node_to_idx, idx_to_node = convert_graph_to_pyg_data(G, df_features)
    return pyg_data, node_to_idx, idx_to_node


if __name__ == "__main__":
    input_filepath = "ml/data/processed_trade_edges_2023.csv"
    print(f"Loading trade graph and building PyG Data from '{input_filepath}'...\n")

    pyg_data, node_to_idx, idx_to_node = load_and_convert_trade_graph(input_filepath)

    print("=" * 60)
    print("PYTORCH GEOMETRIC GRAPH CONVERSION COMPLETED")
    print("=" * 60)
    print(f"Total Nodes:               {pyg_data.num_nodes}")
    print(f"Total Edges:               {pyg_data.num_edges}")
    print(f"Node Feature Matrix (x):   shape {tuple(pyg_data.x.shape)}")
    print(f"Edge Index (edge_index):   shape {tuple(pyg_data.edge_index.shape)}")
    print(f"Edge Attributes (edge_attr): shape {tuple(pyg_data.edge_attr.shape)}")
    print("=" * 60)

    print("\nSample Node-to-Index Mapping (First 5):")
    for country in list(node_to_idx.keys())[:5]:
        print(f"  '{country}' -> {node_to_idx[country]}")

    print("\nSample Index-to-Node Mapping (First 5):")
    for idx in range(min(5, len(idx_to_node))):
        print(f"  {idx} -> '{idx_to_node[idx]}'")

    print("\nSample Edge Connections (First 3):")
    for i in range(min(3, pyg_data.num_edges)):
        src_idx = pyg_data.edge_index[0, i].item()
        dst_idx = pyg_data.edge_index[1, i].item()
        trade_usd = pyg_data.edge_attr[i, 0].item()
        print(f"  Edge {i}: '{idx_to_node[src_idx]}' -> '{idx_to_node[dst_idx]}' (${trade_usd:,.2f})")
