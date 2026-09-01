import os
import sys
import networkx as nx
import pandas as pd

# Ensure repository root is in sys.path for direct script execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.graph.graph_builder import build_trade_graph


def extract_node_features(G: nx.DiGraph) -> pd.DataFrame:
    """
    Calculate node-level graph topological and economic features for each country.

    Parameters:
        G (nx.DiGraph): NetworkX directed trade graph.

    Returns:
        pd.DataFrame: DataFrame containing node features sorted by total trade volume.
    """
    deg_centrality = nx.degree_centrality(G)
    in_deg_centrality = nx.in_degree_centrality(G)
    out_deg_centrality = nx.out_degree_centrality(G)

    feature_records = []

    for node in G.nodes():
        country_code = G.nodes[node].get("code", None)

        # Topological degrees
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        
        # Unique connected trading partners
        predecessors = set(G.predecessors(node))
        successors = set(G.successors(node))
        trading_partners = len(predecessors.union(successors))

        # Weighted degrees (monetary volume in USD)
        weighted_in_deg = sum(
            data.get("trade_value_usd", 0.0) for _, _, data in G.in_edges(node, data=True)
        )
        weighted_out_deg = sum(
            data.get("trade_value_usd", 0.0) for _, _, data in G.out_edges(node, data=True)
        )
        total_trade_volume = weighted_in_deg + weighted_out_deg

        feature_records.append(
            {
                "country": node,
                "country_code": country_code,
                "in_degree": in_deg,
                "out_degree": out_deg,
                "trading_partners": trading_partners,
                "weighted_in_degree_usd": weighted_in_deg,
                "weighted_out_degree_usd": weighted_out_deg,
                "total_trade_volume_usd": total_trade_volume,
                "degree_centrality": round(deg_centrality[node], 4),
                "in_degree_centrality": round(in_deg_centrality[node], 4),
                "out_degree_centrality": round(out_deg_centrality[node], 4),
            }
        )

    df_features = pd.DataFrame(feature_records)
    
    # Sort by total trade volume descending
    df_features = df_features.sort_values(
        by="total_trade_volume_usd", ascending=False
    ).reset_index(drop=True)

    return df_features


def display_node_features(df_features: pd.DataFrame):
    """
    Print a clean, formatted table of node features for visual verification.
    """
    print()
    print("=" * 80)
    print("ATMO GRAPH: COUNTRY NODE FEATURES SUMMARY")
    print("=" * 80)

    display_cols = [
        "country",
        "in_degree",
        "out_degree",
        "trading_partners",
        "weighted_in_degree_usd",
        "weighted_out_degree_usd",
        "total_trade_volume_usd",
        "degree_centrality",
    ]

    df_display = df_features[display_cols].copy()
    
    # Format large monetary numbers for display
    df_display["weighted_in_degree_usd"] = df_display["weighted_in_degree_usd"].apply(
        lambda x: f"${x:,.0f}"
    )
    df_display["weighted_out_degree_usd"] = df_display["weighted_out_degree_usd"].apply(
        lambda x: f"${x:,.0f}"
    )
    df_display["total_trade_volume_usd"] = df_display["total_trade_volume_usd"].apply(
        lambda x: f"${x:,.0f}"
    )

    print(df_display.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    input_filepath = "ml/data/processed_trade_edges_2023.csv"
    print(f"Loading graph and extracting features from '{input_filepath}'...")

    # Reuse graph builder
    trade_graph = build_trade_graph(input_filepath)
    
    # Calculate node features
    features_df = extract_node_features(trade_graph)
    
    # Display results table
    display_node_features(features_df)
