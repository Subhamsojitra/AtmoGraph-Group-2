import os
import networkx as nx
import pandas as pd


def build_trade_graph(df_or_filepath) -> nx.DiGraph:
    """
    Build a NetworkX Directed Graph (DiGraph) from processed trade edge data.

    Parameters:
        df_or_filepath (pd.DataFrame or str): DataFrame containing processed trade edges,
                                              or filepath to processed CSV file.

    Returns:
        nx.DiGraph: Directed graph representing trade flow between countries.
    """
    if isinstance(df_or_filepath, str):
        if not os.path.exists(df_or_filepath):
            raise FileNotFoundError(f"Input file not found: {df_or_filepath}")
        df = pd.read_csv(df_or_filepath)
    else:
        df = df_or_filepath

    G = nx.DiGraph()

    for _, row in df.iterrows():
        reporter = str(row["reporter"])
        partner = str(row["partner"])

        # Add node attributes if not already added
        if not G.has_node(reporter):
            G.add_node(reporter, code=row.get("reporter_code", None))
        if not G.has_node(partner):
            G.add_node(partner, code=row.get("partner_code", None))

        # Extract edge attributes
        edge_attrs = {
            "trade_value_usd": float(row.get("trade_value_usd", 0.0)),
            "edge_weight": float(row.get("edge_weight", row.get("trade_value_usd", 0.0))),
            "log_trade_value": float(row.get("log_trade_value", 0.0)),
            "net_weight_kg": float(row.get("net_weight_kg", 0.0)),
            "commodity": str(row.get("commodity", "TOTAL")),
            "commodity_code": str(row.get("commodity_code", "TOTAL")),
            "year": int(row.get("year", 2023)),
        }

        # Add directed edge: Exporter (reporter) -> Importer (partner)
        G.add_edge(reporter, partner, **edge_attrs)

    return G


def get_graph_summary(G: nx.DiGraph) -> dict:
    """
    Generate summary statistics for the trade graph.

    Parameters:
        G (nx.DiGraph): NetworkX directed trade graph.

    Returns:
        dict: Dictionary of graph summary metrics.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    is_directed = G.is_directed()
    density = nx.density(G) if num_nodes > 1 else 0.0

    total_trade_value = sum(
        data.get("trade_value_usd", 0.0) for _, _, data in G.edges(data=True)
    )

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "is_directed": is_directed,
        "density": density,
        "total_trade_usd": total_trade_value,
    }


def print_graph_summary(G: nx.DiGraph):
    """
    Print formatted graph metrics and sample details.
    """
    summary = get_graph_summary(G)

    print()
    print("=" * 45)
    print("ATMO GRAPH SUMMARY (2023 TRADE NETWORK)")
    print("=" * 45)
    print(f"Total Nodes (Countries):     {summary['num_nodes']}")
    print(f"Total Edges (Trade Flows):  {summary['num_edges']}")
    print(f"Directed Graph:             {summary['is_directed']}")
    print(f"Graph Density:              {summary['density']:.4f}")
    print(f"Total Trade Value (USD):    ${summary['total_trade_usd']:,.2f}")
    print("=" * 45)

    print("\nSample Node Attributes:")
    for node in list(G.nodes)[:5]:
        print(f"  - {node}: {G.nodes[node]}")

    print("\nSample Edge Attributes:")
    for src, dst, data in list(G.edges(data=True))[:3]:
        print(f"  - {src} -> {dst}:")
        for k, v in data.items():
            print(f"      {k}: {v}")


if __name__ == "__main__":
    input_filepath = "ml/data/processed_trade_edges_2023.csv"
    print(f"Loading data and constructing graph from '{input_filepath}'...")

    trade_graph = build_trade_graph(input_filepath)
    print_graph_summary(trade_graph)
