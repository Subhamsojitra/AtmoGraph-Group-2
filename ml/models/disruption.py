import os
import sys
import networkx as nx

# Ensure repository root is in sys.path for direct script execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.graph.graph_builder import build_trade_graph


def apply_disruption(country: str, severity: float, graph=None) -> dict:
    """
    Apply a supply-chain disruption/shock to a specific country in the trade graph.

    Parameters:
        country (str): Name of the disrupted country (must exist in graph).
        severity (float): Disruption intensity between 0.0 (no disruption) and 1.0 (complete disruption).
        graph (nx.DiGraph or str, optional): NetworkX trade graph or path to trade edges CSV.

    Returns:
        dict: Structured dictionary containing disruption metrics and affected outgoing trade edges.
    """
    # 1. Load trade graph if not provided
    if graph is None:
        default_csv = "ml/data/processed_trade_edges_2023.csv"
        graph = build_trade_graph(default_csv)
    elif isinstance(graph, str):
        graph = build_trade_graph(graph)

    # 2. Validate severity bounds
    if not (0.0 <= severity <= 1.0):
        raise ValueError(f"Severity must be between 0.0 and 1.0 (got {severity}).")

    # 3. Validate country existence in graph
    if not graph.has_node(country):
        available = sorted(list(graph.nodes()))
        raise ValueError(
            f"Country '{country}' not found in the trade graph. "
            f"Available countries: {available}"
        )

    # 4. Calculate disruption impact on outgoing trade edges
    affected_edges = []
    total_orig = 0.0
    total_disrupted = 0.0
    total_remaining = 0.0

    for src, partner, data in graph.out_edges(country, data=True):
        orig_val = float(data.get("trade_value_usd", 0.0))
        disrupted_val = orig_val * severity
        remaining_val = orig_val * (1.0 - severity)

        total_orig += orig_val
        total_disrupted += disrupted_val
        total_remaining += remaining_val

        affected_edges.append(
            {
                "exporter": src,
                "importer": partner,
                "original_trade_value_usd": orig_val,
                "disrupted_trade_value_usd": disrupted_val,
                "remaining_trade_value_usd": remaining_val,
            }
        )

    # 5. Construct and return structured output dictionary
    return {
        "disrupted_country": country,
        "severity": float(severity),
        "affected_edges": affected_edges,
        "original_trade_value": total_orig,
        "remaining_trade_value": total_remaining,
        "disrupted_trade_value": total_disrupted,
    }


def display_disruption_summary(disruption_result: dict):
    """
    Format and print a human-readable summary of the disruption impact.
    """
    country = disruption_result["disrupted_country"]
    sev = disruption_result["severity"]
    orig_val = disruption_result["original_trade_value"]
    dis_val = disruption_result["disrupted_trade_value"]
    rem_val = disruption_result["remaining_trade_value"]
    edges = disruption_result["affected_edges"]

    print("\n" + "=" * 65)
    print("ATMO GRAPH SUPPLY-CHAIN DISRUPTION SHOCK SUMMARY")
    print("=" * 65)
    print(f"Target Disrupted Country:   {country}")
    print(f"Disruption Severity:        {sev * 100:.1f}%")
    print(f"Affected Outgoing Edges:    {len(edges)}")
    print(f"Total Original Export:      ${orig_val:,.2f}")
    print(f"Disrupted Export Value:     ${dis_val:,.2f}")
    print(f"Remaining Export Value:     ${rem_val:,.2f}")
    print("=" * 65)

    print("\nAffected Outgoing Trade Edges:")
    for edge in edges:
        exp = edge["exporter"]
        imp = edge["importer"]
        d_val = edge["disrupted_trade_value_usd"]
        r_val = edge["remaining_trade_value_usd"]
        print(f"  - {exp} -> {imp}: Disrupted ${d_val:,.2f} | Remaining ${r_val:,.2f}")
    print("=" * 65)


if __name__ == "__main__":
    test_country = "China"
    test_severity = 0.3
    print(f"Testing apply_disruption('{test_country}', {test_severity})...")

    result = apply_disruption(test_country, test_severity)
    display_disruption_summary(result)
