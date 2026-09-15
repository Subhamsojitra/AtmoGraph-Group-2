import os
import sys
import networkx as nx

# Ensure repository root is in sys.path for direct script execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.graph.graph_builder import build_trade_graph
from ml.models.disruption import apply_disruption


def propagate_disruption(
    country: str,
    severity: float,
    max_hops: int = 2,
    decay_factor: float = 0.5,
    graph=None,
) -> dict:
    """
    Simulate multi-hop cascading shock propagation across the supply-chain trade network.

    Parameters:
        country (str): Origin country where disruption begins.
        severity (float): Initial disruption severity (0.0 to 1.0).
        max_hops (int): Maximum depth of propagation (default 2).
        decay_factor (float): Attenuation factor per hop (default 0.5).
        graph (nx.DiGraph or str, optional): NetworkX trade graph or path to edges CSV.

    Returns:
        dict: Complete structured propagation summary including hop-by-hop impacts,
              country-level cumulative shocks, and global trade losses.
    """
    # 1. Load trade graph if needed
    if graph is None:
        default_csv = "ml/data/processed_trade_edges_2023.csv"
        graph = build_trade_graph(default_csv)
    elif isinstance(graph, str):
        graph = build_trade_graph(graph)

    # 2. Get initial 1st-hop disruption from disruption.py
    initial_summary = apply_disruption(country, severity, graph)

    # Track node shock severity [0.0 to 1.0] for every country in graph
    country_shocks = {node: 0.0 for node in graph.nodes()}
    country_shocks[country] = float(severity)

    hop_impacts = []
    total_global_disrupted_usd = 0.0
    visited_nodes = {country}

    # Helper function to compute total in-degree import volume for a country
    def get_total_imports(c):
        return sum(
            float(data.get("trade_value_usd", 0.0))
            for _, _, data in graph.in_edges(c, data=True)
        )

    # Current frontier of (country, current_shock_severity)
    current_frontier = [(country, float(severity))]

    # 3. Multi-hop propagation loop
    for hop in range(1, max_hops + 1):
        next_frontier = []
        hop_edges = []
        hop_disrupted_usd = 0.0

        for current_country, current_sev in current_frontier:
            if current_sev <= 0.0:
                continue

            for src, partner, data in graph.out_edges(current_country, data=True):
                orig_trade = float(data.get("trade_value_usd", 0.0))
                if orig_trade <= 0:
                    continue

                # Calculate trade loss on this edge
                disrupted_trade = orig_trade * current_sev
                remaining_trade = orig_trade - disrupted_trade
                hop_disrupted_usd += disrupted_trade

                hop_edges.append(
                    {
                        "exporter": src,
                        "importer": partner,
                        "original_trade_value_usd": orig_trade,
                        "disrupted_trade_value_usd": disrupted_trade,
                        "remaining_trade_value_usd": remaining_trade,
                    }
                )

                # Calculate import dependency of partner on exporter
                tot_imports = get_total_imports(partner)
                dependency = (orig_trade / tot_imports) if tot_imports > 0 else 0.0

                # Compute cascading shock to partner for next hop
                cascaded_shock = current_sev * dependency * decay_factor

                # Accumulate country total shock score
                country_shocks[partner] = min(1.0, country_shocks[partner] + cascaded_shock)

                if partner not in visited_nodes and cascaded_shock > 1e-4:
                    visited_nodes.add(partner)
                    next_frontier.append((partner, cascaded_shock))

        hop_impacts.append(
            {
                "hop": hop,
                "disrupted_trade_value": hop_disrupted_usd,
                "affected_edges_count": len(hop_edges),
                "affected_edges": hop_edges,
            }
        )

        total_global_disrupted_usd += hop_disrupted_usd
        current_frontier = next_frontier

    # 4. Construct final summary dictionary
    return {
        "disrupted_country": country,
        "initial_severity": float(severity),
        "max_hops": max_hops,
        "decay_factor": decay_factor,
        "initial_disruption": initial_summary,
        "hop_impacts": hop_impacts,
        "country_total_shocks": {
            k: round(v, 4) for k, v in country_shocks.items() if v > 0
        },
        "total_global_disrupted_usd": total_global_disrupted_usd,
    }


def display_propagation_summary(propagation_result: dict):
    """
    Format and print readable summary table of multi-hop propagation results.
    """
    country = propagation_result["disrupted_country"]
    sev = propagation_result["initial_severity"]
    max_hops = propagation_result["max_hops"]
    decay = propagation_result["decay_factor"]
    total_loss = propagation_result["total_global_disrupted_usd"]
    country_shocks = propagation_result["country_total_shocks"]

    print("\n" + "=" * 70)
    print("ATMO GRAPH MULTI-HOP SHOCK PROPAGATION SUMMARY")
    print("=" * 70)
    print(f"Origin Country:            {country}")
    print(f"Initial Severity:          {sev * 100:.1f}%")
    print(f"Max Hops Simulated:        {max_hops}")
    print(f"Decay Factor per Hop:      {decay}")
    print(f"Total Global Trade Loss:   ${total_loss:,.2f}")
    print("=" * 70)

    for hop_data in propagation_result["hop_impacts"]:
        h = hop_data["hop"]
        h_loss = hop_data["disrupted_trade_value"]
        h_count = hop_data["affected_edges_count"]
        print(f"\nHop {h} Summary:")
        print(f"  - Affected Edges: {h_count}")
        print(f"  - Hop Trade Loss: ${h_loss:,.2f}")

    print("\nCountry Accumulated Shock Exposures:")
    sorted_shocks = sorted(country_shocks.items(), key=lambda x: x[1], reverse=True)
    for c_name, shock_val in sorted_shocks:
        print(f"  - {c_name:<25}: {shock_val * 100:6.2f}% shock")
    print("=" * 70)


if __name__ == "__main__":
    test_country = "China"
    test_severity = 0.3
    print(f"Simulating propagation for '{test_country}' (severity={test_severity}, hops=2, decay=0.5)...")

    result = propagate_disruption(test_country, test_severity, max_hops=2, decay_factor=0.5)
    display_propagation_summary(result)
