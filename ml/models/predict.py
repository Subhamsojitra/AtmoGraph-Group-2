import os
import sys

# Ensure repository root is in sys.path for direct script execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.graph.graph_builder import build_trade_graph
from ml.models.disruption import apply_disruption
from ml.models.propagation import propagate_disruption
from ml.models.risk_score import calculate_risk_scores


def predict_disruption_impact(
    disrupted_country: str,
    severity: float,
    hops: int = 2,
    decay: float = 0.5,
    graph=None,
) -> dict:
    """
    End-to-end AtmoGraph ML pipeline for supply-chain disruption risk analysis.

    Parameters:
        disrupted_country (str): Origin country where trade disruption occurs.
        severity (float): Initial disruption intensity (0.0 to 1.0).
        hops (int): Maximum shock propagation depth (default 2).
        decay (float): Attenuation factor per hop (default 0.5).
        graph (nx.DiGraph or str, optional): NetworkX trade graph or path to trade edges CSV.

    Returns:
        dict: Complete structured dictionary containing initial shock, propagation,
              country-level risk scores, and ranked vulnerability metrics.
    """
    # 1. Load or build trade graph
    if graph is None:
        default_csv = "ml/data/processed_trade_edges_2023.csv"
        graph = build_trade_graph(default_csv)
    elif isinstance(graph, str):
        graph = build_trade_graph(graph)

    # 2. Apply initial 1st-hop disruption
    disruption_res = apply_disruption(disrupted_country, severity, graph=graph)

    # 3. Simulate multi-hop shock propagation
    propagation_res = propagate_disruption(
        disrupted_country,
        severity,
        max_hops=hops,
        decay_factor=decay,
        graph=graph,
    )

    # 4. Compute country risk scores
    risk_res = calculate_risk_scores(propagation_res, graph=graph)

    # 5. Extract affected country names
    risk_ranking = risk_res["risk_ranking"]
    affected_countries = [item["country"] for item in risk_ranking]

    # 6. Combine into unified structured result dictionary
    return {
        "disrupted_country": disrupted_country,
        "severity": float(severity),
        "hops": int(hops),
        "decay": float(decay),
        "original_trade_value": disruption_res["original_trade_value"],
        "disrupted_trade_value": disruption_res["disrupted_trade_value"],
        "remaining_trade_value": disruption_res["remaining_trade_value"],
        "total_global_trade_loss": propagation_res["total_global_disrupted_usd"],
        "affected_countries": affected_countries,
        "country_risk_scores": risk_res["country_risk_scores"],
        "risk_ranking": risk_ranking,
        "highest_risk_country": risk_res["highest_risk_country"],
        "highest_risk_score": risk_res["highest_risk_score"],
        "disruption_details": disruption_res,
        "propagation_details": propagation_res,
    }


def display_predict_report(result: dict):
    """
    Format and print a clean final report of the disruption impact analysis.
    """
    disrupted_country = result["disrupted_country"]
    sev = result["severity"]
    orig_val = result["original_trade_value"]
    lost_val = result["disrupted_trade_value"]
    rem_val = result["remaining_trade_value"]
    ranking = result["risk_ranking"]
    highest_country = result["highest_risk_country"]
    highest_score = result["highest_risk_score"]

    print("\n" + "=" * 65)
    print("ATMOGRAPH FINAL DISRUPTION IMPACT ANALYSIS")
    print("=" * 65)
    print(f"Disrupted Country:      {disrupted_country}")
    print(f"Disruption Severity:    {sev * 100:.1f}%")
    print(f"Original Trade Value:   ${orig_val:,.2f}")
    print(f"Trade Value Lost:       ${lost_val:,.2f}")
    print(f"Remaining Trade Value:  ${rem_val:,.2f}")
    print("=" * 65)

    print("\nAFFECTED COUNTRY RISK RANKING")
    print("-" * 65)
    print(f"{'Rank':<6} {'Country':<25} {'Risk Score':<15} {'Risk Level':<10}")
    print("-" * 65)

    for idx, item in enumerate(ranking, 1):
        country_name = item["country"]
        score = item["risk_score"]
        level = item["risk_level"]
        print(f"{idx:<6} {country_name:<25} {score:<15.4f} {level:<10}")

    print("-" * 65)
    print(f"Highest Risk Country:   {highest_country}")
    print(f"Highest Risk Score:     {highest_score:.4f}")
    print("=" * 65)
    print("Note: This analysis utilizes deterministic graph topology and multi-hop shock")
    print("propagation. It is a graph-based disruption impact and risk assessment model.")
    print("=" * 65)


if __name__ == "__main__":
    disrupted_country = "China"
    severity = 0.3
    hops = 2
    decay = 0.5

    result = predict_disruption_impact(disrupted_country, severity, hops=hops, decay=decay)
    display_predict_report(result)
