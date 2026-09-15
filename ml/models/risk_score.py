import os
import sys

# Ensure repository root is in sys.path for direct script execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.graph.graph_builder import build_trade_graph
from ml.models.propagation import propagate_disruption


def calculate_risk_scores(propagation_result: dict, graph=None) -> dict:
    """
    Convert multi-hop shock propagation results into normalized country risk scores [0 to 1].

    Parameters:
        propagation_result (dict): Output dictionary from propagate_disruption().
        graph (nx.DiGraph or str, optional): NetworkX trade graph or path to trade edges CSV.

    Returns:
        dict: Structured risk analysis containing country scores, ranking, and highest risk node.
    """
    disrupted_country = propagation_result["disrupted_country"]
    country_shocks = propagation_result["country_total_shocks"]
    hop_impacts = propagation_result.get("hop_impacts", [])

    # Aggregate total trade loss ($ USD) received by each importer country
    country_trade_loss = {}
    for hop_data in hop_impacts:
        for edge in hop_data.get("affected_edges", []):
            importer = edge["importer"]
            loss_usd = float(edge.get("disrupted_trade_value_usd", 0.0))
            country_trade_loss[importer] = country_trade_loss.get(importer, 0.0) + loss_usd

    # Compute raw risk scores: Shock Exposure * Trade Loss Volume
    raw_risk_scores = {}
    for country, shock in country_shocks.items():
        # Focus ranking on affected trading partners (exclude origin disrupted country)
        if country == disrupted_country:
            continue
        loss = country_trade_loss.get(country, 0.0)
        raw_risk_scores[country] = shock * loss

    # Normalize risk scores to 0.0 - 1.0 range based on maximum partner exposure
    max_raw_risk = max(raw_risk_scores.values()) if raw_risk_scores else 1.0
    if max_raw_risk == 0.0:
        max_raw_risk = 1.0

    country_risk_scores = {}
    risk_ranking = []

    for country, raw_score in raw_risk_scores.items():
        norm_score = round(raw_score / max_raw_risk, 4)
        country_risk_scores[country] = norm_score

        # Categorize risk level
        if norm_score >= 0.70:
            level = "HIGH"
        elif norm_score >= 0.30:
            level = "MEDIUM"
        else:
            level = "LOW"

        risk_ranking.append(
            {
                "country": country,
                "risk_score": norm_score,
                "risk_level": level,
                "trade_loss_usd": country_trade_loss.get(country, 0.0),
                "shock_exposure": country_shocks.get(country, 0.0),
            }
        )

    # Sort risk ranking by score descending
    risk_ranking.sort(key=lambda x: x["risk_score"], reverse=True)

    highest_risk_country = risk_ranking[0]["country"] if risk_ranking else None
    highest_risk_score = risk_ranking[0]["risk_score"] if risk_ranking else 0.0

    return {
        "disrupted_country": disrupted_country,
        "country_risk_scores": country_risk_scores,
        "risk_ranking": risk_ranking,
        "highest_risk_country": highest_risk_country,
        "highest_risk_score": highest_risk_score,
    }


def display_risk_scores(risk_summary: dict):
    """
    Format and print a readable country risk assessment table.
    """
    disrupted_country = risk_summary["disrupted_country"]
    ranking = risk_summary["risk_ranking"]
    highest_country = risk_summary["highest_risk_country"]
    highest_score = risk_summary["highest_risk_score"]

    print("\n" + "=" * 60)
    print(f"COUNTRY RISK ASSESSMENT (Disruption Origin: {disrupted_country})")
    print("=" * 60)
    print(f"{'Country':<25} {'Risk Score':<15} {'Risk Level':<10}")
    print("-" * 60)

    for item in ranking:
        country_name = item["country"]
        score = item["risk_score"]
        level = item["risk_level"]
        print(f"{country_name:<25} {score:<15.4f} {level:<10}")

    print("=" * 60)
    print(f"Highest Risk Country: {highest_country}")
    print(f"Highest Risk Score:   {highest_score:.4f}")
    print("=" * 60)
    print("Note: Risk scores are derived from graph topological trade volume and shock exposure.")
    print("This is a deterministic graph-based analysis model, not a supervised prediction model.")
    print("=" * 60)


if __name__ == "__main__":
    disrupted_country = "China"
    severity = 0.3
    hops = 2
    decay = 0.5

    print(f"Simulating risk assessment for '{disrupted_country}' (severity={severity}, hops={hops}, decay={decay})...")
    prop_res = propagate_disruption(disrupted_country, severity, max_hops=hops, decay_factor=decay)
    risk_res = calculate_risk_scores(prop_res)
    display_risk_scores(risk_res)
