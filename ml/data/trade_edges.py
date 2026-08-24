import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()

COMTRADE_API_KEY = os.getenv("COMTRADE_API_KEY")

if not COMTRADE_API_KEY:
    raise RuntimeError(
        "COMTRADE_API_KEY is missing from .env"
    )


# ---------------------------------------------------------
# UN Comtrade API
# ---------------------------------------------------------

BASE_URL = "https://comtradeapi.un.org/data/v1/get/C/A/HS"


# ---------------------------------------------------------
# Countries we want for our first AtmoGraph network
# ---------------------------------------------------------

PARTNERS = {
    356: "India",
    842: "United States",
    276: "Germany",
    392: "Japan",
    704: "Vietnam",
    410: "South Korea",
    528: "Netherlands",
    643: "Russia",
    826: "United Kingdom",
    702: "Singapore",
    36: "Australia",
    76: "Brazil",
    124: "Canada",
    784: "United Arab Emirates",
    360: "Indonesia",
    458: "Malaysia",
    764: "Thailand",
    250: "France",
    380: "Italy",
    484: "Mexico"
}


# ---------------------------------------------------------
# Get bilateral trade
# ---------------------------------------------------------

def get_bilateral_trade(
    reporter_code,
    partner_code,
    period=2023,
    flow_code="X"
):

    params = {
        "flowCode": flow_code,
        "reporterCode": reporter_code,
        "partnerCode": partner_code,
        "period": period,
        "cmdCode": "TOTAL",
        "maxRecords": 500,
        "includeDesc": "true",
        "subscription-key": "50036f04cc6c4334999ad5ef85ce445a"

    }

    try:

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        records = result.get("data", [])

        if not records:
            return None

        # We requested TOTAL, so normally one record is enough
        record = records[0]

        return {
            "year": period,
            "reporter_code": record.get("reporterCode"),
            "reporter": record.get("reporterDesc"),
            "partner_code": record.get("partnerCode"),
            "partner": record.get("partnerDesc"),
            "commodity_code": record.get("cmdCode"),
            "commodity": record.get("cmdDesc"),
            "trade_value_usd": record.get("primaryValue"),
            "net_weight_kg": record.get("netWgt")
        }

    except requests.exceptions.RequestException as error:

        print(
            f"API error for partner "
            f"{partner_code}: {error}"
        )

        return None


# ---------------------------------------------------------
# Build trade network
# ---------------------------------------------------------

def build_trade_network(
    reporter_code=156,
    period=2023
):

    edges = []

    print()
    print("Building AtmoGraph trade network")
    print("=" * 45)

    print(f"Reporter: China ({reporter_code})")
    print(f"Year: {period}")
    print(f"Partners: {len(PARTNERS)}")
    print()

    for partner_code, partner_name in PARTNERS.items():

        print(
            f"Fetching China -> {partner_name}..."
        )

        result = get_bilateral_trade(
            reporter_code=reporter_code,
            partner_code=partner_code,
            period=period,
            flow_code="X"
        )

        if result is not None:

            edges.append(result)

            print(
                f"  ✓ Trade value: "
                f"{result['trade_value_usd']}"
            )

        else:

            print("  ✗ No data")

        # Respect API rate limit
        time.sleep(1.1)

    return pd.DataFrame(edges)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    trade_edges = build_trade_network(
        reporter_code=156,
        period=2023
    )

    print()
    print("=" * 45)
    print("TRADE NETWORK COMPLETE")
    print("=" * 45)

    if trade_edges.empty:

        print("No trade edges were created.")

    else:

        print(
            f"Total trade edges: "
            f"{len(trade_edges)}"
        )

        print()
        print(trade_edges.to_string(index=False))

        # Save trade network
        output_file = "ml/data/trade_edges_2023.csv"

        trade_edges.to_csv(
            output_file,
            index=False
        )

        print()
        print(f"Saved trade data to: {output_file}")