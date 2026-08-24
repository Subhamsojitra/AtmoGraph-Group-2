import os
import requests
import pandas as pd
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


# Get API key from .env
COMTRADE_API_KEY = os.getenv("COMTRADE_API_KEY")


# Make sure API key exists
if not COMTRADE_API_KEY:
    raise RuntimeError(
        "COMTRADE_API_KEY is missing from .env"
    )


# UN Comtrade authenticated API
BASE_URL = "https://comtradeapi.un.org/data/v1/get/C/A/HS"


def get_trade_data(
    reporter_code,
    period,
    flow_code="X"
):
    """
    Fetch trade data for one reporting country.

    reporter_code:
        China = 156
        India = 699
        USA = 842

    period:
        Year of trade data.

    flow_code:
        X = Exports
        M = Imports
    """

    # API parameters
    params = {
        "flowCode": flow_code,
        "reporterCode": reporter_code,
        "period": period,
        "partnerCode": 0,
        "cmdCode": "TOTAL",
        "maxRecords": 500,
        "includeDesc": "true",
        "subscription-key": COMTRADE_API_KEY

    }

    print("Requesting UN Comtrade data...")
    print(f"Reporter code: {reporter_code}")
    print(f"Year: {period}")
    print(f"Flow: {flow_code}")

    try:

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=30
        )

        print("\nAPI URL:")
        print(response.url.replace(
            COMTRADE_API_KEY,
            "***HIDDEN***"
        ))

        print("\nHTTP Status:", response.status_code)

        response.raise_for_status()

        result = response.json()

    except requests.exceptions.RequestException as error:

        print("\nAPI request failed:")
        print(error)

        return pd.DataFrame()

    except ValueError:

        print("\nAPI returned invalid JSON.")

        return pd.DataFrame()

    # Get records
    records = result.get("data", [])

    if not records:

        print("\nNo records returned.")

        print("\nAPI response:")
        print(result)

        return pd.DataFrame()

    # Convert API response to DataFrame
    df = pd.DataFrame(records)

    print("\nRaw records received:", len(df))

    # Columns we need
    useful_columns = [
        "period",
        "flowCode",
        "reporterCode",
        "reporterDesc",
        "partnerCode",
        "partnerDesc",
        "cmdCode",
        "cmdDesc",
        "primaryValue",
        "netWgt"
    ]

    # Keep only columns that actually exist
    available_columns = [
        column
        for column in useful_columns
        if column in df.columns
    ]

    df = df[available_columns]

    # Rename columns for AtmoGraph
    rename_columns = {
        "period": "year",
        "flowCode": "flow",
        "reporterCode": "reporter_code",
        "reporterDesc": "reporter",
        "partnerCode": "partner_code",
        "partnerDesc": "partner",
        "cmdCode": "commodity_code",
        "cmdDesc": "commodity",
        "primaryValue": "trade_value_usd",
        "netWgt": "net_weight_kg"
    }

    df = df.rename(columns=rename_columns)

    return df


if __name__ == "__main__":

    # Test:
    # China = 156
    # World = 0

    data = get_trade_data(
        reporter_code=156,
        period=2023,
        flow_code="X"
    )

    if data.empty:

        print("\nNo data received.")

    else:

        print("\nSUCCESS!")
        print("==============================")

        print("Rows:", len(data))

        print("\nColumns:")
        print(data.columns.tolist())

        print("\nTrade relationships:")

        display_columns = [
            column
            for column in [
                "year",
                "reporter",
                "partner",
                "commodity",
                "trade_value_usd",
                "net_weight_kg"
            ]
            if column in data.columns
        ]

        print(
            data[
                display_columns
            ].head(20).to_string(index=False)
        )