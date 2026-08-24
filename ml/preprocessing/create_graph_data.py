import pandas as pd
import numpy as np


def prepare_trade_edges(input_file):
    """
    Prepare UN Comtrade trade data
    for graph construction.
    """

    print("Loading trade data...")

    df = pd.read_csv(input_file)

    print(f"Raw rows: {len(df)}")

    # Convert numeric columns
    df["trade_value_usd"] = pd.to_numeric(
        df["trade_value_usd"],
        errors="coerce"
    )

    df["net_weight_kg"] = pd.to_numeric(
        df["net_weight_kg"],
        errors="coerce"
    )

    # Remove missing trade values
    df = df.dropna(
        subset=[
            "reporter_code",
            "partner_code",
            "trade_value_usd"
        ]
    )

    # Keep only positive trade values
    df = df[
        df["trade_value_usd"] > 0
    ]

    # Create graph edge weight
    df["edge_weight"] = df["trade_value_usd"]

    # Log transformation
    df["log_trade_value"] = np.log1p(
        df["trade_value_usd"]
    )

    print(
        f"Clean rows: {len(df)}"
    )

    return df


if __name__ == "__main__":

    input_file = (
        "ml/data/trade_edges_2023.csv"
    )

    output_file = (
        "ml/data/processed_trade_edges_2023.csv"
    )

    data = prepare_trade_edges(
        input_file
    )

    print()
    print("=" * 45)
    print("PROCESSED TRADE GRAPH DATA")
    print("=" * 45)

    print()

    print(
        data[
            [
                "reporter",
                "partner",
                "trade_value_usd",
                "edge_weight",
                "log_trade_value"
            ]
        ].to_string(index=False)
    )

    data.to_csv(
        output_file,
        index=False
    )

    print()
    print(
        f"Saved processed data to: "
        f"{output_file}"
    )