import pandas as pd
import numpy as np

df = pd.read_csv("data/senate_2026.csv")

print("Columns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

# Compare presidential margins with available polling
polled = df.dropna(
    subset=["pres_margin_r", "poll_margin_d"]
).copy()

# Convert presidential margins to Democratic perspective
polled["pres_margin_d"] = -polled["pres_margin_r"]

# Difference between polling and presidential margin
polled["poll_vs_pres"] = (
    polled["poll_margin_d"] - polled["pres_margin_d"]
)

print("\nCorrelation between presidential margin and polling:")
print(
    polled["pres_margin_d"].corr(
        polled["poll_margin_d"]
    )
)

print("\nPolling compared with 2024 presidential results:")
print(
    polled[
        ["state", "pres_margin_d",
         "poll_margin_d", "poll_vs_pres"]
    ].sort_values("poll_vs_pres").to_string(index=False)
)

print("\nAverage polling difference:")
print(polled["poll_vs_pres"].mean())

print("\nIncumbency values:")
print(df["incumbent_running"].value_counts(dropna=False))

print("\nPolling difference by incumbency:")
print(
    polled.groupby("incumbent_running")["poll_vs_pres"]
    .agg(["count", "mean", "median", "std"])
)

print("\nPolling difference by seat party:")
print(
    polled.groupby("seat_party")["poll_vs_pres"]
    .agg(["count", "mean", "median", "std"])
)

summary = df[
    [
        "state",
        "pres_margin_r",
        "seat_party",
        "incumbent_running",
        "poll_margin_d",
        "betting_dem_probability"
    ]
].copy()

summary["poll_available"] = summary["poll_margin_d"].notna()

summary.to_csv(
    "data/senate_2026_summary.csv",
    index=False
)

print("\nSaved 2026 Senate indicator summary.")