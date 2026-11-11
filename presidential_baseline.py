import pandas as pd

# Load data
pres = pd.read_csv("data/president_results.csv")
senate = pd.read_csv("data/backtest_targets_clean.csv")

# Keep 2020 presidential results for the two major parties
pres = pres[
    (pres["year"] == 2020)
    & (pres["party_simplified"].isin(["DEMOCRAT", "REPUBLICAN"]))
].copy()

# Aggregate votes by state and party
votes = (
    pres.groupby(["state", "party_simplified"])["candidatevotes"]
    .sum()
    .unstack(fill_value=0)
)

# Calculate Democratic two-party margin
votes["pres_margin_d"] = (
    100
    * (votes["DEMOCRAT"] - votes["REPUBLICAN"])
    / (votes["DEMOCRAT"] + votes["REPUBLICAN"])
)

votes = votes.reset_index()

# Merge presidential results with Senate results
df = senate.merge(
    votes[["state", "pres_margin_d"]],
    on="state",
    how="left",
    validate="many_to_one"
)

# Check that all states matched
assert df["pres_margin_d"].notna().all(), "Missing presidential margins!"

# Predict Senate winner from presidential margin
df["predicted_winner"] = df["pres_margin_d"].apply(
    lambda margin: "D" if margin > 0 else "R"
)

# Compare predictions to actual Senate winners
df["correct"] = df["predicted_winner"] == df["actual_winner"]

print("\nBASELINE ACCURACY BY YEAR")
print(df.groupby("year")["correct"].agg(["sum", "count", "mean"]))

print("\nOVERALL ACCURACY")
print(f"{df['correct'].mean():.1%}")

print("\nINCORRECT PREDICTIONS")
print(
    df.loc[
        ~df["correct"],
        ["year", "state", "pres_margin_d",
         "predicted_winner", "actual_winner"]
    ].to_string(index=False)
)

df.to_csv("data/presidential_baseline_results.csv", index=False)

# Group races by how close the previous presidential election was
df["pres_margin_abs"] = df["pres_margin_d"].abs()

df["margin_group"] = pd.cut(
    df["pres_margin_abs"],
    bins=[0, 3, 7, 15, float("inf")],
    labels=["0–3 points", "3–7 points",
            "7–15 points", "15+ points"],
    include_lowest=True
)

print("\nACCURACY BY PRESIDENTIAL MARGIN")
print(
    df.groupby("margin_group", observed=True)["correct"]
    .agg(["sum", "count", "mean"])
    .to_string()
)

print("\nCLOSE STATES (PRESIDENTIAL MARGIN <= 7)")
print(
    df.loc[
        df["pres_margin_abs"] <= 7,
        ["year", "state", "pres_margin_d", "pres_margin_abs",
 "predicted_winner", "actual_winner", "correct"]
    ].sort_values(["year", "pres_margin_abs"])
    .to_string(index=False)
)