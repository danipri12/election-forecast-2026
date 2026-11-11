import pandas as pd

HANDOUT_DATE = "2026-11-01"
POLLING_SOURCE = "RealClearPolling"
BETTING_SOURCE = "Kalshi"
EXTERNAL_SOURCE = "DDHQ"

ours = pd.read_csv("data/senate_final_picks.csv")
external = pd.read_csv("data/external_forecasts.csv")

df = ours.merge(
    external,
    on="state",
    how="inner",
    validate="one_to_one",
    suffixes=("_ours", "_ddhq")
)

# Determine which party DDHQ's probabilities favor
df["ddhq_pick"] = df["dem_probability_ddhq"].apply(
    lambda p: "D" if p > 0.5 else "R"
)

# Compare DDHQ with our original selections
df["agree"] = df["final_pick"] == df["ddhq_pick"]

print("\nFORECAST COMPARISON")
print("Races compared:", len(df))
print("Agreements:", df["agree"].sum())
print("Disagreements:", (~df["agree"]).sum())

print("\nALL COMPARISONS")
print(
    df[
        [
            "state",
            "final_pick",
            "ddhq_pick",
            "betting_dem_probability",
            "dem_probability_ddhq",
            "agree"
        ]
    ]
    .sort_values("state")
    .to_string(index=False)
)

print("\nDISAGREEMENTS ONLY")
print(
    df.loc[
        ~df["agree"],
        ["state", "final_pick", "ddhq_pick"]
    ].to_string(index=False)
)

df["probability_gap"] = (
    df["betting_dem_probability"] -
    df["dem_probability_ddhq"]
).abs()

print("\nLARGEST PROBABILITY DISAGREEMENTS")
print(
    df[
        [
            "state",
            "betting_dem_probability",
            "dem_probability_ddhq",
            "probability_gap"
        ]
    ]
    .sort_values("probability_gap", ascending=False)
    .to_string(index=False)
)

print("\nAVERAGE PROBABILITY GAP")
print(round(df["probability_gap"].mean(), 3))

df["average_dem_probability"] = (
    df["betting_dem_probability"] +
    df["dem_probability_ddhq"]
) / 2

print("\nEQUAL-WEIGHT PROBABILITY COMPARISON")
print(
    df[
        [
            "state",
            "final_pick",
            "average_dem_probability"
        ]
    ]
    .sort_values("average_dem_probability")
    .to_string(index=False)
)

print("\nFINAL EVIDENCE COMPARISON")

columns = [
    "state",
    "pres_margin_r",
    "poll_margin_d",
    "betting_dem_probability",
    "dem_probability_ddhq",
    "average_dem_probability",
    "final_pick"
]

print(
    df[columns]
    .sort_values("average_dem_probability")
    .to_string(index=False)
)

import numpy as np

# Probability assigned to each of our existing picks
df["kalshi_pick_prob"] = np.where(
    df["final_pick"] == "D",
    df["betting_dem_probability"],
    1 - df["betting_dem_probability"]
)

df["ddhq_pick_prob"] = np.where(
    df["final_pick"] == "D",
    df["dem_probability_ddhq"],
    1 - df["dem_probability_ddhq"]
)

print("\nEXPECTED CORRECT PICKS — 8 RACES")
print("Kalshi:", round(df["kalshi_pick_prob"].sum(), 2), "/ 8")
print("DDHQ:", round(df["ddhq_pick_prob"].sum(), 2), "/ 8")

print("\nSOURCE SENSITIVITY BY STATE")
print(
    df[
        [
            "state",
            "final_pick",
            "kalshi_pick_prob",
            "ddhq_pick_prob"
        ]
    ]
    .sort_values("ddhq_pick_prob")
    .to_string(index=False)
)

# Original expected correct picks across all 35 races
original_expected = 30.87

# Replace Kalshi expectations with DDHQ expectations
revised_expected = (
    original_expected
    - df["kalshi_pick_prob"].sum()
    + df["ddhq_pick_prob"].sum()
)

print("\nFULL FORECAST SENSITIVITY")
print("Original expected correct:", round(original_expected, 2))
print("DDHQ-substituted expected correct:", round(revised_expected, 2))
print("DDHQ-substituted expected accuracy:", round(revised_expected / 35 * 100, 1), "%")

df["kalshi_leader"] = df["betting_dem_probability"].apply(
    lambda p: "D" if p > 0.5 else "R"
)

df["ddhq_leader"] = df["dem_probability_ddhq"].apply(
    lambda p: "D" if p > 0.5 else "R"
)

df["both_sources_agree"] = (
    df["kalshi_leader"] == df["ddhq_leader"]
)

print("\nSOURCE AGREEMENT AUDIT")
print(
    df[
        [
            "state",
            "final_pick",
            "kalshi_leader",
            "ddhq_leader",
            "both_sources_agree"
        ]
    ]
    .sort_values("state")
    .to_string(index=False)
)

report = df[
    [
        "state",
        "final_pick",
        "pres_margin_r",
        "poll_margin_d",
        "betting_dem_probability",
        "dem_probability_ddhq",
        "kalshi_leader",
        "ddhq_leader"
    ]
].copy()

report.to_csv(
    "data/forecast_evidence_report.csv",
    index=False
)

print("\nSaved data/forecast_evidence_report.csv")