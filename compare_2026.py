import pandas as pd

df = pd.read_csv("data/senate_final_picks.csv")

df["presidential_pick"] = df["pres_margin_r"].apply(
    lambda margin: "R" if margin > 0 else "D"
)

df["disagree"] = (
    df["final_pick"] != df["presidential_pick"]
)

print("\nPRESIDENTIAL BASELINE VS FINAL FORECAST")
print("Total races:", len(df))
print("Disagreements:", df["disagree"].sum())

print("\nRACES WHERE OUR PICK OVERRIDES THE HISTORICAL BASELINE")
print(
    df.loc[
        df["disagree"],
        [
            "state",
            "pres_margin_r",
            "poll_margin_d",
            "betting_dem_probability",
            "presidential_pick",
            "final_pick"
        ]
    ].to_string(index=False)
)

# Democratic margin in the 2024 presidential election
df["pres_margin_d"] = -df["pres_margin_r"]

# How far Senate polling differs from presidential voting
df["required_shift_d"] = (
    df["poll_margin_d"] - df["pres_margin_d"]
)

overrides = df[df["disagree"]].copy()

print("\n2026 OVERRIDE RISK ANALYSIS")
print(
    overrides[
        [
            "state",
            "required_shift_d",
            "poll_margin_d",
            "betting_dem_probability",
            "final_pick"
        ]
    ]
    .sort_values("required_shift_d", ascending=False)
    .to_string(index=False)
)

print("\nAverage required shift:")
print(overrides["required_shift_d"].mean())