import pandas as pd

df = pd.read_csv("data/senate_final_picks.csv")

print("\nDATASET")
print("Total races:", len(df))
print("Unique states:", df["state"].nunique())

print("\nFINAL PICKS BY PARTY")
print(df["final_pick"].value_counts())

print("\nMISSING VALUES")
print(df[["state", "final_pick"]].isna().sum())

print("\nDUPLICATE STATES")
print(
    df[df["state"].duplicated(keep=False)]
    [["state", "final_pick"]]
    .to_string(index=False)
)

print("\nPICKS CONTRADICTING HANDOUT PROBABILITIES")
df["market_pick"] = df["betting_dem_probability"].apply(
    lambda p: "D" if p > 0.5 else "R"
)

print(
    df[df["final_pick"] != df["market_pick"]]
    [["state", "final_pick",
      "market_pick", "betting_dem_probability"]]
    .to_string(index=False)
)

print("\nALL FINAL PICKS")
print(
    df[["state", "final_pick"]]
    .sort_values("state")
    .to_string(index=False)
)