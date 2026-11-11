import pandas as pd

df = pd.read_csv("data/senate_2026.csv")

# Convert the presidential margin to the Democratic perspective
df["pres_margin_d"] = -df["pres_margin_r"]

# Identify disagreement between the signs of polling and presidential margins
polled = df.dropna(subset=["poll_margin_d"]).copy()

polled["different_sign"] = (
    (polled["pres_margin_d"] > 0)
    != (polled["poll_margin_d"] > 0)
)

print("\nRaces where presidential margins and polling disagree:")
print(
    polled.loc[
        polled["different_sign"],
        [
            "state",
            "pres_margin_d",
            "poll_margin_d",
            "betting_dem_probability"
        ]
    ].to_string(index=False)
)

print("\nPotential betting-data inconsistencies:")
print(
    polled.loc[
        (
            ((polled["poll_margin_d"] > 0) &
             (polled["betting_dem_probability"] < 0.5))
            |
            ((polled["poll_margin_d"] < 0) &
             (polled["betting_dem_probability"] > 0.5))
        ),
        [
            "state",
            "poll_margin_d",
            "betting_dem_probability"
        ]
    ].to_string(index=False)
)