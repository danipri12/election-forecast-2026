import pandas as pd
import numpy as np

df = pd.read_csv("data/senate_final_picks.csv")

# Probability assigned to the party we selected
df["pick_probability"] = np.where(
    df["final_pick"] == "D",
    df["dem_probability"],
    1 - df["dem_probability"]
)

# Expected number of correct predictions
expected_correct = df["pick_probability"].sum()

print("\n2026 FORECAST METRICS")
print(f"Total races: {len(df)}")
print(f"Expected correct picks: {expected_correct:.2f}")
print(f"Expected accuracy: {expected_correct / len(df):.1%}")

print("\nConfidence distribution:")
print(df["pick_probability"].describe())

print("\nLOWEST-CONFIDENCE PICKS")
print(
    df[
        ["state", "final_pick", "pick_probability"]
    ]
    .sort_values("pick_probability")
    .head(10)
    .to_string(index=False)
)

# Approximate uncertainty, assuming independent race outcomes
variance = (
    df["pick_probability"]
    * (1 - df["pick_probability"])
).sum()

print(
    "\nApproximate standard deviation of correct picks:",
    round(np.sqrt(variance), 2)
)