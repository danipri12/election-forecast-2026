import pandas as pd

df = pd.read_csv("data/senate_2026.csv")

# Fix the obvious Massachusetts transcription anomaly
df.loc[df["state"] == "Massachusetts",
       "betting_dem_probability"] = 0.99

# Initial prediction-market baseline
df["dem_probability"] = df["betting_dem_probability"]

df["predicted_winner"] = df["dem_probability"].apply(
    lambda p: "D" if p >= 0.5 else "R"
)

df["confidence"] = df["dem_probability"].apply(
    lambda p: max(p, 1 - p)
)

# Show uncertain races first
df = df.sort_values("confidence")

print("\n2026 SENATE FORECAST — INITIAL BASELINE\n")
print(
    df[
        ["state", "predicted_winner",
         "dem_probability", "confidence"]
    ].to_string(index=False)
)

print("\nPredicted Democratic wins:",
      (df["predicted_winner"] == "D").sum())

print("Predicted Republican wins:",
      (df["predicted_winner"] == "R").sum())

df.to_csv("data/senate_forecast.csv", index=False)

# Independent model's directional picks for the closest races.
# These are NOT additional probabilities to average blindly.
external_picks = {
    "Kansas": "D",       # HDR D, DDHQ toss-up
    "Iowa": "R",         
    "Maine": "D",
    "Texas": "D",
    "Ohio": "D",
    "Alaska": "D",
    "Michigan": "D",
    "Nebraska": "R",     # Independent candidate requires separate review
}

df["external_pick"] = df["state"].map(external_picks)

df["needs_review"] = (
    df["external_pick"].notna()
    & (df["predicted_winner"] != df["external_pick"])
)

print("\nRACES WHERE THE BASELINE DISAGREES WITH NEWER MODELS")
print(
    df.loc[df["needs_review"],
           ["state", "predicted_winner",
            "external_pick", "confidence"]]
    .to_string(index=False)
)

# Revised picks based on the forecast comparison
df["final_pick"] = df["predicted_winner"].copy()

# Record races requiring special attention
df["review_note"] = ""

df.loc[df["state"] == "Iowa", "review_note"] = (
    "Kalshi D=44%; DDHQ D=48%; both narrowly favor R"
)

df.loc[df["state"] == "Kansas", "review_note"] = (
    "Very close; forecasting models disagree"
)

df.loc[df["state"] == "Nebraska", "review_note"] = (
    "Independent candidate; two-party model incomplete"
)

df.loc[df["state"] == "Massachusetts", "review_note"] = (
    "Betting probability manually corrected; verify source"
)

df.to_csv("data/senate_final_picks.csv", index=False)

print("\nFINAL PROVISIONAL SENATE PICKS")
print(
    df[["state", "final_pick", "review_note"]]
    .sort_values("state")
    .to_string(index=False)
)

print("\nDemocratic picks:", (df["final_pick"] == "D").sum())
print("Republican picks:", (df["final_pick"] == "R").sum())