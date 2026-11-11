import pandas as pd

df = pd.read_csv("data/senate_races.csv")

# Select the elections we want to backtest
test = df[df["year"].isin([2022, 2024])].copy()

# Determine actual winners using the two-party margin
test["actual_winner"] = test["dem_margin"].apply(
    lambda x: "D" if x > 0 else "R"
)

print("Columns:")
print(df.columns.tolist())

print("\nNumber of races by year:")
print(test.groupby("year").size())

print("\nHistorical election results:")
print(test.to_string(index=False))

# Identify suspicious or unusual records
print("\nRaces with unusually large margins:")
print(
    test.loc[
        test["dem_margin"].abs() > 50,
        ["year", "state", "special", "dem_margin"]
    ].to_string(index=False)
)

print("\nDuplicate state-year combinations:")
print(
    test.loc[
        test.duplicated(["year", "state"], keep=False),
        ["year", "state", "special", "actual_winner"]
    ].to_string(index=False)
)

# Save provisional backtest targets
test[
    ["year", "state", "special", "dem_margin", "actual_winner"]
].to_csv("data/backtest_targets.csv", index=False)

print("\nSaved data/backtest_targets.csv")

# Exclude 2024 races with important independent candidates
exclude = (
    (test["year"] == 2024)
    & (test["state"].isin(["MAINE", "NEBRASKA"]))
)

clean_test = test.loc[~exclude].copy()

clean_test[
    ["year", "state", "special", "dem_margin", "actual_winner"]
].to_csv("data/backtest_targets_clean.csv", index=False)

print("\nClean backtest races:")
print(clean_test.groupby("year").size())

print("\nTotal:", len(clean_test))
print("Saved data/backtest_targets_clean.csv")