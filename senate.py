import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/senate_races.csv")

# Start with regular elections only
df = df[df["special"] == False].copy()

# Sort chronologically within each state
df = df.sort_values(["state", "year"])

# Previous observed regular Senate election margin
df["previous_margin"] = (
    df.groupby("state")["dem_margin"].shift(1)
)

df["previous_year"] = (
    df.groupby("state")["year"].shift(1)
)

# Keep only states with an earlier observation
df = df.dropna(
    subset=["previous_margin", "previous_year"]
).copy()

# Remove unusually long gaps from missing/filtered elections
df = df[
    (df["year"] - df["previous_year"]).between(2, 12)
].copy()

# Chronological train/test split
train = df[
    (df["year"] >= 2000) &
    (df["year"] <= 2018)
]
test = df[df["year"].between(2020, 2024)]

X_train = train[["previous_margin"]]
y_train = train["dem_margin"]

X_test = test[["previous_margin"]]
y_test = test["dem_margin"]

# Baseline: predict that the margin stays unchanged
baseline_predictions = test["previous_margin"]

# Train ridge regression
model = Ridge(alpha=10.0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Training observations:", len(train))
print("Testing observations:", len(test))

print("\nBaseline MAE:",
      mean_absolute_error(y_test, baseline_predictions))

print("Ridge MAE:",
      mean_absolute_error(y_test, predictions))

print("\nModel coefficient:", model.coef_[0])
print("Model intercept:", model.intercept_)

results = test[
    ["year", "state", "previous_margin", "dem_margin"]
].copy()

results["ridge_estimate"] = predictions
results["absolute_error"] = abs(
    results["dem_margin"] - results["ridge_estimate"]
)

print("\nSample results:")
print(results.head(10).to_string(index=False))