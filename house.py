import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/house_history.csv")

# Historical observations with known outcomes
history = df.dropna(
    subset=["president_party_seat_change"]
).copy()

X = history[["president_approval"]]
y = history["president_party_seat_change"]

loo = LeaveOneOut()

baseline_errors = []
ridge_errors = []

results = []

for train_idx, test_idx in loo.split(X):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Baseline: mean seat change in training elections
    baseline = y_train.mean()

    # Ridge regression
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)

    ridge_prediction = model.predict(X_test)[0]
    actual = y_test.iloc[0]

    baseline_errors.append(abs(actual - baseline))
    ridge_errors.append(abs(actual - ridge_prediction))

    results.append({
        "year": history.iloc[test_idx[0]]["year"],
        "actual_change": actual,
        "baseline_estimate": baseline,
        "ridge_estimate": ridge_prediction
    })

print("Historical cross-validation results:")
print(pd.DataFrame(results).to_string(index=False))

print("\nBaseline MAE:", np.mean(baseline_errors))
print("Ridge MAE:", np.mean(ridge_errors))

print("\nHouse data:")
print(df.to_string(index=False))

# Convert generic ballot to the president's party perspective
history["generic_pres_party"] = np.where(
    history["president_party"] == "D",
    history["generic_ballot_d"],
    -history["generic_ballot_d"]
)

print("\nGeneric ballot vs presidential-party seat change:")
print(
    history[
        ["year", "generic_pres_party",
         "president_party_seat_change"]
    ].to_string(index=False)
)

print("\nCorrelation:")
print(
    history["generic_pres_party"].corr(
        history["president_party_seat_change"]
    )
)