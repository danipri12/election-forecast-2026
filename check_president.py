import pandas as pd

df = pd.read_csv("data/president_results.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nYears:", sorted(df["year"].unique()))
print("\nSample:")
print(df.head().to_string(index=False))