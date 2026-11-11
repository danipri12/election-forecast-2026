import pandas as pd

df = pd.read_csv("data/senate_final_picks.csv")

print("CURRENT SEAT PARTY VALUES")
print(df["seat_party"].value_counts(dropna=False))

print("\nSENATE SEAT CHANGES")
df["seat_flip"] = df["final_pick"] != df["seat_party"]

print(
    df.loc[df["seat_flip"],
           ["state", "seat_party", "final_pick"]]
    .sort_values("state")
    .to_string(index=False)
)

print("\nTOTAL PROJECTED FLIPS:", df["seat_flip"].sum())