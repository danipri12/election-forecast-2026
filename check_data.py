import pandas as pd

senate = pd.read_csv("data/senate_2026.csv")
house = pd.read_csv("data/house_history.csv")

print("Senate races:", len(senate))
print("Senate polls available:",
      senate["poll_margin_d"].notna().sum())

print("House historical rows:",
      house["president_party_seat_change"].notna().sum())

print("\nSenate sample:")
print(senate.head())

print("\nHouse data:")
print(house)