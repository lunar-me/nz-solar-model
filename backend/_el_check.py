import pandas as pd
from pathlib import Path

D = Path(r"d:\lunar-me\nz-solar-model\data")
el = pd.read_csv(D / "Christchurch Electricity - 1h - 2025-08-18 - 2026-08-17.csv")
print("cols:", list(el.columns), "rows:", len(el))
print("head date sample:", el["date"].iloc[0], "|", el["date"].iloc[-1])

# parse timestamps
ts = pd.to_datetime(el["date"], format="%I:%M%p %d %B %Y")
print("first parsed:", ts.iloc[0], "last:", ts.iloc[-1])
usage = pd.to_numeric(el["usage"].str.replace(" kWh", ""), errors="coerce")
dollars = pd.to_numeric(el["dollars"].str.replace("$", ""), errors="coerce")
print("total usage kWh:", round(usage.sum(), 1), " total dollars:", round(dollars.sum(), 2))
print("avg $/kWh:", round(dollars.sum() / usage.sum(), 3))
print("usage[0..2]:", usage.iloc[:3].tolist(), " dollars[0..2]:", dollars.iloc[:3].tolist())
