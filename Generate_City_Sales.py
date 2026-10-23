import pandas as pd
import numpy as np
import plotly.express as px

np.random.seed(42)

n = 1000

cities = ["Bengaluru", "Hyderabad", "Chennai", "Mumbai"]
categories = ["Electronics", "Furniture", "Clothing", "Groceries"]

df = pd.DataFrame({
    "OrderID": [f"ORD{i}" for i in range(1001, 1001 + n)],
    "OrderDate": pd.to_datetime(
        np.random.choice(
            pd.date_range("2026-01-01", "2026-09-30"),
            n
        )
    ),
    "City": np.random.choice(cities, n),
    "Category": np.random.choice(categories, n),
    "Quantity": np.random.randint(1, 11, n),
    "DiscountPct": np.random.choice(
        [0, 5, 10, 15, 20, 25],
        n
    )
})

prices = {
    "Electronics": 5000,
    "Furniture": 8000,
    "Clothing": 1200,
    "Groceries": 500
}

cost_rates = {
    "Electronics": 0.72,
    "Furniture": 0.68,
    "Clothing": 0.55,
    "Groceries": 0.82
}

df["Product"] = df["Category"].map({
    "Electronics": "Headphones",
    "Furniture": "Office Chair",
    "Clothing": "Jacket",
    "Groceries": "Food Pack"
})

df["UnitPrice"] = df["Category"].map(prices)

df["Revenue"] = (
    df["Quantity"] * df["UnitPrice"]
    * (1 - df["DiscountPct"] / 100)
).round(2)

df["Cost"] = (
    df["Quantity"] * df["UnitPrice"]
    * df["Category"].map(cost_rates)
).round(2)

df["Profit"] = (df["Revenue"] - df["Cost"]).round(2)

print(df.head())
print("Rows:", len(df))
print("Missing values:", df.isna().sum().sum())