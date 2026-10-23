import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

df = pd.DataFrame({
    "Order_ID": range(1001, 1001 + n),
    "Sales": np.random.gamma(3, 250, n),
    "Quantity": np.random.randint(1, 11, n),
    "Discount": np.random.choice(
        [0, 0.05, 0.10, 0.20, 0.30, 0.50],
        size=n,
        p=[0.15, 0.15, 0.20, 0.25, 0.15, 0.10]
    ),
    "Category": np.random.choice(
        ["Furniture", "Technology", "Office Supplies"],
        n
    ),
    "Region": np.random.choice(
        ["North", "South", "East", "West"],
        n
    ),
    "Ship_Mode": np.random.choice(
        ["Standard", "Express", "Same Day"],
        n,
        p=[0.60, 0.30, 0.10]
    ),
    "Customer_Satisfaction": np.random.randint(1, 6, n)
})

# Generate profit with a relationship to sales and discount
df["Profit"] = (
    df["Sales"] * (0.25 - df["Discount"] * 0.85)
    + df["Quantity"] * 5
    + np.random.normal(0, 35, n)
).round(2)

# Introduce a few unusual observations
df.loc[10, "Sales"] = 10000
df.loc[20, "Profit"] = -1500

# Introduce missing values
df.loc[[30, 50, 70], "Sales"] = np.nan
df.loc[[40, 80], "Discount"] = np.nan

# Save the dataset
df.to_csv("retail_sales.csv", index=False)

print("Dataset shape:", df.shape)
print(df.head())
print(df.info())