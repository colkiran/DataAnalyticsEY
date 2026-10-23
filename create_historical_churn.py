import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2000

df = pd.DataFrame({
    "TenureMonths": rng.integers(1, 73, n),
    "MonthlySpend": rng.integers(500, 6001, n),
    "SupportCalls": rng.integers(0, 9, n),
    "ContractType": rng.choice(
        ["Monthly", "Annual", "TwoYear"], n
    )
})

# Generate a synthetic churn target for training
logit = (
    1.2
    - 0.035 * df["TenureMonths"]
    + 0.00025 * df["MonthlySpend"]
    + 0.40 * df["SupportCalls"]
    + 0.85 * (df["ContractType"] == "Monthly")
    - 0.65 * (df["ContractType"] == "TwoYear")
)

probability = 1 / (1 + np.exp(-logit))
df["Churned"] = rng.binomial(1, probability)

df.to_csv("historical_churn.csv", index=False)

print("Created historical_churn.csv")
print("Rows:", len(df))
print(df.head())
print(df["Churned"].value_counts())
