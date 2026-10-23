import pandas as pd
from pathlib import Path

df = pd.read_csv(r"C:\Users\Kiran\Data Analytics EY\DataSource\daily_sales.csv")

issues = []

# 1. Required values
for col in ["SalesID", "Product", "Quantity", "UnitPrice", "SalesDate"]:
    mask = df[col].isna() | df[col].astype("string").str.strip().eq("").fillna(False)
    for idx in df.index[mask]:
        issues.append({
            "Row": idx + 2,
            "SalesID": df.at[idx, "SalesID"],
            "Rule": "Completeness",
            "Details": f"{col} is missing"
        })

# 2. Duplicate IDs
mask = df["SalesID"].notna() & df["SalesID"].duplicated(keep=False)
for idx in df.index[mask]:
    issues.append({
        "Row": idx + 2,
        "SalesID": df.at[idx, "SalesID"],
        "Rule": "Uniqueness",
        "Details": "Duplicate SalesID"
    })

# 3. Numeric type checks
for col in ["Quantity", "UnitPrice"]:
    original = df[col].copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = original.notna() & df[col].isna()
    for idx in df.index[mask]:
        issues.append({
            "Row": idx + 2,
            "SalesID": df.at[idx, "SalesID"],
            "Rule": "Data type",
            "Details": f"{col} must be numeric"
        })

# 4. Business rules
for col, condition, message in [
    ("Quantity", df["Quantity"] <= 0, "Quantity must be positive"),
    ("UnitPrice", df["UnitPrice"] < 0, "Price cannot be negative")
]:
    for idx in df.index[condition.fillna(False)]:
        issues.append({
            "Row": idx + 2,
            "SalesID": df.at[idx, "SalesID"],
            "Rule": "Business rule",
            "Details": message
        })

# 5. Date validation
dates = pd.to_datetime(df["SalesDate"], errors="coerce")
mask = df["SalesDate"].notna() & dates.isna()
for idx in df.index[mask]:
    issues.append({
        "Row": idx + 2,
        "SalesID": df.at[idx, "SalesID"],
        "Rule": "Date validation",
        "Details": "Invalid SalesDate"
    })

# 6. Gate publication
if issues:
    pd.DataFrame(issues).to_csv("data_quality_errors.csv", index=False)
    Path("validated_sales.csv").unlink(missing_ok=True)
    raise SystemExit(
        f"BI REFRESH BLOCKED: {len(issues)} quality issues found."
    )

df["SalesDate"] = dates
df["SalesAmount"] = df["Quantity"] * df["UnitPrice"]
df.to_csv("validated_sales.csv", index=False)
print("BI REFRESH APPROVED: validated_sales.csv created.")