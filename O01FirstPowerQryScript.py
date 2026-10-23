import pandas

# power bi passes the current power query table to Python as a Dataframe named dataset

result = dataset.copy()

# Ensure the numeric columns have the correct datatype

result["Quantity"] = pd.to_numeric(result["Quantity"], errors = "coerce" )

result["UnitPrice"] = pd.to_numeric(result["UnitPrice"],errors = "coerce")

result["Discount"] = pd.to_numeric(result["Discount"], errors = "Coerce")

# Replacing the missing numeric values
result["Quantity"] = result["Quantity"].fillna(0)
result["UnitPrice"] = result["UnitPrice"].fillna(0)
result["Discount"] = result["Discount"].fillna(0)


# Calculate the sales after discount
result["NetSales"] = (result["Quantity"] * result["UnitPrice"] * (1 - result["Discount"]))

# Classify each transaction
result["OrderValue"] = pd.cut(
    result["NetSales"],
    bins=[-float("inf"), 10000, 50000, float("inf")],
    labels=["Low", "Medium", "High"],
    right=False
)


# Add a businees rule for reporting
result["SalesStatus"] = result["NetSales"].apply(lambda x: "Review" if x >= 100000 else "Normal" )

# Round Calculated Values
result["NetSales"] = result["NetSales"].round(2)

