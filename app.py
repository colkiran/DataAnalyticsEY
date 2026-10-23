
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="BI Readiness Profiler",
    page_icon="📊",
    layout="wide"
)

REQUIRED = [
    "TransactionID", "TransactionDate", "CustomerID",
    "Product", "Category", "Quantity", "UnitPrice",
    "SalesAmount", "Region"
]


def create_demo_data():
    """Generate 100 retail rows with intentional data-quality defects."""
    rng = np.random.default_rng(42)
    n = 100

    df = pd.DataFrame({
        "TransactionID": [f"TXN{i:04d}" for i in range(1001, 1001 + n)],
        "TransactionDate": pd.date_range(
            "2026-01-01", periods=n, freq="D"
        ).strftime("%Y-%m-%d"),
        "CustomerID": [
            f"C{rng.integers(101, 151)}" for _ in range(n)
        ],
        "Product": rng.choice(
            ["Laptop", "Mouse", "Keyboard", "Monitor", "Headset"], n
        ),
        "Category": rng.choice(
            ["Electronics", "Accessories", "Office"], n
        ),
        "Quantity": rng.integers(1, 6, n),
        "UnitPrice": rng.choice(
            [500, 1200, 1500, 10000, 50000], n
        ).astype(float),
        "Region": rng.choice(
            ["North", "South", "East", "West"], n
        )
    })

    df["SalesAmount"] = df["Quantity"] * df["UnitPrice"]

    # Introduce defects for the workshop.
    df.loc[2, "CustomerID"] = None
    df.loc[7, "Product"] = None
    df.loc[10, "Quantity"] = -1
    df.loc[10, "SalesAmount"] = -df.loc[10, "UnitPrice"]
    df.loc[15, "TransactionDate"] = "not-a-date"
    df.loc[20, "SalesAmount"] = 99999
    df.loc[25, "Category"] = "electronics"
    df.loc[30, "UnitPrice"] = 0
    df.loc[35, "TransactionID"] = None

    # Create a fully duplicated row.
    df.loc[40] = df.loc[39].copy()

    # Create a duplicate transaction ID.
    df.loc[50, "TransactionID"] = df.loc[49, "TransactionID"]

    return df


def add_check(results, name, passed, detail):
    results.append({
        "Check": name,
        "Status": "PASS" if passed else "FAIL",
        "Details": detail
    })


def profile_data(df):
    results = []
    missing_columns = [c for c in REQUIRED if c not in df.columns]

    add_check(
        results, "Required columns",
        len(missing_columns) == 0,
        "All required columns found" if not missing_columns
        else "Missing: " + ", ".join(missing_columns)
    )

    if missing_columns:
        return pd.DataFrame(results)

    # Missing-value percentage: must be <= 1% per column.
    missing_pct = df[REQUIRED].isna().mean() * 100
    worst_col = missing_pct.idxmax()
    worst_pct = missing_pct.max()

    add_check(
        results, "Missing values",
        bool((missing_pct <= 1).all()),
        f"Worst column: {worst_col} ({worst_pct:.1f}% missing)"
    )

    duplicates = int(df.duplicated().sum())
    add_check(
        results, "Duplicate rows",
        duplicates == 0,
        f"{duplicates} fully duplicated rows"
    )

    ids = df["TransactionID"].dropna()
    duplicate_ids = int(ids.duplicated().sum())
    invalid_ids = int(df["TransactionID"].isna().sum())

    add_check(
        results, "Transaction ID uniqueness",
        duplicate_ids == 0 and invalid_ids == 0,
        f"{duplicate_ids} duplicate IDs; {invalid_ids} missing IDs"
    )

    dates = pd.to_datetime(df["TransactionDate"], errors="coerce")
    invalid_dates = int(dates.isna().sum())

    add_check(
        results, "Valid transaction dates",
        invalid_dates == 0,
        f"{invalid_dates} missing or invalid dates"
    )

    quantity = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["UnitPrice"], errors="coerce")
    sales = pd.to_numeric(df["SalesAmount"], errors="coerce")

    invalid_qty = int((quantity.isna() | (quantity <= 0)).sum())
    invalid_price = int((price.isna() | (price <= 0)).sum())

    add_check(
        results, "Positive quantity",
        invalid_qty == 0,
        f"{invalid_qty} invalid or non-positive values"
    )

    add_check(
        results, "Positive unit price",
        invalid_price == 0,
        f"{invalid_price} invalid or non-positive values"
    )

    invalid_sales = sales.isna() | ~np.isfinite(sales)
    expected = quantity * price
    mismatch = (
        ~invalid_sales
        & expected.notna()
        & ~np.isclose(
            sales.fillna(0),
            expected.fillna(0),
            rtol=0,
            atol=0.01
        )
    )
    mismatch_count = int(mismatch.sum())

    add_check(
        results, "Sales calculation",
        not invalid_sales.any() and mismatch_count == 0,
        f"{mismatch_count} calculation mismatches; "
        f"{int(invalid_sales.sum())} invalid sales values"
    )

    return pd.DataFrame(results)


st.title("📊 BI Readiness Profiler")
st.write(
    "Upload retail data, discover quality issues, "
    "visualize patterns, and generate a BI readiness report."
)

uploaded = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx"]
)

use_demo = st.toggle("Use the built-in demo dataset", value=True)

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.success(f"Loaded {uploaded.name}")
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()
elif use_demo:
    df = create_demo_data()
    st.info("Using the demo dataset with intentional quality issues.")
else:
    st.info("Upload a file or enable the demo dataset to continue.")
    st.stop()

st.subheader("1. Dataset overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{len(df.columns):,}")
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

with st.expander("Preview raw data", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

st.subheader("2. Column profiling")

profile = pd.DataFrame({
    "Column": df.columns,
    "Data type": [str(df[c].dtype) for c in df.columns],
    "Missing count": [int(df[c].isna().sum()) for c in df.columns],
    "Missing %": [
        round(df[c].isna().mean() * 100, 2) for c in df.columns
    ],
    "Unique values": [int(df[c].nunique()) for c in df.columns]
})
st.dataframe(profile, use_container_width=True)

st.subheader("3. Visual data-quality checks")

missing_chart = profile[profile["Missing count"] > 0]
if not missing_chart.empty:
    fig = px.bar(
        missing_chart, x="Column", y="Missing count",
        title="Missing values by column"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("No missing values detected.")

if "SalesAmount" in df.columns:
    sales_values = pd.to_numeric(df["SalesAmount"], errors="coerce")
    sales_values = sales_values.dropna()
    sales_values = sales_values[np.isfinite(sales_values)]

    if not sales_values.empty:
        fig = px.histogram(
            x=sales_values, nbins=20,
            title="Sales amount distribution",
            labels={"x": "Sales amount", "y": "Number of rows"}
        )
        st.plotly_chart(fig, use_container_width=True)

if "Category" in df.columns:
    category_counts = (
        df["Category"].fillna("(Missing)")
        .astype(str).value_counts().rename_axis("Category")
        .reset_index(name="Rows")
    )
    fig = px.bar(
        category_counts, x="Category", y="Rows",
        title="Category values as received"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("4. BI readiness assessment")

checks = profile_data(df)
failed = int((checks["Status"] == "FAIL").sum())
passed = int((checks["Status"] == "PASS").sum())
ready = failed == 0

a, b, c = st.columns(3)
a.metric("Checks passed", passed)
b.metric("Checks failed", failed)
c.metric("Readiness", "PASS" if ready else "FAIL")

if ready:
    st.success("PASS — Data meets all mandatory workshop checks.")
else:
    st.error("FAIL — Fix the reported issues before the BI refresh.")

st.dataframe(checks, use_container_width=True)

st.subheader("5. Download reports")

st.download_button(
    "Download quality report (CSV)",
    data=checks.to_csv(index=False).encode("utf-8"),
    file_name="bi_readiness_report.csv",
    mime="text/csv"
)

st.download_button(
    "Download column profile (CSV)",
    data=profile.to_csv(index=False).encode("utf-8"),
    file_name="column_profile.csv",
    mime="text/csv"
)

st.download_button(
    "Download source data (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="profiled_source_data.csv",
    mime="text/csv"
)

st.caption(
    "Workshop prototype: validation uses predefined retail rules. "
    "Passing these checks does not guarantee complete BI correctness."
)
