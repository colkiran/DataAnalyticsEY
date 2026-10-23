
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# 1. Generate sample historical data
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

# 2. Generate a simulated churn target
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

X = df.drop(columns="Churned")
y = df["Churned"]

# 3. Define preprocessing
numeric = [
    "TenureMonths", "MonthlySpend", "SupportCalls"
]
categorical = ["ContractType"]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

# 4. Train/test split and model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# 5. Evaluate the model
scores = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", round(roc_auc_score(y_test, scores), 3))
print(classification_report(
    y_test, model.predict(X_test)
))

# 6. Save the complete preprocessing + model pipeline
joblib.dump(model, "churn_model.joblib")
print("Saved churn_model.joblib")


# install pandas, scikit-learn, joblib, pycaret













