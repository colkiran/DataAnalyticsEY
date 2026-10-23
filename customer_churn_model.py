
import pandas as pd
from pycaret.classification import ClassificationExperiment

# 1. Load historical data
df = pd.read_csv("customer_churn.csv")
df = df.drop(columns=["CustomerID"])

# 2. Initialize experiment using PyCaret 3.x syntax
exp = ClassificationExperiment()

exp.setup(
    data=df,
    target="Churn",
    session_id=42,
    train_size=0.80,
    fold=5,
    verbose=False,
    html=False
)

# 3. Compare classification algorithms
best_model = exp.compare_models(
    sort="AUC",
    n_select=1
)

print("Best model selected:")
print(best_model)

# 4. Tune the selected model
tuned_model = exp.tune_model(
    best_model,
    optimize="AUC",
    n_iter=10
)

# 5. Evaluate predictions on holdout data
holdout_predictions = exp.predict_model(tuned_model)

print("\nHoldout predictions:")
print(holdout_predictions.head())

# 6. Finalize the model using all available historical data
final_model = exp.finalize_model(tuned_model)

# 7. Save the complete model pipeline
exp.save_model(final_model, "churn_model")

print("\nModel saved as churn_model.pkl")
