import pandas as pd

from pycaret.classification import (
setup, compare_models, save_model
)

df = pd.read_csv(r"C:\Users\Kiran\Data Analytics EY\DataSource\historical_churn.csv")

experiment = setup(
	data=df,
    target="Churned",
    session_id=42,
    train_size=0.8,
    fold=5,
    verbose=False,
    html=False
)