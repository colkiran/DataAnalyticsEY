import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Create historical sales data
data = {
    "Advertising": [10, 12, 15, 13, 18, 20, 17, 22],
    "Sales": [100, 115, 130, 125, 155, 170, 160, 185]
}

df = pd.DataFrame(data)

# Step 2: Descriptive analytics
print("Historical data:")
print(df)

print("\nTotal sales:", df["Sales"].sum())
print("Average sales:", df["Sales"].mean())
print("Highest sales:", df["Sales"].max())

# Step 3: Prepare input and target
X = df[["Advertising"]]
y = df["Sales"]

# Step 4: Train the predictive model
model = LinearRegression()
model.fit(X, y)

# Step 5: Predict sales for advertising spend of 24
new_data = pd.DataFrame({"Advertising": [24]})
predicted_sales = model.predict(new_data)[0]

print("\nPredicted sales:", round(predicted_sales, 2))
print("Regression coefficient:", round(model.coef_[0], 2))
print("Intercept:", round(model.intercept_, 2))

# Step 6: Visualize historical data and prediction
plt.scatter(df["Advertising"], df["Sales"],
            label="Historical sales")

plt.plot(
    df["Advertising"],
    model.predict(X),
    label="Regression line"
)

plt.scatter(
    [24], [predicted_sales],
    marker="X", s=150,
    label="Predicted sales"
)

plt.xlabel("Advertising spend (₹000s)")
plt.ylabel("Sales (₹000s)")
plt.title("Retail Sales: Descriptive to Predictive Analytics")
plt.legend()
plt.show()