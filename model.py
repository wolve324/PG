import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

# Load CSV: skip 3 header rows
df = pd.read_csv("BTC-USD.csv", skiprows=3)

# Manually set correct column names
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

# Keep only Date and Close
df = df[["Date", "Close"]]
df.dropna(inplace=True)

# Convert Date to datetime object
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

# Create a numerical "Days" feature for regression
df["Days"] = (df["Date"] - df["Date"].min()).dt.days

# Prepare features and target
X = df[["Days"]]
y = df["Close"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Predict next day's price
next_day = [[X["Days"].max() + 1]]
future_price = model.predict(next_day)
print(f"Predicted price for next day: {future_price[0]}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(df["Date"], y, label="Actual Price")
plt.plot(df["Date"].iloc[len(X_train):], y_pred, label="Predicted Price", linestyle="--")
plt.legend()
plt.xlabel("Date")
plt.ylabel("BTC Price")
plt.title("Bitcoin Price Prediction - Linear Regression")
plt.grid(True)
plt.show()

# Save model to file
with open("crypto_model.pkl", "wb") as f:
    pickle.dump(model, f)
