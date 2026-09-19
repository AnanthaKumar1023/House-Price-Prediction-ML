import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("house_prices.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# --------------------------------------------------
# 2. HANDLE MISSING VALUES
# --------------------------------------------------

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

categorical_columns = df.select_dtypes(include=["object", "string"]).columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


# --------------------------------------------------
# 3. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("price", axis=1)
y = df["price"]


# --------------------------------------------------
# 4. DEFINE FEATURES
# --------------------------------------------------

categorical_features = ["location"]

numeric_features = [
    "area",
    "bedrooms",
    "bathrooms",
    "age"
]


# --------------------------------------------------
# 5. ENCODE CATEGORICAL DATA
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# 6. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 7. CREATE MODEL
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# --------------------------------------------------
# 8. TRAIN MODEL
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed!")


# --------------------------------------------------
# 9. PREDICT TEST DATA
# --------------------------------------------------

y_pred = model.predict(X_test)


print("\nActual vs Predicted:")
for actual, predicted in zip(y_test, y_pred):
    print(
        f"Actual: ₹{actual:,.0f} | "
        f"Predicted: ₹{predicted:,.0f}"
    )


# --------------------------------------------------
# 10. MODEL EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"MAE  : ₹{mae:,.2f}")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# --------------------------------------------------
# 11. ACTUAL VS PREDICTED GRAPH
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    range(len(y_test)),
    y_test.values,
    marker="o",
    label="Actual Price"
)

plt.plot(
    range(len(y_pred)),
    y_pred,
    marker="x",
    label="Predicted Price"
)

plt.xlabel("Test House")
plt.ylabel("House Price")

plt.title("Actual vs Predicted House Prices")

plt.legend()
plt.grid(True)

plt.show()


# --------------------------------------------------
# 12. SCATTER PLOT
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")

plt.title("Actual vs Predicted House Prices")

plt.grid(True)

plt.show()


# --------------------------------------------------
# 13. NEW HOUSE PREDICTION
# --------------------------------------------------

new_house = pd.DataFrame({
    "area": [1800],
    "bedrooms": [3],
    "bathrooms": [2],
    "age": [5],
    "location": ["Chennai"]
})

predicted_price = model.predict(new_house)

print("\n==============================")
print("NEW HOUSE PREDICTION")
print("==============================")

print("Area       : 1800 sq.ft")
print("Bedrooms   : 3")
print("Bathrooms  : 2")
print("Age        : 5 years")
print("Location   : Chennai")

print(f"Predicted Price : ₹{predicted_price[0]:,.2f}")