# ===============================================
# STEP 1: Import Libraries
# ===============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# ===============================================
# STEP 2: Create Synthetic Retail Dataset
# ===============================================
np.random.seed(42)
n_rows = 1000

data = {
    "InvoiceNo": np.random.randint(10000, 20000, size=n_rows),
    "StockCode": np.random.randint(100, 500, size=n_rows),
    "Description": np.random.choice(["Laptop", "Phone", "Tablet", "Headphones", "Charger", "Monitor"], size=n_rows),
    "Quantity": np.random.randint(1, 10, size=n_rows),
    "UnitPrice": np.random.uniform(10, 500, size=n_rows).round(2),
    "InvoiceDate": pd.date_range(start="2023-01-01", periods=n_rows, freq="h"),
    "Country": np.random.choice(["USA", "UK", "Germany", "France", "Canada"], size=n_rows),
    "MarketingSpend": np.random.uniform(500, 5000, size=n_rows).round(2)
}

df = pd.DataFrame(data)

# Feature engineering
df["TotalSales"] = df["Quantity"] * df["UnitPrice"]
df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek  # 0=Mon, 6=Sun
df["Month"] = df["InvoiceDate"].dt.month

print("Dataset Preview:")
print(df.head())

# ===============================================
# STEP 3: Define Features and Target
# ===============================================
features = ["UnitPrice", "Quantity", "MarketingSpend", "DayOfWeek", "Month", "Country"]
target = "TotalSales"

X = df[features]
y = df[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================================
# STEP 4: Preprocessing
# ===============================================

numeric_features = ["UnitPrice", "Quantity", "MarketingSpend", "DayOfWeek", "Month"]
categorical_features = ["Country"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ===============================================
# STEP 5: Build Gradient Boosting Pipeline
# ===============================================
gbr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(random_state=42))
])

# ===============================================
# STEP 6: Hyperparameter Tuning (Optional)
# ===============================================
param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1, 0.2],
    "model__max_depth": [3, 4, 5],
    "model__subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=gbr_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print(f"Best CV R² Score: {grid_search.best_score_:.4f}")

# Use the best model
best_model = grid_search.best_estimator_

# ===============================================
# STEP 7: Model Evaluation
# ===============================================
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nTest R² Score: {r2:.4f}")
print(f"Test RMSE: {rmse:.2f}")

# ===============================================
# STEP 8: Visualization - Actual vs Predicted
# ===============================================
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Gradient Boosting: Actual vs Predicted Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================================
# STEP 9: Predict on New Data
# ===============================================
sample_data = pd.DataFrame({
    "UnitPrice": [220],
    "Quantity": [4],
    "MarketingSpend": [1200],
    "DayOfWeek": [2],  # Wednesday
    "Month": [9],       # September
    "Country": ["USA"]
})

predicted_sales = best_model.predict(sample_data)
print(f"\nPredicted Sales for Sample Input: ${predicted_sales[0]:.2f}")
