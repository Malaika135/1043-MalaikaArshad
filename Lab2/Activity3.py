import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# 1. Load dataset & explore
data = pd.read_csv("Data Set/Car Price Prediction.csv")
print("--- Dataset Head ---")
print(data.head())
print("\n--- Columns Available ---")
print(data.columns.tolist())

# Choose a valid independent feature ('horsepower') and target ('price')
feature_col = 'horsepower'
target_col = 'price'

X = data[[feature_col]].values
y = data[target_col].values

# 2. Apply Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)
print(f"\nLinear Regression R² Score: {r2_score(y, y_pred_lin):.4f}")

# 3. Apply Polynomial Regression (degree = 2, 3, 4) and compare results
degrees = [2, 3, 4]
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='black', label='Actual Data', alpha=0.5)

for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(X)
    
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    y_pred_poly = poly_model.predict(X_poly)
    
    print(f"Polynomial Regression (deg={deg}) R² Score: {r2_score(y, y_pred_poly):.4f}")
    
    # Sort for smooth curve plotting
    sorted_indices = np.argsort(X.ravel())
    plt.plot(X[sorted_indices], y_pred_poly[sorted_indices], label=f'Degree {deg}', linewidth=2)

# 4. Plot fitted curves for different polynomial degrees
plt.xlabel(feature_col.capitalize())
plt.ylabel(target_col.capitalize())
plt.title('Car Price Prediction: Polynomial Regression Comparison')
plt.legend()
plt.show()
