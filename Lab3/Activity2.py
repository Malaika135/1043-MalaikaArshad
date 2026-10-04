import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# 1. Load dataset & explore
data = pd.read_csv("Data Set/Student_Performance.csv")

print("--- Dataset Head ---")
print(data.head())

print("\n--- Columns Available ---")
print(data.columns.tolist())

# Choose independent feature and target
feature_col = 'Hours_Studied'
target_col = 'Exam_Score'

X = data[[feature_col]].values
y = data[target_col].values

# 2. Apply Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

y_pred_lin = lin_reg.predict(X)

print(f"\nLinear Regression R² Score: {r2_score(y, y_pred_lin):.4f}")

# Plot actual data and linear regression line
plt.figure(figsize=(8, 6))

plt.scatter(
    X,
    y,
    color='black',
    label='Actual Data',
    alpha=0.5
)

sorted_indices = np.argsort(X.ravel())

plt.plot(
    X[sorted_indices],
    y_pred_lin[sorted_indices],
    label='Linear Regression',
    linewidth=2
)

plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Student Performance: Linear Regression')
plt.legend()
plt.show()

# 3. Apply Polynomial Regression
# Degrees 2, 3 and 4
degrees = [2, 3, 4]

plt.figure(figsize=(8, 6))

plt.scatter(
    X,
    y,
    color='black',
    label='Actual Data',
    alpha=0.5
)

for deg in degrees:

    # Create polynomial features
    poly = PolynomialFeatures(degree=deg)

    X_poly = poly.fit_transform(X)

    # Create and train model
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)

    # Predict values
    y_pred_poly = poly_model.predict(X_poly)

    # Calculate R² score
    score = r2_score(y, y_pred_poly)

    print(
        f"Polynomial Regression (degree={deg}) "
        f"R² Score: {score:.4f}"
    )

    # Sort values for smooth curve
    sorted_indices = np.argsort(X.ravel())

    plt.plot(
        X[sorted_indices],
        y_pred_poly[sorted_indices],
        label=f'Degree {deg}',
        linewidth=2
    )

# 4. Plot fitted curves
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')

plt.title(
    'Student Performance: Polynomial Regression Comparison'
)

plt.legend()
plt.show()
