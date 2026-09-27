import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load and explore the dataset
df = pd.read_csv("Data Set/Medical Cost Personal Datasets.csv")
print("--- Dataset Head ---")
print(df.head())
print("\n--- Dataset Info ---")
print(df.info())

# Data Exploration & Visualization before training
plt.figure(figsize=(6, 4))
sns.histplot(df['charges'], kde=True, color='green')
plt.title("Distribution of Medical Insurance Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.show()

# 2. Preprocess: Encoding categorical features and scaling numerical features
le = LabelEncoder()
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])
if 'sex' in df.columns:
    df['sex'] = le.fit_transform(df['sex'])

# Features & Target definition
X = df[['age', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train a Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# 4. Evaluate using RMSE & R² score
y_pred = lin_reg.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# 5. Plot Predicted vs Actual Costs
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color='purple', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Insurance Costs')
plt.ylabel('Predicted Insurance Costs')
plt.title('Predicted vs Actual Costs')
plt.legend()
plt.show()
