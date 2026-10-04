import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

# 1. Load dataset & explore

data = load_breast_cancer(as_frame=True)

df = data.frame

print("--- Dataset Head ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Dataset Information ---")
print(df.info())

# Target distribution
plt.figure(figsize=(5, 4))
sns.countplot(x='target', data=df)
plt.title("Breast Cancer Target Distribution")
plt.xlabel("Target (0 = Malignant, 1 = Benign)")
plt.ylabel("Number of Samples")
plt.show()


# 2. Select features and target

X = df.drop('target', axis=1)
y = df['target']

print("\n--- Features ---")
print(X.head())

print("\n--- Target ---")
print(y.head())


# 3. Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. Train Logistic Regression model

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)


# 6. Predictions & Probabilities

y_pred = log_model.predict(X_test)

y_prob = log_model.predict_proba(X_test)[:, 1]


# 7. Evaluation Metrics

print("\n--- Evaluation Metrics ---")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))

print("Recall:", recall_score(y_test, y_pred))

print("F1-score:", f1_score(y_test, y_pred))


# 8. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\n--- Confusion Matrix ---")
print(cm)

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


# 9. ROC Curve

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5, 4))

plt.plot(
    fpr,
    tpr,
    color='darkorange',
    lw=2,
    label=f'ROC curve (AUC = {roc_auc:.2f})'
)

plt.plot(
    [0, 1],
    [0, 1],
    color='navy',
    lw=2,
    linestyle='--'
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic (ROC) Curve')

plt.legend(loc="lower right")

plt.show()
