import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load data and models
df = pd.read_csv("training_features_extended.csv")
rf = joblib.load("models/rf_model_extended.joblib")

# 1. Plot Feature Importance (Top 10)
# This tells you which sensor data actually mattered
importances = rf.feature_importances_
feature_names = df.drop(columns=["student_id", "quality"]).columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Top 10 Most Important Features (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("[OK] Saved feature_importance.png")

# 2. Plot Confusion Matrix
# We need to run a quick prediction to get the matrix
from sklearn.model_selection import train_test_split
X = df.drop(columns=["student_id", "quality"])
y = df["quality"].map({"good": 1, "bad": 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bad', 'Good']).plot(cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.savefig('confusion_matrix.png')
print("[OK] Saved confusion_matrix.png")
plt.show()