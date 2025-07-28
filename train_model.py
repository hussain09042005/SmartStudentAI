import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load training data
df_train = pd.read_csv("sample_data.csv")

# Split features and target
feature_columns = ['Assignment Score', 'Class Participation', 'Midterm Marks', 'Final Exam Marks']
X = df_train[feature_columns]
y = df_train["Final Result"]

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict on training data for evaluation
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)

print(f"✅ Model trained with accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Save the trained model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")
