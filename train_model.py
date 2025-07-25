import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load clean data
df = pd.read_csv("venv/Scripts/student_data.csv", dtype=str)  # Correct file name

# Feature and target columns
features = ["Attendance (%)", "Assignment Score", "Midterm Marks", "Final Exam Marks", "Class Participation"]
target = "Final Result"

# Convert features to numeric
df[features] = df[features].apply(pd.to_numeric, errors='coerce')

# Drop any row where feature or target is missing
df = df.dropna(subset=features + [target])

# Filter valid target labels
df = df[df[target].isin(['Pass', 'Fail'])]

# Encode target
y = df[target].map({'Pass': 1, 'Fail': 0})
X = df[features]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as 'model.pkl'")
