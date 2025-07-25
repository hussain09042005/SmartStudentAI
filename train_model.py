import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load training data
df_train = pd.read_csv("sample_data.csv")

# Split features and target
X = df_train.iloc[:, 2:-1]  # Skip Name & Roll No, take up to 'Final Result'
y = df_train["Final Result"]

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")
