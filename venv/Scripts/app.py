import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="SmartStudent AI", layout="wide")
st.title("📊 SmartStudent AI: Predictive Academic Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload Student Marks CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show raw data
    st.subheader("📄 Uploaded Data")
    st.dataframe(df)

    # Predict if 'Final Result' column not in uploaded file
    if 'Final Result' not in df.columns:
        try:
            features = df[["Attendance (%)", "Assignment Score", "Midterm Marks", "Final Exam Marks", "Class Participation"]]

            predictions = model.predict(features)
            df['Predicted_Result'] = predictions
        except KeyError:
            st.error("⚠️ The uploaded CSV must contain the following columns: 'Attendance (%)', 'Assignment Score', 'Midterm Marks', 'Final Exam Marks', 'Class Participation'")
    else:
        st.success("✅ 'Final Result' already present in uploaded file.")

    # Charts
    st.subheader("📈 Performance Chart")
    try:
        fig, ax = plt.subplots()
        sns.boxplot(data=df[["Attendance (%)", "Assignment Score", "Midterm Marks", "Final Exam Marks", "Class Participation"]], ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"📊 Could not plot chart: {e}")

    # Display prediction result
    st.subheader("✅ Prediction Results")
    st.dataframe(df)

    # Download option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results as CSV", csv, "predicted_results.csv", "text/csv")

else:
    st.info("📎 Please upload a CSV file to get started.")
