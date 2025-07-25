import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("model.pkl")

# App title and team name
st.set_page_config(page_title="SmartStudent AI Dashboard", layout="wide")
st.title("📊 SmartStudent AI: Predictive Academic Dashboard")
st.subheader("By Team Data Decoders")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["Upload CSV & Predict", "Visual Analysis", "About Team"])

# Upload & Predict Section
if menu == "Upload CSV & Predict":
    st.header("📁 Upload Student CSV File")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "Final Result" in df.columns:
            X = df.iloc[:, 2:-1]
        else:
            X = df.iloc[:, 2:]

        try:
            predictions = model.predict(X)
            df["Predicted Result"] = predictions
            st.success("✅ Prediction Completed")

            st.dataframe(df)

            # Download result
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predicted CSV", csv, "predicted_results.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.info("📥 Upload a student dataset in CSV format.")

# Visual Analysis Section
elif menu == "Visual Analysis":
    st.header("📈 Student Performance Visual Analysis")
    st.markdown("Upload a CSV file with student scores and explore visual trends.")

    uploaded_file = st.file_uploader("Upload CSV for Visual Analysis", type="csv", key="viz")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Show correlation heatmap
        st.subheader("Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Plot distributions
        st.subheader("📌 Score Distribution")
        score_columns = df.select_dtypes(include='number').columns.tolist()

        if score_columns:
            selected_col = st.selectbox("Select a column to view distribution", score_columns)
            fig = px.histogram(df, x=selected_col, nbins=20, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig)
        else:
            st.warning("⚠️ No numeric columns found in your file.")

# About Team Section
elif menu == "About Team":
    st.header("👨‍💻 About Team Data Decoders")
    st.markdown("""
    - 📚 This dashboard is built by **Team Data Decoders** from **Vidyalankar College**, as part of the BSc IT final-year project.
    - 🎯 The goal is to help predict student outcomes using machine learning and present interactive visual analysis tools.
    - 💡 Powered by Streamlit, Plotly, Seaborn, and Scikit-learn.
    - 💻 Developed and deployed in 2025.
    """)

    st.image("https://static.streamlit.io/examples/team.png", width=400)
