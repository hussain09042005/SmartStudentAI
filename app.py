import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SmartStudent AI - Academic Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM STYLES ----------------
st.markdown("""
    <style>
        body {
            background-color: #f8fbff;
        }
        .reportview-container .main {
            background-color: #f8fbff;
        }
        .block-container {
            padding: 2rem 2rem;
        }
        h1, h2, h3 {
            color: #0d6efd;
        }
        .stButton > button {
            background-color: #0d6efd;
            color: white;
            border-radius: 10px;
        }
        .stTextInput, .stFileUploader {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("model.pkl")
except:
    model = None
    st.warning("⚠️ Model not found. Please train the model and include model.pkl.")

# ---------------- MAIN HEADER ----------------
st.markdown(
    "<h1 style='text-align: center;'>📊 SmartStudent AI: Predictive Academic Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["Upload CSV & Predict", "Visual Analysis", "About Team"])

# ---------------- SECTION 1: Upload & Predict ----------------
if section == "Upload CSV & Predict":
    st.subheader("📁 Upload Student CSV File")
    uploaded_file = st.file_uploader("Upload your .csv file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.write("### 📄 Uploaded Data Preview:")
        st.dataframe(df.head())

        if model:
            try:
                X = df.iloc[:, 2:]  # Assuming first 2 columns are Name & Roll No
                predictions = model.predict(X)
                df["Predicted Result"] = predictions

                st.write("### ✅ Prediction Results:")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Result CSV", data=csv, file_name="predicted_results.csv", mime='text/csv')
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------- SECTION 2: Visual Analysis ----------------
elif section == "Visual Analysis":
    st.subheader("📊 Student Performance Analysis")

    sample = pd.read_csv("sample_data.csv")  # Use your dataset name here

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Result Distribution")
        fig = px.pie(sample, names="Final Result", title="Pass/Fail Distribution", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig)

    with col2:
        st.markdown("#### 🧠 Marks Comparison")
        marks_columns = sample.columns[2:-1]
        avg_marks = sample[marks_columns].mean().sort_values(ascending=False)
        fig2 = px.bar(x=avg_marks.index, y=avg_marks.values, labels={'x': "Subjects", 'y': "Average Marks"},
                      title="Average Marks per Subject", color=avg_marks.values, color_continuous_scale='Blues')
        st.plotly_chart(fig2)

    st.markdown("#### 📉 Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(sample[marks_columns].corr(), annot=True, cmap="Blues", fmt=".2f")
    st.pyplot(plt)

# ---------------- SECTION 3: About Team ----------------
elif section == "About Team":
    st.subheader("👨‍💻 About Team Data Decoders")

    st.markdown("""
        <div style='background-color: #e9f5ff; padding: 20px; border-radius: 10px;'>
            <h4 style='color:#0d6efd;'>Project Title:</h4>
            <p><strong>SmartStudent AI: Predictive Academic Dashboard</strong></p>

            <h4 style='color:#0d6efd;'>Team Members:</h4>
            <ul>
                <li>💡 Mohammed Hussain Kamal Ahmed Choudhary</li>
                <li>💡 [Add More Team Members]</li>
            </ul>

            <h4 style='color:#0d6efd;'>Tech Stack:</h4>
            <ul>
                <li>📌 Python</li>
                <li>📌 Streamlit</li>
                <li>📌 Scikit-learn</li>
                <li>📌 Pandas, Seaborn, Plotly</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
