import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import base64

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="SmartStudent AI Dashboard",
    page_icon="📘",
    layout="wide"
)

# ------------------- CUSTOM HEADER -------------------
st.markdown("""
    <div style="background: linear-gradient(to right, #4facfe, #00f2fe); padding: 20px 10px; border-radius: 10px;">
        <h2 style="color: white; text-align: center;">📘 SmartStudent AI: Predictive Academic Dashboard</h2>
    </div>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
st.sidebar.title("📚 SmartStudent AI")
page = st.sidebar.radio("Navigate", ["Upload CSV & Predict", "Visual Analysis", "About Team"])

# ------------------- LOAD MODEL -------------------
model = joblib.load("model.pkl")

# ------------------- UPLOAD & PREDICT -------------------
if page == "Upload CSV & Predict":
    st.subheader("📤 Upload Student Data (Without Final Result)")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Prediction
        if "Final Result" not in df.columns:
            input_data = df.copy()
            result = model.predict(input_data.iloc[:, 2:])
            df["Final Result"] = result

            st.success("✅ Prediction complete!")
            st.write(df)

            # Download predicted CSV
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predicted_results.csv">📥 Download Predicted Results</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Your file already contains 'Final Result' column. This is for prediction only.")

# ------------------- VISUAL ANALYSIS -------------------
elif page == "Visual Analysis":
    st.subheader("📈 Academic Performance Overview")

    uploaded_viz = st.file_uploader("Upload Full Dataset (with Final Result)", type=["csv"])

    if uploaded_viz is not None:
        df = pd.read_csv(uploaded_viz)

        if "Final Result" in df.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎓 Total Students", len(df))
            with col2:
                st.metric("✅ Passed", df[df['Final Result'] == 'Pass'].shape[0])
            with col3:
                st.metric("❌ Failed", df[df['Final Result'] == 'Fail'].shape[0])

            # Plotly bar chart
            fig = px.bar(df, x='Name', y='Final Exam Marks', color='Final Result', title='Final Exam Marks by Student')
            st.plotly_chart(fig, use_container_width=True)

            # Seaborn heatmap
            st.subheader("📊 Correlation Heatmap")
            corr = df.select_dtypes(include=np.number).corr()
            fig2, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig2)

        else:
            st.error("❌ Final Result column not found. Please upload complete data for analysis.")

# ------------------- ABOUT TEAM -------------------
elif page == "About Team":
    st.subheader("👨‍💻 Meet the Team")

    st.markdown("""
    <div style="padding: 20px; border-radius: 10px; background-color: #f0f8ff;">
        <h4 style="text-align:center;">🚀 Team Data Decoders</h4>
        <p style="text-align:center;">
            This project is presented as part of an event at <strong>Vidyalankar College</strong>.<br>
            Built using <strong>Python</strong>, <strong>Streamlit</strong>, and <strong>Machine Learning</strong> to help predict student academic performance.
        </p>
        <p style="text-align:center;">✨ Created with 💙 by Mohammed Hussain Kamal Ahmed Choudhary & Team</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------- FOOTER -------------------
st.markdown("""
    <hr style='border: none; border-top: 1px solid #ccc; margin-top: 30px;'>
    <div style='text-align: center; font-size: 13px; color: gray;'>
        🚀 Presented by <strong>Team Data Decoders</strong> | 📍 Vidyalankar College <br>
        💡 Powered by <a href='https://streamlit.io' target='_blank'>Streamlit</a> | 🔗 <a href='https://github.com/hussain09042005/SmartStudentAI' target='_blank'>GitHub Repo</a>
    </div>
""", unsafe_allow_html=True)
