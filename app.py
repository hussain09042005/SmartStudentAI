import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Page config
st.set_page_config(page_title="SmartStudent AI", layout="wide")

# Custom styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f9ff;
        }
        footer {visibility: hidden;}
        .footer {
            position: fixed;
            bottom: 10px;
            left: 0;
            width: 100%;
            text-align: center;
            color: gray;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("## 📊 Welcome to SmartStudent AI")
st.markdown("An interactive academic dashboard that predicts student performance using machine learning.")

# File upload
uploaded_file = st.file_uploader("📁 Upload your student CSV file (without Final Result column):", type="csv")

# Load model
model = joblib.load("model.pkl")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Predict if 'Final Result' is not present
    if "Final Result" not in df.columns:
        X = df.drop(['Student_ID', 'Name'], axis=1)
        df["Final Result"] = model.predict(X)

    st.success("✅ Prediction completed!")

    # Display table
    st.subheader("📋 Student Data with Predictions")
    st.dataframe(df, use_container_width=True)

    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results CSV", data=csv, file_name="SmartStudent_Predictions.csv", mime="text/csv")

    # Charts Section
    st.subheader("📈 Academic Performance Overview")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df, x="Name", y="Final Exam Marks", color="Final Result", title="Final Exam Marks by Student")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pie = px.pie(df, names="Final Result", title="Pass vs Fail Ratio")
        st.plotly_chart(pie, use_container_width=True)

    # Smart Tips
    st.subheader("🎓 Personalized Suggestions")
    for _, row in df.iterrows():
        if row["Final Result"] == "Fail":
            st.warning(f"💡 Tip for {row['Name']}: Improve class participation and assignment score.")

# Footer
st.markdown("""
    <div class="footer">
        🚀 Project by <b>Data Decoders</b> | Presented at <b>Vidyalankar College</b> | Made with ❤️ using Streamlit
    </div>
""", unsafe_allow_html=True)
