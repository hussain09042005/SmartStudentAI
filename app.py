import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import io

# Load trained model
model = joblib.load("model.pkl")

# Streamlit page config
st.set_page_config(page_title="SmartStudent AI", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #cce5ff, #e6f2ff);
    }
    .main {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📚 SmartStudent AI: Predictive Academic Dashboard")
st.markdown("Upload your student records to predict their academic result (Pass/Fail), analyze trends, and get improvement tips.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV file (without Final Result)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    expected_columns = ["Student_ID", "Name", "Attendance (%)", "Assignment Score", "Midterm Marks", "Final Exam Marks", "Class Participation"]
    if not all(col in df.columns for col in expected_columns):
        st.error("Uploaded CSV does not have the required columns.")
    else:
        # Convert necessary columns to numeric
        features = ["Attendance (%)", "Assignment Score", "Midterm Marks", "Final Exam Marks", "Class Participation"]
        df[features] = df[features].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=features)

        # Predict
        predictions = model.predict(df[features])
        df['Final Result'] = predictions
        df['Final Result'] = df['Final Result'].map({1: "Pass", 0: "Fail"})

        # Show results
        st.subheader("📊 Prediction Results")
        st.dataframe(df)

        # Download link
        def get_table_download_link(dataframe):
            csv = dataframe.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download="predicted_results.csv">📥 Download Predicted Results CSV</a>'

        st.markdown(get_table_download_link(df), unsafe_allow_html=True)

        # Charts Section
        st.subheader("📈 Performance Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**✅ Pass vs Fail**")
            result_counts = df['Final Result'].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', startangle=90, colors=["#8fd9a8", "#ff9999"])
            ax1.axis('equal')
            st.pyplot(fig1)

        with col2:
            st.markdown("**📚 Score Distribution**")
            fig2, ax2 = plt.subplots()
            df[features].plot(kind='hist', bins=20, alpha=0.6, ax=ax2)
            st.pyplot(fig2)

        st.markdown("**🧪 Correlation Heatmap**")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        corr = df[features].corr()
        sns.heatmap(corr, annot=True, cmap="Blues", ax=ax3)
        st.pyplot(fig3)

        # Suggestions
        st.subheader("💡 Academic Performance Tips")
        fail_students = df[df['Final Result'] == 'Fail']
        if not fail_students.empty:
            st.warning(f"{len(fail_students)} students failed. Here's how to help them:")
            st.markdown("""
            - 📖 Encourage regular class attendance and participation.
            - 📝 Provide assignment guidance and feedback.
            - 📅 Offer weekly revision sessions.
            - 💬 Conduct one-on-one performance reviews.
            """)
        else:
            st.success("🎉 All students passed! Great job!")

else:
    st.info("Please upload a CSV file to begin.")
st.markdown("""
<hr style='border: 0.5px solid #ccc; margin-top: 50px;'>
<div style='text-align: center; font-size: 14px; color: #888;'>
    📚 <strong>SmartStudent AI</strong> — A project presented by <strong>Team Data Decoders</strong><br>
    🎓 For the Tech Event at <strong>Vidyalankar College</strong><br>
    Built with ❤️ using <a href='https://streamlit.io' target='_blank'>Streamlit</a>
</div>
""", unsafe_allow_html=True)

