import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import csv
from streamlit_lottie import st_lottie
import json
import requests
import time  


# ====== Email + Logging ======
def send_email(name, sender_email, message):
    msg = MIMEMultipart()
    msg["From"] = st.secrets["email"]
    msg["To"] = st.secrets["receiver_email"]
    msg["Subject"] = f"SmartStudent AI - Message from {name}"
    body = f"Name: {name}\nEmail: {sender_email}\n\nMessage:\n{message}"
    msg.attach(MIMEText(body, "plain"))

    auto_reply = MIMEMultipart()
    auto_reply["From"] = st.secrets["email"]
    auto_reply["To"] = sender_email
    auto_reply["Subject"] = "Thanks for contacting SmartStudent AI!"
    auto_body = f"""Dear {name},

Thank you for reaching out to Team Data Decoders!

We have received your message and will get back to you shortly.

Regards,
📊 SmartStudent AI Team
"""
    auto_reply.attach(MIMEText(auto_body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(st.secrets["email"], st.secrets["app_password"])
            server.sendmail(st.secrets["email"], st.secrets["receiver_email"], msg.as_string())
            server.sendmail(st.secrets["email"], sender_email, auto_reply.as_string())

        log_contact_to_csv(name, sender_email, message)
        return True
    except Exception as e:
        st.error(f"❌ Email Error: {e}")
        return False

def log_contact_to_csv(name, email, message):
    file_path = "contact_logs.csv"
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["Name", "Email", "Message", "Timestamp"])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([name, email, message, timestamp])


# ====== Load ML Model ======
model = joblib.load("model.pkl")

# ====== Streamlit Config ======
st.set_page_config(page_title="SmartStudent AI", layout="wide", page_icon="📊")

# ====== UI Styling ======
st.markdown("""
<style>
html, body {
    background: linear-gradient(to right, #dbe9f4, #ffffff);
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4 { color: #2c3e50; }
.stButton>button, .stDownloadButton>button {
    background-color: #3498db; color: white;
    font-weight: bold; border-radius: 10px;
    padding: 0.6rem 1.5rem; border: none;
}
.stButton>button:hover {
    background-color: #2c80b4;
    transform: scale(1.02);
}
.metric-container { display: flex; gap: 2rem; margin-bottom: 1rem; }
.metric-card {
    background-color: white; padding: 1.5rem;
    border-radius: 16px; box-shadow: 0 6px 20px rgba(0,0,0,0.07);
    flex: 1; text-align: center;
}
.metric-label { font-size: 1rem; color: #777; }
.metric-value { font-size: 2rem; font-weight: bold; color: #2c3e50; }
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ====== Header ======
st.markdown("<h1 style='text-align:center;'>📊 SmartStudent AI: Predictive Academic Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Developed by Team Data Decoders</p>", unsafe_allow_html=True)
st.write("---")

def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        else:
            return None
    except:
        return None

# ====== Tabs ======
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📁 Upload & Predict", 
    "📈 Visual Analysis", 
    "📊 Advanced Insights", 
    "🔄 Retrain Model", 
    "👨‍💻 About + Contact",
    "🛡️ Admin Panel"
])

# ====== Upload & Predict ======
with tab1:
    st.subheader("📁 Upload Student CSV File")
    col1, col2 = st.columns([0.6, 0.4])
    uploaded_file = col1.file_uploader("Choose a CSV file", type="csv")
    use_sample = col2.button("🎓 Use Sample Student Data")
    if use_sample:
        df = pd.read_csv("test_student_data.csv")
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = None

    if df is not None:
        required_columns = ['Assignment Score', 'Class Participation', 'Midterm Marks', 'Final Exam Marks']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            st.error("❌ Missing Columns: " + ", ".join(missing))
        else:
            X = df[required_columns]
            try:
                predictions = model.predict(X)
                df["Predicted Result"] = predictions
                total = len(df)
                passed = sum(df["Predicted Result"] == "Pass")
                avg_marks = df["Final Exam Marks"].mean()
                pass_percent = (passed / total) * 100

                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Students</div><div class='metric-value'>{total}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Pass Percentage</div><div class='metric-value'>{pass_percent:.2f}%</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Final Marks</div><div class='metric-value'>{avg_marks:.2f}</div></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.success("✅ Prediction Completed")
                st.dataframe(df, use_container_width=True)
                st.subheader("📋 Summary Statistics")
                st.write(df.describe())

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predicted CSV", csv, "predicted_results.csv", "text/csv")
            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")
    else:
        st.info("📥 Upload a dataset or use the sample.")

# ====== Visual Analysis ======
with tab2:
    st.subheader("📈 Student Performance Visualizations")
    uploaded_file = st.file_uploader("Upload CSV with Predictions", type="csv", key="viz")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 🔷 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.markdown("### 📌 Score Distribution")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select a column", numeric_cols)
            fig = px.histogram(df, x=selected_col, nbins=20, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig)

        if "Final Result" in df.columns:
            result_counts = df["Final Result"].value_counts()
            st.markdown("### 🎯 Pass vs Fail Ratio")
            st.plotly_chart(px.pie(names=result_counts.index, values=result_counts.values, title="Pass vs Fail"))

# ====== Advanced Insights ======
with tab3:
    st.subheader("📊 Advanced Insights")
    uploaded_file = st.file_uploader("Upload CSV with Predictions", type="csv", key="adv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if all(col in df.columns for col in ['Assignment Score', 'Class Participation', 'Midterm Marks', 'Final Exam Marks', 'Final Result']):
            st.plotly_chart(px.box(df, y=['Assignment Score', 'Class Participation', 'Midterm Marks', 'Final Exam Marks'], title="Boxplot of Scores"))
        else:
            st.warning("Upload CSV with all required columns.")

# ====== Retrain Model ======
with tab4:
    st.subheader("🔄 Retrain the ML Model")
    if st.button("Train Now"):
        try:
            df_train = pd.read_csv("sample_data.csv")
            X = df_train[['Assignment Score', 'Class Participation', 'Midterm Marks', 'Final Exam Marks']]
            y = df_train["Final Result"]
            model_new = RandomForestClassifier(n_estimators=100)
            model_new.fit(X, y)
            joblib.dump(model_new, "model.pkl")
            st.success("✅ Model retrained successfully.")
        except Exception as e:
            st.error(f"❌ Error retraining model: {e}")

# ====== About + Contact ======
with tab5:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("👨‍💻 Team Data Decoders")
        st.markdown("""
- 🎓 Vidyalankar College Project
- 🧠 Predicts academic success using ML
- 📊 Powered by Seaborn & Plotly
- 💡 Built with Streamlit
        """)
        st.image("team.jpg", width=280)
    with col2:
        st.subheader("📬 Get in Touch with Team")
        with st.form("contact_form"):
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
            message = st.text_area("Your Message")
            send = st.form_submit_button("📨 Send Email")
            if send and name and email and message:
                if send_email(name, email, message):
                    st.success("✅ Message sent successfully!")
            elif send:
                st.warning("⚠️ Please fill all fields.")

# ====== Admin Panel (Protected) ======
# ====== Admin Panel (Protected) ======
# ====== Admin Panel (Protected) ======
# ====== Admin Panel (Protected) ======
# ====== Admin Panel ======
# ====== Admin Panel ======
# ====== Admin Panel ======
# ====== Admin Panel ======
# ==== Admin Login Section (in "🛡️ Admin Panel" tab) ====

# ==== Admin Panel ====
with tab6:
    st.subheader("🔐 Admin Login")

    # Initialize session state
    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False

    # Login Form
    if not st.session_state["admin_logged_in"]:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "hussain123":
                st.session_state["admin_logged_in"] = True
                st.success("✅ Welcome, Admin!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

    # ==== Show Admin Tabs only if logged in ====
    if st.session_state["admin_logged_in"]:
        st.markdown("### 🛡️ Admin Dashboard - Contact Messages")

        admin_tab1, admin_tab2, admin_tab3 = st.tabs(["📨 Messages", "📊 Logs", "📥 Downloads"])

        with admin_tab1:
            try:
                df = pd.read_csv("contact_logs.csv")
                df.columns = ["Name", "Email", "Message", "Timestamp"]

                for index, row in df.iterrows():
                    st.markdown(f"""
                    <div style="padding: 1rem; background-color: #fff; border-radius: 10px; margin-bottom: 10px; color: black;">
                        <strong>👤 Name:</strong> {row['Name']}<br>
                        <strong>📧 Email:</strong> {row['Email']}<br>
                        <strong>📝 Message:</strong> {row['Message']}<br>
                        <strong>🕒 Timestamp:</strong> {row['Timestamp']}
                    </div>
                    """, unsafe_allow_html=True)

                    delete_btn = st.button(f"🗑️ Delete Message {index+1}", key=f"delete_{index}")
                    if delete_btn:
                        df.drop(index, inplace=True)
                        df.to_csv("contact_logs.csv", index=False)
                        st.success("✅ Message deleted successfully!")
                        st.experimental_rerun()
            except Exception as e:
                st.warning("⚠️ Could not load contact messages.")
                st.text(str(e))

        with admin_tab2:
            st.info("📊 Logs will be added soon.")

        with admin_tab3:
            try:
                with open("contact_logs.csv", "rb") as f:
                    st.download_button("⬇️ Download contact_logs.csv", f, file_name="contact_logs.csv")
            except:
                st.warning("⚠️ Download unavailable.")






