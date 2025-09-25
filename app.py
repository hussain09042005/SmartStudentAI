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
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==================== Streamlit Page Config ====================
st.set_page_config(page_title="SmartStudent AI", layout="wide", page_icon="📊")

# ==================== CSS Styling ====================
st.markdown("""
<style>
html, body { background: linear-gradient(to right, #f0f4f8, #ffffff); font-family: 'Segoe UI', sans-serif; }
h1,h2,h3,h4 { color: #2c3e50; }
.stButton>button, .stDownloadButton>button {
    background-color: #3498db; color: white;
    font-weight: bold; border-radius: 12px;
    padding: 0.6rem 1.5rem; border: none; transition: 0.2s;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background-color: #2c80b4; transform: scale(1.03);
}
.metric-container { display: flex; gap: 1.5rem; margin-bottom: 1rem; }
.metric-card { background-color: white; padding: 1.5rem; border-radius: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.07); flex:1; text-align:center; }
.metric-label { font-size: 1rem; color: #777; }
.metric-value { font-size: 2rem; font-weight:bold; color:#2c3e50; }
.new-badge { display:inline-block; background:red; color:white; border-radius:50%; padding:2px 7px; font-size:0.8rem; font-weight:bold; margin-left:5px;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==================== Load ML Model ====================
if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
else:
    model = RandomForestClassifier(n_estimators=100)

# ==================== Email + Logging ====================
def send_email(name, sender_email, message):
    try:
        msg = MIMEMultipart()
        msg["From"] = st.secrets["email"]
        msg["To"] = st.secrets["receiver_email"]
        msg["Subject"] = f"SmartStudent AI - Message from {name}"
        msg.attach(MIMEText(f"Name: {name}\nEmail: {sender_email}\n\nMessage:\n{message}", "plain"))

        auto_reply = MIMEMultipart()
        auto_reply["From"] = st.secrets["email"]
        auto_reply["To"] = sender_email
        auto_reply["Subject"] = "Thanks for contacting SmartStudent AI!"
        auto_reply.attach(MIMEText(f"Dear {name},\n\nThank you for reaching out!\n\nRegards,\n📊 SmartStudent AI Team", "plain"))

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

def log_contact_to_csv(name, email, message, reply=""):
    file_path = "contact_logs.csv"
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["Name", "Email", "Message", "Reply", "Timestamp", "Seen"])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([name, email, message, reply, timestamp, "No"])

# ==================== Sidebar Navigation ====================
try:
    df_logs = pd.read_csv("contact_logs.csv")
    new_msgs_count = len(df_logs[df_logs["Seen"]=="No"])
except:
    new_msgs_count = 0

menu = ["Dashboard", "Visual Analysis", "Advanced Insights", "Retrain Model", "About + Contact", "Admin Panel"]
choice = st.sidebar.selectbox("Navigation", menu)

# Show new messages badge
if new_msgs_count > 0:
    st.sidebar.markdown(f"**🆕 New Messages: {new_msgs_count}**")


# ==================== Dashboard ====================
if choice == "Dashboard":
    st.header("📁 Upload & Predict")
    col1, col2 = st.columns([0.7,0.3])
    uploaded_file = col1.file_uploader("Choose CSV", type="csv")
    use_sample = col2.button("🎓 Use Sample Data")
    if use_sample:
        df = pd.read_csv("test_student_data.csv")
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = None

    if df is not None:
        required_columns = ['Assignment Score','Class Participation','Midterm Marks','Final Exam Marks']
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            st.error(f"❌ Missing Columns: {', '.join(missing)}")
        else:
            try:
                with st.spinner("🔄 Predicting..."):
                    X = df[required_columns]
                    predictions = model.predict(X)
                    df["Predicted Result"] = predictions
                    total = len(df)
                    passed = sum(df["Predicted Result"]=="Pass")
                    avg_marks = df["Final Exam Marks"].mean()
                    pass_percent = (passed/total)*100

                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Students</div><div class='metric-value'>{total}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Pass %</div><div class='metric-value'>{pass_percent:.2f}%</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Marks</div><div class='metric-value'>{avg_marks:.2f}</div></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.success("✅ Prediction Completed")
                st.dataframe(df, use_container_width=True)
                st.download_button("📥 Download Predicted CSV", df.to_csv(index=False).encode('utf-8'), "predicted_results.csv")
            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")
    else:
        st.info("📥 Upload a dataset or use sample data.")
# ==================== Visual Analysis ====================
elif choice == "Visual Analysis":
    st.header("📈 Student Performance")
    uploaded_file = st.file_uploader("Upload CSV with Predictions", type="csv", key="viz")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔷 Score Distribution")
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select Column", numeric_cols)
                fig = px.histogram(df, x=selected_col, nbins=20, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "Final Result" in df.columns:
                result_counts = df["Final Result"].value_counts()
                st.markdown("### 🎯 Pass vs Fail")
                st.plotly_chart(px.pie(names=result_counts.index, values=result_counts.values, title="Pass vs Fail"))

# ==================== Advanced Insights ====================
elif choice == "Advanced Insights":
    st.header("📊 Advanced Insights")
    uploaded_file = st.file_uploader("Upload CSV with Predictions", type="csv", key="adv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        numeric_cols = ['Assignment Score', 'Class Participation', 'Midterm Marks', 'Final Exam Marks']
        if all(col in df.columns for col in numeric_cols+["Final Result"]):
            st.plotly_chart(px.box(df, y=numeric_cols, title="Boxplot of Scores"))
            st.plotly_chart(px.imshow(df[numeric_cols].corr(), text_auto=True, color_continuous_scale="Blues", title="Correlation Heatmap"))
        else:
            st.warning("Upload CSV with all required columns.")

# ==================== Retrain Model ====================
elif choice == "Retrain Model":
    st.header("🔄 Retrain Model")
    uploaded_file = st.file_uploader("Upload CSV for Retraining", type="csv", key="retrain")
    if uploaded_file:
        df_train = pd.read_csv(uploaded_file)
        if st.button("Train Now"):
            with st.spinner("⚡ Training Model..."):
                X = df_train[['Assignment Score', 'Class Participation', 'Midterm Marks', 'Final Exam Marks']]
                y = df_train["Final Result"]
                model_new = RandomForestClassifier(n_estimators=100)
                model_new.fit(X, y)
                joblib.dump(model_new, "model.pkl")
                st.success("✅ Model retrained successfully!")
                y_pred = model_new.predict(X)
                acc = accuracy_score(y, y_pred)
                st.info(f"Training Accuracy: {acc*100:.2f}%")
                cm = confusion_matrix(y, y_pred)
                st.write("Confusion Matrix:")
                st.write(cm)

# ==================== About + Contact ====================
elif choice == "About + Contact":
    st.header("📬 Contact Us")
    st.markdown("Reach out to us for any queries or suggestions.")
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message", height=120)
    if st.button("Send Message"):
        if name and email and message:
            if send_email(name, email, message):
                st.success("✅ Message sent successfully!")
            else:
                st.error("❌ Failed to send message. Check email settings.")
        else:
            st.warning("⚠️ Please fill all fields.")

# ------------------- Admin Panel -------------------
elif choice == "Admin Panel":
    st.subheader("🔐 Admin Login")

    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False

    # Admin login form
    if not st.session_state["admin_logged_in"]:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")

        if login_btn:
            if username == "admin" and password == "hussain123":
                st.session_state["admin_logged_in"] = True
                st.success("✅ Welcome, Admin!")
            else:
                st.error("❌ Invalid credentials.")

    if st.session_state["admin_logged_in"]:
        st.markdown("### 🛡️ Admin Dashboard - Overview")

        # ==================== Load messages safely ====================
        try:
            df_logs = pd.read_csv("contact_logs.csv")
        except:
            df_logs = pd.DataFrame(columns=["Name", "Email", "Message", "Reply", "Timestamp", "Seen"])

        # Ensure required columns exist
        for col in ["Seen", "Reply", "Timestamp"]:
            if col not in df_logs.columns:
                if col == "Seen":
                    df_logs[col] = "No"
                elif col == "Reply":
                    df_logs[col] = ""
                else:
                    df_logs[col] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure Reply and Timestamp columns are clean
        df_logs['Reply'] = df_logs['Reply'].apply(lambda x: str(x) if isinstance(x, str) else "")
        df_logs['Timestamp'] = df_logs['Timestamp'].apply(
            lambda x: str(x) if pd.notna(x) else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        # ==================== Metrics Cards ====================
        def get_counts():
            total_msgs = len(df_logs)
            today_msgs = len(df_logs[df_logs["Timestamp"].str.startswith(
                datetime.datetime.now().strftime("%Y-%m-%d"), na=False)])
            new_msgs_count = len(df_logs[df_logs["Seen"] == "No"])
            return total_msgs, today_msgs, new_msgs_count

        total_msgs, today_msgs, new_msgs_count = get_counts()

        st.markdown(f"""
        <div style="display:flex; gap:1.5rem; margin-bottom:1rem;">
            <div style="flex:1; background-color:#3498db; color:white; padding:1.5rem; border-radius:16px; text-align:center;">
                <div>Total Messages</div>
                <div style="font-size:2rem; font-weight:bold;">{total_msgs}</div>
            </div>
            <div style="flex:1; background-color:#2ecc71; color:white; padding:1.5rem; border-radius:16px; text-align:center;">
                <div>Today</div>
                <div style="font-size:2rem; font-weight:bold;">{today_msgs}</div>
            </div>
            <div style="flex:1; background-color:#e74c3c; color:white; padding:1.5rem; border-radius:16px; text-align:center;">
                <div>New Messages</div>
                <div style="font-size:2rem; font-weight:bold;">{new_msgs_count}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ==================== Tabs ====================
        admin_tab1, admin_tab2, admin_tab3 = st.tabs([f"📨 Messages ({new_msgs_count} new)", "📊 Logs", "📥 Downloads"])

        # ---------------- Messages Tab -----------------
        with admin_tab1:
            st.markdown("### 💌 Contact Messages")
            if df_logs.empty:
                st.info("No messages yet.")
            else:
                for index, row in df_logs.iterrows():
                    name = str(row.get("Name", "Unknown"))
                    email = str(row.get("Email", ""))
                    message = str(row.get("Message", ""))
                    timestamp = str(row.get("Timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    reply_value = str(row.get("Reply", ""))

                    # Show expander with timestamp
                    with st.expander(f"👤 {name} - {timestamp}", expanded=False):
                        st.markdown(f"**📧 Email:** {email}  \n**📝 Message:** {message}")

                        # Mark as Seen dynamically
                        if df_logs.at[index, 'Seen'] == 'No':
                            df_logs.at[index, 'Seen'] = 'Yes'
                            df_logs.to_csv("contact_logs.csv", index=False)
                            total_msgs, today_msgs, new_msgs_count = get_counts()
                            st.experimental_rerun()  # refresh the badge dynamically

                        # Reply text area
                        reply_text = st.text_area("✉️ Reply", key=f"reply_{index}", height=100, value=reply_value)
                        send_reply = st.button("Send Reply", key=f"send_{index}", use_container_width=True)

                        if send_reply and reply_text.strip():
                            try:
                                msg = MIMEMultipart()
                                msg["From"] = st.secrets["email"]
                                msg["To"] = email
                                msg["Subject"] = "Reply from SmartStudent AI"
                                msg.attach(MIMEText(reply_text, "plain"))

                                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                                    server.starttls()
                                    server.login(st.secrets["email"], st.secrets["app_password"])
                                    server.sendmail(st.secrets["email"], email, msg.as_string())

                                st.success(f"✅ Reply sent to {name}!")
                                df_logs.at[index, "Reply"] = reply_text
                                df_logs.to_csv("contact_logs.csv", index=False)

                                # Refresh counts after reply
                                total_msgs, today_msgs, new_msgs_count = get_counts()
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to send reply: {e}")

        # ---------------- Logs Tab -----------------
        with admin_tab2:
            st.markdown("### 📊 Contact Logs")
            if df_logs.empty:
                st.info("No logs available.")
            else:
                st.dataframe(df_logs.sort_values(by="Timestamp", ascending=False), use_container_width=True)

        # ---------------- Downloads Tab -----------------
        with admin_tab3:
            st.markdown("### ⬇️ Download Logs")
            if not df_logs.empty:
                with open("contact_logs.csv", "rb") as f:
                    st.download_button("Download contact_logs.csv", f, file_name="contact_logs.csv")
            else:
                st.warning("⚠️ No data to download.")





