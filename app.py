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
import time
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ==================== Streamlit Page Config ====================
st.set_page_config(page_title="SmartStudent AI", layout="wide", page_icon="📊")

# ==================== Global Header (Project Name) ====================
st.markdown("""
    <div style="background:linear-gradient(90deg,#3498db,#2ecc71);
                padding:1rem 2rem; border-radius:10px; text-align:center;
                color:white; font-size:1.8rem; font-weight:bold;
                box-shadow:0 4px 15px rgba(0,0,0,0.2); margin-bottom:20px;">
        📊 SmartStudent AI: Predictive Academic Dashboard
    </div>
""", unsafe_allow_html=True)


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


# ==================== DASHBOARD (Professional Version) ====================
if choice == "Dashboard":
    st.markdown("""
    <div style="background:linear-gradient(90deg,#2980b9,#6dd5fa,#ffffff);
                padding:1rem 2rem; border-radius:10px; text-align:center;
                color:white; font-size:1.6rem; font-weight:bold; 
                box-shadow:0 4px 15px rgba(0,0,0,0.15); margin-bottom:25px;">
        🎓 SmartStudent AI - Predictive Dashboard
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#555; font-size:1.1rem; text-align:center;">
        Upload student performance data to predict academic outcomes and visualize insights dynamically.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([0.7, 0.3])
    uploaded_file = col1.file_uploader("📂 Upload Student Dataset (CSV)", type="csv")
    use_sample = col2.button("🎓 Use Sample Data")

    if use_sample:
        df = pd.read_csv("test_student_data.csv")
        st.success("✅ Sample data loaded successfully!")
    elif uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        df = None

    if df is not None:
        required_columns = ['Assignment Score', 'Class Participation', 'Midterm Marks', 'Final Exam Marks']
        missing = [c for c in required_columns if c not in df.columns]

        if missing:
            st.error(f"❌ Missing Required Columns: {', '.join(missing)}")
        else:
            try:
                with st.spinner("🧠 Analyzing data and predicting results..."):
                    progress_bar = st.progress(0)
                    for pct in range(0, 101, 10):
                        progress_bar.progress(pct)
                        time.sleep(0.05)


                    X = df[required_columns]
                    predictions = model.predict(X)
                    df["Predicted Result"] = predictions

                    total = len(df)
                    passed = sum(df["Predicted Result"] == "Pass")
                    avg_marks = df["Final Exam Marks"].mean()
                    pass_percent = (passed / total) * 100

                # 🎯 Dashboard Metrics
                st.markdown("""
                <div style="display:flex; gap:1.5rem; margin-top:1.5rem;">
                    <div style="flex:1; background-color:#3498db; color:white; padding:1.5rem; border-radius:15px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.1);">
                        <div style="font-size:1.2rem;">👥 Total Students</div>
                        <div style="font-size:2rem; font-weight:bold;">{}</div>
                    </div>
                    <div style="flex:1; background-color:#2ecc71; color:white; padding:1.5rem; border-radius:15px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.1);">
                        <div style="font-size:1.2rem;">✅ Pass Percentage</div>
                        <div style="font-size:2rem; font-weight:bold;">{:.2f}%</div>
                    </div>
                    <div style="flex:1; background-color:#f1c40f; color:white; padding:1.5rem; border-radius:15px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.1);">
                        <div style="font-size:1.2rem;">📊 Average Final Marks</div>
                        <div style="font-size:2rem; font-weight:bold;">{:.2f}</div>
                    </div>
                </div>
                """.format(total, pass_percent, avg_marks), unsafe_allow_html=True)

                st.balloons()
                st.success("🎉 Prediction Completed Successfully!")

                # 🎨 Predicted Table
                st.markdown("### 🧾 Predicted Results")
                st.dataframe(df, use_container_width=True)

                # 📥 Download Option
                st.download_button(
                    "📥 Download Predicted CSV",
                    df.to_csv(index=False).encode('utf-8'),
                    "predicted_results.csv",
                    use_container_width=True
                )

                # 📈 Quick Visualization
                st.markdown("### 📈 Pass vs Fail Overview")
                if "Predicted Result" in df.columns:
                    fig = px.pie(df, names="Predicted Result", title="Pass vs Fail Ratio",
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"⚠️ Error while predicting: {e}")
    else:
        st.info("📥 Upload a dataset or click 'Use Sample Data' to begin.")


## ==================== Visual Analysis (Professional Version) ====================
elif choice == "Visual Analysis":
    st.markdown("""
    <div style="background:linear-gradient(90deg,#9b59b6,#8e44ad);
                padding:1rem 2rem; border-radius:10px; text-align:center;
                color:white; font-size:1.5rem; font-weight:bold; margin-bottom:25px;">
        📈 Visual Analysis of Student Performance
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload CSV with Predictions", type="csv", key="viz")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.markdown("### 🔍 Select Visualization Options")
        analysis_type = st.selectbox("Choose Analysis Type:", [
            "Overall Score Distribution",
            "Correlation Heatmap",
            "Pass vs Fail Ratio",
            "Performance Comparison",
            "Custom Column Analysis"
        ])

        # ================== DISTRIBUTION ==================
        if analysis_type == "Overall Score Distribution":
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select a column for distribution:", numeric_cols)
                fig = px.histogram(df, x=selected_col, nbins=20,
                                   color_discrete_sequence=["#3498db"],
                                   title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No numeric columns found.")

        # ================== CORRELATION HEATMAP ==================
        elif analysis_type == "Correlation Heatmap":
            numeric_df = df.select_dtypes(include='number')
            if not numeric_df.empty:
                st.markdown("### 🔷 Correlation Heatmap of Scores")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
                st.pyplot(fig)
            else:
                st.warning("⚠️ No numeric data found for correlation analysis.")

        # ================== PASS VS FAIL RATIO ==================
        elif analysis_type == "Pass vs Fail Ratio":
            if "Predicted Result" in df.columns:
                st.markdown("### 🎯 Pass vs Fail Overview")
                result_counts = df["Predicted Result"].value_counts()
                fig = px.pie(names=result_counts.index, values=result_counts.values,
                             color=result_counts.index,
                             color_discrete_map={"Pass": "#2ecc71", "Fail": "#e74c3c"},
                             title="Overall Pass/Fail Ratio")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Please include the 'Predicted Result' column in your data.")

        # ================== PERFORMANCE COMPARISON ==================
        elif analysis_type == "Performance Comparison":
            if all(col in df.columns for col in ['Assignment Score', 'Midterm Marks', 'Final Exam Marks']):
                st.markdown("### 📊 Comparative Performance Overview")
                melted_df = df.melt(value_vars=['Assignment Score', 'Midterm Marks', 'Final Exam Marks'],
                                    var_name='Assessment Type', value_name='Score')
                fig = px.box(melted_df, x="Assessment Type", y="Score", color="Assessment Type",
                             title="Performance Distribution Across Assessments",
                             color_discrete_sequence=["#3498db", "#9b59b6", "#e67e22"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Required columns missing: 'Assignment Score', 'Midterm Marks', 'Final Exam Marks'.")

        # ================== CUSTOM COLUMN ANALYSIS ==================
        elif analysis_type == "Custom Column Analysis":
            st.markdown("### 🔧 Explore Relationships Between Any Two Columns")
            all_cols = df.columns.tolist()
            x_col = st.selectbox("Select X-axis column:", all_cols)
            y_col = st.selectbox("Select Y-axis column:", all_cols, index=min(1, len(all_cols)-1))
            color_col = st.selectbox("Color by (optional):", [None] + all_cols)
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                             title=f"Relationship between {x_col} and {y_col}",
                             color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📁 Upload your **Predicted CSV** file to begin visual analysis.")


# ==================== Advanced Insights (Professional Version) ====================
elif choice == "Advanced Insights":
    st.markdown("""
    <div style="background:linear-gradient(90deg,#1abc9c,#16a085);
                padding:1rem 2rem; border-radius:10px; text-align:center;
                color:white; font-size:1.5rem; font-weight:bold; margin-bottom:25px;">
        📊 Advanced Insights & Trend Analysis
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Upload CSV with Predictions", type="csv", key="adv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.markdown("### 🔍 Choose Insight Type")
        insight_type = st.selectbox("Select Analysis:", [
            "Trend Over Students",
            "Score Comparison (Multi Metric)",
            "Correlation Heatmap",
            "Boxplot Insights",
            "Prediction Accuracy & Confusion Matrix"
        ])

        # ================== TREND ANALYSIS ==================
        if insight_type == "Trend Over Students":
            st.markdown("### 📈 Trend of Scores Across Students")
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select a column to visualize trend:", numeric_cols)
                df["Student Index"] = range(1, len(df) + 1)
                fig = px.line(df, x="Student Index", y=selected_col,
                              title=f"Trend of {selected_col} Over Students",
                              markers=True, line_shape="spline",
                              color_discrete_sequence=["#1abc9c"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No numeric columns found in dataset.")

        # ================== MULTI-METRIC COMPARISON ==================
        elif insight_type == "Score Comparison (Multi Metric)":
            st.markdown("### 📊 Comparative View of Student Scores")
            cols = ['Assignment Score', 'Class Participation', 'Midterm Marks', 'Final Exam Marks']
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                melted_df = df.melt(value_vars=available_cols,
                                    var_name='Score Type', value_name='Value')
                fig = px.violin(melted_df, x='Score Type', y='Value', box=True, points='all',
                                color='Score Type', color_discrete_sequence=px.colors.qualitative.Safe,
                                title="Score Comparison across Assessments")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Required columns not found for comparison.")

        # ================== CORRELATION HEATMAP ==================
        elif insight_type == "Correlation Heatmap":
            st.markdown("### 🔷 Correlation Between All Score Columns")
            numeric_df = df.select_dtypes(include='number')
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
                st.pyplot(fig)
            else:
                st.warning("⚠️ No numeric data available.")

        # ================== BOXPLOT INSIGHTS ==================
        elif insight_type == "Boxplot Insights":
            st.markdown("### 📦 Boxplot Insights - Detect Outliers & Variability")
            numeric_cols = ['Assignment Score', 'Class Participation', 'Midterm Marks', 'Final Exam Marks']
            available_cols = [c for c in numeric_cols if c in df.columns]
            if available_cols:
                fig = px.box(df, y=available_cols, color_discrete_sequence=px.colors.sequential.Mint,
                             title="Boxplot of Student Performance")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No required columns found.")

        # ================== PREDICTION ACCURACY & CONFUSION MATRIX ==================
        elif insight_type == "Prediction Accuracy & Confusion Matrix":
            st.markdown("### 🧠 Model Evaluation Insights")
            if "Predicted Result" in df.columns and "Final Result" in df.columns:
                correct = (df["Predicted Result"] == df["Final Result"]).sum()
                total = len(df)
                accuracy = (correct / total) * 100
                st.metric("🎯 Prediction Accuracy", f"{accuracy:.2f}%")

                cm = pd.crosstab(df["Final Result"], df["Predicted Result"], rownames=['Actual'], colnames=['Predicted'])
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
                st.pyplot(fig)
            else:
                st.warning("⚠️ Both 'Final Result' and 'Predicted Result' columns are required.")
    else:
        st.info("📁 Upload your **Predicted CSV** file to explore advanced insights.")


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

        # Load messages
        try:
            df_logs = pd.read_csv("contact_logs.csv", encoding='utf-8-sig')
        except:
            df_logs = pd.DataFrame(columns=["Name", "Email", "Message", "Reply", "Timestamp", "Seen"])

        # Clean data
        df_logs['Timestamp'] = pd.to_datetime(df_logs.get('Timestamp'), errors='coerce')
        df_logs['Timestamp'] = df_logs['Timestamp'].fillna(datetime.datetime.now())
        df_logs['Seen'] = df_logs.get('Seen', 'No').fillna('No')
        df_logs['Reply'] = df_logs.get('Reply', '').fillna('')
        df_logs['Reply'] = df_logs['Reply'].apply(lambda x: '' if str(x).lower() == 'nan' else str(x))

        # Metrics
        def get_counts():
            total_msgs = len(df_logs)
            today_msgs = len(df_logs[df_logs["Timestamp"].dt.date == datetime.datetime.now().date()])
            new_msgs_count = len(df_logs[df_logs["Seen"] == "No"])
            return total_msgs, today_msgs, new_msgs_count

        total_msgs, today_msgs, new_msgs_count = get_counts()

        # Display metrics
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

        # Tabs
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
                    timestamp = row['Timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                    reply_value = str(row.get("Reply", ""))

                    # Track seen status
                    if f"seen_{index}" not in st.session_state:
                        st.session_state[f"seen_{index}"] = df_logs.at[index, 'Seen'] == 'Yes'

                    # Unique key for expander
                    expander_key = f"exp_{index}_{row['Timestamp'].strftime('%Y%m%d%H%M%S')}"

                    with st.expander(f"👤 {name} - {timestamp}", expanded=False):
                        st.markdown(f"**📧 Email:** {email}  \n**📝 Message:** {message}")

                        # Mark as seen
                        if not st.session_state[f"seen_{index}"]:
                            df_logs.at[index, 'Seen'] = 'Yes'
                            df_logs.to_csv("contact_logs.csv", index=False, encoding='utf-8-sig')
                            st.session_state[f"seen_{index}"] = True
                            total_msgs, today_msgs, new_msgs_count = get_counts()

                        # Reply
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
                                df_logs.to_csv("contact_logs.csv", index=False, encoding='utf-8-sig')

                                total_msgs, today_msgs, new_msgs_count = get_counts()
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








