import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import smtplib
import os
import csv
import time

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ==================== Streamlit Page Config ====================
st.set_page_config(
    page_title="SmartStudent AI",
    layout="wide",
    page_icon="📊"
)

# ===================== AUTHENTICATION SYSTEM =====================

# -------- Session Initialization --------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0

if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()

if "remember_until" not in st.session_state:
    st.session_state.remember_until = None


# -------- Simple User Storage --------
def login_user(username, password):
    users = {
        "student": {"password": "stud123", "role": "Student"},
        "faculty": {"password": "fac123", "role": "Faculty"},
        "admin": {"password": "admin123", "role": "Admin"}
    }

    if username in users and users[username]["password"] == password:
        return users[username]["role"]

    return None


# ==================== REMEMBER ME SYSTEM ====================

current_time = time.time()

# Auto-login if remember session valid
if (
    st.session_state.remember_until is not None
    and current_time < st.session_state.remember_until
):
    st.session_state.logged_in = True

# Expire remember session
elif (
    st.session_state.remember_until is not None
    and current_time >= st.session_state.remember_until
):
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.remember_until = None


# ===================== LOGIN SCREEN =====================

if not st.session_state.logged_in:

    st.title("🔐 SmartStudent AI Secure Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    remember_me = st.checkbox("Remember me for 20 minutes")

    if st.button("Login", use_container_width=True):

        role = login_user(username, password)

        if role:
            st.session_state.logged_in = True
            st.session_state.role = role
            st.session_state.login_attempts = 0
            st.session_state.last_activity = time.time()

            # 🔥 Remember for 20 minutes
            if remember_me:
                st.session_state.remember_until = time.time() + (20 * 60)

            st.success(f"Welcome {role} ✅")
            st.rerun()

        else:
            st.session_state.login_attempts += 1
            st.error("Invalid Credentials ❌")

            if st.session_state.login_attempts >= 3:
                st.warning("Too many failed attempts. Try again later.")
                st.stop()

    st.stop()


# ==================== INACTIVITY TIMEOUT (10 MIN) ====================

# Apply inactivity timeout ONLY if remember is NOT active
if st.session_state.remember_until is None:

    if current_time - st.session_state.last_activity > 600:
        st.session_state.logged_in = False
        st.session_state.role = None
        st.warning("Session expired due to inactivity.")
        st.rerun()

# Update activity time
st.session_state.last_activity = current_time


# ==================== SHOW REMEMBER TIMER ====================

if st.session_state.remember_until:
    remaining = int(st.session_state.remember_until - time.time())
    if remaining > 0:
        st.sidebar.info(f"⏳ Remember session: {remaining//60} min left")


# ==================== EMAIL FUNCTION ====================
def send_email(name, sender_email, message):
    try:
        # ================== ADMIN EMAIL ==================
        admin_msg = MIMEMultipart("alternative")
        admin_msg["From"] = st.secrets["email"]
        admin_msg["To"] = st.secrets["receiver_email"]
        admin_msg["Subject"] = f"📩 New Message from {name}"
        admin_msg["Reply-To"] = sender_email

        admin_html = f"""
        <html>
        <body style="font-family:Arial; background:#f4f6f9; padding:20px;">
            <div style="background:white; padding:20px; border-radius:10px;">
                <h2 style="color:#2c3e50;">New Contact Form Submission</h2>
                <p><b>Name:</b> {name}</p>
                <p><b>Email:</b> {sender_email}</p>
                <p><b>Message:</b></p>
                <div style="background:#ecf0f1; padding:15px; border-radius:5px;">
                    {message}
                </div>
            </div>
        </body>
        </html>
        """

        admin_msg.attach(MIMEText(admin_html, "html"))

        # ================== AUTO REPLY TO USER ==================
        user_msg = MIMEMultipart("alternative")
        user_msg["From"] = st.secrets["email"]
        user_msg["To"] = sender_email
        user_msg["Subject"] = "✅ SmartStudent AI – Message Received"

        user_html = f"""
        <html>
        <body style="font-family:Arial; background:#f8f9fa; padding:20px;">
            <div style="background:white; padding:25px; border-radius:10px;">
                <h2 style="color:#2c3e50;">Thank You {name} 👋</h2>
                <p>We have successfully received your message.</p>
                <p>Our team will respond shortly.</p>
                <hr>
                <p style="color:gray; font-size:14px;">
                    🚀 SmartStudent AI Support Team
                </p>
            </div>
        </body>
        </html>
        """

        user_msg.attach(MIMEText(user_html, "html"))

        # ================== SEND BOTH EMAILS ==================
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(
                st.secrets["email"],
                st.secrets["app_password"]
            )

            # Send to Admin
            server.sendmail(
                st.secrets["email"],
                st.secrets["receiver_email"],
                admin_msg.as_string()
            )

            # Send confirmation to User
            server.sendmail(
                st.secrets["email"],
                sender_email,
                user_msg.as_string()
            )

        # ================== SAVE TO CSV ==================
        log_file = "contact_logs.csv"
        file_exists = os.path.isfile(log_file)

        with open(log_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["Name", "Email", "Message", "Reply", "Timestamp", "Seen"])

            writer.writerow([
                name,
                sender_email,
                message,
                "",
                datetime.datetime.now(),
                "No"
            ])

        return True

    except Exception as e:
        st.error(f"Email Error: {e}")
        return False


# ===================== SIDEBAR =====================

# Safety check – role must exist
if st.session_state.role is None:
    st.warning("Session invalid. Please login again.")
    st.session_state.logged_in = False
    st.rerun()

role = st.session_state.role

st.sidebar.markdown(f"### 👤 Logged in as: {role}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.remember_until = None
    st.rerun()


# ===================== ROLE BASED MENU =====================

if role == "Student":
    menu = [
        "Dashboard",
        "Manual Prediction",
        "Visual Analysis",
        "About + Contact"
    ]

elif role == "Faculty":
    menu = [
        "Dashboard",
        "Manual Prediction",
        "Visual Analysis",
        "Advanced Insights",
        "About + Contact"
    ]

elif role == "Admin":
    menu = [
        "Dashboard",
        "Manual Prediction",
        "Visual Analysis",
        "Advanced Insights",
        "Retrain Model",
        "Admin Panel",
        "About + Contact"
    ]

else:
    # Extra safety
    st.error("Invalid role detected.")
    st.stop()


choice = st.sidebar.selectbox("Navigation", menu)

# ==================== DASHBOARD ====================
if choice == "Dashboard":

    import numpy as np

    # ===== Header =====
    st.markdown("""
    <div style="background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);
                padding:1.8rem; border-radius:18px; text-align:center;
                color:white; font-size:2rem; font-weight:600;
                box-shadow:0 10px 30px rgba(0,0,0,0.3); margin-bottom:35px;">
        📊 SmartStudent AI – Executive Academic Dashboard
    </div>
    """, unsafe_allow_html=True)

    # ===== Model Overview =====
    st.markdown("## 🤖 Model Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Type", "Random Forest")
    col2.metric("Input Features", "4 Academic Metrics")
    col3.metric("Model Status",
                "Loaded ✅" if os.path.exists("model.pkl") else "Not Trained ❌")

    st.markdown("---")



    # ===== System Info =====
    st.markdown("## 📌 System Information")

    colA, colB = st.columns(2)

    with colA:
        st.info("""
        🔹 Predicts academic outcomes  
        🔹 Uses Random Forest AI  
        🔹 Evaluates 4 academic inputs  
        """)

    with colB:
        st.warning("""
        ⚠ Use Manual Prediction for real analysis  
        ⚠ Upload dataset for visual insights  
        """)

    st.markdown("---")

    st.success("🚀 System Ready for Academic Intelligence Analysis")


        # ===== Clean Professional Footer Message =====
    st.markdown("""
        <div style="text-align:center; margin-top:30px; 
                    padding:15px; border-radius:10px;
                    background-color:#f8f9fa; font-size:0.95rem; color:#555;">
            🚀 SmartStudent AI is ready. Use the navigation menu to begin analysis.
        </div>
        """, unsafe_allow_html=True)
# ==================== MANUAL PREDICTION – ENTERPRISE VERSION ====================
elif choice == "Manual Prediction":

    # ---------- Load Model Safely ----------
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
    else:
        st.error("❌ Model not found. Please retrain the model first.")
        st.stop()

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
                padding:2rem; border-radius:18px; text-align:center;
                color:white; font-size:1.8rem; font-weight:600;
                box-shadow:0 10px 30px rgba(0,0,0,0.4);
                margin-bottom:35px;">
        🎯 Executive Academic Prediction Engine
    </div>
    """, unsafe_allow_html=True)

    # ================= INPUT SECTION =================
    st.markdown("### 📊 Academic Input Parameters")

    col1, col2 = st.columns(2)

    with col1:
        assignment = st.slider("📘 Assignment Score", 0, 100, 70)
        participation = st.slider("🗣 Class Participation", 0, 100, 75)

    with col2:
        midterm = st.slider("📝 Midterm Marks", 0, 100, 72)
        final_exam = st.slider("🎓 Final Exam Marks", 0, 100, 80)

    avg_score = (assignment + participation + midterm + final_exam) / 4

    # ---------- Input Validation ----------
    if any(score < 0 or score > 100 for score in 
           [assignment, participation, midterm, final_exam]):
        st.error("Invalid score detected. Scores must be between 0 and 100.")
        st.stop()

    st.markdown("---")

    # ================= KPI CARDS =================
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Assignment", assignment)
    k2.metric("Participation", participation)
    k3.metric("Midterm", midterm)
    k4.metric("Final Exam", final_exam)

    st.markdown("### 📈 Academic Strength Index")
    st.progress(avg_score / 100)

    # ================= GRADE ENGINE =================
    if avg_score >= 90:
        grade = "A+"
    elif avg_score >= 80:
        grade = "A"
    elif avg_score >= 70:
        grade = "B"
    elif avg_score >= 60:
        grade = "C"
    else:
        grade = "D"

    st.metric("🎓 Academic Grade", grade)

    # ================= RISK ENGINE =================
    if avg_score >= 85:
        risk_level = "Low Risk"
        badge_color = "#2ecc71"
    elif avg_score >= 60:
        risk_level = "Moderate Risk"
        badge_color = "#f39c12"
    else:
        risk_level = "High Risk"
        badge_color = "#e74c3c"

    st.markdown(f"""
        <div style="background:{badge_color};
                    padding:12px; border-radius:12px;
                    text-align:center; color:white;
                    font-weight:bold; margin-top:10px;">
            🚦 Risk Assessment: {risk_level}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ================= STABILITY INDEX =================
    variance = pd.Series(
        [assignment, participation, midterm, final_exam]
    ).std()

    st.metric("📊 Performance Stability Index", f"{variance:.2f}")

    if variance < 5:
        st.success("Stable Performance Pattern")
    elif variance < 15:
        st.warning("Moderate Variability Detected")
    else:
        st.error("High Academic Instability")

    st.markdown("---")

    # ================= PREDICTION =================
    if st.button("🚀 Generate AI Prediction", use_container_width=True):

        input_df = pd.DataFrame([{
            "Assignment Score": assignment,
            "Class Participation": participation,
            "Midterm Marks": midterm,
            "Final Exam Marks": final_exam
        }])

        prediction = model.predict(input_df)[0]

        # ---------- Feature Importance ----------
        if hasattr(model, "feature_importances_"):

            st.subheader("📊 Model Feature Importance")

            importance_df = pd.DataFrame({
                "Feature": input_df.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            fig_imp = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Contribution to Prediction"
            )

            st.plotly_chart(fig_imp, use_container_width=True)

        # ---------- Probability ----------
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df).max() * 100
        else:
            prob = 85

        colA, colB = st.columns(2)
        colA.metric("🎯 Predicted Outcome", prediction)
        colB.metric("🔎 Confidence Score", f"{prob:.2f}%")

        # ---------- Confidence Indicator ----------
        st.markdown("### 🤖 AI Confidence Level")
        st.progress(prob / 100)

        if prob > 80:
            st.success("High AI Confidence")
        elif prob > 60:
            st.warning("Moderate AI Confidence")
        else:
            st.error("Low AI Confidence")

        # ---------- Projection Engine ----------
        projected_next_term = min(100, avg_score + (avg_score * 0.05))
        st.metric("📈 Projected Next Term Score",
                  f"{projected_next_term:.2f}%")

        # ---------- Donut Chart ----------
        fig = px.pie(
            names=["Performance", "Gap"],
            values=[avg_score, 100 - avg_score],
            hole=0.7,
            title="Academic Strength Composition"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------- Recommendations ----------
        st.markdown("## 🧠 Strategic Academic Recommendations")

        if risk_level == "Low Risk":
            recommendation = """
• Maintain current academic rhythm  
• Introduce advanced practice materials  
• Prepare for competitive examinations  
"""
        elif risk_level == "Moderate Risk":
            recommendation = """
• Increase revision frequency  
• Focus on weak assessment components  
• Schedule weekly performance review  
"""
        else:
            recommendation = """
• Immediate academic mentoring required  
• Daily supervised study schedule  
• Faculty intervention recommended  
"""

        st.info(recommendation)

        # ---------- Final Result Message ----------
        if prediction == "Pass":
            st.success("🎉 AI Model predicts PASS with strong probability.")
        else:
            st.error("⚠ AI Model predicts FAIL — Intervention Recommended.")

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

                X = df_train[['Assignment Score', 'Class Participation',
                              'Midterm Marks', 'Final Exam Marks']]
                y = df_train["Final Result"]

                model_new = RandomForestClassifier(n_estimators=100)
                model_new.fit(X, y)

                joblib.dump(model_new, "model.pkl")
                st.success("✅ Model retrained successfully!")

                y_pred = model_new.predict(X)
                acc = accuracy_score(y, y_pred)
                st.info(f"Training Accuracy: {acc*100:.2f}%")

                from sklearn.metrics import classification_report

                st.subheader("📊 Detailed Model Performance")

                report = classification_report(y, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                cm = confusion_matrix(y, y_pred)

                fig_cm, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                st.pyplot(fig_cm)



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

# ==================== 🔐 PROFESSIONAL ADMIN PANEL ====================
elif choice == "Admin Panel":

    st.markdown("""
        <h2 style='text-align:center; color:#2c3e50;'>🛡️ Admin Control Panel</h2>
        <hr style='border:1px solid #3498db; margin-bottom:15px;'>
    """, unsafe_allow_html=True)

    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False

    # ---------- Login Section ----------
    if not st.session_state["admin_logged_in"]:

        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "hussain123":
                st.session_state["admin_logged_in"] = True
                st.success("✅ Welcome, Admin!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials.")

    # ---------- Admin Dashboard ----------
    else:

        st.markdown("""
            <div style="background:#ecf6ff; padding:1rem 1.5rem; border-radius:12px; margin-bottom:15px;">
                <h4 style="color:#2c3e50; margin:0;">📊 Dashboard Overview</h4>
            </div>
        """, unsafe_allow_html=True)

        try:
            df_logs = pd.read_csv("contact_logs.csv", encoding="utf-8-sig")
        except:
            df_logs = pd.DataFrame(
                columns=["Name", "Email", "Message", "Reply", "Timestamp", "Seen"]
            )

        # Safe handling
        df_logs["Timestamp"] = pd.to_datetime(
            df_logs.get("Timestamp"), errors="coerce"
        ).fillna(datetime.datetime.now())

        df_logs["Seen"] = df_logs.get("Seen", "No").fillna("No")

        if "Reply" not in df_logs.columns:
            df_logs["Reply"] = ""

        df_logs["Reply"] = df_logs["Reply"].astype(str)
        df_logs["Reply"] = df_logs["Reply"].apply(
            lambda x: "" if x.lower() in ["nan"] else x
        )

        # Metrics
        total_msgs = len(df_logs)
        today_msgs = len(
            df_logs[df_logs["Timestamp"].dt.date == datetime.datetime.now().date()]
        )
        new_msgs_count = len(df_logs[df_logs["Seen"] == "No"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Messages", total_msgs)
        col2.metric("Today", today_msgs)
        col3.metric("New Messages", new_msgs_count)

        st.markdown("---")

        tabs = st.tabs([
            f"📨 Messages ({new_msgs_count} new)",
            "📊 Logs",
            "⬇️ Downloads"
        ])

        # ================= Messages Tab =================
        with tabs[0]:

            if df_logs.empty:
                st.info("No messages yet.")
            else:
                for index, row in df_logs.iterrows():

                    name = row["Name"]
                    email = row["Email"]
                    message = row["Message"]
                    timestamp = row["Timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                    reply_value = row["Reply"]

                    with st.expander(f"👤 {name} - {timestamp}"):

                        st.markdown(f"""
                        **📧 Email:** {email}  
                        **📝 Message:** {message}
                        """)

                        reply_text = st.text_area(
                            "✉️ Reply",
                            key=f"reply_{index}",
                            value=reply_value,
                            height=120
                        )

                        if st.button("Send Reply", key=f"send_{index}"):

                            if reply_text.strip():
                                try:
                                    msg = MIMEMultipart()
                                    msg["From"] = st.secrets["email"]
                                    msg["To"] = email
                                    msg["Subject"] = "Reply from SmartStudent AI"
                                    msg.attach(MIMEText(reply_text, "plain"))

                                    with smtplib.SMTP("smtp.gmail.com", 587) as server:
                                        server.starttls()
                                        server.login(
                                            st.secrets["email"],
                                            st.secrets["app_password"]
                                        )
                                        server.sendmail(
                                            st.secrets["email"],
                                            email,
                                            msg.as_string()
                                        )

                                    df_logs.at[index, "Reply"] = reply_text
                                    df_logs.at[index, "Seen"] = "Yes"
                                    df_logs.to_csv(
                                        "contact_logs.csv",
                                        index=False,
                                        encoding="utf-8-sig"
                                    )

                                    st.success("Reply sent successfully!")
                                    st.rerun()

                                except Exception as e:
                                    st.error(f"Email error: {e}")

        # ================= Logs Tab =================
        with tabs[1]:
            st.dataframe(df_logs.sort_values("Timestamp", ascending=False))

        # ================= Download Tab =================
        with tabs[2]:
            if not df_logs.empty:
                with open("contact_logs.csv", "rb") as f:
                    st.download_button(
                        "⬇ Download contact_logs.csv",
                        f,
                        file_name="contact_logs.csv"
                    )











