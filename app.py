from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image
import sqlite3
import hashlib
from datetime import datetime
import joblib

# =========================
# PATH SETUP
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "users.db"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
USER_RESULT_DIR = PROJECT_ROOT / "user_results"
USER_RESULT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_FILE = OUTPUT_DIR / "model_results.csv"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Audio Deepfake Detection Dashboard",
    page_icon="🎧",
    layout="wide"
)

# =========================
# STYLE
# =========================
st.markdown(
    """
    <style>
    .main { background-color: #F5F7FB; }
    .hero-box {
        background: linear-gradient(135deg, #0F172A, #2563EB);
        padding: 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }
    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 16px;
        color: #DBEAFE;
    }
    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: #111827;
        margin-top: 25px;
        margin-bottom: 10px;
    }
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetricLabel"] {
        color: #374151;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #1D4ED8;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# DATABASE FUNCTIONS
# =========================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)

    cursor.execute("PRAGMA table_info(users)")
    user_columns = [column[1] for column in cursor.fetchall()]
    if "role" not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        dataset_name TEXT NOT NULL,
        feature_type TEXT NOT NULL,
        classifier TEXT NOT NULL,
        total_records INTEGER,
        real_count INTEGER,
        fake_count INTEGER,
        result_file TEXT,
        upload_time TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS print_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            dataset_name TEXT,
            feature_type TEXT,
            classifier TEXT,
            total_records INTEGER,
            real_count INTEGER,
            fake_count INTEGER,
            print_time TEXT
        )
    """)

    conn.commit()
    conn.close()


def register_user(username: str, password: str):
    username = username.strip()
    role = "admin" if username.lower() == "admin" else "user"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, hash_password(password), role)
        )
        conn.commit()
        conn.close()
        return True, "Registration successful. You can now login."
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists."


def verify_user(username: str, password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username.strip(),))
    result = cursor.fetchone()
    conn.close()

    if result is None:
        return False
    return result[0] == hash_password(password)


def get_user_role(username: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else "user"


def save_dataset_result(username, dataset_name, feature_type, classifier, total_records, real_count, fake_count, result_file):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO dataset_results (
            username, dataset_name, feature_type, classifier,
            total_records, real_count, fake_count, result_file, upload_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        dataset_name,
        feature_type,
        classifier,
        total_records,
        real_count,
        fake_count,
        result_file,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_user_dataset_history(username):
    conn = sqlite3.connect(DB_PATH)
    history_df = pd.read_sql_query(
        """
        SELECT 
            id,
            dataset_name,
            feature_type,
            classifier,
            total_records,
            real_count,
            fake_count,
            result_file,
            upload_time
        FROM dataset_results
        WHERE username = ?
        ORDER BY upload_time DESC
        """,
        conn,
        params=(username,)
    )
    conn.close()
    return history_df


def get_all_dataset_history():
    conn = sqlite3.connect(DB_PATH)
    history_df = pd.read_sql_query(
        """
        SELECT username, dataset_name, feature_type, classifier,
               total_records, real_count, fake_count, upload_time
        FROM dataset_results
        ORDER BY upload_time DESC
        """,
        conn
    )
    conn.close()
    return history_df


def save_print_log(username, dataset_name, feature_type, classifier, total_records, real_count, fake_count):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO print_logs (
            username, dataset_name, feature_type, classifier,
            total_records, real_count, fake_count, print_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username, dataset_name, feature_type, classifier,
        total_records, real_count, fake_count,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def get_print_logs():
    conn = sqlite3.connect(DB_PATH)
    logs_df = pd.read_sql_query(
        "SELECT * FROM print_logs ORDER BY print_time DESC",
        conn
    )
    conn.close()
    return logs_df

# =========================
# MODEL / FILE HELPERS
# =========================
def safe_name(text):
    return text.lower().replace(" ", "_").replace("+", "plus").replace("-", "_")


def get_model_file(feature_type, classifier):
    name_map = {
        ("MFCC", "SVM"): "mfcc_svm_within.pkl",
        ("MFCC", "Random Forest"): "mfcc_random_forest_within.pkl",
        ("MFCC", "XGBoost"): "mfcc_xgboost_within.pkl",
        ("WavLM", "SVM"): "wavlm_svm_within.pkl",
        ("WavLM", "Random Forest"): "wavlm_random_forest_within.pkl",
        ("WavLM", "XGBoost"): "wavlm_xgboost_within.pkl",
        ("MFCC + WavLM", "SVM"): "mfcc_plus_wavlm_svm_within.pkl",
        ("MFCC + WavLM", "Random Forest"): "mfcc_plus_wavlm_random_forest_within.pkl",
        ("MFCC + WavLM", "XGBoost"): "mfcc_plus_wavlm_xgboost_within.pkl",
    }
    return MODEL_DIR / name_map[(feature_type, classifier)]


def get_required_feature_columns(feature_type):
    if feature_type == "MFCC":
        return [f"mfcc_{i}" for i in range(1, 81)]
    if feature_type == "WavLM":
        return [f"wavlm_{i}" for i in range(1, 769)]
    return [f"mfcc_{i}" for i in range(1, 81)] + [f"wavlm_{i}" for i in range(1, 769)]


def load_results_file():
    if not RESULTS_FILE.exists():
        return None
    df = pd.read_csv(RESULTS_FILE)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    return df

# =========================
# UI HELPERS
# =========================
def show_header():
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">🎧 Audio Deepfake Detection Dashboard</div>
            <div class="hero-subtitle">
                Final Year Project dashboard for dataset checking, feature-based prediction,
                model performance viewing, and user result history.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def auth_page():
    init_db()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    if st.session_state.authenticated:
        return True

    st.markdown("### User Access")
    st.write("Login or register to access the dashboard.")

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        auth_tab = st.radio("Choose option", ["Login", "Register"], horizontal=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if auth_tab == "Register":
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.button("Register", use_container_width=True):
                if username.strip() == "" or password.strip() == "":
                    st.error("Username and password cannot be empty.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(password) < 5:
                    st.error("Password must be at least 5 characters.")
                else:
                    success, message = register_user(username, password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        else:
            if st.button("Login", use_container_width=True):
                if verify_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username.strip()
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    return False

# =========================
# PAGE SECTIONS
# =========================
def show_user_dataset_dashboard():
    st.markdown('<div class="section-title">My Dataset Dashboard</div>', unsafe_allow_html=True)

    history_df = get_user_dataset_history(st.session_state.username)

    if history_df.empty:
        st.info("No dataset has been submitted yet. Upload a feature CSV dataset to generate your first result.")

    else:
        st.success("Previous dataset results found.")

        latest = history_df.iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Latest Total Records", int(latest["total_records"]))

        with col2:
            st.metric("Latest Predicted Real", int(latest["real_count"]))

        with col3:
            st.metric("Latest Predicted Fake", int(latest["fake_count"]))

        st.write("Dataset Result History")

        display_history_df = history_df.copy()

        if "result_file" in display_history_df.columns:
            display_history_df = display_history_df.drop(columns=["result_file"])

        st.dataframe(
            display_history_df,
            use_container_width=True,
            hide_index=True
        )

        if "result_file" in history_df.columns:
            st.subheader("Download Previous Prediction Result")

            history_options = (
                history_df["id"].astype(str)
                + " - "
                + history_df["dataset_name"].astype(str)
                + " - "
                + history_df["feature_type"].astype(str)
                + " - "
                + history_df["classifier"].astype(str)
            )

            selected_history = st.selectbox(
                "Select previous result",
                history_options
            )

            selected_id = int(selected_history.split(" - ")[0])
            selected_row = history_df[history_df["id"] == selected_id].iloc[0]

            if pd.notna(selected_row["result_file"]):
                result_file_path = PROJECT_ROOT / selected_row["result_file"]

                if result_file_path.exists():
                    result_df = pd.read_csv(result_file_path)

                    st.write("Selected Previous Prediction Result")
                    st.dataframe(result_df, use_container_width=True)

                    previous_csv_output = result_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Selected Previous Prediction Result CSV",
                        data=previous_csv_output,
                        file_name=result_file_path.name,
                        mime="text/csv",
                        key=f"download_previous_result_{selected_id}"
                    )
                else:
                    st.warning("Saved result file was not found.")

    st.divider()
    st.subheader("Submit New Dataset")
    st.info(
        "Upload a feature CSV file to classify audio samples as real or fake. "
        "The uploaded file must already contain extracted feature columns."
    )

    MAX_FILE_SIZE_MB = 5
    MAX_ROWS = 1000

    upload_feature_types = st.multiselect(
        "Select Feature Type(s) for Uploaded Dataset",
        ["MFCC", "WavLM", "MFCC + WavLM"],
        default=["MFCC"],
        key="upload_feature_types"
    )

    upload_classifiers = st.multiselect(
        "Select Classifier(s) for Uploaded Dataset",
        ["SVM", "Random Forest", "XGBoost"],
        default=["SVM"],
        key="upload_classifiers"
    )
    uploaded_file = st.file_uploader("Upload Feature CSV", type=["csv"])

    if uploaded_file is None:
        return

    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.")
        return

    try:
        user_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read CSV file: {e}")
        return

    if len(user_df) > MAX_ROWS:
        st.error(f"Too many rows. Maximum allowed rows is {MAX_ROWS}.")
        return

    st.success("Dataset uploaded successfully.")
    st.write("Dataset Preview")
    st.dataframe(user_df.head(), use_container_width=True)

    if len(upload_feature_types) == 0:
        st.error("Please select at least one feature type.")
        return

    if len(upload_classifiers) == 0:
        st.error("Please select at least one classifier.")
        return

    all_summary_rows = []
    all_prediction_outputs = []

    for selected_feature_type in upload_feature_types:
        required_cols = get_required_feature_columns(selected_feature_type)
        missing_cols = [col for col in required_cols if col not in user_df.columns]

        if missing_cols:
            st.warning(
                f"{selected_feature_type} was skipped because the uploaded CSV does not contain the required feature columns."
            )
            continue

        X_user = user_df[required_cols]

        for selected_classifier in upload_classifiers:
            try:
                model_path = get_model_file(selected_feature_type, selected_classifier)

                if not model_path.exists():
                    st.warning(f"Model file not found: {model_path.name}. Skipped.")
                    continue

                model = joblib.load(model_path)
                predictions = model.predict(X_user)

                real_count = int((predictions == 0).sum())
                fake_count = int((predictions == 1).sum())
                total_records = len(predictions)

                all_summary_rows.append({
                    "Dataset Name": uploaded_file.name,
                    "Feature Type": selected_feature_type,
                    "Classifier": selected_classifier,
                    "Total Records": total_records,
                    "Predicted Real": real_count,
                    "Predicted Fake": fake_count
                })

                temp_result_df = user_df.copy()
                temp_result_df["feature_type"] = selected_feature_type
                temp_result_df["classifier"] = selected_classifier
                temp_result_df["prediction"] = predictions
                temp_result_df["prediction_label"] = temp_result_df["prediction"].map({
                    0: "Real",
                    1: "Fake"
                })

                safe_dataset_name = uploaded_file.name.replace(".csv", "").replace(" ", "_")
                safe_feature_name = selected_feature_type.replace(" ", "_").replace("+", "plus")
                safe_classifier_name = selected_classifier.replace(" ", "_")

                result_filename = f"{st.session_state.username}_{safe_dataset_name}_{safe_feature_name}_{safe_classifier_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                result_path = USER_RESULT_DIR / result_filename

                temp_result_df.to_csv(result_path, index=False)

                save_dataset_result(
                    st.session_state.username,
                    uploaded_file.name,
                    selected_feature_type,
                    selected_classifier,
                    total_records,
                    real_count,
                    fake_count,
                    str(result_path.relative_to(PROJECT_ROOT))
                )

                all_prediction_outputs.append(temp_result_df)

            except Exception as e:
                st.error(f"Prediction failed for {selected_feature_type} + {selected_classifier}: {e}")

    if len(all_summary_rows) > 0:
        st.success("Prediction completed successfully.")

        summary_df = pd.DataFrame(all_summary_rows)

        st.subheader("Prediction Summary")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.subheader("Uploaded Dataset Result Dashboard")

        total_real = int(summary_df["Predicted Real"].sum())
        total_fake = int(summary_df["Predicted Fake"].sum())
        total_predictions = int(summary_df["Total Records"].sum())

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Predictions", total_predictions)

        with col2:
            st.metric("Total Predicted Real", total_real)

        with col3:
            st.metric("Total Predicted Fake", total_fake)

        chart_data = pd.DataFrame({
            "Class": ["Real", "Fake"],
            "Count": [total_real, total_fake]
        }).set_index("Class")

        st.bar_chart(chart_data)

        st.info(
            "This result shows the classification output for the uploaded dataset. "
            "Each selected feature-classifier combination produces a prediction summary showing how many records were classified as real or fake."
        )

        if len(all_prediction_outputs) > 0:
            combined_result_df = pd.concat(all_prediction_outputs, ignore_index=True)

            st.subheader("Detailed Prediction Results")
            st.dataframe(combined_result_df, use_container_width=True)

            csv_output = combined_result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download All Prediction Results CSV",
                data=csv_output,
                file_name="all_prediction_results.csv",
                mime="text/csv",
                key="download_all_prediction_results_main"
            )
        else:
            st.warning("No detailed prediction output was generated.")

        if st.button("Generate Printable Report"):
            for row in all_summary_rows:
                save_print_log(
                    st.session_state.username,
                    row["Dataset Name"],
                    row["Feature Type"],
                    row["Classifier"],
                    row["Total Records"],
                    row["Predicted Real"],
                    row["Predicted Fake"]
                )

            st.success("Printable report generated and logged.")

            st.markdown("### Printable Report")
            st.write(f"Username: {st.session_state.username}")
            st.write(f"Dataset Name: {uploaded_file.name}")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.info("Use Ctrl + P to print this report from the browser.")
    else:
        st.error("No prediction was completed. Please check the uploaded CSV and selected feature types.")


def show_experiment_dashboard():
    results_df = load_results_file()
    if results_df is None:
        st.error("model_results.csv not found. Please run python src/train_models.py first.")
        return

    st.markdown('<div class="section-title">Experiment Dashboard</div>', unsafe_allow_html=True)

    evaluation_type = st.sidebar.selectbox(
        "Evaluation Type",
        sorted(results_df["Evaluation Type"].unique())
    )
    feature_type = st.sidebar.selectbox(
        "Feature Type",
        sorted(results_df["Feature"].unique())
    )
    classifier = st.sidebar.selectbox(
        "Classifier",
        sorted(results_df["Classifier"].unique())
    )
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    metric = st.sidebar.selectbox("Metric for Chart", metric_cols)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Datasets", "2")
    with col2:
        st.metric("Feature Sets", "3")
    with col3:
        st.metric("Classifiers", "3")
    with col4:
        st.metric("Evaluation Types", "2")

    st.subheader("Dataset Overview")
    dataset_df = pd.DataFrame({
        "Dataset": ["ASVspoof2019 LA", "Fake-or-Real"],
        "Audio Type / Category": [
            "Bonafide + Logical Access spoof audio",
            "Real + TTS-generated fake audio"
        ],
        "Real / Bonafide Samples": [500, 500],
        "Fake / Spoof Samples": [500, 500],
        "Total Samples": [1000, 1000],
    })
    st.dataframe(dataset_df, use_container_width=True, hide_index=True)

    st.subheader("Feature Extraction Output")
    feature_df = pd.DataFrame({
        "Feature Type": ["MFCC", "WavLM", "MFCC + WavLM"],
        "Number of Features": [80, 768, 848],
        "Description": [
            "Handcrafted acoustic features",
            "Learning-driven speech embeddings",
            "Concatenation of MFCC and WavLM features"
        ]
    })
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.subheader("Model Performance Results")
    filtered_df = results_df[results_df["Evaluation Type"] == evaluation_type].copy()
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    st.subheader("Selected Model Summary")
    selected_row = results_df[
        (results_df["Evaluation Type"] == evaluation_type) &
        (results_df["Feature"] == feature_type) &
        (results_df["Classifier"] == classifier)
    ]

    if not selected_row.empty:
        row = selected_row.iloc[0]
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", row["Accuracy"])
        with col2:
            st.metric("Precision", row["Precision"])
        with col3:
            st.metric("Recall", row["Recall"])
        with col4:
            st.metric("F1-Score", row["F1-Score"])
        with col5:
            st.metric("ROC-AUC", row["ROC-AUC"])

    st.subheader(f"{metric} Comparison")
    chart_df = filtered_df.copy()
    chart_df["Model"] = chart_df["Feature"] + " - " + chart_df["Classifier"]
    st.bar_chart(chart_df[["Model", metric]].set_index("Model"))

    st.subheader("Confusion Matrix")
    eval_short = "within" if evaluation_type == "Within-dataset ASVspoof" else "cross"
    cm_filename = f"confusion_matrix_{safe_name(feature_type)}_{safe_name(classifier)}_{eval_short}.png"
    cm_path = OUTPUT_DIR / cm_filename
    st.write(f"Selected confusion matrix: `{cm_filename}`")
    if cm_path.exists():
        image = Image.open(cm_path)

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(image, width=600)
    else:
        st.warning("Confusion matrix image not found in outputs folder.")

    st.subheader("Cross-Dataset Generalization Summary")
    within_df = results_df[results_df["Evaluation Type"] == "Within-dataset ASVspoof"]
    cross_df = results_df[results_df["Evaluation Type"] == "Cross-dataset ASVspoof to FoR"]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Within-Dataset Accuracy", round(within_df["Accuracy"].mean(), 4))
    with col2:
        st.metric("Average Cross-Dataset Accuracy", round(cross_df["Accuracy"].mean(), 4))
    st.info(
        "Cross-dataset evaluation shows whether a model trained on ASVspoof2019 LA can generalize to Fake-or-Real. "
        "A large performance drop indicates weak generalization across datasets."
    )


def show_admin_dashboard():
    st.markdown('<div class="section-title">Admin Dashboard</div>', unsafe_allow_html=True)

    st.subheader("All User Dataset Results")
    all_history_df = get_all_dataset_history()
    if all_history_df.empty:
        st.info("No user dataset submissions yet.")
    else:
        st.dataframe(all_history_df, use_container_width=True, hide_index=True)

    st.subheader("Print Report Logs")
    print_logs_df = get_print_logs()
    if print_logs_df.empty:
        st.info("No printable report logs yet.")
    else:
        st.dataframe(print_logs_df, use_container_width=True, hide_index=True)

# =========================
# MAIN APP
# =========================
show_header()

if not auth_page():
    st.stop()

st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.latest_result = None
    st.rerun()

role = get_user_role(st.session_state.username)
page_options = ["My Dataset Dashboard", "Experiment Dashboard"]
if role == "admin":
    page_options.append("Admin Dashboard")

user_role = get_user_role(st.session_state.username)

if user_role == "admin":
    page_options = [
        "My Dataset Dashboard",
        "Experiment Dashboard",
        "Admin Dashboard"
    ]
else:
    page_options = [
        "My Dataset Dashboard"
    ]

selected_page = st.sidebar.radio("Navigation", page_options)

if selected_page == "My Dataset Dashboard":
    show_user_dataset_dashboard()

elif selected_page == "Experiment Dashboard":
    if user_role != "admin":
        st.error("Access denied. This page is only available for admin.")
        st.stop()
    show_experiment_dashboard()

elif selected_page == "Admin Dashboard":
    if user_role != "admin":
        st.error("Access denied. This page is only available for admin.")
        st.stop()
    show_admin_dashboard()
