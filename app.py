import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "final_gb_classifier.pkl"
DATA_PATH = BASE_DIR / "Telco-Customer-Churn.csv"

# ------------------------------------------------------------------
# Page config (must be first Streamlit command)
# ------------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(120deg, #6a11cb 0%, #2575fc 55%, #00c6ff 100%);
        padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 22px;
        box-shadow: 0 8px 24px rgba(37,117,252,0.25);
    }
    .hero h1 { margin: 0; font-size: 2.2rem; color: white; }
    .hero p  { margin: 6px 0 0 0; font-size: 1.05rem; opacity: 0.92; }
    .result-card { padding: 26px 30px; border-radius: 18px; color: white; text-align: center;
                   box-shadow: 0 8px 24px rgba(0,0,0,0.18); margin-top: 10px; }
    .churn-yes { background: linear-gradient(135deg, #ff416c, #ff4b2b); }
    .churn-no  { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .result-card h2 { margin: 0; color: white; font-size: 1.8rem; }
    .result-card p  { margin: 8px 0 0 0; font-size: 1.05rem; }
    .prob-wrap { background: rgba(128,128,128,0.25); border-radius: 999px; height: 22px;
                 overflow: hidden; margin: 10px 0 4px 0; }
    .prob-bar  { height: 100%; border-radius: 999px;
                 background: linear-gradient(90deg, #38ef7d 0%, #f9d423 50%, #ff4b2b 100%); }
    .tip { border-left: 5px solid #6a11cb; background: rgba(106,17,203,0.08);
           padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; }
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(90deg, #6a11cb, #2575fc); color: white; border: none;
        padding: 0.7rem 2rem; font-size: 1.1rem; font-weight: 600; border-radius: 12px;
        width: 100%; transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 6px 18px rgba(37,117,252,0.4); color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Load model + data (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


model = load_model()
FEATURES = list(model.feature_names_in_)

# ------------------------------------------------------------------
# Encoding maps (identical to LabelEncoder in the notebook = alphabetical)
# ------------------------------------------------------------------
GENDER_MAP = {"Female": 0, "Male": 1}
YES_NO_MAP = {"No": 0, "Yes": 1}
MULTI_MAP = {"No": 0, "No phone service": 1, "Yes": 2}
ADDON_MAP = {"No": 0, "No internet service": 1, "Yes": 2}
INTERNET_MAP = {"DSL": 0, "Fiber optic": 1, "No": 2}
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT_MAP = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3,
}

# UI label helpers (model still gets numeric codes)
GENDER = {0: "Female", 1: "Male"}
YES_NO = {0: "No", 1: "Yes"}
MULTI_LINES = {0: "No", 1: "No phone service", 2: "Yes"}
NET_ADDON = {0: "No", 1: "No internet service", 2: "Yes"}
TENURE = {
    0: "0 - 11 months",
    1: "12 - 23 months",
    2: "24 - 35 months",
    3: "36 - 47 months",
    4: "48 - 59 months",
    5: "60+ months",
}


def radio(label, mapping):
    return st.radio(label, list(mapping.keys()), format_func=lambda k: mapping[k], horizontal=True)


def preprocess_input(data):
    """Single customer dict (UI values) -> model-ready DataFrame."""
    df = pd.DataFrame(data, index=[0])
    df["InternetService"] = df["InternetService"].map(INTERNET_MAP)
    df["Contract"] = df["Contract"].map(CONTRACT_MAP)
    df["PaymentMethod"] = df["PaymentMethod"].map(PAYMENT_MAP)
    return df[FEATURES]


def encode_raw(raw):
    """Raw Telco-style DataFrame (text values) -> model-ready DataFrame."""
    df = raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df = df.dropna(subset=["TotalCharges", "MonthlyCharges", "tenure"])

    out = pd.DataFrame(index=df.index)
    out["gender"] = df["gender"].map(GENDER_MAP)
    out["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        out[c] = df[c].map(YES_NO_MAP)
    out["MultipleLines"] = df["MultipleLines"].map(MULTI_MAP)
    out["InternetService"] = df["InternetService"].map(INTERNET_MAP)
    for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies"]:
        out[c] = df[c].map(ADDON_MAP)
    out["Contract"] = df["Contract"].map(CONTRACT_MAP)
    out["PaymentMethod"] = df["PaymentMethod"].map(PAYMENT_MAP)
    out["MonthlyCharges"] = df["MonthlyCharges"]
    out["TotalCharges"] = df["TotalCharges"]
    out["tenure_group"] = (df["tenure"] // 12).clip(0, 5).astype(int)
    out = out.dropna()
    return out[FEATURES].astype(float).astype({c: int for c in FEATURES if c not in ["MonthlyCharges", "TotalCharges"]})


def risk_label(p):
    if p < 0.33:
        return "🟢 Low"
    if p < 0.66:
        return "🟡 Medium"
    return "🔴 High"


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 About")
    st.write(
        "Predicts whether a telecom customer is likely to **churn** using a tuned "
        "**Gradient Boosting Classifier**."
    )
    st.markdown("---")
    st.markdown("### 🧭 Pages")
    st.write("🔮 **Single** - one customer\n\n📂 **Batch** - upload a CSV\n\n📈 **Insights** - data & model")
    st.markdown("---")
    st.caption("Model: GradientBoostingClassifier · Dataset: Telco Customer Churn")

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📉 Customer Churn Prediction</h1>
        <p>Predict churn for one customer, a whole CSV, or explore what drives churn.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

page1, page2, page3 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction", "📈 Insights"])

# ==================================================================
# PAGE 1 - SINGLE PREDICTION
# ==================================================================
with page1:
    tab1, tab2, tab3 = st.tabs(["👤 Customer Profile", "📡 Services", "💳 Billing & Contract"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            gender = radio("Gender", GENDER)
            partner = radio("Partner", YES_NO)
            senior_citizen = radio("Senior Citizen", YES_NO)
        with c2:
            dependents = radio("Dependents", YES_NO)
            tenure_group = st.selectbox("Tenure Group", list(TENURE.keys()), format_func=lambda k: TENURE[k])

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            phone_service = radio("Phone Service", YES_NO)
            multiple_lines = radio("Multiple Lines", MULTI_LINES)
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            streaming_tv = radio("Streaming TV", NET_ADDON)
            streaming_movies = radio("Streaming Movies", NET_ADDON)
        with c2:
            online_security = radio("Online Security", NET_ADDON)
            online_backup = radio("Online Backup", NET_ADDON)
            device_protection = radio("Device Protection", NET_ADDON)
            tech_support = radio("Tech Support", NET_ADDON)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            paperless_billing = radio("Paperless Billing", YES_NO)
        with c2:
            monthly_charges = st.number_input("Monthly Charges", value=70.0, min_value=0.0, step=1.0)
            total_charges = st.number_input("Total Charges", value=500.0, min_value=0.0, step=10.0)

    st.markdown("")

    if st.button("🔮 Predict", key="predict_single"):
        user_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "tenure_group": tenure_group,
        }
        processed_data = preprocess_input(user_data)

        with st.spinner("Analysing customer..."):
            prediction = model.predict(processed_data)
            churn_prob = float(model.predict_proba(processed_data)[0][1])

        st.markdown("---")
        left, right = st.columns([3, 2])

        with left:
            if prediction[0] == 1:
                st.markdown(
                    '<div class="result-card churn-yes"><h2>⚠️ The customer is likely to churn.</h2>'
                    "<p>Consider a retention offer soon.</p></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-card churn-no"><h2>✅ The customer is likely to stay.</h2>'
                    "<p>Looks like a loyal customer.</p></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("#### Churn probability")
            st.markdown(
                f'<div class="prob-wrap"><div class="prob-bar" style="width:{churn_prob*100:.1f}%"></div></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"{churn_prob*100:.1f}% chance of churning")

        with right:
            m1, m2 = st.columns(2)
            m1.metric("Churn Risk", risk_label(churn_prob))
            m2.metric("Stay Probability", f"{(1 - churn_prob)*100:.1f}%")
            m3, m4 = st.columns(2)
            m3.metric("Monthly Charges", f"{monthly_charges:,.2f}")
            m4.metric("Tenure", TENURE[tenure_group].split(" months")[0] + " m")

        tips = []
        if contract == "Month-to-month":
            tips.append("📝 Month-to-month contract: offer a discount for a 1- or 2-year plan.")
        if internet_service == "Fiber optic":
            tips.append("🌐 Fiber optic users churn more often: check speed/price satisfaction.")
        if payment_method == "Electronic check":
            tips.append("💳 Electronic check payers churn more: encourage auto-pay with a small incentive.")
        if internet_service != "No" and tech_support == 0:
            tips.append("🛠️ No tech support: a free trial of support can improve retention.")
        if tenure_group == 0:
            tips.append("🆕 New customer (first year): onboarding calls and early offers help a lot.")
        if tips:
            st.markdown("#### 💡 Retention ideas")
            for t in tips:
                st.markdown(f'<div class="tip">{t}</div>', unsafe_allow_html=True)

        with st.expander("🔍 See processed input sent to the model"):
            st.dataframe(processed_data)

# ==================================================================
# PAGE 2 - BATCH PREDICTION
# ==================================================================
with page2:
    st.subheader("📂 Predict for many customers at once")
    st.write(
        "Upload a CSV in the **same format as Telco-Customer-Churn.csv** "
        "(columns: gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, "
        "InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, "
        "StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    sample_data = load_data()
    use_sample = False
    if sample_data is not None:
        use_sample = st.checkbox("Or try with the bundled Telco dataset (first 300 rows)")

    raw = None
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
    elif use_sample:
        raw = sample_data.head(300)

    if raw is not None:
        required = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
        missing = [c for c in required if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            try:
                encoded = encode_raw(raw)
                probs = model.predict_proba(encoded)[:, 1]
                preds = (probs >= 0.5).astype(int)

                result = raw.loc[encoded.index].copy()
                result["Churn_Probability_%"] = (probs * 100).round(1)
                result["Churn_Prediction"] = np.where(preds == 1, "Yes", "No")
                result["Risk"] = [risk_label(p) for p in probs]

                dropped = len(raw) - len(result)
                if dropped:
                    st.warning(f"{dropped} row(s) skipped because of missing/invalid values.")

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Customers", f"{len(result):,}")
                k2.metric("Predicted to churn", f"{int(preds.sum()):,}")
                k3.metric("Churn rate", f"{preds.mean()*100:.1f}%")
                k4.metric("Avg churn probability", f"{probs.mean()*100:.1f}%")

                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("##### Risk distribution")
                    st.bar_chart(result["Risk"].value_counts())
                with cc2:
                    st.markdown("##### Churn probability histogram")
                    hist, edges = np.histogram(probs * 100, bins=10, range=(0, 100))
                    st.bar_chart(pd.DataFrame({"customers": hist}, index=[f"{int(e)}-{int(e+10)}%" for e in edges[:-1]]))

                st.markdown("##### 🔥 Top 15 highest-risk customers")
                top = result.sort_values("Churn_Probability_%", ascending=False).head(15)
                st.dataframe(top)

                with st.expander("📋 Full results"):
                    st.dataframe(result)

                buf = io.StringIO()
                result.to_csv(buf, index=False)
                st.download_button("⬇️ Download results as CSV", buf.getvalue(),
                                   file_name="churn_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Could not process the file: {e}")

# ==================================================================
# PAGE 3 - INSIGHTS
# ==================================================================
with page3:
    st.subheader("📈 What drives churn?")
    data = load_data()

    if data is not None:
        d = data.copy()
        d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
        d = d.dropna(subset=["TotalCharges"])
        d["ChurnFlag"] = (d["Churn"] == "Yes").astype(int)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Customers", f"{len(d):,}")
        k2.metric("Churn rate", f"{d['ChurnFlag'].mean()*100:.1f}%")
        k3.metric("Avg monthly charges", f"{d['MonthlyCharges'].mean():.1f}")
        k4.metric("Avg tenure (months)", f"{d['tenure'].mean():.1f}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Churn rate by Contract (%)")
            st.bar_chart(d.groupby("Contract")["ChurnFlag"].mean() * 100)
        with c2:
            st.markdown("##### Churn rate by Internet Service (%)")
            st.bar_chart(d.groupby("InternetService")["ChurnFlag"].mean() * 100)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("##### Churn rate by Payment Method (%)")
            st.bar_chart(d.groupby("PaymentMethod")["ChurnFlag"].mean() * 100)
        with c4:
            st.markdown("##### Churn rate by Tenure (years)")
            years = (d["tenure"] // 12).astype(int).clip(0, 5)
            st.bar_chart(d.groupby(years)["ChurnFlag"].mean() * 100)
    else:
        st.info("Place Telco-Customer-Churn.csv next to app.py to see dataset charts.")

    st.markdown("##### 🧠 Model feature importance")
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
        st.bar_chart(imp, horizontal=True)
