import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

st.markdown("""
<style>

/* ================= BRIGHT BACKGROUND ================= */
.stApp {
    background: linear-gradient(120deg, #f0f9ff, #e0f7ff, #c7f9ff, #dbeafe);
    color: #1e293b;
}

/* ================= HERO ================= */
.hero {
    text-align:center;
    background: linear-gradient(90deg, #2563EB, #06B6D4);
    border-radius:18px;
    padding:30px 20px;
    color:white;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.1);
}

/* ================= INPUT FIELDS ================= */
input, .stNumberInput input {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5f5 !important;
    border-radius: 10px !important;
    padding: 8px !important;
}

/* ================= REMOVE ALL BLUE / CLICK EFFECT ================= */


/* Hover (very light) */

/* ================= SELECT BOX ================= */
.stSelectbox div {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border-radius: 10px !important;
}

/* ================= LABELS ================= */
label {
    color: #334155 !important;
    font-weight: 500;
    font-size: 14px;
}

/* ================= SLIDER ================= */
.stSlider > div {
    background: #cbd5f5 !important;
}

/* ================= TABS ================= */
.stTabs [role="tab"] {
    background: rgba(255,255,255,0.6);
    border-radius: 10px;
    padding: 10px;
}

.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5f5;
}

/* ================= CARDS ================= */
.card {
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(0,0,0,0.05);
    margin-bottom: 20px;
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
}

/* ================= SECTION TITLE ================= */
.section-banner {
    text-align:center;
    font-size:22px;
    font-weight:600;
    margin:20px 0;
    padding:8px;
    border-radius:8px;
    background: linear-gradient(90deg, #2563EB, #06B6D4);
    color:white;
}

/* ================= BUTTON ================= */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    font-size: 16px;
    background: linear-gradient(90deg, #2563EB, #06B6D4);
    color: white;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
}

/* ================= REMOVE EXTRA BAR ================= */
.block-container {
    padding-top: 1rem;
}

[data-testid="stHorizontalBlock"] > div {
    background: transparent !important;
}
            
        * {
    outline: none !important;
    box-shadow: none !important;
}

/* ================= RADIO BUTTONS ================= */
.stRadio > div {
    display: flex;
    gap: 10px;
}

/* Default style */
.stRadio label {
    background: transparent !important;
    color: #1e293b !important;
    padding: 6px 10px;
    border-radius: 0px;
    border: none !important;
    cursor: pointer;
}

/* SELECTED → NO CHANGE AT ALL */
.stRadio input:checked + div {
    background: transparent !important;
    color: #1e293b !important;
    border: none !important;
}

/* Remove hover effect */
.stRadio label:hover {
    background: transparent !important;
}
            

/* ================= SUGGESTION BOX ================= */
.suggestion-box {
    background: linear-gradient(135deg, #ffffff, #f0f9ff);
    border-radius: 12px;
    padding: 14px 16px;
    margin: 10px 0;
    border-left: 5px solid #2563EB;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    font-size: 14px;
    color: #1e293b;
    transition: 0.3s;
}

/* Hover effect */
.suggestion-box:hover {
    transform: translateX(5px);
    box-shadow: 0px 6px 15px rgba(0,0,0,0.08);
}

/* Icon style */
.suggestion-box span {
    font-weight: 600;
}


</style>
""", unsafe_allow_html=True)
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="AI Preventive Health Dashboard",
    layout="wide"
)


# =======================
# LOAD MODELS (CACHED)
# =======================
import gdown
import os
import joblib
import streamlit as st

@st.cache_resource
def load_models():

    files = {
        "heart": ("heart.pkl", "16V_bk9eFzXCqBtOeIlK55vZISxyZ7BR2"),
        "diabetes": ("diabetes.pkl", "1I9f6ACGRy2FTFJe38LpZbIQn2J2nsOTh"),
        "obesity": ("obesity.pkl", "1LuaEQgttlDnV-kK7X5ujEwRqC0c9jsV5"),
        "hypertension": ("hypertension.pkl", "1Ub29aUOQNI4ZQQWeKsZSdTQBQ9wmqoum"),
    }

    models = {}

    for key, (filename, file_id) in files.items():

        if not os.path.exists(filename):
            file_id = "1AbCDeFgHiJK12345LmNoPq"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=False)

        models[key] = joblib.load(filename)

    return models


with st.spinner("📥 Loading AI models..."):
    models = load_models()

# =======================
# HELPER FUNCTIONS
# =======================
def build_input(bundle, input_dict):
    df = pd.DataFrame([input_dict])
    if "features" not in bundle:
        raise ValueError("Model bundle does not contain feature list.")
    df = df.reindex(columns=bundle["features"], fill_value=0)
    return df

def hybrid_predict(bundle, df):
    features = bundle.get("features")
    scaler = bundle.get("scaler")
    lr = bundle.get("lr")
    rf = bundle.get("rf")
    xgb = bundle.get("xgb")
    df = df[features]
    if scaler is not None:
        scaled = scaler.transform(df)
        lr_prob = lr.predict_proba(scaled)[:, 1]
    else:
        lr_prob = lr.predict_proba(df)[:, 1]
    rf_prob = rf.predict_proba(df)[:, 1]
    xgb_prob = xgb.predict_proba(df)[:, 1]
    final_prob = 0.3 * lr_prob + 0.3 * rf_prob + 0.4 * xgb_prob
    return float(final_prob[0])

import shap
import matplotlib.pyplot as plt

def shap_explain(bundle, input_df):
    try:
        model = bundle.get("xgb") or bundle.get("rf")

        if model is None:
            st.warning("No valid model found for SHAP")
            return

        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error("SHAP failed internally")
        st.text(str(e))

def risk_category(prob):
    percent = round(prob * 100, 2)
    if percent < 25:
        return percent, "Low", "#22C55E"
    elif percent < 50:
        return percent, "Moderate", "#F59E0B"
    elif percent < 75:
        return percent, "High", "#FB923C"
    else:
        return percent, "Very High", "#EF4444"

# =======================
# HERO SECTION
# =======================
st.markdown("""
<style>
.hero {
    text-align:center;
    background: linear-gradient(90deg, #2563EB, #06B6D4);
    border-radius:18px;
    padding:30px 20px;  /* 👈 reduced height */
    color:white;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class='hero'>
    <h1 style='font-size:38px;'>🧠 AI Health Intelligence System</h1>
    <p style='font-size:16px;'>Predict • Prevent • Improve Your Lifestyle</p>
</div>
""", unsafe_allow_html=True)

# =======================
# FORM SECTION
# =======================
import streamlit as st

# Initialize session state for empty fields
def init_state():
    fields = [
        "age","gender","height","weight",
        "smoke","alcohol","activity","walking","sleep","stress",
        "fruits","veggies","diet","calories","carb",
        "pollution","walk","food_access",
        "highbp","highchol","stroke","healthcheck"
    ]
    for f in fields:
        if f not in st.session_state:
            st.session_state[f] = None

init_state()

st.markdown("<div class='card'>", unsafe_allow_html=True)

with st.form("health_form"):

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["👤 Profile", "🏃 Lifestyle", "🥗 Diet", "🌍 Environment", "🏥 Health History"]
    )

    # ---------- PROFILE ----------
    with tab1:
        st.markdown("<div style='background: linear-gradient(90deg, #E0F7FA, #B2EBF2); padding:20px; border-radius:15px;'>", unsafe_allow_html=True)

        age = st.number_input("Age", 18, 90, value=st.session_state.age if st.session_state.age else 18)
        gender = st.radio("Gender", ["Male", "Female"], index=None, key="gender")

        height = st.number_input("Height (cm)", 140, 210, value=st.session_state.height if st.session_state.height else 140)
        weight = st.number_input("Weight (kg)", 40, 200, value=st.session_state.weight if st.session_state.weight else 40)

        bmi = None
        if height and weight:
            bmi = round(weight / ((height / 100) ** 2), 2)
            st.info(f"BMI: {bmi}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- LIFESTYLE ----------
    with tab2:
        st.markdown("<div style='background: linear-gradient(90deg, #FFF3E0, #FFE0B2); padding:20px; border-radius:15px;'>", unsafe_allow_html=True)

        smoke = st.radio("Do you smoke?", ["No", "Yes"], index=None, key="smoke")
        alcohol = st.radio("Alcohol Consumption", ["Never", "Occasional", "Frequent"], index=None, key="alcohol")
        activity = st.radio("Physical Activity", ["Sedentary", "Moderate", "Active"], index=None, key="activity")
        walking = st.radio("Daily Walking Level", ["Low", "Medium", "High"], index=None, key="walking")
        sleep = st.radio("Sleep Duration", ["<5", "5-6", "7-8", ">8"], index=None, key="sleep")
        stress = st.radio("Stress Level", ["Low", "Moderate", "High"], index=None, key="stress")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- DIET ----------
    with tab3:
        st.markdown("<div style='background: linear-gradient(90deg, #E8F5E9, #C8E6C9); padding:20px; border-radius:15px;'>", unsafe_allow_html=True)

        fruits = st.radio("Fruits Daily?", ["Yes", "No"], index=None, key="fruits")
        veggies = st.radio("Vegetables Daily?", ["Yes", "No"], index=None, key="veggies")
        diet = st.radio("Diet Type", ["Healthy", "Mixed", "Junk Heavy"], index=None, key="diet")
        calories = st.radio("Daily Calorie Range", ["<1800", "1800-2500", ">2500"], index=None, key="calories")
        carb = st.radio("Carbohydrate Intake", ["Low", "Moderate", "High"], index=None, key="carb")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- ENVIRONMENT ----------
    with tab4:
        st.markdown("<div style='background: linear-gradient(90deg, #FFFDE7, #FFF9C4); padding:20px; border-radius:15px;'>", unsafe_allow_html=True)

        pollution = st.radio("Air Quality", ["Clean", "Moderate", "Polluted"], index=None, key="pollution")
        walk = st.radio("Walk-Friendly Area?", ["Poor", "Average", "Good"], index=None, key="walk")
        food_access = st.radio("Healthy Food Access?", ["Yes", "No"], index=None, key="food_access")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- HEALTH HISTORY ----------
    with tab5:
        st.markdown("<div style='background: linear-gradient(90deg, #F3E5F5, #E1BEE7); padding:20px; border-radius:15px;'>", unsafe_allow_html=True)

        highbp = st.radio("High BP Diagnosed?", ["No", "Yes"], index=None, key="highbp")
        highchol = st.radio("High Cholesterol Diagnosed?", ["No", "Yes"], index=None, key="highchol")
        stroke = st.radio("History of Stroke?", ["No", "Yes"], index=None, key="stroke")
        healthcheck = st.radio("Regular Health Checkups?", ["No", "Yes"], index=None, key="healthcheck")

        st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍 Analyze My Health Risk")


# ================= VALIDATION =================
if submitted:

    missing = [k for k, v in st.session_state.items() if v is None]

    if missing:
        st.error("⚠️ Please fill all inputs before prediction!")
    else:
        st.success("✅ All inputs filled! Running prediction...")

    
# =======================
# AFTER SUBMISSION
# =======================
if submitted:
    # =======================
    # INPUT CONVERSION
    # =======================
    sex = 1 if gender == "Male" else 0
    smoker = 1 if smoke == "Yes" else 0
    alcohol_val = 2 if alcohol == "Frequent" else 1 if alcohol == "Occasional" else 0
    activity_val = 0 if activity == "Sedentary" else 1 if activity == "Moderate" else 2
    steps_val = 2000 if walking == "Low" else 6000 if walking == "Medium" else 10000
    sleep_val = 4 if sleep == "<5" else 5.5 if sleep == "5-6" else 7.5 if sleep == "7-8" else 8.5
    stress_val = {"Low": 1, "Moderate": 3, "High": 5}[stress]
    fruits_val = 1 if fruits == "Yes" else 0
    veggies_val = 1 if veggies == "Yes" else 0
    pollution_val = {"Clean": 20, "Moderate": 50, "Polluted": 90}[pollution]
    walk_val = {"Poor": 30, "Average": 60, "Good": 90}[walk]
    food_desert = 1 if food_access == "No" else 0
    highbp_val = 1 if highbp == "Yes" else 0
    highchol_val = 1 if highchol == "Yes" else 0
    stroke_val = 1 if stroke == "Yes" else 0

    # =======================
    # BUILD INPUTS
    # =======================
    heart_df = build_input(models["heart"], {
        "Smoking": smoker, "Diabetic": highbp_val, "PhysicalActivity": activity_val,
        "AlcoholDrinking": alcohol_val, "SleepTime": sleep_val, "BMI": bmi,
        "MentalHealth": stress_val, "Pollution_PM25": pollution_val,
        "Walkability": walk_val, "FoodDesertIndex": food_desert
    })

    diabetes_df = build_input(models["diabetes"], {
        "HighBP": highbp_val, "HighChol": highchol_val,
        "CholCheck": 1 if healthcheck == "Yes" else 0,
        "BMI": bmi, "Smoker": smoker, "Stroke": stroke_val,
        "PhysActivity": activity_val, "Fruits": fruits_val, "Veggies": veggies_val,
        "HvyAlcoholConsump": 1 if alcohol == "Frequent" else 0,
        "Age": age, "AvgSteps": steps_val,
        "DietCalories": 2000 if calories == "1800-2500" else 2500,
        "CarbScore": 1 if carb == "Low" else 2 if carb == "Moderate" else 3
    })

    obesity_df = build_input(models["obesity"], {
        "smoking": smoker,
        "diet_pattern": 1 if diet == "Healthy" else 2 if diet == "Mixed" else 3,
        "physical_activity": activity_val,
        "alcohol_use": alcohol_val,
        "sleep_hours": sleep_val,
        "stress_level": stress_val,
    })

    hypertension_df = build_input(models["hypertension"], {
        "age": age, "gender": sex, "BMI": bmi, "cholesterol": highchol_val,
        "gluc": highbp_val, "smoke": smoker, "alco": alcohol_val,
        "active": activity_val, "Sleep_Duration": sleep_val,
        "Pollution_PM25": pollution_val, "Walkability": walk_val,
        "AvgSteps": steps_val
    })

    with st.spinner("🧠 AI is analyzing your health patterns..."):
        import time
        time.sleep(1.5)
    # =======================
    # PREDICTIONS

    heart_prob = hybrid_predict(models["heart"], heart_df)
    diabetes_prob = hybrid_predict(models["diabetes"], diabetes_df)
    obesity_prob = hybrid_predict(models["obesity"], obesity_df)
    hyper_prob = hybrid_predict(models["hypertension"], hypertension_df)


    # =======================
    # RESULTS UI
    # =======================
    # ===============================
# COMBINED RESULTS
# ===============================
    st.markdown("<br><br>", unsafe_allow_html=True)

    diseases = {
    "❤️ Heart Disease": heart_prob,
    "🩸 Diabetes": diabetes_prob,
    "⚖️ Obesity": obesity_prob,
    "💓 Hypertension": hyper_prob
    }

    st.markdown("<div class='section-banner'>🏥 AI Health Risk Dashboard</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    for i, (name, prob) in enumerate(diseases.items()):
        percent, level, color = risk_category(prob)

        card = f"""
        <div class="card">
        <h3 style="text-align:center;">{name}</h3>

        <h1 style="text-align:center; color:{color}; font-size:40px;">
        {percent}%
        </h1>

        <div style='height:12px; background:#1e293b; border-radius:10px; overflow:hidden;'>
            <div style='width:{percent}%; height:12px; background:{color};
            box-shadow:0 0 15px {color};'></div>
        </div>

        <p style="text-align:center; margin-top:10px; font-size:18px;">
            <b>{level} Risk</b>
        </p>
        </div>
        """

        if i % 2 == 0:
            col1.markdown(card, unsafe_allow_html=True)
        else:
            col2.markdown(card, unsafe_allow_html=True)


    avg_risk = (heart_prob + diabetes_prob + obesity_prob + hyper_prob) / 4
    percent, level, color = risk_category(avg_risk)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div class='section-banner'>📊 Overall Health Score</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <h2 style="text-align:center;">Your Overall Risk</h2>
        <h1 style="text-align:center; color:{color}; font-size:45px;">{percent}%</h1>
        <p style="text-align:center; font-size:20px;"><b>{level}</b></p>
    </div>
    """, unsafe_allow_html=True)


    # =======================
    # SHAP (COMPACT 2 COLUMN)
    # =======================
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div class='section-banner'>🧠 AI Explainability</div>", unsafe_allow_html=True)

    def shap_plot(title, bundle, df):
        st.markdown(f"<h4 style='text-align:center;'>{title}</h4>", unsafe_allow_html=True)
        try:
            model = bundle.get("xgb") or bundle.get("rf")   # ✅ FIXED

            if model is None:
                st.warning("Not available")
                return

            explainer = shap.Explainer(model)
            vals = explainer(df)

            fig = plt.figure(figsize=(4,2.5))
            shap.plots.waterfall(vals[0], show=False)
            st.pyplot(fig)
            plt.close()

        except:
            st.warning("Not available")

    c1,c2 = st.columns(2)
    with c1:
        shap_plot("❤️ Heart Disease ",models["heart"],heart_df)
        shap_plot("⚖️ Obesity ",models["obesity"],obesity_df)
    with c2:
        shap_plot("🩸 Diabetes ",models["diabetes"],diabetes_df)
        shap_plot("💓 Hypertension ",models["hypertension"],hypertension_df)

    # =======================
    # SMART AI SUGGESTIONS
    # =======================
    # ================= SUGGESTIONS =================

    st.markdown("<br><br>", unsafe_allow_html=True)

# Create empty list FIRST
    suggestions = []

# Add conditions
    if bmi > 25:
        suggestions.append("⚖️ Reduce weight through balanced diet & exercise")

    if smoker:
        suggestions.append("🚭 Quit smoking to reduce heart risk")

    if activity_val == 0:
        suggestions.append("🏃 Increase physical activity (at least 30 min/day)")

    if sleep_val < 6:
        suggestions.append("😴 Improve sleep (7-8 hours recommended)")

    if alcohol_val == 2:
        suggestions.append("🍺 Reduce alcohol consumption")


# DISPLAY SECTION (after adding suggestions)
    if suggestions:
        st.markdown("<div class='section-banner'>💡 Personalized Suggestions</div>", unsafe_allow_html=True)

        for s in suggestions:
            st.markdown(f"""
            <div class='suggestion-box'>
                {s}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.success("✅ Your lifestyle looks good! Keep maintaining it.")

    # =======================
    # DIGITAL TWIN (COMPACT)
    # =======================
    # =======================
# DIGITAL TWIN 🔮 (GRID STYLE - 2 PER ROW)
# =======================

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div class='section-banner'>🔮 Digital Twin Simulation</div>", unsafe_allow_html=True)

    def simulation_chart(title, prob):
        factors = ["BMI", "Smoking", "Sleep", "Activity"]

        values = [
        bmi,
        smoker * 100,
        (8 - sleep_val) * 10,
        (2 - activity_val) * 50
        ]

    # Normalize values
        current = [prob * v / 100 for v in values]
        improved = [c * 0.7 for c in current]

        current = [c * 100 for c in current]
        improved = [i * 100 for i in improved]

        fig, ax = plt.subplots(figsize=(4, 3))

    # ===== CLEAN LOOK =====
        ax.set_facecolor('#f8fafc')
        fig.patch.set_facecolor('#f8fafc')

    # Remove grid & borders
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ===== MODERN LINE PLOT =====
        ax.plot(factors, current, marker='o', linewidth=2, label="Current")
        ax.plot(factors, improved, marker='o', linestyle='--', linewidth=2, label="Improved")

    # Fill between (digital twin feel)
        ax.fill_between(factors, current, improved, alpha=0.1)

    # ===== VALUE LABELS =====
        for i, v in enumerate(current):
            ax.text(i, v + 2, f"{int(v)}", fontsize=7, ha='center')

        for i, v in enumerate(improved):
            ax.text(i, v - 5, f"{int(v)}", fontsize=7, ha='center')

    # ===== AXIS SETTINGS =====
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=10, weight='bold')
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=7)

    # Legend (clean)
        ax.legend(fontsize=7, frameon=False)

        st.pyplot(fig)
        plt.close(fig)


# ===== GRID LAYOUT =====
    col1, col2 = st.columns(2)

    with col1:
        simulation_chart("❤️ Heart Disease", heart_prob)
        simulation_chart("⚖️ Obesity", obesity_prob)

    with col2:
        simulation_chart("🩸 Diabetes", diabetes_prob)
        simulation_chart("💓 Hypertension", hyper_prob)
