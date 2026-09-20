"""
Streamlit web app for cardiovascular risk prediction.
"""

import logging

import pandas as pd
import streamlit as st

from inference import (
    build_prediction_result,
    format_download_text,
    load_models,
    load_preprocessor,
)

logging.basicConfig(level=logging.INFO)

CONTRIBUTION_DESCRIPTIONS = {
    "rf": (
        "A forest of 200 randomised decision trees voting on CHD risk."
    ),
    "gb": (
        "Sequentially constructed decision trees, each one correcting the errors "
        "of the previous ones."
    ),
    "xgb": (
        "An optimised gradient-boosting library with built-in regularisation."
    ),
    "age": "The single strongest CHD risk factor in the Framingham literature.",
    "sysBP": "Systolic blood pressure (mmHg). ≥140 is hypertensive.",
    "totChol": "Total cholesterol (mg/dL). >200 is elevated.",
    "currentSmoker": (
        "Active smoking status. One of the strongest modifiable CHD risk factors."
    ),
    "diabetes": "Diabetes diagnosis or treatment. Roughly 2–4× CHD risk.",
}

NEXT_STEPS = {
    "Low": {
        "header": "Keep Up the Good Work!",
        "steps": [
            (
                "1. Maintain Current Lifestyle",
                "Continue with your healthy habits. Regular exercise, balanced "
                "diet, and stress management are key.",
            ),
            (
                "2. Regular Health Checkups",
                "Schedule annual check-ups with your healthcare provider to "
                "monitor your cardiovascular health.",
            ),
            (
                "3. Monitor Key Metrics",
                "Keep track of blood pressure, cholesterol levels, and weight "
                "to catch any changes early.",
            ),
            (
                "4. Healthy Lifestyle Tips",
                "• Stay physically active (150 min moderate exercise/week)\n"
                "• Eat a heart-healthy diet rich in fruits and vegetables\n"
                "• Maintain a healthy weight\n"
                "• Don't smoke and avoid secondhand smoke",
            ),
        ],
    },
    "Moderate": {
        "header": "Take Action Now",
        "steps": [
            (
                "1. Schedule a Doctor's Appointment",
                "Book an appointment with your primary care physician or "
                "cardiologist to discuss your results.",
            ),
            (
                "2. Lifestyle Modifications",
                "• Increase physical activity to 150+ minutes per week\n"
                "• Adopt a Mediterranean or DASH diet\n"
                "• Reduce sodium intake\n"
                "• Manage stress through meditation or yoga",
            ),
            (
                "3. Review Medications",
                "Your doctor may recommend medications for blood pressure, "
                "cholesterol, or other risk factors.",
            ),
            (
                "4. Quit Smoking",
                "If you smoke, quitting is one of the most important steps "
                "to reduce your risk.",
            ),
            (
                "5. Monitor & Follow-up",
                "Plan to reassess your risk every 6-12 months and track your "
                "progress.",
            ),
        ],
    },
    "High": {
        "header": "Seek Medical Attention",
        "steps": [
            (
                "1. Contact Your Doctor Immediately",
                "Schedule an urgent appointment with your healthcare provider "
                "or cardiologist to discuss these results.",
            ),
            (
                "2. Comprehensive Medical Evaluation",
                "Your doctor may recommend additional tests such as EKG, "
                "echocardiogram, or stress tests.",
            ),
            (
                "3. Medication Management",
                "Your doctor will likely prescribe medications to manage blood "
                "pressure, cholesterol, and other risk factors.",
            ),
            (
                "4. Aggressive Lifestyle Changes",
                "• Increase physical activity with medical supervision\n"
                "• Strict diet changes (consult with nutritionist)\n"
                "• Reduce stress significantly\n"
                "• Completely eliminate smoking and alcohol",
            ),
            (
                "5. Consider Preventive Procedures",
                "Your cardiologist may recommend coronary calcium scoring, "
                "carotid ultrasound, or other diagnostic procedures.",
            ),
            (
                "6. Frequent Monitoring",
                "Plan follow-up appointments every 3-6 months to assess "
                "progress and adjust treatment.",
            ),
        ],
    },
}


@st.cache_resource
def get_preprocessor():
    return load_preprocessor()


@st.cache_resource
def get_models_and_metrics():
    return load_models()


def render_next_steps(risk_level):
    block = NEXT_STEPS.get(risk_level, NEXT_STEPS["Moderate"])
    st.markdown(f"**{block['header']}**")
    for title, body in block["steps"]:
        st.markdown(f"**{title}**")
        st.markdown(body)


def render_explanation(explanation):
    if not explanation or not explanation.get("contributions"):
        return

    title = (
        f"Why this prediction? ({explanation['model']}, "
        f"predicted {explanation['probability']}%)"
    )
    with st.container(border=True):
        st.subheader(title)
        st.caption(
            "The CISE meta-learner is a logistic regression with eight inputs. "
            "Each input contributes additively in logit space."
        )

        rows = []
        for c in explanation["contributions"]:
            kind_label = (
                "Base learner" if c["kind"] == "base_learner" else "Clinical feature"
            )
            sign = "+" if c["delta_pp"] >= 0 else ""
            rows.append(
                {
                    "Contributor": c["label"],
                    "Type": kind_label,
                    "Effect on Risk": f"{sign}{c['delta_pp']:.1f} pp",
                    "Description": CONTRIBUTION_DESCRIPTIONS.get(c["name"], ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            '**Reading the numbers:** "+8 pp" means this contributor added 8 '
            "percentage points to the final risk score relative to a patient with "
            "that value at the training-set average."
        )


def render_results(result):
    primary = result["primary_prediction"]
    prob = primary["probability"]
    level = primary["risk_level"]

    st.header("Results")

    with st.container(border=True):
        st.subheader("Your 10-Year Risk")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Risk", f"{prob}%", help=f"Primary model: {primary['model']}")
        with col2:
            st.progress(min(prob / 100.0, 1.0))
            if level == "Low":
                st.success(f"**{level}** risk")
            elif level == "Moderate":
                st.warning(f"**{level}** risk")
            else:
                st.error(f"**{level}** risk")

    with st.container(border=True):
        st.subheader("Model Predictions Comparison")
        chart_df = pd.DataFrame(result["all_predictions"])[["model", "probability"]]
        chart_df = chart_df.set_index("model")
        st.bar_chart(chart_df, height=350)

    render_explanation(result.get("explanation"))

    with st.container(border=True):
        st.subheader("Recommended Next Steps")
        render_next_steps(level)

    with st.container(border=True):
        st.subheader("Detailed Model Results")
        table_rows = []
        for pred in result["all_predictions"]:
            auc_pct = f"{pred['auc'] * 100:.1f}%"
            table_rows.append(
                {
                    "Model": pred["model"],
                    "Risk Probability": f"{pred['probability']}%",
                    "Risk Level": pred["risk_level"],
                    "ROC-AUC Score": auc_pct,
                }
            )
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    st.warning(
        "This assessment tool is for **educational and informational purposes only**. "
        "It should **NOT** be used for medical diagnosis or to replace professional "
        "medical advice. Please consult with a qualified healthcare provider."
    )

    st.download_button(
        "Download Results",
        data=format_download_text(result),
        file_name="cardiovascular_risk_assessment.txt",
        mime="text/plain",
    )


def assessment_page():
    st.title("Cardiovascular Risk Assessment")
    st.markdown(
        "Enter clinical values to estimate the 10-year coronary heart disease risk."
    )

    scaler, imputer, feature_names = get_preprocessor()
    models, metrics = get_models_and_metrics()

    if not models or scaler is None:
        st.error(
            "Models or preprocessor not loaded. Run `python -m src.run_pipeline` "
            "from the project root first."
        )
        return

    st.info(
        "Numeric inputs are constrained to physiologically realistic ranges that "
        "match the training data. The models were trained on the Framingham cohort "
        "and are unreliable on values outside these ranges."
    )

    # Not wrapped in st.form: widgets inside forms do not rerun until submit, so
    # Current Smoker could never re-enable Cigarettes per Day.
    with st.container(border=True):
        st.subheader("Personal Information")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input(
                "Age (years)",
                min_value=20,
                max_value=100,
                value=55,
                help="Training data: 32–70 years.",
            )
        with c2:
            male = st.selectbox("Gender", ["Female", "Male"])

    with st.container(border=True):
        st.subheader("Blood Pressure")
        c3, c4 = st.columns(2)
        with c3:
            sys_bp = st.number_input(
                "Systolic BP (mmHg)", min_value=80, max_value=220, value=130
            )
        with c4:
            dia_bp = st.number_input(
                "Diastolic BP (mmHg)", min_value=40, max_value=130, value=80
            )

    with st.container(border=True):
        st.subheader("Cholesterol & Glucose")
        c5, c6 = st.columns(2)
        with c5:
            tot_chol = st.number_input(
                "Total Cholesterol (mg/dL)",
                min_value=100,
                max_value=400,
                value=240,
            )
        with c6:
            glucose = st.number_input(
                "Glucose (mg/dL)", min_value=50, max_value=400, value=100
            )
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=14.0,
            max_value=55.0,
            value=25.0,
            step=0.1,
        )

    with st.container(border=True):
        st.subheader("Lifestyle")
        c7, c8 = st.columns(2)
        with c7:
            heart_rate = st.number_input(
                "Resting Heart Rate (bpm)", min_value=40, max_value=150, value=70
            )
        with c8:
            current_smoker = st.selectbox("Current Smoker", ["No", "Yes"])
            cigs_per_day = st.number_input(
                "Cigarettes per Day",
                min_value=0,
                max_value=60,
                value=0,
                disabled=(current_smoker == "No"),
            )

    with st.container(border=True):
        st.subheader("Medical History")
        c9, c10 = st.columns(2)
        with c9:
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            bp_meds = st.selectbox("Blood Pressure Medication", ["No", "Yes"])
        with c10:
            prevalent_hyp = st.selectbox("Hypertension", ["No", "Yes"])
            prevalent_stroke = st.selectbox("History of Stroke", ["No", "Yes"])

        education = st.selectbox(
            "Education Level",
            [
                "Some High School",
                "High School / GED",
                "Some College",
                "College or Higher",
            ],
        )

    submitted = st.button("Calculate Risk", type="primary")

    if submitted:
        if current_smoker == "No":
            cigs_per_day = 0

        req = {
            "male": 1 if male == "Male" else 0,
            "age": age,
            "education": [
                "Some High School",
                "High School / GED",
                "Some College",
                "College or Higher",
            ].index(education)
            + 1,
            "currentSmoker": 1 if current_smoker == "Yes" else 0,
            "cigsPerDay": cigs_per_day,
            "BPMeds": 1 if bp_meds == "Yes" else 0,
            "prevalentStroke": 1 if prevalent_stroke == "Yes" else 0,
            "prevalentHyp": 1 if prevalent_hyp == "Yes" else 0,
            "diabetes": 1 if diabetes == "Yes" else 0,
            "totChol": tot_chol,
            "sysBP": sys_bp,
            "diaBP": dia_bp,
            "BMI": bmi,
            "heartRate": heart_rate,
            "glucose": glucose,
        }

        result = build_prediction_result(
            req, models, metrics, scaler, imputer, feature_names
        )
        render_results(result)


st.set_page_config(
    page_title="CVD Risk Predictor",
    page_icon="❤️",
    layout="wide",
)

pg = st.navigation(
    [
        st.Page(assessment_page, title="Assessment", default=True),
        st.Page("pages/about.py", title="About"),
        st.Page("pages/disclaimer.py", title="Disclaimer"),
    ]
)
pg.run()
