"""About page (project background and live model metrics)."""

import pandas as pd
import streamlit as st

from inference import get_model_info, load_models

st.title("About this project")


@st.cache_resource
def _models_and_metrics():
    return load_models()


with st.container(border=True):
    st.subheader("What this is")
    st.markdown(
        "An educational tool that predicts 10-year cardiovascular disease (CHD) risk "
        "using machine learning models trained on the Framingham Heart Study dataset. "
        "The Framingham study began in 1948 and is the source of most of the "
        "risk-factor knowledge we have for CHD today.\n\n"
        "**Educational only.** Not for clinical decisions. See the Disclaimer page."
    )

with st.container(border=True):
    st.subheader("Dataset")
    st.markdown("""
- 4,240 participants, 15 clinical / demographic variables
- Outcome: 10-year CHD occurrence (binary)
- Class split: 84.8% no-CHD, 15.2% CHD — significant imbalance
- 80/20 stratified train/test split (3,392 train / 848 test)
""")

with st.container(border=True):
    st.subheader("Models")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Reference**")
        st.markdown("""
- Logistic Regression (linear baseline)
- Random Forest (bagged trees)
- Gradient Boosting (sequential trees)
- XGBoost (with `scale_pos_weight`)
- SVM (RBF kernel)
- 3-layer MLP (neural network)

All trained with cost-sensitive class weighting (the SMOTE alternative was tried and
didn't help on this dataset).
""")
    with col2:
        st.markdown("**Custom contribution**")
        st.markdown("""
**Clinically-Informed Stacking Ensemble (CISE).** A custom stacking architecture: the
Logistic Regression meta-learner sees not only the three base learner predictions
(RF, GB, XGBoost) but also five clinically-validated Framingham risk factors — age,
systolic BP, total cholesterol, smoker status, diabetes. The eight resulting
coefficients are directly inspectable.
""")

with st.container(border=True):
    st.subheader("Live performance")
    st.caption(
        "Loaded from the most recent training run (`results/model_results.csv`). "
        "The primary risk number on the Assessment page comes from whichever model "
        "has the highest ROC-AUC."
    )

    models, metrics = _models_and_metrics()
    if not metrics:
        st.warning("No model metrics found. Run the training pipeline first.")
    else:
        info = get_model_info(models, metrics)
        sorted_info = sorted(info, key=lambda m: m["auc"], reverse=True)
        rows = [
            {
                "Model": m["name"],
                "ROC-AUC": f"{m['auc'] * 100:.1f}%",
                "Recall": f"{m['recall'] * 100:.1f}%",
                "Precision": f"{m['precision'] * 100:.1f}%",
                "F1": f"{m['f1'] * 100:.1f}%",
            }
            for m in sorted_info
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        cise = next((m for m in info if m.get("meta_coefficients")), None)
        if cise:
            st.markdown("**CISE meta-learner coefficients**")
            st.caption(
                "The first three weights say how much the meta-learner trusts "
                "each base learner. The last five say how much residual weight "
                "it gives the clinical features."
            )
            coef_rows = []
            for c in cise["meta_coefficients"]:
                is_base = c["name"] in {"rf", "gb", "xgb"}
                coef_rows.append(
                    {
                        "Feature": c["name"],
                        "Weight": c["weight"],
                        "Source": "Base learner" if is_base else "Clinical feature",
                    }
                )
            st.dataframe(pd.DataFrame(coef_rows), width="stretch", hide_index=True)

with st.container(border=True):
    st.subheader("How it was built")
    st.markdown("""
- Median-imputed missing values, StandardScaler, one-hot encoding for education
- Stratified 5-fold cross-validation
- Cost-sensitive class weighting (5.59:1) instead of SMOTE
- Out-of-fold base predictions feed CISE's meta-learner (no leakage)
- Same held-out 848-sample test set used for every model
""")

st.info(
    "**Source:** Framingham Heart Study, "
    "[www.framinghamheartstudy.org](https://www.framinghamheartstudy.org)"
)
