"""Legal disclaimer and limitations."""

import streamlit as st

st.title("Legal Disclaimer & Important Notice")

st.error(
    "**This tool is strictly for educational and research purposes only.** "
    "It is NOT intended for clinical diagnosis, treatment, or "
    "clinical decision making."
)

with st.container(border=True):
    st.subheader("NOT Medical Advice")
    st.markdown("""
The predictions provided by this tool are simulated estimates based on machine
learning models trained on the Framingham Heart Study dataset. They are **not**:

- Medical diagnoses
- Clinical recommendations
- Substitutes for professional medical evaluation
- Approved for clinical use or medical decision-making
- Validated for individual patient risk assessment

**If you have cardiovascular concerns, please consult a qualified healthcare
professional who can conduct a proper clinical evaluation.**
""")

with st.container(border=True):
    st.subheader("Limitations")
    st.markdown("**Model Limitations**")
    st.markdown("""
- **Fixed Features:** Model trained on specific Framingham dataset features;
  may not apply to all populations
- **Temporal Gap:** Framingham data is historical; modern risk factors may differ
- **Population-Specific:** Primarily derived from North American cohort;
  generalizability uncertain
- **Predictive Performance:** ROC-AUC ~0.70 indicates moderate accuracy
- **Missing Information:** Many clinical variables not captured in this tool
- **Class Imbalance:** Baseline CHD prevalence (15%) may differ in clinical settings
""")
    st.markdown("**User Input Limitations**")
    st.markdown("""
- Tool cannot verify accuracy of self-reported medical information
- Single-point-in-time assessment; risk evolves over time
- Does not account for recent medical treatment changes
- Cannot capture qualitative clinical judgment
""")

with st.container(border=True):
    st.subheader("When to Seek Medical Attention")
    st.markdown("""
**Consult a healthcare provider if you experience:**

- Chest pain or pressure
- Shortness of breath
- Rapid or irregular heartbeat
- Fainting or severe dizziness
- Persistent fatigue or weakness
- New or worsening symptoms

**Emergency:** Call emergency services (911 in US) for severe chest pain or
other signs of heart attack.
""")

st.caption("For Educational Purposes Only*")
