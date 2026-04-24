 Predicting Cardiovascular Risk Using the Framingham Heart Study


---

## 1. Introduction

Cardiovascular disease (CVD) remains a leading cause of mortality worldwide, with early identification of high-risk individuals being essential for prevention and intervention. The **Framingham Heart Study** provides a rich, longitudinal dataset of anonymised patient information, including lifestyle factors, clinical measurements, and health outcomes, that can be leveraged for cardiovascular risk prediction.  

Machine learning (ML) offers powerful tools to identify patterns and predict outcomes from complex datasets. This project aims to combine statistical analysis and ML techniques to create a predictive model for cardiovascular risk that can be integrated into a user-friendly web interface for educational purposes.

---

## 2. Aim and Objectives

**Aim:**  
To develop and evaluate machine learning models for cardiovascular risk prediction using the Framingham Heart Study dataset and deploy a web-based interface demonstrating the model’s functionality.  

**Objectives:**  
1. Perform exploratory data analysis on the Framingham dataset to identify relevant features.  
2. Preprocess data for ML modeling, including handling missing values, normalization, and feature selection.  
3. Train and evaluate multiple ML models, including Logistic Regression, Random Forest, XGBoost, and Artificial Neural Networks (ANN) and anything else to predict cardiovascular risk.  
4. Develop a responsive website where users can input anonymised patient information to see simulated risk predictions.  
5. Document model performance using accuracy, precision, recall, F1-score, and ROC-AUC metrics.  

---

## 3. Methodology

1. **Data Acquisition:**  
   Access anonymised Framingham Heart Study data following all ethical guidelines. Only relevant features for cardiovascular risk prediction (e.g., age, cholesterol, blood pressure, smoking status) will be used.

2. **Data Preprocessing:**  
   - Handle missing values using imputation techniques.  
   - Normalize continuous variables and encode categorical variables.  
   - Split data into training and testing sets.  

3. **Modeling:**  
   - Implement Logistic Regression, Random Forest, XGBoost, ANN models, ect.
   - Evaluate models using cross-validation and performance metrics.  
   - Select the best-performing model based on predictive accuracy and interpretability.

4. **Web Interface Development:**  
   - Build a simple web application using Python, html and css or JavaScript frameworks.  
   - Users can enter sample anonymised information to simulate cardiovascular risk prediction.  
   - The website will include disclaimers stating the system is for educational purposes only and not a medical diagnostic tool.

5. **Evaluation and Documentation:**  
   - Compare model performance with standard metrics.  
   - Document limitations, ethical considerations, and potential applications in research and education.

**Software/Tools:**  
- Python: Pandas, NumPy, Scikit-learn, TensorFlow/Keras  ect
- Web development: Flask or Django, HTML/CSS/JavaScript  ect


---

## 4. Expected Outcomes

1. A set of trained and validated ML models for cardiovascular risk prediction.  
2. Performance evaluation of each model with clear reporting of metrics.  
3. A web-based educational tool demonstrating how the models predict risk based on user input.  
4. A detailed  documenting methodology, results, and ethical considerations.

---

## 5. Ethical Considerations

- The dataset is fully anonymised; no identifiable personal information will be used.  
- The project will be strictly educational, with no clinical or commercial use.  
- Disclaimers will clarify that predictions are simulated and not a medical diagnosis.  
- Compliance with data governance and ethical guidelines for secondary research is ensured.  

---




---

## 7. References

- Framingham Heart Study. Available at: https://framinghamheartstudy.org/  
