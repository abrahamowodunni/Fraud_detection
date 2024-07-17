# ![Fraud Detection](https://i.pinimg.com/564x/de/9c/f3/de9cf3532a1811d8818c59cd67fd6a63.jpg) Fraud Detection Project 🕵️‍♂️💳

## Overview
This project aims to detect fraud in financial transactions using machine learning techniques to enhance security and reduce risks. The project covers fraud analytics, credit risk analysis, market services, and consumer services.

## Beneficial Analysis for Experian 📊
Fraud detection at Experian involves analyzing transaction patterns and customer behaviors to identify anomalies indicative of fraudulent activities.

## Research Links 🔍
- [Experian Insights on Fraud Detection and Machine Learning](https://www.experian.com/blogs/insights/fraud-detection-and-machine-learning/)
- [AWS Fraud Detector Samples](https://github.com/aws-samples/aws-fraud-detector-samples)
- [Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation)
- [Experian Advanced Analytics with ML & AI](https://www.experian.com/business/solutions/advanced-analytics/machine-learning-ai-analytics)
- [Experian Machine Learning in Business Information](https://www.experian.com/business-information/landing/machine-learning)

## Feature Engineering 🛠️
- **Data Preparation:** Removed unnecessary features such as `Trans_date_trans_time`, `Cc_num`, `First`, `Last`, `Gender`, `Street`, `State`, `City`, `Zipcode`, `Trans_num`, and converted `DoB` into categorical age groups.
- **Data Scaling:** Applied label encoding and scaling to numerical data for model compatibility.

## Choice of Model 🤖
To address fraud detection efficiently:
- **Criteria:** Prioritized models based on speed, interpretability, and appropriate metrics (accuracy, F1 score, AUC).
- **Metric Emphasis:** Focused on recall and precision to ensure comprehensive fraud detection without disrupting legitimate transactions.

## Customer Perspective 🧑‍💼
- **Trust and Security:** Ensured high recall to detect most fraud cases.
- **Minimal Disruption:** Maintained high precision to minimize false positives and customer inconvenience.
- **Balanced Approach:** Aimed for a balanced F1 score to foster customer loyalty and confidence.

## Model Research and Selection 🔍
- **Initial Models:** Evaluated models such as Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, AdaBoost, Support Vector Machine, XGBoost, and CatBoost.
- **Model Criteria:** Speed, interpretability, and performance metrics.
- **Final Selection:** Chose XGBoost for its superior performance in F1 score and recall after fine-tuning and using SMOTE for balancing.

## Model Fine-Tuning 🔧
- **Challenges:** Faced GPU limitations for GridSearchCV and Google Colab timeouts.
- **Approach:** Manually tuned parameters (learning_rate, n_estimators, max_depth) to achieve optimal results.
- **Threshold Adjustment:** Lowered the decision threshold to improve recall, achieving an F1 score improvement from 83 to 88 while maintaining high precision.

## Integration for Fraud Detection 🌐
- **API Development:** Built a Flask API to integrate the fraud detection model into existing systems.
- **Integration Points:** Facilitated real-time fraud detection and decision-making capabilities.

## Recommendations 🌟
- **Data Enhancement:** Gather more transaction data and include additional features like transaction descriptions and geolocation patterns.
- **Advanced Techniques:** Explore anomaly detection and clustering for enhanced fraud detection.
- **Additional Information:** Incorporate details about stolen cards and historical transaction behavior for more comprehensive analysis.

## Code Structure 📦
- **Version Control:** Used Git and GitHub Actions for version control and CI/CD pipelines.
- **Virtual Environment:** Employed virtual environments for package management and dependency isolation.
- **Project Template:** Streamlined project setup with templates for initial files and folders.
- **Setup.py:** Managed project metadata, dependencies, packaging, and installation processes.
