# Task4-AI-ML
# Logistic Regression Classifier - Breast Cancer Diagnosis

#Objective
To Build a binary classification model using Logistic Regression to predict whether a tumor is malignant or benign based on the Breast Cancer Wisconsin Dataset.

#Dataset
- Source: [Breast Cancer Wisconsin Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- Target variable: `diagnosis` (M = malignant, B = benign)
- Preprocessing included:
  - Dropping unnecessary columns (`id`, `Unnamed: 32`)
  - Label encoding diagnosis: Malignant → 1, Benign → 0
  - Feature scaling using `StandardScaler`

#Tools Used
- Python
- Scikit-learn (modeling & evaluation)
- Pandas (data manipulation)
- Matplotlib (visualizations)

#Model: Logistic Regression
Logistic regression uses the sigmoid function to estimate the probability that a sample belongs to the positive class:
We trained the model using `LogisticRegression()` from scikit-learn, and evaluated on test data using metrics like precision, recall, F1-score, and ROC-AUC.

# Evaluation Metrics
#Confusion Matrix (Threshold = 0.5)
- Accuracy: 97%
- Precision (Malignant): 98%
- Recall (Malignant): 95%
- F1-Score: 96%

#ROC-AUC Score:
- 0.997 → Excellent discriminatory power

#ROC Curve:
Plotted using predicted probabilities to visualize TPR vs. FPR.

#Threshold Tuning & Impact
- Lower threshold (0.3) favors recall→ better for catching all positive cases.
- Higher threshold (0.7) favors precision → better when false positives are more costly.
