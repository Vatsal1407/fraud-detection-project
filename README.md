# Bank Transaction Fraud Detection

A comprehensive machine learning project implementing fraud detection using Stratified K-Fold Cross-Validation, with systematic analysis of class imbalance handling techniques.

## 🎯 Project Overview

This project demonstrates end-to-end machine learning workflow for fraud detection, including:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Multiple model comparison (Logistic Regression, Random Forest, XGBoost)
- Class imbalance handling (SMOTE, class weights, threshold optimization)
- Systematic debugging and model evaluation

**Dataset:** 200,000 bank transactions with 5.04% fraud rate

## 📊 Key Results

| Model | Strategy | Threshold | Recall | Precision | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|-----------|----------|---------|
| Logistic Regression | Class Weights | Default | 48.13% | 5.00% | 9.05% | 0.498 |
| Random Forest | SMOTE | 0.1 | 69.08% | 5.05% | 9.41% | 0.498 |
| XGBoost | SMOTE | 0.1 | 5.58% | 5.13% | 5.30% | 0.499 |

**Final Model:** Random Forest with threshold 0.1
- **Recall:** 69% (catches 7 out of 10 fraud cases)
- **Precision:** 5% (95% false positive rate)
- **ROC-AUC:** 0.498 (indicates dataset limitations)

## 🔍 Key Findings

### Data Quality Issue
- All models achieved **ROC-AUC ≈ 0.50** (random guessing level)
- Dataset features cannot effectively distinguish fraud from non-fraud
- Confirmed through probability distribution analysis: fraud and non-fraud distributions overlap completely
- Maximum fraud probability: 0.37 (never reaches decision threshold of 0.5)

### Methodology Success
Despite poor model performance, the project demonstrates:
- ✅ Proper Stratified K-Fold Cross-Validation implementation
- ✅ Comprehensive testing of class imbalance strategies
- ✅ Systematic threshold optimization
- ✅ Thorough debugging and root cause analysis
- ✅ Professional model evaluation and documentation

## 📁 Project Structure
```
fraud_detection_project/
├── data/
│   ├── Bank_Transaction_Fraud_Detection.csv
│   └── preprocessed/
│       ├── X_train_full.npy
│       ├── X_test.npy
│       ├── y_train_full.npy
│       ├── y_test.npy
│       ├── scaler.pkl
│       ├── feature_names.txt
│       └── preprocessing_params.pkl
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb          # Feature Engineering & Preprocessing
│   └── 03_modeling.ipynb               # Model Training & Evaluation
├── models/
│   ├── random_forest_fraud_detector.pkl
│   ├── scaler_final.pkl
│   ├── model_metadata.pkl
│   └── model_evaluation.png
├── requirements.txt
└── README.md
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip package manager

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/fraud-detection-project.git
cd fraud-detection-project
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 📦 Dependencies
```txt
pandas==2.1.4
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
scikit-learn==1.3.2
xgboost==2.0.3
imbalanced-learn==0.11.0
jupyter==1.0.0
joblib==1.3.2
```

## 🚀 Usage

### 1. Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_eda.ipynb
```
Explore the dataset, visualize distributions, and identify patterns.

### 2. Preprocessing
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```
Feature engineering, encoding, scaling, and train-test split.

### 3. Model Training
```bash
jupyter notebook notebooks/03_modeling.ipynb
```
Train models using Stratified K-Fold CV and evaluate performance.

### 4. Load and Use Model
```python
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('models/random_forest_fraud_detector.pkl')
scaler = joblib.load('models/scaler_final.pkl')

# Prepare your data (must have 54 features in correct order)
# Scale features
X_scaled = scaler.transform(X)

# Predict probabilities
fraud_probability = model.predict_proba(X_scaled)[:, 1]

# Apply threshold
threshold = 0.1
predictions = (fraud_probability >= threshold).astype(int)
```

## 🔬 Methodology

### Data Preprocessing
1. **Feature Engineering:**
   - Time-based features (Hour, Day, Month, Is_Weekend, Is_Night, Is_Business_Hours)
   - Transaction features (Transaction_to_Balance_Ratio, Is_High_Value, Is_Low_Balance)
   - Frequency encoding for high-cardinality features (State, City, Transaction_Description)

2. **Encoding:**
   - One-hot encoding for categorical features (7 features → 38 encoded features)
   - Final feature count: 54 features

3. **Train-Test Split:**
   - 80% training (160,000 samples) for K-Fold CV
   - 20% hold-out test set (40,000 samples) for final evaluation
   - Stratified split to maintain 5.04% fraud rate

### Model Training Strategy

**Stratified K-Fold Cross-Validation (5 folds):**
- Each fold maintains ~5% fraud rate
- Training: 128,000 samples per fold
- Validation: 32,000 samples per fold
- SMOTE applied inside each fold (prevents data leakage)
- Features scaled inside each fold (prevents data leakage)

**Class Imbalance Handling:**
- Tested SMOTE (sampling_strategy=0.5)
- Tested class_weight='balanced'
- Tested threshold optimization (0.01, 0.1, 0.3, 0.5)

### Models Tested
1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble of 100 decision trees
3. **XGBoost** - Gradient boosting model

## 📈 Detailed Results

### Cross-Validation Results (5-Fold)

**Logistic Regression (Class Weights):**
- Training time: 2.17 seconds
- Avg Recall: 48.13% ± 0.60%
- Avg F1-Score: 9.05% ± 0.08%
- Avg ROC-AUC: 0.498 ± 0.003

**Random Forest (Threshold 0.1):**
- Training time: 70 seconds
- Avg Recall: ~50%
- Avg F1-Score: ~10%
- Avg ROC-AUC: 0.500 ± 0.009

**XGBoost (Threshold 0.1):**
- Training time: 48.70 seconds
- Avg Recall: 5.58% ± 0.76%
- Avg F1-Score: 5.30% ± 0.25%
- Avg ROC-AUC: 0.499 ± 0.005

### Hold-out Test Set (Final Model)

**Random Forest (Threshold 0.1):**
```
Confusion Matrix:
[[11753 26229]  ← 11,753 correct non-fraud, 26,229 false positives
 [  624  1394]] ← 624 missed fraud, 1,394 caught fraud

Classification Report:
              precision    recall  f1-score   support
   Non-Fraud       0.95      0.31      0.47     37982
       Fraud       0.05      0.69      0.09      2018
    accuracy                           0.33     40000
```

## ⚠️ Limitations & Recommendations

### Dataset Limitations
- **ROC-AUC ≈ 0.50:** Model cannot distinguish fraud from non-fraud better than random chance
- **Weak features:** EDA showed all features have ~5% fraud rate (no discrimination)
- **Probability overlap:** Fraud and non-fraud probability distributions are identical
- **Likely synthetic data:** Real fraud datasets typically have clearer patterns

### Model Limitations
- **High false positive rate (95%):** Out of 100 fraud predictions, only 5 are actual fraud
- **Not suitable for automatic fraud blocking:** Would block too many legitimate transactions
- **Suitable for flagging only:** Can be used to flag transactions for manual review

### Recommendations
1. **Improve data quality:**
   - Collect additional features (IP address, device fingerprint, transaction velocity)
   - Include external data sources (geolocation, merchant reputation)
   - Engineer domain-specific features with fraud expert input

2. **Alternative approaches:**
   - Anomaly detection (Isolation Forest, One-Class SVM)
   - Deep learning with sequential data (LSTM for transaction history)
   - Graph-based methods (detect fraud networks)

3. **Ensemble with rules:**
   - Combine ML model with rule-based system
   - Use expert-defined suspicious patterns

## 💡 Key Learnings

1. **Proper methodology doesn't guarantee good results** - Data quality is paramount
2. **ROC-AUC is critical** - Don't rely solely on accuracy with imbalanced data
3. **Understanding failure is valuable** - Systematic debugging reveals root causes
4. **K-Fold CV prevents overfitting claims** - Robust evaluation across multiple folds
5. **Feature engineering matters** - Weak features limit all models equally

## 🎓 Skills Demonstrated

- ✅ End-to-end ML project workflow
- ✅ Stratified K-Fold Cross-Validation
- ✅ Class imbalance handling (SMOTE, class weights, threshold tuning)
- ✅ Multiple algorithm comparison
- ✅ Systematic debugging and root cause analysis
- ✅ Model evaluation and interpretation
- ✅ Professional documentation
- ✅ Git version control

## 📚 References

1. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.

## 📧 Contact

**Author:** [Your Name]
**Email:** your.email@example.com
**LinkedIn:** [Your LinkedIn Profile]
**GitHub:** [Your GitHub Profile]

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dataset: [Kaggle - Bank Transaction Fraud Detection](https://www.kaggle.com/datasets/marusagar/bank-transaction-fraud-detection)
- Scikit-learn documentation
- Imbalanced-learn documentation

---

**Note:** This project demonstrates proper ML methodology and systematic analysis. While model performance is limited by dataset quality (ROC-AUC ≈ 0.50), the project showcases professional ML practices, thorough evaluation, and honest assessment of limitations - all critical skills for real-world data science.