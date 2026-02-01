# 📉 Customer Churn Prediction – End-to-End Machine Learning Pipeline 🧠

Customer churn is a critical business problem in subscription-based industries.
Acquiring new customers is significantly more expensive than retaining existing ones, so early identification of churn risk is essential.

This project builds an end-to-end machine learning pipeline to predict customer churn using behavioral, contract, and service usage data.

The goal is not only high accuracy, but high recall and probability-based risk scoring, enabling proactive customer retention strategies.

# 🎯 Objectives
* Predict which customers are likely to churn.

* Maximize recall (catch as many churners as possible), while trying to achieve a reasonable precision.

* Produce probabilities, not just hard classifications

* Build a robust, production-ready pipeline.

* Compare multiple models and evaluate stacking performance

# Tools & Technologies 🧰

* Python 🐍
  - scikit-learn 🧠
  - imblearn ⚖️
  - XGBoost ⚡
  - pandas 🐼  / numpy 0️⃣1️⃣
  - matplotlib / seaborn 📈

# Methodology 📓

1. EDA 🔎
2. Data Cleaning 🧹
3. Feature Engineering 🔧
4. Handling Class Imbalance ⚖️
5. Model Selection and Hyperparameter Tuning 🤖 (RF, XGB, LR, Stack)
6. Result Visualization and Analysis 📊

# Model Performance 🔝

| Model | ROC-AUC (Test) | ROC-AUC (5-Fold CV) | PR-AUC (Test) | Test-CV Gap | Ranking | Business Impact |
|-------|----------------|---------------------|---------------|-------------|---------|-----------------|
| **Stacking Ensemble** | **0.8847** 🥇 | **0.8587** | **0.6861** 🥇 | 2.60% | **1st** | Identifies 89% of churners with 53% precision |
| Logistic Regression | 0.8831 🥈 | 0.8595 | 0.6824 🥈 | 2.36% | 2nd | Nearly identical performance, 10x faster |
| XGBoost | 0.8808 | 0.8354 | 0.6419 | 4.54% | 3rd | Good but unstable (4.5% test-CV gap) |
| Random Forest | 0.8763 | 0.8386 | 0.6405 | 3.77% | 4th | Lowest PR-AUC, prone to false positives |
| *Random Baseline* | *0.5000* | - | *0.2650* | - | - | - |

**Key Metrics:**
- **ROC-AUC:** Measures overall discrimination ability (0.5 = random, 1.0 = perfect)
- **PR-AUC:** Precision-Recall AUC - more informative for imbalanced datasets (baseline = 26.5% churn rate)
- **Test-CV Gap:** Difference between test and cross-validation scores (lower = better generalization)

**Winner:** Stacking Ensemble achieves the best performance on both metrics while maintaining excellent stability (2.60% gap).

### Why These Metrics Matter

**ROC-AUC (Receiver Operating Characteristic)**
- Measures how well the model distinguishes between churners and non-churners
- Best model: **88.47%** (excellent discrimination)

**PR-AUC (Precision-Recall)**
- More reliable for imbalanced data (26.5% churn rate)
- Shows real-world performance: **68.61%** average precision across all recall levels
- **2.59x better** than random predictions

### Model Selection

**For Production:** Logistic Regression (0.6824 PR-AUC)
- Only 0.54% worse than Stacking
- 10x faster inference time
- Fully interpretable for business stakeholders

**Technical Achievement:** Stacking Ensemble (0.6861 PR-AUC)
- State-of-the-art performance
- Best for high-stakes predictions

# Results 🏆




