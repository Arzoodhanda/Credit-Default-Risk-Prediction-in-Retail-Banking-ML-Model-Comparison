# Credit-Default-Risk-Prediction-in-Retail-Banking-ML-Model-Comparison
# Introduction & Motivation
RESEARCH CONTEXT
Credit risk assessment is fundamental to modern banking.
Inaccurate assessment leads to financial losses and regulatory
scrutiny.
Traditional Logistic Regression struggles with complex, nonlinear relationships in modern financial data.
THE PROBLEM
ML models show promise but raise interpretability concerns
critical in regulated banking.
Class imbalance makes accuracy alone misleading — most
borrowers don't default.
RESEARCH GAP
No consistent comparative framework exists that evaluates predictive performance, robustness, and interpretability together under
standardised conditions.
This study compares Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
RESEARCH QUESTIONS
1. How effectively can ML models predict credit default?
2. How does LR compare with DT, RF, and GB?
3. Which model balances accuracy, robustness, and interpretability?
4. Which features are the most significant predictors?
# Dataset Overview
Property Value
Observations 30,000
Predictor Variables 23 (after ID removal)
Target Variable default (binary)
Missing Values None
Data Source UCI Credit Card Default
VARIABLE GROUPS
Demographic (4): SEX, EDUCATION, MARRIAGE, AGE
Credit Capacity (1): LIMIT_BAL
Repayment Status (6): PAY_0 to PAY_6
Bill Amounts (6): BILL_AMT1 to BILL_AMT6
Payment Amounts (6): PAY_AMT1 to PAY_AMT6
78%
22%
Non-Default
Default
Moderate class imbalance justifies multi-metric evaluation beyond
accuracy alone — precision, recall, F1-score, and ROC-AUC are essential.
Literature Review & Research Gap
ML Outperforms Traditional
RF and GB consistently outperform LR in
predictive accuracy on structured financial data
(Brown & Mues, 2012; Lessmann et al., 2015)
Class Imbalance Challenge
Accuracy is misleading for imbalanced credit
data. Precision, recall, F1, and ROC-AUC are
essential (He & Garcia, 2009)
Interpretability Trade-Off
The performance vs. explainability trade-off
remains unresolved in regulated banking (DoshiVelez & Kim, 2017)
# IDENTIFIED RESEARCH GAPS
1
Inconsistent Comparisons
Prior studies use different preprocessing, validation, and metrics
— making direct comparison unreliable
2
Unbalanced Framework
No unified evaluation that simultaneously assesses accuracy,
robustness, and interpretability
3
Imbalance Neglected
Class imbalance is acknowledged but not systematically addressed in
model comparison studies
This study fills all three gaps through a consistent, multi-metric
comparative framework
# Methodology & Research Design
RESEARCH DESIGN
Quantitative, positivist, deductive design using secondary archival data.
CRISP-DM structured workflow ensures reproducibility.
Data
Understanding
Data
Preparation
Modelling
4 ML Models
Evaluation
Multi-Metric
FOUR MODELS COMPARED
Logistic Regression
Transparent statistical baseline.
Highest interpretability, weakest
discrimination.
Decision Tree
Rule-based interpretability. Captures
non-linearity but prone to overfitting.
Random Forest
Ensemble of trees. Best practical
balance of performance and
interpretability.
Gradient Boosting
Sequential error correction. Highest
predictive power, lowest
transparency.
EVALUATION METRICS
Accuracy
Overall correctness
Precision
Reliability of positives
Recall
Defaulter detection
F1-Score
Precision + Recall
ROC-AUC
Ranking ability
EDA Key Findings
01
Class Imbalance Confirmed
77.88% non-defaulters vs 22.12% defaulters.
This moderate imbalance means accuracy
alone is insufficient — multi-metric evaluation
is essential for fair assessment.
02
Repayment Behaviour Dominates
Recent repayment-status variables (PAY_0,
PAY_2, PAY_3, PAY_4) show the strongest
correlation with default. Behavioural signals
are the key drivers of risk.
03
Demographics Matter Less
SEX, EDUCATION, MARRIAGE, and AGE
show weak predictive signal. Static profile
characteristics are far less informative than
dynamic behaviour.
CORRELATION ANALYSIS INSIGHT
0
0
0
0
0
0
0
0 0.05 0.1 0.15 0.2 0.25 0.3 0.35
PAY_0
PAY_2
PAY_3
PAY_4
LIMIT_BAL
PAY_AMT1
AGE The correlation analysis reveals a clear hierarchy:
PAY_0 (most recent repayment status) is the strongest individual predictor
of default.
Recent repayment history consistently outweighs all other variable groups
— confirming that behavioural signals, not static demographics, drive
default risk.
Model Performance Comparison
Model ROC-AUC F1-Score Accuracy Key Strength
Gradient Boosting 0.7772 0.5200 Highest Best
discrimination
Random Forest 0.7700 0.5407 High Best practical
balance
Decision Tree 0.6800 0.4700 Moderate Interpretable
rules
Logistic Regression 0.6700 0.4200 Moderate Most
transparent
1
1
1
1
0.6
0.65
0.7
0.75
0.8
GB RF DT LR
# KEY FINDINGS
Gradient Boosting achieves the highest ROC-AUC (~0.7772)
and accuracy — best for pure predictive ranking
Random Forest delivers the highest F1-score (~0.5407) — best
practical overall balance
Decision Tree & LR offer greater interpretability but weaker
discrimination — valuable as explainable baselines
All ensemble methods significantly outperform the linear baseline,
confirming the value of ML in credit risk
Feature Importance & Interpretability
FEATURE IMPORTANCE RANKING
0
0
0
0
0
0
0
0
0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4
PAY_0
PAY_2
PAY_3
LIMIT_BAL
PAY_4
BILL_AMT1
PAY_AMT1
AGE
INTERPRETABILITY TRADE-OFF
Logistic Regression — Coefficient-based interpretation. Highest
transparency.
Decision Tree — Readable decision paths and threshold-based rules.
Random Forest — Aggregate feature importance. Moderate interpretability.
Gradient Boosting — Strongest prediction, lowest direct transparency.
KEY INSIGHT: BEHAVIOURAL vs DEMOGRAPHIC
Behavioural variables (repayment status, bill amounts) consistently
outperform demographic features across all models.
PAY_0 alone is the strongest predictor — recent account conduct is more
informative than static customer descriptors.
Model selection is a governance issue, not just a technical one.
A lender may prefer a slightly weaker model if it is easier to explain,
validate, and defend in a regulated environment.
# Conclusions & Contributions
ML Models Are Effective
All four models captured meaningful predictive structure from structured
financial data, with ensemble methods showing clear advantages.
Best Model Depends on Priority
Gradient Boosting for discrimination, Random Forest for practical balance,
LR for transparency.
Behavioural Signals Dominate
Recent repayment behaviour (especially PAY_0) is more predictive than
demographic attributes. Lenders should prioritise account conduct data.
Multi-Dimensional Quality
Model quality is multidimensional — ROC-AUC alone is insufficient.
Transparency, governance, and error costs all matter in practice.
# LIMITATIONS & FUTURE WORK
Limitations:
Single benchmark dataset limits generalisability
No advanced imbalance handling (SMOTE, class weighting)
No SHAP or LIME explainability techniques applied
Future Research:
Test across multiple credit datasets
Apply SMOTE, cost-sensitive learning
Integrate SHAP/LIME for model explanation
