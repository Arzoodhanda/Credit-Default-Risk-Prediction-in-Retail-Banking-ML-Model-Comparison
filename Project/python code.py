
# ----------------------------
# 1. Imports
# ----------------------------
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Optional for prettier heatmap
import seaborn as sns

# ----------------------------
# 2. Configuration
# ----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
OUTPUT_DIR = "credit_default_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 3. Load Data
# ----------------------------
file_path = "UCI_Credit_Card.csv"   # change if needed
df = pd.read_csv(file_path)

print("Initial shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ----------------------------
# 4. Basic Cleaning
# ----------------------------
# Standardise column names
df.columns = [c.strip() for c in df.columns]

# Rename target if present
if "default.payment.next.month" in df.columns:
    df = df.rename(columns={"default.payment.next.month": "default"})

# Drop ID if present
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

print("\nShape after dropping ID (if present):", df.shape)

# ----------------------------
# 5. Data Screening
# ----------------------------
print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Save descriptive statistics
desc_stats = df.describe().T
desc_stats.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"))

# ----------------------------
# 6. Define Features and Target
# ----------------------------
target_col = "default"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found.")

X = df.drop(columns=[target_col])
y = df[target_col]

print("\nFeature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

# ----------------------------
# 7. Class Distribution
# ----------------------------
class_counts = y.value_counts().sort_index()
class_props = y.value_counts(normalize=True).sort_index() * 100

class_summary = pd.DataFrame({
    "count": class_counts,
    "percentage": class_props.round(2)
})
class_summary.index = ["Non-default (0)", "Default (1)"]
print("\nClass distribution:")
print(class_summary)

class_summary.to_csv(os.path.join(OUTPUT_DIR, "class_distribution.csv"))

plt.figure(figsize=(6, 4))
ax = sns.countplot(x=y)
ax.set_title("Class Distribution: Credit Default")
ax.set_xlabel("Default (0 = No, 1 = Yes)")
ax.set_ylabel("Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_1_class_distribution.png"), dpi=300)
plt.close()

# ----------------------------
# 8. Correlation Analysis
# ----------------------------
corr = df.corr(numeric_only=True)

# Correlation with target
target_corr = corr[target_col].drop(target_col).sort_values(key=np.abs, ascending=False)
target_corr.to_csv(os.path.join(OUTPUT_DIR, "target_correlations.csv"), header=["correlation_with_default"])

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_2_correlation_heatmap.png"), dpi=300)
plt.close()

# ----------------------------
# 9. Group Mean Comparison
# ----------------------------
group_means = df.groupby(target_col).mean(numeric_only=True).T
group_means.columns = ["Non-default", "Default"]
group_means["Difference(Default - Non-default)"] = group_means["Default"] - group_means["Non-default"]
group_means.to_csv(os.path.join(OUTPUT_DIR, "group_mean_comparison.csv"))

top_mean_diff = group_means["Difference(Default - Non-default)"].abs().sort_values(ascending=False).head(10).index
plot_data = group_means.loc[top_mean_diff, ["Non-default", "Default"]]

plot_data.plot(kind="bar", figsize=(12, 6))
plt.title("Top Mean Differences: Defaulters vs Non-defaulters")
plt.xlabel("Feature")
plt.ylabel("Mean value")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_3_group_mean_comparison.png"), dpi=300)
plt.close()

# ----------------------------
# 10. Selected Boxplots
# ----------------------------
selected_features = [f for f in ["PAY_0", "PAY_2", "PAY_3", "LIMIT_BAL", "BILL_AMT1", "PAY_AMT1"] if f in df.columns]

for feature in selected_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=target_col, y=feature, data=df)
    plt.title(f"{feature} by Default Status")
    plt.xlabel("Default (0 = No, 1 = Yes)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{feature}.png"), dpi=300)
    plt.close()

# ----------------------------
# 11. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# ----------------------------
# 12. Model Definitions
# ----------------------------
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10
        ))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            n_jobs=-1
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        ))
    ])
}

# ----------------------------
# 13. Model Training and Evaluation
# ----------------------------
results = []
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

roc_curve_fig = plt.figure(figsize=(8, 6))

confusion_table_rows = []
feature_importance_frames = []

for model_name, pipeline in models.items():
    print(f"\n--- Training {model_name} ---")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # Probability scores for ROC-AUC
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        # Fallback for models with decision_function only
        y_proba = pipeline.decision_function(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "Test ROC-AUC": roc_auc,
        "CV ROC-AUC Mean": cv_scores.mean(),
        "CV ROC-AUC Std": cv_scores.std()
    })

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(OUTPUT_DIR, f"classification_report_{model_name.replace(' ', '_')}.csv")
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    confusion_table_rows.append({
        "Model": model_name,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.replace(' ', '_')}.png"), dpi=300)
    plt.close()

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_proba, name=model_name)

    # Feature importance / coefficients
    model_obj = pipeline.named_steps["model"]

    if model_name == "Logistic Regression":
        importances = np.abs(model_obj.coef_[0])
        fi = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

    elif model_name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
        importances = model_obj.feature_importances_
        fi = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

    else:
        fi = None

    if fi is not None:
        fi["Model"] = model_name
        fi.to_csv(
            os.path.join(OUTPUT_DIR, f"feature_importance_{model_name.replace(' ', '_')}.csv"),
            index=False
        )
        feature_importance_frames.append(fi)

        # Plot top 10
        top10 = fi.head(10).iloc[::-1]
        plt.figure(figsize=(8, 5))
        plt.barh(top10["Feature"], top10["Importance"])
        plt.title(f"Top 10 Features - {model_name}")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"feature_importance_plot_{model_name.replace(' ', '_')}.png"), dpi=300)
        plt.close()

# Save combined ROC curve
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curves for All Models")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_5_roc_curves_all_models.png"), dpi=300)
plt.close()

# ----------------------------
# 14. Results Tables
# ----------------------------
results_df = pd.DataFrame(results).sort_values("Test ROC-AUC", ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_performance_summary.csv"), index=False)

confusion_df = pd.DataFrame(confusion_table_rows)
confusion_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix_summary.csv"), index=False)

print("\nModel Performance Summary:")
print(results_df)

print("\nConfusion Matrix Summary:")
print(confusion_df)

# ----------------------------
# 15. Combined Feature Importance Summary
# ----------------------------
if feature_importance_frames:
    combined_fi = pd.concat(feature_importance_frames, ignore_index=True)
    combined_fi.to_csv(os.path.join(OUTPUT_DIR, "combined_feature_importance_all_models.csv"), index=False)

    # Average rank-based importance summary
    combined_fi["Rank"] = combined_fi.groupby("Model")["Importance"].rank(ascending=False, method="average")
    avg_rank = combined_fi.groupby("Feature")["Rank"].mean().sort_values()
    avg_rank_df = avg_rank.reset_index()
    avg_rank_df.columns = ["Feature", "Average_Rank"]
    avg_rank_df.to_csv(os.path.join(OUTPUT_DIR, "average_feature_rank_summary.csv"), index=False)

    print("\nTop features by average rank across models:")
    print(avg_rank_df.head(15))

# ----------------------------
# 16. Dissertation-Ready Summary Text Output
# ----------------------------
best_roc_model = results_df.iloc[0]["Model"]
best_f1_model = results_df.sort_values("F1-score", ascending=False).iloc[0]["Model"]

summary_lines = [
    "Dissertation Results Summary",
    "============================",
    f"Best model by Test ROC-AUC: {best_roc_model}",
    f"Best model by F1-score: {best_f1_model}",
    "",
    "Interpretive guidance:",
    "- Gradient Boosting is expected to perform best in pure ranking/discrimination terms.",
    "- Random Forest often provides the best practical balance across precision and recall.",
    "- Logistic Regression offers baseline transparency but weaker overall discrimination.",
    "- Recent repayment-status features are expected to dominate predictive importance."
]

with open(os.path.join(OUTPUT_DIR, "results_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\nAll outputs saved to:", OUTPUT_DIR)