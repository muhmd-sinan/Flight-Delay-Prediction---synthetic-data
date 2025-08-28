"""
Flight Delay Prediction - Model Evaluation Metrics Generator
Generates comprehensive evaluation metrics and visualizations for dashboard
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    mean_squared_error, r2_score, mean_absolute_error,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve
import xgboost as xgb

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Directories
DATA_DIR = Path("data")
VIZ_DIR = Path("evaluation")
VIZ_DIR.mkdir(exist_ok=True)

# Load data
print("Loading dataset...")
df = pd.read_csv(DATA_DIR / "airline_delay_dataset.csv")

# Preprocess (same as training)
df["airport_traffic_last_hour"] = df["dep_flights_last_hr"] + df["arr_flights_last_hr"]
df["route"] = df["origin"] + "_" + df["destination"]
df = df.drop(columns=['origin', 'destination', 'flight_number', 'dep_flights_last_hr', 'arr_flights_last_hr'])

# Train/test split
train = df[df['year'] < 2023]
test = df[df['year'] == 2023]

X_train = train.drop(columns=['delay_minutes', 'delay_15'])
y_train = train['delay_15']
b_train = train['delay_minutes']

X_test = test.drop(columns=['delay_minutes', 'delay_15'])
y_test = test['delay_15']
b_test = test['delay_minutes']

# Target encode categorical columns
from category_encoders import TargetEncoder
cat_cols = ['airline', 'route']
te = TargetEncoder(cols=cat_cols)
X_train = te.fit_transform(X_train, y_train)
X_test = te.transform(X_test)

# Load models
print("Loading models...")
with open(DATA_DIR / "xgb_delay_classifier.pkl", 'rb') as f:
    clf = pickle.load(f)

with open(DATA_DIR / "xgb_delay_regressor.pkl", 'rb') as f:
    reg = pickle.load(f)

results = {}

# ============================================
# CLASSIFIER EVALUATION
# ============================================
print("Evaluating classifier...")

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Basic metrics
clf_metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
    'average_precision': float(average_precision_score(y_test, y_pred_proba)),
    'brier_score': float(brier_score_loss(y_test, y_pred_proba))
}

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
clf_metrics['precision'] = report['weighted avg']['precision']
clf_metrics['recall'] = report['weighted avg']['recall']
clf_metrics['f1_score'] = report['weighted avg']['f1-score']

# Per-class metrics
clf_metrics['precision_class_0'] = report['0']['precision']
clf_metrics['precision_class_1'] = report['1']['precision']
clf_metrics['recall_class_0'] = report['0']['recall']
clf_metrics['recall_class_1'] = report['1']['recall']
clf_metrics['f1_class_0'] = report['0']['f1-score']
clf_metrics['f1_class_1'] = report['1']['f1-score']

results['classifier'] = clf_metrics

# Confusion Matrix
print("  Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Delay (≤15min)', 'Delay (>15min)'],
            yticklabels=['No Delay (≤15min)', 'Delay (>15min)'])
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
ax.set_title('Confusion Matrix - Delay Classifier')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC Curve
print("  Generating ROC curve...")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {clf_metrics["roc_auc"]:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - Delay Classifier')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# Precision-Recall Curve
print("  Generating PR curve...")
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall, precision, label=f'PR Curve (AP = {clf_metrics["average_precision"]:.3f})', linewidth=2)
ax.axhline(y=y_test.mean(), color='r', linestyle='--', label=f'Baseline ({y_test.mean():.2f})')
ax.fill_between(recall, precision, alpha=0.3)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve - Delay Classifier')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'pr_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# Calibration Curve
print("  Generating calibration curve...")
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(prob_pred, prob_true, 's-', label='Model', linewidth=2, markersize=8)
ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curve - Delay Classifier')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'calibration_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature Importance
print("  Generating feature importance...")
importance = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
importance.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Feature Importance')
ax.set_title('Top Features - Delay Classifier')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Learning Curves (subset for speed)
print("  Generating learning curves...")
train_sizes, train_scores, val_scores = learning_curve(
    xgb.XGBClassifier(learning_rate=0.03, max_depth=5, n_estimators=100, objective='binary:logistic'),
    X_train.iloc[:5000], y_train.iloc[:5000],
    cv=3, scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=1
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(train_sizes, train_mean, 'o-', label='Training Score', color='blue')
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
ax.plot(train_sizes, val_mean, 'o-', label='Validation Score', color='green')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='green')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('ROC AUC Score')
ax.set_title('Learning Curves - Delay Classifier')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================
# REGRESSOR EVALUATION
# ============================================
print("Evaluating regressor...")

# Filter only delayed flights for regressor (as done in training)
delayed_test = test[test['delay_15'] == 1]
A_test = delayed_test.drop(columns=['delay_minutes', 'delay_15'])
b_test_delayed = delayed_test['delay_minutes']

# Apply same encoding
A_test = te.transform(A_test)

b_pred = reg.predict(A_test)

# Basic metrics
reg_metrics = {
    'mse': float(mean_squared_error(b_test_delayed, b_pred)),
    'rmse': float(np.sqrt(mean_squared_error(b_test_delayed, b_pred))),
    'mae': float(mean_absolute_error(b_test_delayed, b_pred)),
    'r2': float(r2_score(b_test_delayed, b_pred)),
    'mape': float(np.mean(np.abs((b_test_delayed - b_pred) / b_test_delayed)) * 100)
}

results['regressor'] = reg_metrics

# Actual vs Predicted Scatter
print("  Generating actual vs predicted plot...")
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(b_pred, b_test_delayed, alpha=0.5, s=20)
min_val = min(b_pred.min(), b_test_delayed.min())
max_val = max(b_pred.max(), b_test_delayed.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Predicted Delay (minutes)')
ax.set_ylabel('Actual Delay (minutes)')
ax.set_title('Actual vs Predicted - Delay Regressor')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ_DIR / 'actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()

# Residuals Distribution
print("  Generating residuals distribution...")
residuals = b_test_delayed - b_pred
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Residuals histogram
ax1.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax1.set_xlabel('Residuals (Actual - Predicted)')
ax1.set_ylabel('Frequency')
ax1.set_title('Residuals Distribution')
ax1.grid(True, alpha=0.3)

# Residuals vs Predicted
ax2.scatter(b_pred, residuals, alpha=0.5, s=20)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Delay (minutes)')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals vs Predicted')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'residuals.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature Importance (Regressor)
print("  Generating regressor feature importance...")
reg_importance = pd.Series(reg.feature_importances_, index=X_train.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
reg_importance.plot(kind='barh', ax=ax, color='coral')
ax.set_xlabel('Feature Importance')
ax.set_title('Top Features - Delay Regressor')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'regressor_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Save metrics JSON
print("Saving metrics...")
with open(VIZ_DIR / 'metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n[OK] Evaluation complete!")
print(f"Metrics saved to: {VIZ_DIR / 'metrics.json'}")
print(f"Visualizations saved to: {VIZ_DIR}")
print("\nGenerated files:")
for file in VIZ_DIR.iterdir():
    print(f"  - {file.name}")
