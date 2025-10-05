# ===========================================================
# Kepler's Ghosts – Advanced Stacking Ensemble Model
# ===========================================================
# Features:
# - Uses 10 ENRICHED features (including koi_model_snr).
# - Robust Stacking Ensemble Model (RF, LGBM, ERT, LR).
# - Retains all 4 visualization plots.
# ===========================================================

import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
sns.set(style="whitegrid")

# ===========================================================
# 1. Setup Paths and Data (Enriched Features)
# ===========================================================
DATA_PATH = r"C:\Users\KimoStore\Desktop\py4e\NASA\nasssa5\trail\koi_dataset.csv"


# Output directory for model artifacts
OUTDIR = Path("./models_v8_enriched")
OUTDIR.mkdir(exist_ok=True)

# Enriched feature list (10 features)
FEATURES = [
    'koi_period', 'koi_prad', 'koi_teq', 'koi_srad', 'koi_impact',
    'koi_duration', 'koi_depth', 'koi_steff', 'koi_prad_err1', 'koi_model_snr'
]
CATEGORY_MAP = {'FALSE POSITIVE': 0, 'CANDIDATE': 1, 'CONFIRMED': 2}
CLASS_NAMES = list(CATEGORY_MAP.keys())

print(f"📂 Loading data from: {DATA_PATH} using {len(FEATURES)} enriched features.")
df = pd.read_csv(DATA_PATH, comment="#")
df = df[df['koi_disposition'].isin(CATEGORY_MAP.keys())]
df['target'] = df['koi_disposition'].map(CATEGORY_MAP)

# Filter data using the ten enriched features and target
required_columns = FEATURES + ['target']
df_clean = df[required_columns].dropna()

X = df_clean[FEATURES]
y = df_clean['target']
print(f"✅ Initial dataset size: {len(df)} | Cleaned size: {len(X)}")

# ===========================================================
# 2. Data Splitting and Preprocessing
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURES)

# ===========================================================
# 3. Stacking Ensemble Model Setup (Robust Structure)
# ===========================================================
# Using the proven robust Stacking architecture
estimators = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42)),
    ('lgb', lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, class_weight='balanced', random_state=42)),
    ('ert', ExtraTreesClassifier(n_estimators=300, max_depth=12, class_weight='balanced', random_state=42))
]

model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    n_jobs=-1
)

print("🚀 Training Advanced Stacking Ensemble Model...")
model.fit(X_train_scaled, y_train)

# ===========================================================
# 4. Evaluation
# ===========================================================
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
auc_score = roc_auc_score(y_test, y_proba, multi_class="ovr")

print("\n================ Model Evaluation Metrics ================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"AUC Score : {auc_score:.4f}")
print("==========================================================")

# ===========================================================
# 5. Confusion Matrix (Plot 1)
# ===========================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix (Advanced Model)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()
plt.savefig(OUTDIR / "confusion_matrix_v8.png")
print("🟦 Saved: confusion_matrix_v8.png")

# ===========================================================
# 6. Feature Importance (Plot 2)
# ===========================================================
# Using the RF base estimator inside Stacking for feature importance
base_rf = model.named_estimators_['rf']
feature_importance = pd.Series(base_rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance.values, y=feature_importance.index,
            hue=feature_importance.index, legend=False, palette="viridis")
plt.title("Feature Importance (Advanced Model - Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
plt.savefig(OUTDIR / "feature_importance_v8.png")
print("🟩 Saved: feature_importance_v8.png")

# ===========================================================
# 7. Performance Metrics Visualization (Plot 3)
# ===========================================================
metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'AUC': auc_score}
plt.figure(figsize=(7, 4))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()),
            hue=list(metrics.keys()), legend=False, palette="mako")
plt.title("Performance Metrics Summary (Advanced Model)")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
plt.savefig(OUTDIR / "performance_metrics_v8.png")
print("🟨 Saved: performance_metrics_v8.png")

# ===========================================================
# 8. ROC Curves (Optional Plot)
# ===========================================================
plt.figure(figsize=(7, 6))
for i, name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.title("ROC Curves (Advanced Model)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
plt.savefig(OUTDIR / "roc_curves_v8.png")
print("🟥 Saved: roc_curves_v8.png")

# ===========================================================
# 9. Learning Curve (Plot 4)
# ===========================================================
print("📊 Generating learning curve...")
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_scaled, y_train,
    cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    shuffle=True, random_state=42
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', color='green', label='Training Accuracy')
plt.plot(train_sizes, test_mean, 'o-', color='blue', label='Validation Accuracy')
plt.fill_between(train_sizes, train_mean - np.std(train_scores, axis=1),
                 train_mean + np.std(train_scores, axis=1), alpha=0.1, color='green')
plt.fill_between(train_sizes, test_mean - np.std(test_scores, axis=1),
                 test_mean + np.std(test_scores, axis=1), alpha=0.1, color='blue')
plt.title("Learning Curve – Kepler’s Ghosts (Advanced Model)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(OUTDIR / "learning_curve_v8.png")
print("🧩 Saved: learning_curve_v8.png")

# ===========================================================
# 10. Save Model and Components
# ===========================================================
joblib.dump(model, OUTDIR / "model_v8.joblib")
joblib.dump(scaler, OUTDIR / "scaler_v8.joblib")
joblib.dump(FEATURES, OUTDIR / "features_v8.joblib")

print("\n✅ Training, evaluation, and results extraction completed successfully!")
print(f"📂 All model results saved inside folder: {OUTDIR.resolve()}")
