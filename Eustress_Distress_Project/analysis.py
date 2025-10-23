import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, 
    precision_recall_curve, PrecisionRecallDisplay, roc_auc_score, roc_curve, auc
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import seaborn as sns

# Setting up data for analysis, using random_state=17 (same used in model training)
df = pd.read_csv("Data/StressAppraisal.csv")
data = df.drop(columns=["Productivity", "Mood", "Stress_Numeric", "Stress"])
labels = df["Stress"]
labels_encoded, uniques = pd.factorize(labels) # 0 - Boredom, 1 - Distress, 2 - Eustress, 3 - Eustress-distress coexistence
data_train, data_test, labels_train, labels_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=17, stratify=labels_encoded)
labels_test_bin = label_binarize(labels_test, classes=range(len(uniques)))

# Loading both the XGBoost models, one that uses SMOTE and one without
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_model.json")
xgb_model_smote = xgb.XGBClassifier()
xgb_model_smote.load_model("xgb_model_smote.json")

# Creating the predicted labels along with the probability estimates
labels_pred = xgb_model.predict(data_test)
labels_pred_proba = xgb_model.predict_proba(data_test)
labels_pred_smote = xgb_model_smote.predict(data_test)
labels_pred_proba_smote = xgb_model_smote.predict_proba(data_test)

### Standard Statistics 
# No SMOTE
print("\nClassification Report:\n", classification_report(labels_test, labels_pred, target_names=uniques))
# SMOTE
print("\nClassification Report (SMOTE):\n", classification_report(labels_test, labels_pred_smote, target_names=uniques))


# Confusion Matrix
# No SMOTE
cm = confusion_matrix(labels_test, labels_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=uniques)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (XGBClassifier)")
plt.show()

# SMOTE
cm_smote = confusion_matrix(labels_test, labels_pred_smote)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_smote, display_labels=uniques)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (XGBClassifier with SMOTE)")
plt.show()


### Precision-Recall Curve
# No SMOTE
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(4):
    precision, recall, _ = precision_recall_curve(labels_test_bin[:, i], labels_pred_proba[:, i])
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot(ax=ax, name=f"Class {uniques[i]}")
ax.set_title("Precision-Recall Curve for Each Class (OVA)")
plt.show()

# SMOTE
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(4):
    precision, recall, _ = precision_recall_curve(labels_test_bin[:, i], labels_pred_proba_smote[:, i])
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot(ax=ax, name=f"Class {uniques[i]}")
ax.set_title("Precision-Recall Curve for Each Class (OVA, SMOTE)")
plt.show()


# AUROC and AUC-ROC Curve
# No SMOTE
auroc_macro_ovr = roc_auc_score(labels_test, labels_pred_proba, multi_class='ovr', average='macro')
print(f"Multiclass AUROC (OVR, Macro): {auroc_macro_ovr:.3f}")
auroc_weighted_ovo = roc_auc_score(labels_test, labels_pred_proba, multi_class='ovo', average='weighted')
print(f"Multiclass AUROC (OVO, Weighted): {auroc_weighted_ovo:.3f}")

# SMOTE
auroc_macro_ovr_smote = roc_auc_score(labels_test, labels_pred_proba_smote, multi_class='ovr', average='macro')
print(f"Multiclass AUROC (OVR, Macro, SMOTE): {auroc_macro_ovr_smote:.3f}")
auroc_weighted_ovo_smote = roc_auc_score(labels_test, labels_pred_proba, multi_class='ovo', average='weighted')
print(f"Multiclass AUROC (OVO, Weighted, SMOTE): {auroc_weighted_ovo_smote:.3f}\n")

# No SMOTE
false_positive_rate = dict()
true_positive_rate = dict()
roc_auc = dict()

for i in range(4):
    false_positive_rate[i], true_positive_rate[i], _ = roc_curve(labels_test_bin[:, i], labels_pred_proba[:, i])
    roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])

plt.figure(figsize=(8, 6))
for i in range(4):
    plt.plot(false_positive_rate[i], true_positive_rate[i], label=f'ROC curve of class {uniques[i]} (area = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUR-ROC (OVA)')
plt.legend(loc="lower right")
plt.show()

# SMOTE
false_positive_rate_smote = dict()
true_positive_rate_smote = dict()
roc_auc_smote = dict()

for i in range(4):
    false_positive_rate_smote[i], true_positive_rate_smote[i], _ = roc_curve(labels_test_bin[:, i], labels_pred_proba_smote[:, i])
    roc_auc_smote[i] = auc(false_positive_rate_smote[i], true_positive_rate_smote[i])

plt.figure(figsize=(8, 6))
for i in range(4):
    plt.plot(false_positive_rate_smote[i], true_positive_rate_smote[i], label=f'ROC curve of class {uniques[i]} (area = {roc_auc_smote[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUR-ROC (OVA, SMOTE)')
plt.legend(loc="lower right")
plt.show()


# 1-Sample T-Test
# Loading in Cross Validation Results
cv_results = pd.read_csv("cv_results_xgb.csv")
cv_results_smote = pd.read_csv("cv_results_xgb_smote.csv")
cv_accuracy = cv_results["test_accuracy"].values
cv_f1 = cv_results["test_f1_macro"].values
cv_accuracy_smote = cv_results["test_f1_macro"].values
cv_f1_smote = cv_results["test_accuracy"].values

# Values from paper
sample_accuracy_mean = 0.8278
sample_f1_mean = 0.8228

# Calculating 95% Confidence Intervals for Cross Validation Data
# No SMOTE
acc_mean = np.mean(cv_f1)
acc_se = stats.sem(cv_f1)
acc_h = acc_se * stats.t.ppf((1 + 0.95) / 2, 9)
f1_mean = np.mean(cv_f1)
f1_se = stats.sem(cv_f1)
f1_h = f1_se * stats.t.ppf((1 + 0.95) / 2, 9)
print(f"My F1: {f1_mean:.3f} (95% CI: {f1_mean-f1_h:.3f}–{f1_mean+f1_h:.3f}), Paper: {sample_f1_mean}")
print(f"My Accuracy: {acc_mean:.3f} (95% CI: {acc_mean-acc_h:.3f}–{acc_mean+acc_h:.3f}), Paper: {sample_accuracy_mean}")

# SMOTE
acc_mean_smote = np.mean(cv_accuracy_smote)
acc_se_smote = stats.sem(cv_accuracy_smote)
acc_h_smote = acc_se_smote * stats.t.ppf((1 + 0.95) / 2, 9)
f1_mean_smote = np.mean(cv_f1_smote)
f1_se_smote = stats.sem(cv_f1_smote)
f1_h_smote = f1_se_smote * stats.t.ppf((1 + 0.95) / 2, 9)
print(f"My F1 with SMOTE: {f1_mean_smote:.3f} (95% CI: {f1_mean_smote-f1_h_smote:.3f}–{f1_mean_smote+f1_h_smote:.3f}), Paper: {sample_f1_mean}")
print(f"My Accuracy with SMOTE: {acc_mean_smote:.3f} (95% CI: {acc_mean_smote-acc_h_smote:.3f}–{acc_mean_smote+acc_h_smote:.3f}), Paper: {sample_accuracy_mean}\n")

# Printing T-Test comparison values
# No SMOTE
t_stat_f1, p_val_f1 = stats.ttest_1samp(cv_f1, sample_f1_mean)
t_stat_acc, p_val_acc = stats.ttest_1samp(cv_accuracy, sample_accuracy_mean)
print(f"F1 comparison: t = {t_stat_f1:.3f}, p = {p_val_f1:.3f}")
print(f"Accuracy comparison: t = {t_stat_acc:.3f}, p = {p_val_acc:.3f}")

# SMOTE
t_stat_f1_smote, p_val_f1_smote = stats.ttest_1samp(cv_f1_smote, sample_f1_mean)
t_stat_acc_smote, p_val_acc_smote = stats.ttest_1samp(cv_accuracy_smote, sample_accuracy_mean)
print(f"F1 comparison with SMOTE: t = {t_stat_f1_smote:.3f}, p = {p_val_f1_smote:.3f}")
print(f"Accuracy comparison with SMOTE: t = {t_stat_acc_smote:.3f}, p = {p_val_acc_smote:.3f}")

# Plotting the 1-Sample T-Test, No SMOTE
plot_data = pd.DataFrame({
    "Metric": ["F1"] * len(cv_f1) + ["Accuracy"] * len(cv_accuracy),
    "Score": np.concatenate([cv_f1, cv_accuracy])
})

plt.figure(figsize=(7, 5))
sns.boxplot(x="Metric", y="Score", data=plot_data, width=0.5)
sns.swarmplot(x="Metric", y="Score", data=plot_data, color="k", alpha=0.6, size=5)
plt.axhline(sample_f1_mean, color="red", linestyle="--", label="Paper F1 Mean (0.8278)")
plt.axhline(sample_accuracy_mean, color="blue", linestyle="--", label="Paper Accuracy Mean (0.8228)")
plt.title("Cross-Validation Scores vs Paper's Reported Means")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.show()

# Plotting the 1-Sample T-Test, SMOTE
plot_data_smote = pd.DataFrame({
    "Metric": ["F1"] * len(cv_f1_smote) + ["Accuracy"] * len(cv_accuracy_smote),
    "Score": np.concatenate([cv_f1_smote, cv_accuracy_smote])
})

plt.figure(figsize=(7, 5))
sns.boxplot(x="Metric", y="Score", data=plot_data_smote, width=0.5)
sns.swarmplot(x="Metric", y="Score", data=plot_data_smote, color="k", alpha=0.6, size=5)
plt.axhline(sample_f1_mean, color="red", linestyle="--", label="Paper F1 Mean (0.8278)")
plt.axhline(sample_accuracy_mean, color="blue", linestyle="--", label="Paper Accuracy Mean (0.8228)")
plt.title("Cross-Validation Scores vs Paper's Reported Means, SMOTE")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.show()