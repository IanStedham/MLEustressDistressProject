import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import make_scorer, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

df = pd.read_csv("Data/StressAppraisal.csv")
data = df.drop(columns=["Productivity", "Mood", "Stress_Numeric", "Stress"])
labels = df["Stress"]
labels_encoded, uniques = pd.factorize(labels) # 0 - Boredom, 1 - Distress, 2 - Eustress, 3 - Eustress-distress coexistence
data_train, data_test, labels_train, labels_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=17, stratify=labels_encoded)

# Creating a model for Cross Validation, a final mode trained with SMOTE, and a final model trained without SMOTE
xgb_cv_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=17,
    n_estimators=100
)
xgb_final_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=17,
    n_estimators=100
)
xgb_final_model_smote = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=17,
    n_estimators=100
)

# Using accuracy and F1 to measure Cross Validation Folds
scoring = {
    'accuracy': 'accuracy',
    'f1_macro': make_scorer(f1_score, average='macro')
}

smote = SMOTE(random_state=17)
pipeline = Pipeline([
    ('smote', smote),
    ('xgb', xgb_cv_model)
])

# Training Cross Validation models with and without SMOTE
results = cross_validate(
    xgb_cv_model,
    data, labels_encoded,
    cv=10,
    scoring=scoring,
    return_train_score=True
)
results_smote = cross_validate(
    pipeline,
    data, labels_encoded,
    cv=10,
    scoring=scoring,
    return_train_score=True
)

# Printing Cross Validation model results without SMOTE
for i in range(10):
    print(f"Fold {i+1}:")
    print(f"  Training Accuracy  = {results['train_accuracy'][i]:.3f}")
    print(f"  Test Accuracy      = {results['test_accuracy'][i]:.3f}")
    print(f"  Training F1        = {results['train_f1_macro'][i]:.3f}")
    print(f"  Test F1            = {results['test_f1_macro'][i]:.3f}\n")
print(f"Mean Training Accuracy: {results['train_accuracy'].mean():.3f}")
print(f"Mean Test Accuracy:     {results['test_accuracy'].mean():.3f}\n")
print(f"Mean Training F1: {results['train_f1_macro'].mean():.3f}")
print(f"Mean Test F1:     {results['test_f1_macro'].mean():.3f}\n\n")

# Printing Cross Validation model results with SMOTE
for i in range(10):
    print(f"Fold {i+1}:")
    print(f"  Training Accuracy SMOTE = {results_smote['train_accuracy'][i]:.3f}")
    print(f"  Test Accuracy SMOTE     = {results_smote['test_accuracy'][i]:.3f}")
    print(f"  Training F1 SMOTE       = {results_smote['train_f1_macro'][i]:.3f}")
    print(f"  Test F1 SMOTE           = {results_smote['test_f1_macro'][i]:.3f}\n")
print(f"Mean Training Accuracy with SMOTE: {results_smote['train_accuracy'].mean():.3f}")
print(f"Mean Test Accuracy with SMOTE:     {results_smote['test_accuracy'].mean():.3f}\n")
print(f"Mean Training F1 with SMOTE: {results_smote['train_f1_macro'].mean():.3f}")
print(f"Mean Test F1 with SMOTE:     {results_smote['test_f1_macro'].mean():.3f}\n")

# Saving cross validation results
df_cv_results = pd.DataFrame(results)
df_cv_results_smote = pd.DataFrame(results_smote)
df_cv_results.to_csv("cv_results_xgb.csv", index=False)
df_cv_results_smote.to_csv("cv_results_xgb_smote.csv", index=False)

# Training and saving final models with and without SMOTE
data_train_smote, labels_train_smote = smote.fit_resample(data_train, labels_train)
xgb_final_model.fit(data_train, labels_train)
xgb_final_model.save_model("xgb_model.json")
xgb_final_model_smote.fit(data_train_smote, labels_train_smote)
xgb_final_model_smote.save_model("xgb_model_smote.json")
