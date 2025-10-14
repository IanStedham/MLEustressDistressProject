import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# they used SMOT to redistribute classes

df = pd.read_csv("Data/StressAppraisal.csv")
df = df.sample(frac=1, random_state=17).reset_index(drop=True)
data = df.drop(columns=["Productivity", "Mood", "Stress_Numeric", "Stress", "Electrodermal activity, 75th percentile ", "Electrodermal activity, maximum "])
labels = df["Stress"]
labels_encoded, uniques = pd.factorize(labels) # 0 - Boredom, 1 - Distress, 2 - Eustress, 3 - Eustress-distress coexistence
data_train, data_test, labels_train, labels_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=17, stratify=labels_encoded)
dtrain = xgb.DMatrix(data_train, label=labels_train)
dtest = xgb.DMatrix(data_test, label=labels_test)


# sns.boxplot(x=labels, y=df["Electrodermal activity, 75th percentile "])
# plt.show()
# sns.boxplot(x=labels, y=df["Electrodermal activity, maximum "])
# plt.show()
# mi = mutual_info_classif(data, labels_encoded, discrete_features='auto')
# print(sorted(zip(data.columns, mi), key=lambda x: -x[1])[:10])

params = {
    "objective": "multi:softprob",
    "num_class": 4,       
    "eval_metric": "mlogloss",      # multiclass log loss is standard
    "eta": 0.1,                     # learning rate
    "max_depth": 3,                
    "subsample": 0.8,               # subsampling rows
    "colsample_bytree": 0.8,        # subsampling features
    "gamma": 0,                     # min loss reduction
    "lambda": 1,                    # L2 regularization
    "alpha": 0,                     # L1 regularization
    "nthread": -1,                  
    "seed": 42
}

num_round = 1  # boosting iterations, can tune
bst = xgb.train(params, dtrain, num_round, evals=[(dtest, "test")])

# softprob returns probabilities
labels_pred_proba = bst.predict(dtest)  # shape: (num_samples, 4)
labels_pred = labels_pred_proba.argmax(axis=1)  # pick class with max probability

print("Accuracy:", accuracy_score(labels_test, labels_pred))
