import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from visualization import graph_roc_curve, graph_precision_recall_curve, graph_importances, graph_shap
from models import apply_random_forest, apply_regression, apply_xgboost

file = "./data/creditcard.csv"

df = pd.read_csv(file)

df["Amount_log"] = np.log1p(df["Amount"])

scaler = StandardScaler()

df["Amount_log_scaled"] = scaler.fit_transform(df[["Amount_log"]])

X = df.drop(["Class", "Amount", "Amount_log"], axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

smote = SMOTE()
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

regression, regression_report = apply_regression(X_train, y_train, X_test, y_test)
xgb, xgboost_report = apply_xgboost(X_train_over, y_train_over, X_test, y_test)
rf, random_forest_report = apply_random_forest(X_train_over, y_train_over, X_test, y_test)

y_probs_regression = regression.predict_proba(X_test)[:,1]
y_probs_xgb = xgb.predict_proba(X_test)[:,1]
y_probs_rf = rf.predict_proba(X_test)[:,1]

graph_roc_curve(y_test, y_probs_regression, "Regression")
graph_precision_recall_curve(y_test, y_probs_regression, "Regression")

graph_roc_curve(y_test, y_probs_xgb, "XGBoost")
graph_precision_recall_curve(y_test, y_probs_xgb, "XGBoost")

graph_roc_curve(y_test, y_probs_rf, "Random Forest")
graph_precision_recall_curve(y_test, y_probs_rf, "Random Forest")

graph_importances(rf, "Random Forest")
graph_importances(xgb, "XGBoost")

print(f"Logistic Regression: \n{regression_report}")
print(f"Random Forest: \n{random_forest_report}")
print(f"XGBoost: \n{xgboost_report}")

graph_shap(xgb, X_test, "XGBoost")