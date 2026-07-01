from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import shap

def graph_roc_curve(y_test, y_probs, model):
    fpr, tpr, _= roc_curve(y_test, y_probs)

    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve - {model}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    print(f"{model} - AUC: {roc_auc_score(y_test, y_probs)}")
    return plt.show()

def graph_precision_recall_curve(y_test, y_probs, model):
    precision, recall, _= precision_recall_curve(y_test, y_probs)

    plt.plot(recall, precision)
    plt.title(f"Precision-Recall Curve - {model}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    return plt.show()

def graph_importances(model, name_model):
    try:
        importances = model.feature_importances_
        plt.bar(range(len(importances)), importances)
        plt.title(f"Importance of variables - {name_model}")
        plt.show()
    except AttributeError:
        print("Only the XGBoost and Random Forest models have this method.")

def graph_shap(model, X_test, name_model):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test[:100])

    shap.plots.bar(shap_values, show=False)

    plt.title(f"Shap Values - {name_model}")
    plt.show()