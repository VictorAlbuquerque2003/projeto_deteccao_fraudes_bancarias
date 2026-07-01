from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def apply_regression( X_train, y_train, X_test, y_test):
    regression = LogisticRegression(max_iter=1000, random_state = 42)

    regression.fit(X_train, y_train)

    y_pred = regression.predict(X_test)

    return regression, classification_report(y_test, y_pred)

def apply_xgboost(X_train, y_train, X_test, y_test):
    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state = 42
    )

    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)

    return xgb, classification_report(y_test, y_pred)

def apply_random_forest(X_train, y_train, X_test, y_test):
    random_forest = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        class_weight="balanced",
        n_jobs =- 1,
        random_state=42
    )

    random_forest.fit(X_train, y_train)

    y_pred_rf = random_forest.predict(X_test)

    return random_forest, classification_report(y_test, y_pred_rf)