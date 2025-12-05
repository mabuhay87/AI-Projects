
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log_model = LogisticRegression(max_iter=2000)
    log_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42
    )
    rf_model.fit(X_train, y_train)

    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    return (log_model, rf_model, xgb_model), (X_train, X_test, y_train, y_test)
