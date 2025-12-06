
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def split_and_resample(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, X_test, y_train_res, y_test

def train_models(X_train_res, y_train_res):
    log_model = LogisticRegression(max_iter=2000)
    log_model.fit(X_train_res, y_train_res)

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=42
    )
    rf_model.fit(X_train_res, y_train_res)

    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=3
    )
    xgb_model.fit(X_train_res, y_train_res)

    return log_model, rf_model, xgb_model
