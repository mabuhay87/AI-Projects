
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def build_preprocessor(df: pd.DataFrame, target_col: str = "is_fraud"):
    X = df.drop(columns=[target_col, "claim_id", "person_id"])
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    return preprocessor, cat_cols, num_cols
