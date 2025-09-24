import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(path: str):
    """Load dataset from CSV"""
    return pd.read_csv(path)

def get_feature_columns(df: pd.DataFrame, target_col: str = "satisfaction"):
    """Split dataframe into features and target, also return column types"""
    X = df.drop(columns=[target_col])
    y = df[target_col].apply(lambda x: 1 if x == "satisfied" else 0)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    """Preprocessing pipeline"""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    return preprocessor
