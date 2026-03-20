import joblib
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression


def load_data(file_path: str):
    df = pd.read_csv(file_path)
    return df


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def train_model(data_path: str, target_col: str):
    df = load_data(data_path)

    # Drop customer ID since it is not predictive
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Clean TotalCharges if it is stored as text with blanks
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Convert target to binary
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\nModel Performance")
    print("------------------")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/model.pkl")

    print("\nModel saved to models/model.pkl")


if __name__ == "__main__":
    train_model("data/churn.csv", "Churn")