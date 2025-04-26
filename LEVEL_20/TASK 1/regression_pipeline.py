import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple
from zenml import pipeline, step
from zenml.client import Client


@step
def load_data() -> pd.DataFrame:
    df = pd.read_csv("customer_shopping_data.csv")
    return df


@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df["total_spent"] = df["quantity"] * df["price"]
    df = df.drop(columns=["invoice_no", "customer_id", "invoice_date"])
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df = pd.get_dummies(df, columns=["category", "payment_method", "shopping_mall"], drop_first=True)
    return df


@step
def train_model(df: pd.DataFrame) -> Tuple[LinearRegression, pd.DataFrame, pd.Series]:
    X = df.drop(columns=["total_spent"])
    y = df["total_spent"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test


@step
def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"mse": mse, "r2": r2}


@pipeline
def regression_pipeline():
    df = load_data()
    clean_df = clean_data(df)
    model, X_test, y_test = train_model(clean_df)
    results = evaluate_model(model, X_test, y_test)
    return results


if __name__ == "__main__":
    # Check if ZenML is initialized
    try:
        client = Client()
        client.get_pipeline("regression_pipeline")
    except Exception:
        print("ZenML repository not found. Run 'zenml init' in this directory first.")
        exit(1)

    # Run the pipeline
    regression_pipeline()

    # Get the last successful run's evaluate_model output
    client = Client()
    run = client.get_pipeline("regression_pipeline").last_successful_run

    if run:
        metrics = run.steps["evaluate_model"].output.load()
        print(f"\n📊 MSE: {metrics['mse']}")
        print(f"📊 R2 Score: {metrics['r2']}")
    else:
        print("❌ No successful pipeline run found.")