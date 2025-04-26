import pandas as pd
import mlflow
from sklearn.datasets import load_diabetes
from zenml import step

from model_dev import LinearRegressionModel, RandomForestModel
from evaluation import Evaluation

@step
def load_data() -> pd.DataFrame:
    diabetes = load_diabetes()
    df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    return df

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

@step
def train_models(df: pd.DataFrame) -> dict:
    X = df.drop("target", axis=1)
    y = df["target"]

    mlflow.sklearn.autolog()

    models = {
        "LinearRegression": LinearRegressionModel(),
        "RandomForest": RandomForestModel()
    }

    trained_models = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            trained_model = model.train(X, y)
            trained_models[name] = trained_model

    return trained_models

@step
def evaluate_models(trained_models: dict, df: pd.DataFrame) -> dict:
    X = df.drop("target", axis=1)
    y = df["target"]

    results = {}

    for name, model in trained_models.items():
        with mlflow.start_run(run_name=f"{name}_eval"):
            y_pred = model.predict(X)
            metrics = Evaluation.evaluate(y, y_pred)

            # Log to MLflow
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{name}_{metric_name}", value)

            results[name] = metrics
    return results
