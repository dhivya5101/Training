from zenml import pipeline, step
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.linear_model import LinearRegression

@step
def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the diabetes dataset."""
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    return X, y

@step
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Train a linear regression model."""
    from model_dev import LinearRegressionModel
    model = LinearRegressionModel()
    trained_model = model.train(X_train, y_train)
    return trained_model

@step
def evaluate_model(model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate the model and return R2 score."""
    from evaluation import R2
    y_pred = model.predict(X_test)
    r2_eval = R2()
    return r2_eval.calculate_scores(y_test, y_pred)

@step
def deployment_trigger(r2_score: float) -> bool:
    """Trigger deployment based on R2 score."""
    return r2_score >= 0.5

@pipeline
def continuous_deployment_pipeline():
    """Define the continuous deployment pipeline."""
    X, y = load_data()
    model = train_model(X_train=X, y_train=y)
    r2_score = evaluate_model(model=model, X_test=X, y_test=y)
    deploy = deployment_trigger(r2_score=r2_score)
    mlflow_model_deployer_step(model=model)

if __name__ == "__main__":
    run_response = continuous_deployment_pipeline()  
    print(f"Pipeline run status: {run_response.status}") 
