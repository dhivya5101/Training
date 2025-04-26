import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from zenml import step, pipeline

@step
def load_data() -> pd.DataFrame:
    """Load the diabetes dataset."""
    diabetes = load_diabetes()
    df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    return df

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by dropping null values."""
    df_cleaned = df.dropna()
    return df_cleaned

@step
def train_model(df: pd.DataFrame) -> LinearRegression:
    """Train a linear regression model with MLflow tracking."""
    mlflow.sklearn.autolog()  # Enable MLflow autologging for sklearn
    X = df.drop('target', axis=1)
    y = df['target']
    model = LinearRegression()
    model.fit(X, y)
    return model

@step
def evaluate_model(model: LinearRegression, df: pd.DataFrame) -> dict:
    """Evaluate the model and log metrics to MLflow."""
    X = df.drop('target', axis=1)
    y = df['target']
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Log metrics to MLflow
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    
    return {"mse": mse, "r2": r2}

@pipeline(
    enable_cache=False,
    name="mlflow_model_tracker_pipeline"
)
def mlflow_model_tracker_pipeline():
    """Define the regression pipeline with MLflow tracking."""
    data = load_data()
    cleaned_data = clean_data(data)
    model = train_model(cleaned_data)
    metrics = evaluate_model(model, cleaned_data)

if __name__ == "__main__":
    # Run the pipeline
    try:
        print("Ensure MLflow tracker is in the active stack. Run: `zenml stack update -e mlflow_tracker`")
        print("Check active stack with: `zenml stack describe`")
        mlflow_model_tracker_pipeline()
        print("Pipeline completed successfully.")
        print("View results in MLflow UI: `mlflow ui` (http://localhost:5000)")
        print("View pipeline runs in ZenML dashboard: `zenml dashboard`")
    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")