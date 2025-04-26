from zenml import step
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import pandas as pd

@step
def train_models(cleaned_data: pd.DataFrame):
    X = cleaned_data.drop("target", axis=1)
    y = cleaned_data["target"]

    # Define model and parameter grid
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None]
    }

    # Use GridSearchCV to find best parameters
    pipeline = SKPipeline([("scaler", StandardScaler()), ("rf", rf)])
    grid_search = GridSearchCV(
        pipeline, 
        param_grid={"rf__n_estimators": param_grid["n_estimators"], "rf__max_depth": param_grid["max_depth"]},
        cv=3,
        scoring="r2",
        n_jobs=-1
    )

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_

    # Log best model and parameters to MLflow
    with mlflow.start_run():
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_r2_score", grid_search.best_score_)

    return best_model
