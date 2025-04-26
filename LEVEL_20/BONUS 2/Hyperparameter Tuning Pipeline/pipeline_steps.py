import pandas as pd
import mlflow
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from zenml import step


@step
def load_data() -> pd.DataFrame:
    diabetes = load_diabetes()
    df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target
    return df


@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


@step
def train_random_forest(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    rf = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1
    )

    with mlflow.start_run(run_name="RandomForest_GridSearch"):
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_

        # Log best parameters and score
        mlflow.log_params(best_params)
        mlflow.log_metric("best_neg_mse", best_score)

    return best_model


@step
def evaluate_model(model, df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = df["target"]
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    with mlflow.start_run(run_name="Final_Evaluation"):
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2_Score", r2)

    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    return {"MSE": mse, "R2": r2}
