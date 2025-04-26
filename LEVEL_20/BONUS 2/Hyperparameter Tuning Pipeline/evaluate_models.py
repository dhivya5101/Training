from zenml import step
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

@step
def evaluate_models(model, cleaned_data: pd.DataFrame):
    X = cleaned_data.drop("target", axis=1)
    y = cleaned_data["target"]

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
