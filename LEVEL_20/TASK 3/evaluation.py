from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class Evaluation:
    @staticmethod
    def evaluate(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {"MSE": mse, "RMSE": rmse, "R2": r2}
