import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from zenml.steps import step

@step
def evaluate_model(model: LogisticRegression, data: pd.DataFrame) -> float:
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    print(f"Model Accuracy: {accuracy}")
    return accuracy
