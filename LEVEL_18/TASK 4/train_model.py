import pandas as pd
from sklearn.linear_model import LogisticRegression
from zenml.steps import step

@step
def train_model(data: pd.DataFrame) -> LogisticRegression:
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    
    model = LogisticRegression()
    model.fit(X, y)
    
    return model
