import pandas as pd
from zenml.steps import step

@step
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna(subset=["Survived"])
    data = data.fillna(method="ffill")
    return data
