import pandas as pd
from zenml.steps import step

@step
def load_data() -> pd.DataFrame:
    df = pd.read_csv("Titanic-Dataset.csv")
    print("Data types of the columns:\n", df.dtypes)

    df = df.select_dtypes(include=['float64', 'int64']) 
    return df
