from zenml import pipeline, step
import json
import pandas as pd
import mlflow
import numpy as np

# Step: Load and prepare a sample row from the CSV file
@step
def dynamic_importer() -> str:
    file_path = r"C:\Users\lenovo\Desktop\my_project\melb_data.csv"
    df = pd.read_csv(file_path)

    # Drop the target column if present, assuming it's 'Price'
    if 'Price' in df.columns:
        df = df.drop(columns=['Price'])

    # Fill missing values to avoid issues during inference
    df = df.fillna(method='ffill').fillna(0)

    sample_data = df.iloc[0].to_dict()  # First row as example
    return json.dumps(sample_data)

# Step: Load the MLflow model and perform prediction
@step
def model_predictor(data: str) -> np.ndarray:
    data_dict = json.loads(data)

    # Ensure the order of features matches training
    features = [data_dict[feature] for feature in data_dict.keys()]

    # Load model from MLflow Model Registry (make sure model is registered)
    model_uri = "file:///C:/Users/lenovo/Desktop/my_project/mlruns/0/c16075ce44a74fa496f483509fe8d7e3/artifacts/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Perform prediction (ensure the result is a NumPy array)
    prediction = model.predict([features])
    return prediction  # Return as a NumPy array

# Pipeline: Connect the steps
@pipeline
def inference_pipeline():
    data = dynamic_importer()
    prediction = model_predictor(data=data)
    return prediction

# Entry point
if __name__ == "__main__":
    # Execute the pipeline and capture the result
    pipeline_instance = inference_pipeline.run()
    # Extract the result from the pipeline
    result = pipeline_instance.steps[-1].output
    print("Predicted price:", result)
