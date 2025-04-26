import pandas as pd
import mlflow
import mlflow.sklearn
from zenml.steps import step
from zenml.pipelines import pipeline

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Step 1: Load model
@step
def load_model():
    model_uri = "models:/online_retail_data/1"  # Replace with your actual model name/version
    model = mlflow.sklearn.load_model(model_uri)
    return model

# Step 2: Preprocess data
@step
def preprocess_data() -> pd.DataFrame:
    # Simulated test data
    new_data = pd.DataFrame({
        'Quantity': [10, 5],
        'UnitPrice': [20, 50],
    })
    new_data['TotalPrice'] = new_data['Quantity'] * new_data['UnitPrice']

    # You can extend preprocessing here if needed (e.g., one-hot encoding)
    return new_data

# Step 3: Predict
@step
def make_predictions(model, test_data: pd.DataFrame):
    # Assuming your model uses these 3 features
    X = test_data[['Quantity', 'UnitPrice', 'TotalPrice']]
    predictions = model.predict(X)
    print("Predictions:", predictions)
    return predictions

# Step 4: Define pipeline
@pipeline
def inference_pipeline(
    load_model_step,
    preprocess_data_step,
    make_predictions_step
):
    model = load_model_step()
    test_data = preprocess_data_step()
    make_predictions_step(model=model, test_data=test_data)

# Step 5: Instantiate and run pipeline
if __name__ == "__main__":
    inference_pipeline_instance = inference_pipeline(
        load_model_step=load_model(),
        preprocess_data_step=preprocess_data(),
        make_predictions_step=make_predictions()
    )
    inference_pipeline_instance.run()
