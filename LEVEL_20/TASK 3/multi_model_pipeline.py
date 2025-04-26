from zenml import pipeline
from pipeline_steps import load_data, clean_data, train_models, evaluate_models

@pipeline(enable_cache=False)
def multi_model_comparison_pipeline():
    data = load_data()
    cleaned_data = clean_data(data)
    models = train_models(cleaned_data)
    evaluate_models(models, cleaned_data)

if __name__ == "__main__":
    try:
        print("Make sure MLflow tracker is in your ZenML stack.")
        print("Run this command if not: `zenml stack update -e mlflow_tracker`")
        multi_model_comparison_pipeline()
        print("Pipeline completed.")
        print("View in MLflow UI: `mlflow ui`")
        print("View in ZenML dashboard: `zenml dashboard`")
    except Exception as e:
        print(f"Error: {str(e)}")
