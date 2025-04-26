from zenml import pipeline
from pipeline_steps import load_data, clean_data, train_random_forest, evaluate_model

@pipeline(enable_cache=False)
def hyperparameter_tuning_pipeline():
    df = load_data()
    cleaned_df = clean_data(df)
    best_model = train_random_forest(cleaned_df)
    evaluate_model(best_model, cleaned_df)

if __name__ == "__main__":
    print("Starting Hyperparameter Tuning Pipeline...")
    hyperparameter_tuning_pipeline()
    print("Pipeline complete! Run `mlflow ui` to view results.")
