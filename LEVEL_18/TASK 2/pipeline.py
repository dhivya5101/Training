from steps.load_data import load_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from zenml import pipeline

@pipeline
def titanic_pipeline():
    df = load_data()
    df_clean = clean_data(df)
    model = train_model(df_clean)
    evaluate_model(model, df_clean)

if __name__ == "__main__":
    pipe = titanic_pipeline()
    run_response = pipe.run()  
    print(run_response)
