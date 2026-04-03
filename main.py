from src.data_preprocessing import load_and_preprocess
from src.model_training import train_model
from src.prediction import evaluate_model

def main():
    data = load_and_preprocess()
    model, X_test, y_test = train_model(data)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()