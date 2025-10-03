import pandas as pd
from model import get_models
from preprocess import load_and_preprocess_data
from evaluate import evaluate_model

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data("../data/sample_annotations.csv")

    models = get_models()
    for name, model in models.items():
        print("="*50)
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
