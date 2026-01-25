"""
Main integration script for the AI Spam Message Classifier.
"""

import os
import numpy as np
from load_data import load_data
from preprocess import clean_text
from features import extract_features
from model import train_model
from evaluate import evaluate_model
from predict import predict_message
from model_io import save_model, load_model


MODEL_PATH = "spam_model.pkl"


def main() -> None:
    """
    Run the complete spam classification pipeline.
    """
    # Step 1: Load data
    data = load_data("../data/spam.csv")

    if data is None:
        print("Failed to load data. Exiting.")
        return

    # Step 2: Preprocess text
    data["cleaned_message"] = data["message"].apply(clean_text)

    # Step 3: Feature extraction
    features, vectorizer = extract_features(
        data["cleaned_message"].tolist()
    )

    # Step 4: Train model
    labels = np.array(data["label"])
    model, x_test, y_test = train_model(features, labels)

    # Step 5: Evaluate model
    evaluate_model(model, x_test, y_test)

    # Step 6: Save model
    save_model(model, vectorizer, MODEL_PATH)

    # Step 7: Load model
    loaded = load_model(MODEL_PATH)
    loaded_model = loaded["model"]
    loaded_vectorizer = loaded["vectorizer"]

    # Step 8: Predict new message using loaded model
    new_message = "Congratulations! You won a free lottery prize."
    result = predict_message(
        loaded_model,
        loaded_vectorizer,
        new_message
    )

    print("\nNew Message Prediction (Loaded Model)")
    print("-----------------------------------")
    print("Message :", new_message)
    print("Result  :", result)


if __name__ == "__main__":
    main()
