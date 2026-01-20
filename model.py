"""
Model training module for spam message classification.
"""

from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import csr_matrix
import numpy as np


def train_model(
    features: csr_matrix,
    labels: np.ndarray
) -> Tuple[MultinomialNB, csr_matrix, np.ndarray]:
    """
    Train a Naive Bayes model on extracted features.

    Args:
        features (csr_matrix): Feature matrix
        labels (np.ndarray): Target labels

    Returns:
        Tuple: Trained model, test features, test labels
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(x_train, y_train)

    return model, x_test, y_test


def main() -> None:
    """
    Test model training with dummy data.
    """
    # Simple example data
    sample_features = np.array([
        [2, 1, 0],
        [0, 1, 2],
        [1, 0, 1],
        [2, 0, 0]
    ])

    sample_labels = np.array([1, 0, 0, 1])

    model, _, _ = train_model(sample_features, sample_labels)

    print("Model trained successfully:", model)


if __name__ == "__main__":
    main()
