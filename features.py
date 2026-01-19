"""
Feature extraction module for spam message classification.
"""

from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer


def extract_features(text_data: List[str]) -> Tuple:
    """
    Convert text data into numerical features using Bag of Words.

    Args:
        text_data (List[str]): List of cleaned text messages

    Returns:
        Tuple: Feature matrix and fitted vectorizer
    """
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(text_data)

    return features, vectorizer


def main() -> None:
    """
    Test the feature extraction function.
    """
    sample_texts = [
        "win free prize now",
        "hey are we meeting today",
        "call me later",
        "free offer just for you"
    ]

    features, vectorizer = extract_features(sample_texts)

    print("Feature matrix shape:", features.shape)
    print("Vocabulary size:", len(vectorizer.vocabulary_))


if __name__ == "__main__":
    main()
