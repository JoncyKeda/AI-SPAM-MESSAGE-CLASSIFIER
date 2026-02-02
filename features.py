"""
Feature extraction module for spam message classification.
Uses TF-IDF for better text representation.
"""

from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features(text_data: List[str]) -> Tuple:
    """
    Convert text data into numerical features using TF-IDF.

    Args:
        text_data (List[str]): List of cleaned text messages

    Returns:
        Tuple: Feature matrix and fitted vectorizer
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    features = vectorizer.fit_transform(text_data)

    return features, vectorizer


def main() -> None:
    """
    Test TF-IDF feature extraction.
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
