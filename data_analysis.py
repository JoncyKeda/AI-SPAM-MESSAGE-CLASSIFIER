"""
Basic dataset analysis for the spam message classifier.
"""

from load_data import load_data
from config import DATA_PATH


def analyze_dataset() -> None:
    """
    Perform simple analysis on the dataset.
    """
    data = load_data(DATA_PATH)

    if data is None:
        print("Failed to load dataset.")
        return

    total_messages = len(data)
    spam_count = (data["label"] == 1).sum()
    not_spam_count = (data["label"] == 0).sum()

    print("Dataset Analysis")
    print("----------------")
    print(f"Total messages : {total_messages}")
    print(f"Spam messages  : {spam_count}")
    print(f"Not spam msgs  : {not_spam_count}")


if __name__ == "__main__":
    analyze_dataset()
