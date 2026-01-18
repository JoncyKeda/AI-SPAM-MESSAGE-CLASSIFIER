"""
Text preprocessing module for spam message classification.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean input text by lowercasing and removing special characters.

    Args:
        text (str): Raw input message

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main() -> None:
    """
    Test the text preprocessing function.
    """
    sample_text = "WIN a FREE Prize now!!!"
    cleaned_text = clean_text(sample_text)
    print("Original:", sample_text)
    print("Cleaned :", cleaned_text)


if __name__ == "__main__":
    main()
