# Import libraries and dependencies
import re
import string
import pandas as pd
from pandarallel import pandarallel
import spacy
from spacy import prefer_gpu

# Initialize pandarallel paralelism
pandarallel.initialize(progress_bar=True)


# Compile regex patterns
PUNCT_DIGIT_TABLE = str.maketrans("", "", string.punctuation + string.digits)

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
                "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
                "ninety", "hundred", "thousand", "million", "billion"]

DIGIT_PATTERN = re.compile(r"\b\d+([.,]\d+)?\b")
NUMBER_WORD_PATTERN = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)


# Loads a lightweight spaCy model
if prefer_gpu():
    # Loads a model optimized for GPU acceleration
    spacy_model = spacy.load("en_core_web_trf", disable=["parser", "lemmatizer"])
else:
    # Loads a model devised to run without GPU acceleration
    spacy_model = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])


# Dealing with numbers in text


def contains_number_regex(text: str) -> bool:
    """
    Checks if text contains digits or common written number words.

    Args:
        text (str): Input text.

    Returns:
        bool: True if text contains numeric patterns or words such as
              'one', 'hundred', 'million', etc.; False otherwise.
    """

    if not isinstance(text, str):
        return False

    return bool(DIGIT_PATTERN.search(text) or NUMBER_WORD_PATTERN.search(text))


def has_spacy_number_entity(text: str) -> bool:
    """
    Detects numeric-related entities using spaCy NER.

    Recognizes entity types such as CARDINAL, ORDINAL, MONEY, DATE,
    TIME, PERCENT, and QUANTITY.

    Args:
        text (str): Input text.

    Returns:
        bool: True if spaCy finds any numeric-like entities, False otherwise.
    """

    if not isinstance(text, str) or len(text) < 3:
        return False

    target_ents = {"CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT", "DATE", "TIME"}
    doc = spacy_model(text)

    for ent in doc.ents:
        if ent.label_ in target_ents:
            return True

    return False


def filter_numbers(df: pd.DataFrame, fast_mode: bool = False) -> pd.DataFrame:
    """
    Filter dataset rows that contain numeric entities or expressions.

    Combines results from both regex-based and spaCy-based detection
    using a logical union. This ensures coverage of both literal and
    semantic numeric mentions.

    Args:
        df (pd.DataFrame): Input dataframe containing a 'text' column.
        fast_mode (bool): If True, runs only the regex detector for
                          faster, less precise filtering.

    Returns:
        pd.DataFrame: Subset of rows containing numeric-related phrases.
    """

    mask_regex = df["text"].parallel_apply(contains_number_regex)

    if fast_mode:
        combined_mask = mask_regex
    else:
        mask_spacy = df["text"].parallel_apply(has_spacy_number_entity)
        combined_mask = mask_regex | mask_spacy

    return df[combined_mask]


# Main pipeline


def curate_datasets(df: pd.DataFrame, fast_mode: bool = False) -> None:

    # Keeping a stable reference to the original dataset
    df = df.copy()
    df["original_index"] = df.index

    # Combining question + answer for richer context
    df["text"] = df["question"].fillna("") + " " + df["answer"].fillna(" ")

    # Step 1 : Filtering for numbers and numeric phrases

    df_numbers = filter_numbers(df, fast_mode=fast_mode)

    # Summary of the subset of datasets:

    print("\nSummary of unique matches:")
    print(f"Phrases containing numbers: {len(df_numbers)}")


# Entry point


if __name__ == "__main__":
    """
    Entry point for direct script execution.

    Loads the JEOPARDY_QUESTIONS1.json dataset, validates columns,
    and executes the curation pipeline with the chosen performance mode.
    """

    df = pd.read_json("questions/JEOPARDY_QUESTIONS1.json")
    df.columns = [col.lower() for col in df.columns]
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("Dataset must contain both 'question' and 'answer' columns!")

    curate_datasets(df, fast_mode=False)
