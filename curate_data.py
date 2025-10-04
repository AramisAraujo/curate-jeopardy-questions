# Import libraries and dependencies
import re
import string
import nltk
import pandas as pd
from pandarallel import pandarallel
from nltk.corpus import words, wordnet
from nltk.tokenize import wordpunct_tokenize
import spacy


# Initialize pandarallel paralelism
pandarallel.initialize(progress_bar=True)

# Loading NLTK models
nltk.download("words")
nltk.download("wordnet")
nltk.download("punkt")

ENGLISH_WORDS = set(word.lower() for word in words.words())
ENGLISH_WORDS.update(word.lower() for word in wordnet.words())


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
if spacy.prefer_gpu():
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


# Dealing with non-English words

def contains_non_english_words(text: str, min_ratio: float = 0.8) -> bool:
    """
    Determines if text contains non-English words by comparing tokens
    with a known English vocabulary.

    Args:
        text (str): Input text.
        min_ratio (float): Minimum ratio of tokens that must appear in
                           the English dictionary for the text to be
                           considered English.

    Returns:
        bool: True if fewer than 'min_ratio' of tokens are recognized
              English words, indicating likely presence of non-English words.
    """

    if not isinstance(text, str):
        return False

    text = text.translate(PUNCT_DIGIT_TABLE).lower()
    tokens = [token for token in wordpunct_tokenize(text) if len(token) > 1]

    if not tokens:
        return False

    english_words_count = sum(1 for token in tokens if token in ENGLISH_WORDS)
    ratio = english_words_count / len(tokens)

    return ratio < min_ratio


def filter_non_english_words(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters dataset rows containing non-English words using a wordlist approach.

    Args:
        df (pd.DataFrame): Input dataframe with a 'text' column.

    Returns:
        pd.DataFrame: Subset of rows likely containing non-English text.
    """

    filtered_df = df[df["text"].parallel_apply(contains_non_english_words)]
    return filtered_df


# Main pipeline


def curate_datasets(df: pd.DataFrame, fast_mode: bool = False, strict_stratification: bool = True) -> None:

    """
    Execute the full curation process to generate three NER validation subsets.

    Args:
        df (pd.DataFrame): Input dataframe with 'question' and 'answer' columns.
        fast_mode (bool): If True, uses regex-only detection for numbers to accelerate processing.
        strict_stratification (bool): If True, stratifies classes without duplicates within themselves.

    """

    # Keeping a stable reference to the original dataset
    df = df.copy()
    df["original_index"] = df.index

    # Combining question + answer for richer context
    df["text"] = df["question"].fillna("") + " " + df["answer"].fillna(" ")

    # Step 1 : Filtering for numbers and numeric phrases

    df_numbers = filter_numbers(df, fast_mode=fast_mode)

    # Step 2: Filtering phrases with non-English words

    if strict_stratification:
        used_ids = set(df_numbers.index)
        df_non_english = df[~df.index.isin(used_ids)]
        df_non_english = filter_non_english_words(df_non_english)
    else:
        df_non_english = filter_non_english_words(df)

    # Summary of the subset of datasets:

    print("\nSummary of unique matches:")
    print(f"Phrases containing numbers: {len(df_numbers)}")
    print(f"Phrases non-English words: {len(df_non_english)}")


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

    use_fast_mode = False
    use_strict_stratification = False

    curate_datasets(df, fast_mode=use_fast_mode, strict_stratification=use_strict_stratification)
