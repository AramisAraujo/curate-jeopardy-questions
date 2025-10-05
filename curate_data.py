# Import libraries and dependencies
import re
from bs4 import BeautifulSoup
import string
import nltk
import pandas as pd
from pandarallel import pandarallel
from nltk.corpus import words, wordnet
from nltk.tokenize import wordpunct_tokenize
from spacy import prefer_gpu
from wordfreq import top_n_list
import spacy
import logging

# Configure logging

logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for more details
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",)

logger = logging.getLogger(__name__)

# Initialize pandarallel paralelism
pandarallel.initialize(progress_bar=True)

# Loading NLTK models
nltk.download("words")
nltk.download("wordnet")
nltk.download("punkt")

ENGLISH_WORDS = set(word.lower() for word in words.words())
ENGLISH_WORDS.update(word.lower() for word in wordnet.words())
ENGLISH_WORDS.update(top_n_list("en", 50_000))


# Compile regex patterns
PUNCT_DIGIT_TABLE = str.maketrans("", "", string.punctuation + string.digits)

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
                "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
                "hundred", "thousand", "million", "billion"]

DIGIT_PATTERN = re.compile(r"\b\d+([.,]\d+)?\b")
NUMBER_WORD_PATTERN = re.compile(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)


# Loads a lightweight spaCy model
if prefer_gpu():
    # Loads a model optimized for GPU acceleration
    spacy_model = spacy.load("en_core_web_trf", disable=["parser", "lemmatizer"])
else:
    # Loads a model devised to run without GPU acceleration
    spacy_model = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])


# Handling HTML elements in text
def clean_text(text: str) -> str:
    """
    Cleans raw text by removing HTML tags and URLs.

    Args:
        text (str): The input string, which may contain HTML or URLs.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    
    # Using BeautifulSoup to remove HTML tags
    # The .get_text() method extracts all the human-readable text
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Using regex to remove URLs
    # This pattern looks for http/https and www prefixes
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    return text.strip()


# Handling numbers in text

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


# Handling non-English words

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


# Handling unusual proper nouns

def has_unusual_proper_noun(text: str) -> bool:
    """
    Identify whether text contains unusual proper nouns, based on NER.

    Detects entities such as PERSON, ORG, GPE, LOC, or WORK_OF_ART
    that are not common English words.

    Args:
        text (str): Input text.

    Returns:
        bool: True if text contains a proper noun not in the English
              dictionary, False otherwise.
    """
    if not isinstance(text, str) or len(text) < 3:
        return False
    doc = spacy_model(text)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART"}:
            token_lower = ent.text.strip().lower()
            if len(token_lower) > 2 and token_lower not in ENGLISH_WORDS:
                return True
    return False


def filter_proper_nouns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset rows containing unusual or rare proper nouns.

    Args:
        df (pd.DataFrame): Input dataframe with a 'text' column.

    Returns:
        pd.DataFrame: Subset of rows containing unusual proper nouns.
    """

    filtered_df = df[df["text"].parallel_apply(has_unusual_proper_noun)]
    return filtered_df

# Main pipeline

def curate_datasets(df: pd.DataFrame, fast_mode: bool = False, strict_stratification: bool = True, random_state: int = 42) -> None:

    """
    Execute the full curation process to generate three NER validation subsets.

    Adds 'original_index' to preserve reference to the source dataset.

    Processes:
      1. Add 'original_index' to map curated samples to source file rows.
      2. Combines 'question' and 'answer' text into a unified field.
      3. Filter entries into three strata:
         - Numeric entities (numbers, money, dates, etc.)
         - Non-English words
         - Unusual proper nouns
      4. Optionally removes duplicates between strata.
      5. Randomly sample up to 1,000 examples from each.
      6. Save as JSONL files.

    Args:
        df (pd.DataFrame): Input dataframe with 'question' and 'answer' columns.
        fast_mode (bool): If True, uses regex-only detection for numbers to accelerate processing.
        strict_stratification (bool): If True, stratifies classes without duplicates within themselves.
        random_state (int): Random state seed for reproducibility.

    Outputs:
        subset_numbers.jsonl, subset_non_english.jsonl, subset_proper_nouns.jsonl

    """

    # Keeping a stable reference to the original dataset

    df = df.copy()
    df["original_index"] = df.index

    # Combining question + answer for richer context
    logger.info("\nApplying data cleaning steps")
    df["question_clean"] = df["question"].parallel_apply(clean_text)
    df["answer_clean"] = df["answer"].parallel_apply(clean_text)

    df["text"] = df["question_clean"].fillna("") + " " + df["answer_clean"].fillna(" ")

    # Step 1 : Filtering for numbers and numeric phrases

    logger.info("\nFiltering phrases containing numbers")
    df_numbers = filter_numbers(df, fast_mode=fast_mode)

    # Step 2: Filtering phrases with non-English words

    logger.info("\nFiltering phrases containing non-English words")
    if strict_stratification:
        used_ids = set(df_numbers.index)
        df_non_english = df[~df.index.isin(used_ids)]
        df_non_english = filter_non_english_words(df_non_english)
    else:
        df_non_english = filter_non_english_words(df)

    # Step 3: Filtering phrases with unusual proper nouns

    logger.info("\nFiltering phrases containing unusual proper nouns")
    if strict_stratification:
        used_ids |= set(df_non_english.index)
        df_proper = df[~df.index.isin(used_ids)]
        df_proper = filter_proper_nouns(df_proper)
    else:
        df_proper = filter_proper_nouns(df)

    # Summary of the subset of datasets:

    logger.info("\nSummary of matches:")
    logger.info(f"Examples with phrases containing numbers: {len(df_numbers)}")
    logger.info(f"Examples with phrases containing non-English words: {len(df_non_english)}")
    logger.info(f"Examples with phrases containing unusual proper nouns: {len(df_proper)}")

    # Saves 1000 samples per stratum, keeping original_index

    logger.info("Saving sample strata dataset..")

    columns_to_save = ["original_index", "question", "answer", "text",
                       "category", "value", "round", "show_number", "air_date"]

    df_numbers.sample(min(1000, len(df_numbers)), random_state=random_state)[columns_to_save]\
        .to_json("outputs/subset_numbers.jsonl", orient="records", lines=True, force_ascii=False)

    df_non_english.sample(min(1000, len(df_non_english)), random_state=random_state)[columns_to_save]\
        .to_json("outputs/subset_non_english.jsonl", orient="records", lines=True, force_ascii=False)

    df_proper.sample(min(1000, len(df_proper)), random_state=random_state)[columns_to_save]\
        .to_json("outputs/subset_proper_nouns.jsonl", orient="records", lines=True, force_ascii=False)
    
    logger.info("Pipeline concluded.")


# Entry point

if __name__ == "__main__":
    """
    Entry point for direct script execution.

    Loads the JEOPARDY_QUESTIONS1.json dataset, validates columns,
    and executes the curation pipeline with the chosen performance mode and stratification strategy.
    """

    df = pd.read_json("questions/JEOPARDY_QUESTIONS1.json")
    df.columns = [col.lower() for col in df.columns]
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("Dataset must contain both 'question' and 'answer' columns!")

    use_fast_mode = False
    use_strict_stratification = False
    seed = 42

    curate_datasets(df, fast_mode=use_fast_mode,
                     strict_stratification=use_strict_stratification,
                     random_state=seed)
