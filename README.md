# Jeopardy Dataset Curation for NER Evaluation

This repository contains a Python script to curate structured subsets from the **Jeopardy Questions** dataset for a **Named Entity Recognition (NER)** model evaluation.  
The curated subsets isolate linguistic cases that commonly challenge NER systems.

## Overview

The source dataset (`JEOPARDY_QUESTIONS1.json`) contains approximately **200,000** Jeopardy questions.  
This script produces three balanced subsets (up to 1,000 samples each):

1. **subset_numbers.jsonl**  
   Questions or answers containing numeric expressions (digits, written numbers, money, dates, percentages, etc.).

2. **subset_non_english.jsonl**  
   Questions or answers containing non-English words or mixed-language phrases.

3. **subset_proper_nouns.jsonl**  
   Questions or answers containing unusual or rare proper nouns, such as uncommon people, places, or organizations.

Each entry in the output includes the `original_index` field, which maps back to the source dataset for reproducibility and traceability.

## Features

- Efficient processing of large text datasets using:
  - **spaCy** for Named Entity Recognition (NER)
  - **pandarallel** for CPU parallelization
  - **GPU acceleration** via `spacy.prefer_gpu()` (if available)
- UTF-8 JSONL output for downstream ML workflows
- Fully documented and modular code structure

## Curation Process

The curation process is divided into several steps designed to extract linguistically distinct subsets suitable for testing NER models:

1. **Loading Data**  
   The script loads `JEOPARDY_QUESTIONS1.json` and standardizes column names.  
   Each question and answer pair is merged into a single text field to provide richer linguistic context for entity detection.

2. **Numeric Phrase Detection**  
   Two complementary methods identify questions involving numbers:  
   - **Regex patterns** for explicit digits and written numbers (e.g., “10”, “ten”, “million”).  
   - **spaCy NER** to detect numeric entities such as `CARDINAL`, `MONEY`, `DATE`, `PERCENT`, and `QUANTITY`.  
   The union of both captures both surface and semantic number mentions.

3. **Non-English Phrase Detection**  
   Tokenized words are compared to the NLTK English wordlist (`words` + `wordnet`).  
   Texts with a low ratio of recognized English tokens are classified as containing non-English content.

4. **Unusual Proper Noun Detection**  
   Using spaCy NER, the script identifies entities labeled as `PERSON`, `ORG`, `GPE`, `LOC`, or `WORK_OF_ART`.  
   Entities that are not in the English vocabulary are flagged as unusual or rare proper nouns.

5. **Stratification and Sampling**  
   Each detected stratum may optionally be deduplicated to prevent overlap across categories.  
   Up to 1,000 random examples per category are selected and saved as JSONL files.

6. **Traceability and Outputs**  
   Each row retains its `original_index` from the source dataset, ensuring reproducibility.  
   The final subsets are written to the `outputs/` directory in UTF-8 JSONL format.

This process ensures that each curated subset highlights a specific linguistic phenomenon relevant for evaluating NER robustness, while maintaining a clean, reproducible, and interpretable workflow.

## Curation Results

After analyzing the full dataset of 216,930 questions, the following number of examples were identified for each stratum:

| Stratum                     | Total Examples Found |
|-----------------------------|----------------------|
| Phrases with Numbers        | 105,319              |
| Phrases with Non-English    | 8,519                |
| Phrases with Unusual Nouns  | 129,940              |


## Requirements

Python 3.10 or higher is recommended.

### Python Dependencies

```bash
pip install -r requirements.txt
```

### Optional GPU Support

To enable GPU acceleration:

```bash
# For CUDA 12
pip install -U spacy[cuda12x]

# For CUDA 11
pip install -U spacy[cuda11x]
```

### Download spaCy Models

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
```

## Input

Download and place the source dataset in the same directory:

- [Jeopardy Questions Dataset (Reddit link)](https://www.reddit.com/r/datasets/comments/1uyd0t/200000_jeopardy_questions_in_a_json_file/)
- The file name should be:  
  `JEOPARDY_QUESTIONS1.json`

The JSON file should contain at least the columns:
`question`, `answer`, `category`, `value`, `air_date`, `round`, and `show_number`.

## Usage

Run the curation script directly:

```bash
python curate_data.py
```

The script will:

1. Detect and use GPU if available.
2. Combine question and answer text for richer context.
3. Filter and create three subsets for NER validation.
4. Save the output under the `outputs/` directory.

## Output Files

| File | Description |
|------|--------------|
| `outputs/subset_numbers.jsonl` | Entries with numeric entities or expressions |
| `outputs/subset_non_english.jsonl` | Entries with non-English or mixed-language text |
| `outputs/subset_proper_nouns.jsonl` | Entries with unusual or rare proper nouns |

Each line in these JSONL files represents one curated record.

## Example Output

```json
{
  "original_index": 15327,
  "question": "This Italian city is home to the Leaning Tower.",
  "answer": "Pisa",
  "category": "Geography",
  "show_number": 2431,
  "value": "$400",
  "round": "Jeopardy!",
  "air_date": "1999-03-18",
  "text": "This Italian city is home to the Leaning Tower. Pisa"
}
```

To enable fast mode (for testing), edit the last line of the script:

```python

curate_datasets(df,fast_mode=True, strict_stratification=use_strict_stratification, random_state=seed)
```

To enable strict stratification (no overlap between strata datasets), edit the last line of the script:

```python

curate_datasets(df,fast_mode=use_fast_mode, strict_stratification=True, random_state=seed)
```

