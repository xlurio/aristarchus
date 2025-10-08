from __future__ import annotations
import pathlib
import spacy
import argparse
import sys
from collections import Counter

from aristarchus.constants import (
    MIN_COUNT_ADVS_N_ADJS,
    MIN_COUNT_NOUNS_N_VERBS,
)
from aristarchus.types import (
    LANGUAGES_SUPPORTED,
    TextEditorNamespace,
)
from aristarchus.metrics import compute_stylistic_metrics
from aristarchus.word_processing import (
    format_word_counts_with_examples,
    process_tokens_for_docs_n_gramatical_functions_with_examples,
    process_manner_adverbs_and_quality_adjectives_with_examples,
)


def edit_fiction(file_path: pathlib.Path, language: LANGUAGES_SUPPORTED = "pt") -> str:
    nlp = spacy.load("pt_core_news_sm" if language == "pt" else "en_core_web_sm")
    paragraphs = [
        paragraph.replace("\n", "").strip()
        for paragraph in file_path.read_text(encoding="utf-8").split("\n\n")
        if paragraph.strip()
    ]
    documents = list(nlp.pipe(paragraphs))

    nouns_n_verbs_lemmas, nouns_n_verbs_examples = (
        process_tokens_for_docs_n_gramatical_functions_with_examples(
            iter(documents), ["NOUN", "VERB"]
        )
    )
    nouns_n_verbs_counter = Counter(nouns_n_verbs_lemmas)

    adverbs_n_adjectives_lemmas, adverbs_n_adjectives_examples = (
        process_manner_adverbs_and_quality_adjectives_with_examples(iter(documents))
    )
    adverbs_n_adjectives_counter = Counter(adverbs_n_adjectives_lemmas)

    result = compute_stylistic_metrics(documents)
    result += format_word_counts_with_examples(
        "Nouns and Verbs",
        nouns_n_verbs_counter,
        nouns_n_verbs_examples,
        MIN_COUNT_NOUNS_N_VERBS,
    )
    result += format_word_counts_with_examples(
        "Manner Adverbs and Quality Adjectives",
        adverbs_n_adjectives_counter,
        adverbs_n_adjectives_examples,
        MIN_COUNT_ADVS_N_ADJS,
    )

    return result


def run():
    parser = argparse.ArgumentParser(description="Edit your fiction")
    parser.add_argument(
        "file", type=pathlib.Path, help="The file containing the text to edit"
    )
    parser.add_argument(
        "--language", "-l", type=str, default="pt", help="Language of the text"
    )
    args = parser.parse_args(sys.argv[1:], namespace=TextEditorNamespace())

    print(edit_fiction(args.file, language=args.language))


if __name__ == "__main__":
    run()
