from __future__ import annotations
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING
from collections import Counter

from aristarchus.constants import (
    ENGLISH_EXCLUDED_ADV_ADJ,
    EXCLUDED_LY_ADVERBS,
    EXCLUDED_MENTE_ADVERBS,
    EXCLUDED_TAGS,
    PORTUGUESE_EXCLUDED_ADV_ADJ,
)
from aristarchus.types import GRAMATICAL_FUNCTIONS

if TYPE_CHECKING:
    from spacy.tokens.doc import Doc
    from spacy.tokens.token import Token


def format_word_counts_with_examples(
    title: str,
    counter: Counter,
    word_examples: dict[str, list[str]],
    min_count: int = 0,
) -> str:
    result = f"{title}:\n"
    for word, count in counter.items():
        if count > min_count:
            examples = ", ".join(sorted(set(word_examples[word])))
            result += f"  {word}: {count} → {examples}\n"
    result += "\n"
    return result


def process_tokens_for_docs_n_gramatical_functions_with_examples(
    docs: Iterator[Doc], gram_funcs: Sequence[GRAMATICAL_FUNCTIONS]
) -> tuple[Iterator[str], dict[str, list[str]]]:
    word_examples = {}
    lemmas = []

    for doc in docs:
        for token in doc:
            if token.pos_ in gram_funcs:
                lemma = token.lemma_
                original_word = token.text.lower()

                if lemma not in word_examples:
                    word_examples[lemma] = []
                word_examples[lemma].append(original_word)
                lemmas.append(lemma)

    return iter(lemmas), word_examples


def is_adverb_of_manner(token: Token) -> bool:
    """Check if an adverb is an adverb of manner."""
    if token.pos_ != "ADV":
        return False

    # Portuguese: ends with -mente
    if token.text.lower().endswith("mente"):
        return token.lemma_.lower() not in EXCLUDED_MENTE_ADVERBS

    # English: typically ends with -ly (but exclude some common non-manner adverbs)
    if token.text.lower().endswith("ly"):
        return token.lemma_.lower() not in EXCLUDED_LY_ADVERBS

    return False


def is_adjective_of_quality(token: Token) -> bool:
    """Check if an adjective is a quality adjective (descriptive, not demonstrative/possessive)."""
    if token.pos_ != "ADJ":
        return False

    excluded_lemmas = PORTUGUESE_EXCLUDED_ADV_ADJ | ENGLISH_EXCLUDED_ADV_ADJ

    return (
        token.tag_ not in EXCLUDED_TAGS and token.lemma_.lower() not in excluded_lemmas
    )


def process_manner_adverbs_and_quality_adjectives_with_examples(
    docs: Iterator[Doc],
) -> tuple[Iterator[str], dict[str, list[str]]]:
    word_examples = {}
    lemmas = []

    for doc in docs:
        for token in doc:
            if is_adverb_of_manner(token) or is_adjective_of_quality(token):
                lemma = token.lemma_
                original_word = token.text.lower()

                if lemma not in word_examples:
                    word_examples[lemma] = []
                word_examples[lemma].append(original_word)
                lemmas.append(lemma)

    return iter(lemmas), word_examples


def analyze_word_classes(documents: list[Doc]) -> tuple[int, int]:
    """Count nouns+verbs vs manner adverbs+quality adjectives."""
    nouns_verbs_count = 0
    adjs_advs_count = 0

    for doc in documents:
        for token in doc:
            if token.pos_ in ["NOUN", "VERB"]:
                nouns_verbs_count += 1
            elif is_adverb_of_manner(token) or is_adjective_of_quality(token):
                adjs_advs_count += 1

    return nouns_verbs_count, adjs_advs_count
