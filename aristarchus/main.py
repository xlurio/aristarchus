from __future__ import annotations
from collections.abc import Iterator, Sequence
import pathlib
import statistics
from typing import TYPE_CHECKING, Literal
import spacy
import argparse
import sys
from collections import Counter

if TYPE_CHECKING:
    from spacy.tokens.doc import Doc
    from spacy.tokens.token import Token


MIN_COUNT_NOUNS_N_VERBS = 3
MIN_COUNT_ADVS_N_ADJS = 1
GRAMATICAL_FUNCTIONS = Literal["NOUN", "VERB", "ADJ", "ADV"]


class TextEditorNamespace(argparse.Namespace):
    file: pathlib.Path


def _format_word_counts_with_examples(
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


def _process_tokens_for_docs_n_gramatical_functions_with_examples(
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


def _detect_passive_voice(tokens: list[Token]) -> bool:
    """Detect if a sentence uses passive voice."""
    return any(t.dep_ == "nsubjpass" for t in tokens)


def _extract_sentence_opening(tokens: list[Token]) -> str:
    """Extract the opening words of a sentence."""
    if len(tokens) >= 2:
        return " ".join([tokens[0].lemma_, tokens[1].lemma_]).lower()
    elif tokens:
        return tokens[0].lemma_.lower()
    return ""


def _analyze_sentences(
    documents: list[Doc],
) -> tuple[list[int], int, Counter, int, set, list[str]]:
    """Analyze sentences for length, passive voice, openings, tokens, and unique lemmas."""
    sentence_lengths = []
    passive_sentence_texts = []
    openings = Counter()
    total_tokens = 0
    unique_lemmas = set()

    for doc in documents:
        for sent in doc.sents:
            tokens = [t for t in sent if not t.is_punct and not t.is_space]
            sentence_lengths.append(len(tokens))
            total_tokens += len(tokens)
            unique_lemmas.update([t.lemma_.lower() for t in tokens if t.is_alpha])

            if _detect_passive_voice(tokens):
                passive_sentence_texts.append(sent.text.strip())

            opening = _extract_sentence_opening(tokens)
            if opening:
                openings[opening] += 1

    return (
        sentence_lengths,
        openings,
        total_tokens,
        unique_lemmas,
        passive_sentence_texts,
    )


def _format_repeated_openings(openings: Counter) -> str:
    """Format repeated sentence openings."""
    result = "  Repeated sentence openings:\n"
    for opening, count in openings.most_common():
        if count > 2:
            result += f"    {opening}: {count}\n"
    result += "\n"
    return result


def _compute_stylistic_metrics(documents: list[Doc]) -> str:
    """Compute sentence-level stylistic statistics and readability metrics."""
    sentence_lengths, openings, total_tokens, unique_lemmas, passive_sentence_texts = (
        _analyze_sentences(documents)
    )

    avg_len = statistics.mean(sentence_lengths) if sentence_lengths else 0
    var_len = statistics.pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
    lexical_div = len(unique_lemmas) / total_tokens if total_tokens else 0

    result = "Stylistic Metrics:\n"
    result += f"  Avg. sentence length: {avg_len:.2f} (10-25) tokens\n"
    result += f"  Sentence length variation: {var_len:.2f} (approximate the avg)\n"
    result += f"  Lexical diversity: {lexical_div:.3f} (0.4-0.6)\n\n"

    if passive_sentence_texts:
        result += "Passive voice sentences:\n"
        for sentence in passive_sentence_texts:
            result += f"    • {sentence}\n"
        result += "\n"

    result += _format_repeated_openings(openings)

    return result


def edit_fiction(file_path: pathlib.Path) -> str:
    nlp = spacy.load("pt_core_news_sm")
    paragraphs = [
        paragraph.replace("\n", "").strip()
        for paragraph in file_path.read_text(encoding="utf-8").split("\n\n")
        if paragraph.strip()
    ]
    documents = list(nlp.pipe(paragraphs))

    nouns_n_verbs_lemmas, nouns_n_verbs_examples = (
        _process_tokens_for_docs_n_gramatical_functions_with_examples(
            iter(documents), ["NOUN", "VERB"]
        )
    )
    nouns_n_verbs_counter = Counter(nouns_n_verbs_lemmas)

    adverbs_n_adjectives_lemmas, adverbs_n_adjectives_examples = (
        _process_tokens_for_docs_n_gramatical_functions_with_examples(
            iter(documents), ["ADV", "ADJ"]
        )
    )
    adverbs_n_adjectives_counter = Counter(adverbs_n_adjectives_lemmas)

    result = _compute_stylistic_metrics(documents)
    result += _format_word_counts_with_examples(
        "Nouns and Verbs",
        nouns_n_verbs_counter,
        nouns_n_verbs_examples,
        MIN_COUNT_NOUNS_N_VERBS,
    )
    result += _format_word_counts_with_examples(
        "Adverbs and Adjectives",
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
    args = parser.parse_args(sys.argv[1:], namespace=TextEditorNamespace())

    print(edit_fiction(args.file))


if __name__ == "__main__":
    run()
