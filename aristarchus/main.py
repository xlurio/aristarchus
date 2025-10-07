from __future__ import annotations
from collections.abc import Iterator, Sequence
import pathlib
from typing import TYPE_CHECKING, Literal
import spacy
import argparse
import sys
from collections import Counter


if TYPE_CHECKING:
    from spacy.tokens.doc import Doc
    from spacy.tokens.token import Token


MAX_COUNT_NOUNS_N_VERBS = 3
GRAMATICAL_FUNCTIONS = Literal["NOUN", "VERB", "ADJ", "ADV"]


class TextEditorNamespace(argparse.Namespace):
    file: pathlib.Path


def _process_tokens_for_doc_n_gramatical_functions(
    doc: Doc, gram_funcs: Sequence[GRAMATICAL_FUNCTIONS]
) -> Iterator[Token]:
    for token in doc:
        if token.pos_ in gram_funcs:
            yield token


def _process_tokens_for_docs_n_gramatical_functions(
    docs: Iterator[Doc], gram_funcs: Sequence[GRAMATICAL_FUNCTIONS]
) -> Iterator[Token]:
    for doc in docs:
        yield from _process_tokens_for_doc_n_gramatical_functions(doc, gram_funcs)


def edit_fiction(file_path: pathlib.Path) -> str:
    nlp = spacy.load("pt_core_news_sm")
    paragraphs = [
        paragraph.replace("\n", "").strip()
        for paragraph in file_path.read_text(encoding="utf-8").split("\n\n")
    ]
    documents = nlp.pipe(paragraphs)

    nouns_n_verbs_counter = Counter(_process_tokens_for_docs_n_gramatical_functions(documents, ["NOUN", "VERB"]))
    adverbs_n_adjectives_counter = Counter(_process_tokens_for_docs_n_gramatical_functions(documents, ["ADV", "ADJ"]))

    result = _format_word_counts(
        "Nouns and Verbs", nouns_n_verbs_counter, MAX_COUNT_NOUNS_N_VERBS
    )
    result += _format_word_counts(
        "Adverbs and Adjectives", adverbs_n_adjectives_counter
    )

    return result


def _format_word_counts(title: str, counter: Counter, min_count: int = 1) -> str:
    result = f"{title}:\n"
    for word, count in counter.items():
        if count >= min_count:
            result += f"  {word}: {count}\n"
    result += "\n"
    return result


def run():
    parser = argparse.ArgumentParser(description="Edit your fiction")
    parser.add_argument(
        "file", type=pathlib.Path, help="The file containing the text to edit"
    )
    args = parser.parse_args(*sys.argv[1:], namespace=TextEditorNamespace())

    print(edit_fiction(args.file))
