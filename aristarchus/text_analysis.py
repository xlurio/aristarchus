from __future__ import annotations
from typing import TYPE_CHECKING
from collections import Counter

if TYPE_CHECKING:
    from spacy.tokens.doc import Doc
    from spacy.tokens.token import Token


def detect_passive_voice(tokens: list[Token]) -> bool:
    """Detect if a sentence uses passive voice."""
    return any(t.dep_ == "nsubjpass" for t in tokens)


def extract_sentence_opening(tokens: list[Token]) -> str:
    """Extract the opening words of a sentence."""
    if len(tokens) >= 2:
        return " ".join([tokens[0].lemma_, tokens[1].lemma_]).lower()
    elif tokens:
        return tokens[0].lemma_.lower()
    return ""


def analyze_sentences(
    documents: list[Doc],
) -> tuple[list[int], Counter, int, set, list[str]]:
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

            if detect_passive_voice(tokens):
                passive_sentence_texts.append(sent.text.strip())

            opening = extract_sentence_opening(tokens)
            if opening:
                openings[opening] += 1

    return (
        sentence_lengths,
        openings,
        total_tokens,
        unique_lemmas,
        passive_sentence_texts,
    )


def format_repeated_openings(openings: Counter) -> str:
    """Format repeated sentence openings."""
    result = "  Repeated sentence openings:\n"
    for opening, count in openings.most_common():
        if count > 2:
            result += f"    {opening}: {count}\n"
    result += "\n"
    return result
