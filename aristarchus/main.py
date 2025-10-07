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
LANGUAGES_SUPPORTED = Literal["pt", "en"]
EXCLUDED_TAGS = ["DET"]  # Determiners are often tagged as adjectives in some models
PORTUGUESE_EXCLUDED_ADV_ADJ = {
    "algum",
    "alguma",
    "algumas",
    "alguns",
    "anterior",
    "aquela",
    "aquelas",
    "aquele",
    "aqueles",
    "essa",
    "essas",
    "esse",
    "esses",
    "esta",
    "estas",
    "este",
    "estes",
    "meu",
    "meus",
    "minha",
    "minhas",
    "muita",
    "muitas",
    "muito",
    "muitos",
    "nenhum",
    "nenhuma",
    "nenhumas",
    "nenhuns",
    "nossa",
    "nossas",
    "nosso",
    "nossos",
    "outra",
    "outras",
    "outro",
    "outros",
    "pouca",
    "poucas",
    "pouco",
    "poucos",
    "primeiro",
    "próximo",
    "quaisquer",
    "qualquer",
    "segunda",
    "seu",
    "seus",
    "sua",
    "suas",
    "tanta",
    "tantas",
    "tanto",
    "tantos",
    "terceiro",
    "teu",
    "teus",
    "toda",
    "todas",
    "todo",
    "todos",
    "tua",
    "tuas",
    "último",
    "vossa",
    "vossas",
    "vosso",
    "vossos",
}
ENGLISH_EXCLUDED_ADV_ADJ = {
    "a",
    "all",
    "an",
    "another",
    "any",
    "both",
    "different",
    "double",
    "each",
    "either",
    "enough",
    "every",
    "few",
    "first",
    "her",
    "his",
    "its",
    "last",
    "least",
    "less",
    "little",
    "many",
    "more",
    "most",
    "much",
    "my",
    "neither",
    "next",
    "one",
    "other",
    "our",
    "previous",
    "same",
    "second",
    "several",
    "single",
    "some",
    "such",
    "that",
    "the",
    "their",
    "these",
    "third",
    "this",
    "those",
    "three",
    "triple",
    "two",
    "your",
}
EXCLUDED_MENTE_ADVERBS = {
    "somente",
}
EXCLUDED_LY_ADVERBS = {
    "absolutely",
    "actually",
    "basically",
    "certainly",
    "completely",
    "currently",
    "definitely",
    "early",
    "entirely",
    "exactly",
    "finally",
    "formerly",
    "fully",
    "generally",
    "gradually",
    "immediately",
    "initially",
    "likely",
    "normally",
    "only",
    "originally",
    "perfectly",
    "possibly",
    "previously",
    "probably",
    "quickly",
    "really",
    "recently",
    "simply",
    "slowly",
    "suddenly",
    "totally",
    "typically",
    "usually",
}


class TextEditorNamespace(argparse.Namespace):
    file: pathlib.Path
    language: LANGUAGES_SUPPORTED


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


def _is_adverb_of_manner(token: Token) -> bool:
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


def _is_adjective_of_quality(token: Token) -> bool:
    """Check if an adjective is a quality adjective (descriptive, not demonstrative/possessive)."""
    if token.pos_ != "ADJ":
        return False

    excluded_lemmas = PORTUGUESE_EXCLUDED_ADV_ADJ | ENGLISH_EXCLUDED_ADV_ADJ

    return (
        token.tag_ not in EXCLUDED_TAGS and token.lemma_.lower() not in excluded_lemmas
    )


def _process_manner_adverbs_and_quality_adjectives_with_examples(
    docs: Iterator[Doc],
) -> tuple[Iterator[str], dict[str, list[str]]]:
    word_examples = {}
    lemmas = []

    for doc in docs:
        for token in doc:
            if _is_adverb_of_manner(token) or _is_adjective_of_quality(token):
                lemma = token.lemma_
                original_word = token.text.lower()

                if lemma not in word_examples:
                    word_examples[lemma] = []
                word_examples[lemma].append(original_word)
                lemmas.append(lemma)

    return iter(lemmas), word_examples


def _analyze_word_classes(documents: list[Doc]) -> tuple[int, int]:
    """Count nouns+verbs vs manner adverbs+quality adjectives."""
    nouns_verbs_count = 0
    adjs_advs_count = 0

    for doc in documents:
        for token in doc:
            if token.pos_ in ["NOUN", "VERB"]:
                nouns_verbs_count += 1
            elif _is_adverb_of_manner(token) or _is_adjective_of_quality(token):
                adjs_advs_count += 1

    return nouns_verbs_count, adjs_advs_count


def _compute_stylistic_metrics(documents: list[Doc]) -> str:
    """Compute sentence-level stylistic statistics and readability metrics."""
    sentence_lengths, openings, total_tokens, unique_lemmas, passive_sentence_texts = (
        _analyze_sentences(documents)
    )

    nouns_verbs_count, adjs_advs_count = _analyze_word_classes(documents)

    avg_len = statistics.mean(sentence_lengths) if sentence_lengths else 0
    var_len = statistics.pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
    var_len_percentage = (var_len / avg_len * 100) if avg_len else 0
    lexical_div = len(unique_lemmas) / total_tokens if total_tokens else 0

    # Calculate nouns+verbs to adjectives+adverbs ratio
    if adjs_advs_count > 0:
        nv_to_aa_ratio = nouns_verbs_count / adjs_advs_count
    else:
        nv_to_aa_ratio = float("inf") if nouns_verbs_count > 0 else 0

    result = "Stylistic Metrics:\n"
    result += f"  Avg. sentence length: {avg_len:.2f} (10-25) tokens\n"
    result += f"  Sentence length variation: {var_len:.2f} ({var_len_percentage:.2f}%) (approximate the avg)\n"
    result += f"  Lexical diversity: {lexical_div:.3f} (0.4-0.6)\n"
    result += f"  Nouns+Verbs to Adjectives+Adverbs ratio: {nv_to_aa_ratio:.2f} (>3.5 for clear prose)\n\n"

    if passive_sentence_texts:
        result += "Passive voice sentences:\n"
        for sentence in passive_sentence_texts:
            result += f"    • {sentence}\n"
        result += "\n"

    result += _format_repeated_openings(openings)

    return result


def edit_fiction(file_path: pathlib.Path, language: LANGUAGES_SUPPORTED = "pt") -> str:
    nlp = spacy.load("pt_core_news_sm" if language == "pt" else "en_core_web_sm")
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
        _process_manner_adverbs_and_quality_adjectives_with_examples(iter(documents))
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
