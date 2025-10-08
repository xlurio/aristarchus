from __future__ import annotations
import statistics
from typing import TYPE_CHECKING

from aristarchus.text_analysis import analyze_sentences, format_repeated_openings
from aristarchus.word_processing import analyze_word_classes

if TYPE_CHECKING:
    from spacy.tokens.doc import Doc


def compute_stylistic_metrics(documents: list[Doc]) -> str:
    """Compute sentence-level stylistic statistics and readability metrics."""
    sentence_lengths, openings, total_tokens, unique_lemmas, passive_sentence_texts = (
        analyze_sentences(documents)
    )

    nouns_verbs_count, adjs_advs_count = analyze_word_classes(documents)

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

    result += format_repeated_openings(openings)

    return result
