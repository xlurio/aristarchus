from __future__ import annotations
import statistics
from typing import TYPE_CHECKING

import textstat

from aristarchus.text_analysis import analyze_sentences
from aristarchus.word_processing import analyze_word_classes

if TYPE_CHECKING:
    from spacy.tokens.doc import Doc


def _compute_readability_metrics(full_text: str) -> str:
    """Compute readability indices with ideal ranges."""
    result = "Readability Indices (ideal ranges in parentheses):\n"
    try:
        readability_metrics = {
            "Flesch Reading Ease": (
                textstat.flesch_reading_ease(full_text),
                "80-90 (easy to read fiction)",
            ),
            "Flesch-Kincaid Grade": (
                textstat.flesch_kincaid_grade(full_text),
                "4-5 (typical fiction range)",
            ),
            "Gunning Fog Index": (
                textstat.gunning_fog(full_text),
                "< 7 (plain, fluent prose)",
            ),
            "SMOG Index": (textstat.smog_index(full_text), "< 8 (good readability)"),
            "Coleman-Liau Index": (
                textstat.coleman_liau_index(full_text),
                "5-7 (smooth general prose)",
            ),
            "Automated Readability Index": (
                textstat.automated_readability_index(full_text),
                "4-6",
            ),
        }

        for name, (value, ideal) in readability_metrics.items():
            result += f"  {name}: {value:.2f} — ideal {ideal}\n"

    except Exception as e:
        result += f"  [!] Could not compute readability metrics: {e}\n"

    result += "\n"
    return result



def compute_stylistic_metrics(documents: list[Doc]) -> str:
    """Compute sentence-level stylistic statistics and readability metrics."""
    sentence_lengths, openings, total_tokens, unique_lemmas, passive_sentence_texts = (
        analyze_sentences(documents)
    )

    nouns_verbs_count, adjs_advs_count = analyze_word_classes(documents)

    full_text = "\n".join([doc.text for doc in documents])
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
    result += f"  Avg. sentence length: {avg_len:.2f} (9-11) tokens\n"
    result += f"  Sentence length variation: {var_len:.2f} ({var_len_percentage:.2f}%) (7-8)\n"
    result += f"  Lexical diversity: {lexical_div:.3f} (0.2-0.3)\n"
    result += f"  Nouns+Verbs to Adjectives+Adverbs ratio: {nv_to_aa_ratio:.2f} (>6 for clear prose)\n\n"

    if passive_sentence_texts:
        result += "Passive voice sentences:\n"
        for sentence in passive_sentence_texts:
            result += f"    • {sentence}\n"
        result += "\n"

    result += _compute_readability_metrics(full_text)

    return result