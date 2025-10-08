from __future__ import annotations
import argparse
from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from pathlib import Path


GRAMATICAL_FUNCTIONS = Literal["NOUN", "VERB", "ADJ", "ADV"]
LANGUAGES_SUPPORTED = Literal["pt", "en"]


class TextEditorNamespace(argparse.Namespace):
    file: Path
    language: LANGUAGES_SUPPORTED
