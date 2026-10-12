"""Clean spoken English before machine translation.

Speech transcripts keep hesitations ("uh", "um") and false starts
("I think I think"). Translation models treat them as content and often
produce nonsense for them, so they are removed from the text that is sent
to the translator. The English shown on screen is left as transcribed.
"""

import re

_FILLERS = re.compile(r"\b(?:mm-hmm|uh-huh|uh+|um+|uhm|erm|hmm+|mm+|mhm)\b[,.]?", re.IGNORECASE)
# The same 1-3 word phrase repeated back to back: "and and", "I think I think"
_REPEATS = re.compile(r"\b((?:[\w']+\s+){0,2}[\w']+)(?:[\s,]+\1\b)+", re.IGNORECASE)


def clean_for_translation(text: str) -> str:
    text = _FILLERS.sub(" ", text)
    text = _REPEATS.sub(r"\1", text)
    text = re.sub(r"\s+([,.?!])", r"\1", text)      # space before punctuation
    text = re.sub(r"([,.?!])(?:\s*[,.])+", r"\1", text)  # ", ," or ". ," left behind
    text = re.sub(r"\s+", " ", text).strip().lstrip(",. ")
    # The same whole sentence repeated: "Okay. Okay. Okay." -> "Okay."
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for i, s in enumerate(sentences) if i == 0 or s.lower() != sentences[i - 1].lower()]
    return " ".join(kept)
