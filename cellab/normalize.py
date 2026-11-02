"""Text preprocessing: sentence splitting and synonym replacement.

The grammar only knows one spelling for every keyword, so before parsing we
lower-case the sentence and rewrite the common variants to that spelling.
"""

import re
from dataclasses import dataclass

# Order matters: longer phrases must be replaced before their sub-phrases.
SYNONYMS = [
    (r"\bneighbou?rhood\b", "neighbourhood"),
    (r"\bneighbou?rs\b", "neighbours"),
    (r"\bneighbou?r\b", "neighbour"),
    (r"\bvon\s+neumann\b", "vonneumann"),
    (r"\b(turns|changes|transforms)\s+(into|to)\b", "becomes"),
    (r"\b(turns|changes)\b", "becomes"),
    (r"\b(becomes|become)\b", "becomes"),
    (r"\b(remains|stays|stay|remain)\b", "stays"),
    (r"\b(is\s+)?unchanged\b", "stays the same"),
    (r"\b(less|fewer)\s+than\b", "fewer than"),
    (r"\b(greater|more)\s+than\b", "more than"),
    (r"\bmin(imum)?\b", "at least"),
    (r"\bmax(imum)?\b", "at most"),
    (r"\b(wraps?|wrapping|wrapped)(\s+around)?\b", "wrap"),
    (r"\b(toroidal|periodic|circular)\b", "wrap"),
    (r"\b(closed|bounded|fixed)\b", "fixed"),
    (r"\bpercent\b", "%"),
    (r"\b(iterations|generation|iteration|step)\b", "generations"),
]

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_SIZE = re.compile(r"(\d+)\s*(?:x|×|\*|by)\s*(\d+)")
# A sentence ends at a '.' that is not inside a number (0.3) or at a newline.
_SENTENCE_END = re.compile(r"\.(?!\d)|[\n;]")


@dataclass
class Sentence:
    text: str        # original text, shown to the user in error messages
    normalized: str  # what is actually fed to the parser
    line: int        # 1-based line number in the description


def normalize(text: str) -> str:
    s = text.lower().strip()
    s = _SIZE.sub(r"\1 by \2", s)
    for pattern, replacement in SYNONYMS:
        s = re.sub(pattern, replacement, s)
    for word, value in NUMBER_WORDS.items():
        s = re.sub(rf"\b{word}\b", str(value), s)
    s = re.sub(r"(\d)\s*%", r"\1 %", s)
    return re.sub(r"\s+", " ", s).strip()


def split_sentences(description: str) -> list[Sentence]:
    description = re.sub(r"#[^\n]*", "", description)  # comments
    sentences = []
    start = 0
    for match in list(_SENTENCE_END.finditer(description)) + [None]:
        end = match.start() if match else len(description)
        chunk = description[start:end]
        if chunk.strip():
            line = description.count("\n", 0, start + len(chunk) - len(chunk.lstrip())) + 1
            sentences.append(Sentence(chunk.strip(), normalize(chunk), line))
        if match:
            start = match.end()
    return sentences
