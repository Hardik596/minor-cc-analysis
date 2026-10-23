"""Repair text extracted from BBT PDFs that use the legacy Balaram font.

These PDFs embed two legacy 8-bit fonts:

  1. Balaram, for romanised Sanskrit/Bengali (IAST). Extraction yields Latin-1
     lookalikes -- "Kåñëa" is really "Kṛṣṇa", "Çré" is really "Śrī". This is
     fully recoverable via a codepoint map (`BALARAM_MAP`).

  2. A Bengali display font, for the original verses. Extraction yields
     unrecoverable noise ("ıËp¡‰¬±ı˛œ", "’±ÀıÀ˙"). Those lines are detected and
     dropped -- every such verse has an English translation alongside it, so no
     narrative content is lost.

The map must be applied with `str.translate` in a SINGLE pass, never as chained
`replace` calls: the mapping is not acyclic (`ï`->`ñ` while `ñ`->`ṣ`), so
sequential replacement would turn every `ï` into `ṣ`.
"""

import re
import unicodedata

# Balaram byte -> IAST character. Verified against known words in the corpus:
# Kåñëa/Kṛṣṇa, Çré/Śrī, paëòita/paṇḍita, ajïäta/ajñāta, saìkértana/saṅkīrtana,
# véryataù/vīryataḥ, saàsära/saṁsāra, Öhäkura/Ṭhākura, Ñaòbhuja/Ṣaḍbhuja.
BALARAM_MAP = {
    "ä": "ā", "Ä": "Ā",
    "é": "ī", "É": "Ī",
    "ü": "ū", "Ü": "Ū",
    "å": "ṛ", "Å": "Ṛ",
    "è": "ḷ", "È": "Ḷ",
    "ì": "ṅ", "Ì": "Ṅ",
    "ï": "ñ", "Ï": "Ñ",
    "ö": "ṭ", "Ö": "Ṭ",
    "ò": "ḍ", "Ò": "Ḍ",
    "ë": "ṇ", "Ë": "Ṇ",
    "ç": "ś", "Ç": "Ś",
    "ñ": "ṣ", "Ñ": "Ṣ",
    "à": "ṁ", "À": "Ṁ",
    "ù": "ḥ", "Ù": "Ḥ",
}

_TRANSLATE = str.maketrans(BALARAM_MAP)

# Codepoints emitted by the Bengali display font. None of these belong to the
# Balaram IAST set, so their presence marks a line as unrecoverable verse text.
_BENGALI_NOISE = set(
    "±√˛¬ıø˘Ã˝¡˜Ó˚Œ¸≈ˆ◊Û∂õÊ‰·«œÒ¤¶Î˙û∆ÔÂ¯‡μˇ§∏‚´Í»ƒ‹ÕÈß÷¿¢≥Ø®Ù"
    "ıŒœƒˆˇ˘˙˚˛˜˝"
    # NB: U+2013 EN DASH is deliberately NOT here. It is ordinary punctuation in
    # this edition -- verse-range headings are written "TEXTS 65–66". Including
    # it caused every such heading to be discarded as Bengali: at 11 characters
    # with one flagged character the ratio test (1/11 = 0.09) tripped the 0.06
    # threshold. 54 headings per volume were being lost.
    "‡‰‹‚∂∆≈≥∏√◊"
)

# Running heads / footers that repeat on nearly every page.
_RUNNING_HEAD = re.compile(
    r"^\s*(Ādi|Madhya|Antya|Śrī\s+Caitanya[- ]caritāmṛta)"
    r"[\s.,-]*\d*[\s.,-]*\d*\s*$",
    re.IGNORECASE,
)
_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")


def fix_encoding(text):
    """Map Balaram codepoints onto real IAST characters in one pass."""
    return text.translate(_TRANSLATE)


def is_bengali_noise(line, min_hits=3, ratio=0.06, min_len_for_ratio=20):
    """True if `line` is Bengali display-font output rather than readable text.

    The ratio test is only applied to lines long enough for a ratio to mean
    anything, and always needs at least two flagged characters. On a short line a
    single stray character is 5-10% of it, which was enough to discard legitimate
    headings before this guard existed.
    """
    if not line:
        return False
    hits = sum(1 for ch in line if ch in _BENGALI_NOISE)
    if hits >= min_hits:
        return True
    return (hits >= 2
            and len(line) >= min_len_for_ratio
            and (hits / len(line)) > ratio)


def clean_page(text):
    """Repair encoding and strip page furniture from one page of extracted text.

    Returns a list of kept lines.
    """
    kept = []
    for line in fix_encoding(text).splitlines():
        line = line.strip()
        if not line:
            continue
        if is_bengali_noise(line):
            continue
        if _PAGE_NUMBER.match(line) or _RUNNING_HEAD.match(line):
            continue
        kept.append(line)
    return kept


def unwrap(lines):
    """Rejoin hard-wrapped lines into paragraphs.

    The PDFs wrap mid-sentence at a fixed column, so a line that does not end in
    sentence-final punctuation continues the next one. Without this, NLTK's
    sentence tokeniser splits on the wrap points instead of on real boundaries.
    """
    paragraphs, buf = [], []
    for line in lines:
        # Structural markers (TEXT 21, SYNONYMS, PURPORT) always start fresh.
        if re.match(r"^(TEXT|TEXTS|SYNONYMS|TRANSLATION|PURPORT|CHAPTER)\b", line):
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
            paragraphs.append(line)
            continue
        if line.endswith("-"):  # hyphenated across the wrap
            buf.append(line[:-1])
            continue
        buf.append(line)
        if re.search(r"[.!?][\"'”’)\]]*$", line):
            paragraphs.append(" ".join(buf))
            buf = []
    if buf:
        paragraphs.append(" ".join(buf))
    return paragraphs


# Word-by-word glossary blocks ("mätä—the mother; bhakta-gaëera—of the...").
# These are reference apparatus, not narrative, and would badly skew TF-IDF and
# the word-adjacency graphs if left in.
_SYNONYM_LINE = re.compile(r"—")


def is_synonym_block(paragraph, min_dashes=3):
    return len(_SYNONYM_LINE.findall(paragraph)) >= min_dashes


def normalise(text):
    """Final tidy-up: NFC, curly quotes -> straight, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"'))
    return re.sub(r"[ \t]+", " ", text).strip()
