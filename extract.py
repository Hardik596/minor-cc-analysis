"""Extract the three Caitanya-caritāmṛta līlās from PDF into clean UTF-8 text.

Extraction is slow (~11k pages), so results are cached under `corpus/`. Delete
that directory to force a re-extract.
"""

import os
import re
import sys

import pymupdf

import balaram

PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "Sri Caitanya Caritamrta")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")

# The three līlās are the units of analysis.
LILAS = {
    "Adi": "Sri Caitanya Caritamrta Adi.pdf",
    "Madhya": "Sri Caitanya Caritamrta Madhya.pdf",
    "Antya": "Sri Caitanya Caritamrta Antya.pdf",
}

# Apparatus that is not narrative prose.
_SKIP = re.compile(
    r"^(TEXT|TEXTS|SYNONYMS|WORD-FOR-WORD MEANINGS|CHAPTER|PURPORT|TRANSLATION)\b"
    r"\s*\d*\s*$"
)

# Inline scriptural cross-references: "[Cc. Madhya 22.83]", "(Cc. Antya 7.11)",
# "Bg. 4.7", "SB 1.2.3". There are ~5,500 of these. They are citation apparatus,
# but NER reads the līlā name as an entity -- left in, "Antya" ranks as the
# second most-mentioned "character" in Antya-līlā.
_CITATION = re.compile(
    r"[\[(]?\s*(?:Cc\.|Bg\.|SB|Śrīmad-Bhāgavatam|Bhāg\.)\s*"
    r"(?:Ādi|Madhya|Antya)?\s*[\d.–-]+\s*[\])]?",
    re.IGNORECASE,
)
# Bare references not prefixed by a work abbreviation.
_BARE_REF = re.compile(r"\b(?:Ādi|Madhya|Antya)\s+\d+\.\d+\b")

# Section headings that mark whose words follow.
_SECTION_HEAD = re.compile(
    r"^(TRANSLATION|PURPORT|SYNONYMS|WORD-FOR-WORD MEANINGS|TEXTS?|CHAPTER)\b"
)

# Any structural marker of the main body.
_BODY_MARKER = re.compile(
    r"^\s*(TEXTS?|SYNONYMS|WORD-FOR-WORD MEANINGS|TRANSLATION|PURPORT)\b", re.M)


def find_body_end(doc):
    """Last page index belonging to the main body.

    Each PDF appends a "Reviews" section that reprints EVERY translation as a
    reference list ("Ādi 13.33 - Lord Caitanya Mahāprabhu lived..."). In the Ādi
    volume this runs from roughly page 1930 to 2438 -- a fifth of the file -- and
    it repeats the translations twice over, so a naive read yields three copies
    of every translated verse.

    The review pages carry none of the body's structural markers, so the last
    page bearing one marks the end of the body.
    """
    last = 0
    for i, page in enumerate(doc):
        if _BODY_MARKER.search(balaram.fix_encoding(page.get_text())):
            last = i
    return last


def extract_lila(name, pdf_path, keep_synonyms=False, progress=True,
                 sections=("TRANSLATION", "PURPORT", "OTHER")):
    """Return cleaned narrative text for one līlā.

    `sections` selects which parts of the edition to keep:

      TRANSLATION  Kṛṣṇadāsa Kavirāja's 16th-century text, in translation
      PURPORT      Bhaktivedanta Swami Prabhupāda's 20th-century commentary
      OTHER        chapter summaries, front matter, unmarked prose

    This matters for any claim about "the Caitanya-caritāmṛta": in Ādi-līlā the
    purports are 39.4% of the extracted text against the translation's 35.2%, so
    analysing the default mix measures a 16th-century Bengali work and a modern
    English commentary on it together. Use `sections=("TRANSLATION",)` to isolate
    the text itself.
    """
    doc = pymupdf.open(pdf_path)
    body_end = find_body_end(doc)
    lines = []
    for i, page in enumerate(doc):
        if i > body_end:
            break                     # the rest is the duplicate Reviews section
        # Unwrapping happens once over the whole book, not per page, so that
        # paragraphs running across a page break are rejoined rather than cut.
        lines.extend(balaram.clean_page(page.get_text()))
        if progress and i % 500 == 0:
            print(f"  {name}: page {i}/{body_end}", file=sys.stderr)
    doc.close()

    wanted = set(sections)
    paragraphs, current = [], "OTHER"
    for para in balaram.unwrap(lines):
        head = _SECTION_HEAD.match(para)
        if head:
            marker = head.group(1).upper()
            # SYNONYMS/TEXT blocks are apparatus; treat what follows as OTHER
            # until the next TRANSLATION or PURPORT heading.
            current = marker if marker in ("TRANSLATION", "PURPORT") else "OTHER"
            continue
        if _SKIP.match(para):
            continue
        if not keep_synonyms and balaram.is_synonym_block(para):
            continue
        if current not in wanted:
            continue
        para = _BARE_REF.sub(" ", _CITATION.sub(" ", para))
        para = balaram.normalise(para)
        if len(para) > 30:  # drop stray fragments
            paragraphs.append(para)
    return "\n\n".join(paragraphs)


DEFAULT_SECTIONS = ("TRANSLATION", "PURPORT", "OTHER")


def _cache_path(name, sections):
    """Cache file per section selection, so variants do not overwrite each other."""
    if tuple(sections) == DEFAULT_SECTIONS:
        return os.path.join(CACHE_DIR, f"{name}.txt")
    tag = "-".join(s.lower() for s in sorted(sections))
    return os.path.join(CACHE_DIR, f"{name}.{tag}.txt")


def load_corpus(rebuild=False, sections=DEFAULT_SECTIONS):
    """Return {lila_name: text}, extracting and caching on first use."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    corpus = {}
    for name, pdf in LILAS.items():
        cached = _cache_path(name, sections)
        if os.path.exists(cached) and not rebuild:
            with open(cached, encoding="utf-8") as fh:
                corpus[name] = fh.read()
            continue
        print(f"Extracting {name} ({'+'.join(sections)})...", file=sys.stderr)
        text = extract_lila(name, os.path.join(PDF_DIR, pdf), sections=sections)
        with open(cached, "w", encoding="utf-8") as fh:
            fh.write(text)
        corpus[name] = text
    return corpus


if __name__ == "__main__":
    sections = (("TRANSLATION",) if "--translations-only" in sys.argv
                else DEFAULT_SECTIONS)
    corpus = load_corpus(rebuild="--rebuild" in sys.argv, sections=sections)
    for name, text in corpus.items():
        print(f"{name:8s} {len(text):>10,} chars")
