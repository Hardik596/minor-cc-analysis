"""Split the Caitanya-caritāmṛta into its 62 chapters.

The Ramcharitmanas study used the seven kāṇḍas as its unit. The CC's structural
analogue is the three līlās, but three points are too few for sentiment
trajectories or word-frequency comparisons, so chapters are the working unit:
Ādi 17 + Madhya 25 + Antya 20 = 62.

Chapter boundaries are taken from **verse-number resets**, not from "Chapter N"
headings. The headings are unreliable in these PDFs -- Madhya yields 24 of its 25
and Antya 21 of its 20, with the numbering inconsistently spelled ("Chapter Two"
beside "Chapter 3"). Verse numbering resets to 1 at every chapter and is exact:
16 + 24 + 19 resets, giving 17 + 25 + 20 chapters.
"""

import os
import re
import sys

import pymupdf

import balaram
import extract

CACHE = os.path.join(extract.CACHE_DIR, "chapters")

# "TEXT 5", "TEXTS 65–66" -- the verse marker that opens each verse block.
_VERSE = re.compile(r"^TEXTS?\s+([0-9]+)\s*(?:[-–—]\s*([0-9]+))?\s*$", re.I)
_SECTION_HEAD = extract._SECTION_HEAD


def split_chapters(pdf_path, sections=("TRANSLATION",), progress=True):
    """Return [chapter_text, ...] for one līlā, in order.

    Only the requested sections are kept (default: the translation, i.e.
    Kṛṣṇadāsa Kavirāja's text rather than the modern commentary).
    """
    doc = pymupdf.open(pdf_path)
    body_end = extract.find_body_end(doc)
    lines = []
    for i, page in enumerate(doc):
        if i > body_end:
            break                     # duplicate Reviews section -- see extract
        lines.extend(balaram.clean_page(page.get_text()))
        if progress and i % 800 == 0:
            print(f"    page {i}/{body_end}", file=sys.stderr)
    doc.close()

    wanted = set(sections)
    chapters, current_chapter, buf = [], [], []
    section, last_verse = "OTHER", 0

    for para in balaram.unwrap(lines):
        verse = _VERSE.match(para)
        if verse:
            start = int(verse.group(1))
            end = int(verse.group(2) or start)
            if start < last_verse and current_chapter:
                chapters.append("\n\n".join(current_chapter))
                current_chapter = []
            last_verse = end
            section = "OTHER"          # verse body is not translation or purport
            continue

        head = _SECTION_HEAD.match(para)
        if head:
            marker = head.group(1).upper()
            section = marker if marker in ("TRANSLATION", "PURPORT") else "OTHER"
            continue

        if section not in wanted or extract._SKIP.match(para):
            continue
        if balaram.is_synonym_block(para):
            continue
        para = extract._BARE_REF.sub(" ", extract._CITATION.sub(" ", para))
        para = balaram.normalise(para)
        if len(para) > 30:
            current_chapter.append(para)

    if current_chapter:
        chapters.append("\n\n".join(current_chapter))
    return chapters


def load_chapters(rebuild=False, sections=("TRANSLATION",)):
    """Return an ordered list of dicts: lila, chapter number, label, text."""
    os.makedirs(CACHE, exist_ok=True)
    out = []
    for lila, pdf in extract.LILAS.items():
        cached = os.path.join(CACHE, f"{lila}.txt")
        if os.path.exists(cached) and not rebuild:
            with open(cached, encoding="utf-8") as fh:
                texts = fh.read().split("\n\f\n")
        else:
            print(f"Splitting {lila} into chapters...", file=sys.stderr)
            texts = split_chapters(os.path.join(extract.PDF_DIR, pdf), sections)
            with open(cached, "w", encoding="utf-8") as fh:
                fh.write("\n\f\n".join(texts))
        for i, text in enumerate(texts, start=1):
            out.append({"lila": lila, "chapter": i,
                        "label": f"{lila} {i}", "text": text})
    return out


if __name__ == "__main__":
    chapters = load_chapters(rebuild="--rebuild" in sys.argv)
    by_lila = {}
    for ch in chapters:
        by_lila.setdefault(ch["lila"], []).append(ch)
    for lila, chs in by_lila.items():
        total = sum(len(c["text"]) for c in chs)
        print(f"{lila:8s} {len(chs):3d} chapters  {total:>10,} chars")
        sizes = [len(c['text']) for c in chs]
        print(f"         smallest {min(sizes):,} / largest {max(sizes):,}")
    print(f"TOTAL    {len(chapters):3d} chapters")
