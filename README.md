# Graph-theoretic and sentiment analysis of Śrī Caitanya-caritāmṛta

Port of an earlier Ramcharitmanas notebook to the Caitanya-caritāmṛta corpus.
Unit of analysis: the three **līlās** — Ādi, Madhya, Antya.

## Layout

| File | Role |
|---|---|
| `balaram.py` | Repairs the legacy font encoding in the PDFs; strips page furniture and Bengali verse noise |
| `extract.py` | PDF → clean UTF-8 text per līlā, cached under `corpus/` |
| `characters.py` | Canonical character resolution (honorifics, aliases, non-character stoplist) |
| `project.py` | The analysis pipeline; writes `results/analysis.json` |

## Running

```bash
pip install nltk spacy networkx scikit-learn matplotlib pymupdf
python -m spacy download en_core_web_sm

python extract.py          # one-off, ~2 min; caches to corpus/
python project.py          # full analysis
```

NLTK corpora download themselves on first run. `python extract.py --rebuild`
forces re-extraction.

## The two things that made this non-trivial

### 1. The PDFs do not contain readable text

They embed two legacy 8-bit fonts, so naive extraction yields mojibake:

```
extracted:  "Kåñëa consciousness"   "Çré Rüpa and Sanätana Gosvämés"
actual:     "Kṛṣṇa consciousness"   "Śrī Rūpa and Sanātana Gosvāmīs"
```

* **Balaram** (romanised Sanskrit) is recoverable via a codepoint map —
  `balaram.BALARAM_MAP`, verified against known words in the corpus.
  It must be applied with `str.translate` in one pass: the mapping is cyclic
  (`ï`→`ñ` while `ñ`→`ṣ`), so chained `.replace()` calls would corrupt it.
* **The Bengali display font** is not recoverable. Those lines are detected and
  dropped; every such verse has an English translation beside it, so no
  narrative content is lost.

Extraction also drops the word-by-word glossary blocks
(`mätä—the mother; bhakta-gaëera—of the…`), which are reference apparatus and
would badly skew TF-IDF and the word-adjacency graphs, and the ~5,500 inline
scriptural cross-references (`[Cc. Madhya 22.83]`, `(Cc. Antya 7.11)`). Left in,
NER reads the līlā name as an entity and `Antya` ranks as the second most
mentioned "character" in Antya-līlā.

Net: 12.5M raw chars → 7.4M chars of narrative prose.

### Extraction completeness audit

| | Ādi | Madhya | Antya | Total |
|---|---|---|---|---|
| Chapters detected | 17 ✓ | 25 ✓ | 20 ✓ | **62 ✓** |
| TRANSLATION blocks | 2,253 | 5,930 | 3,176 | **11,359** |
| …yielding content | 2,253 | 5,930 | 3,176 | **11,359 (100%)** |
| Verses covered | 2,311 | 6,015 | 3,228 | **11,554** |
| Verse-sequence anomalies | 0 | 1 | 0 | **1** |

No TRANSLATION block is lost to the filters, and the chapter counts match the
work exactly. Verse coverage is 11,554 against a canonical 11,555; the single
discrepancy is a heading mangled in the PDF's own text layer (`e 34` instead of
`TEXT 34`, Madhya) — **that verse's translation is still captured**, which is why
Madhya has one more TRANSLATION heading than TEXT heading.

### 2. spaCy's entity labels are unreliable on this corpus

`en_core_web_sm` is trained on contemporary newswire. Measured on ~500k
characters of Ādi-līlā:

| Surface form | spaCy label(s) |
|---|---|
| Kṛṣṇa | PERSON 275 **and** ORG 188 |
| Caitanya | **GPE** 236 |
| Caitanya Mahāprabhu | PERSON 87 |
| Nityānanda | ORG 42 |
| Śrīmad-Bhāgavatam | PERSON 61 — a scripture |
| Vedānta | PERSON 132 — a philosophy |

Filtering on `label_ == "PERSON"` therefore *loses the protagonist* and admits
scriptures. `characters.py` instead pools candidates across labels and resolves
them by name (honorific stripping → alias map → non-character stoplist).

**Limitation to state in any write-up:** this is lexical resolution, not
coreference. Pronouns (`He`, `the Lord`) are unattributed, so mention counts are
lower bounds, and epithets such as "the Supreme Personality of Godhead" are
excluded rather than resolved to Kṛṣṇa.

## Changes from the original script

The original was entirely commented out and had never run. Beyond the port:

| Issue | Original | Now |
|---|---|---|
| Antagonist | Picked the **least**-mentioned entity — i.e. whichever name appeared once | Lowest mean sentiment among frequently-mentioned characters, and reports *no* antagonist when none scores negative (see below) |
| "Characters" | Concatenated spaCy PERSONs with the argmax-TF-IDF word of each sentence — the sentence's **rarest** word, not a character | NER candidates only, canonicalised; TF-IDF repurposed to contrast the three līlās |
| Significance ranking | Summed four centrality dicts; PageRank always sums to 1.0 and the rest scale with node count, so it ranked by **sentence length** | z-scored within node-count bins, so length is controlled rather than ranked (see below) |
| Cross-text comparison | Per-sentence metrics compared directly between texts | 30-token moving windows (Mota et al. 2016), because the per-sentence metrics are a length proxy |
| Metric naming | "Compactness" — a different published metric, and not a speech-graph attribute | Renamed `global_efficiency` (Latora & Marchiori 2001), which is what it computes |
| Dominating sentiment | `max()` over `neg/neu/pos/compound` together — different scales, so it returned `neu` every time | `neu/neg/pos` compared as one distribution; `compound` reported separately as valence |
| Entropy | Divided by zero on edgeless sentences | Guarded, and normalised by `log2(n)` so it is not a proxy for length |
| Lemmatisation | Lemmatised, *then* POS-tagged the lemmas; WordNet defaults to noun, so verbs were never lemmatised | POS-tag first, lemmatise with that tag |
| Memory | Retained a graph object for every sentence (~100k) to use 100 | Graph-level measures streamed; graphs rebuilt only for the top 100 |
| spaCy input | `nlp(text)` on a whole book — exceeds the 1M-char `max_length` | Chunked on paragraph boundaries |
| Per-sentence cost | Rebuilt the stopword set and lemmatiser inside the loop | Built once |
| NLTK resources | `punkt`, `averaged_perceptron_tagger`; `vader_lexicon` never downloaded | Post-3.8.2 names with fallback; VADER included |
| Paths | Hardcoded Colab `/content/drive/...`, `.txt` only | Repo-relative, PDF-driven, cached |

### The length confound

The per-sentence graph metrics are **not** size-invariant, despite looking like
normalised ratios. Measured on 4,854 Ādi-līlā sentences:

```
r(nodes, edge_density)      = -0.725
r(nodes, global_efficiency) = -0.836
r(edge_density, 2/n)        = +0.972   <- chain-graph prediction
```

A word-adjacency graph over *n* distinct tokens is nearly a chain (*n*−1 edges),
so density ≈ 2/*n*. Per-sentence edge density is therefore close to a reciprocal
length measure.

This is not academic. Under length control the **rank order of the three līlās
reverses**: uncontrolled, Ādi has the lowest edge density; controlled, it is in
the middle and Antya is lowest — Ādi merely has longer sentences (9.64 mean nodes
vs 8.89). For entropy the between-līlā spread collapses from 2.06% to 0.23%.

Use `measure_windowed()` for anything comparing texts, and
`length_controlled_scores()` for ranking sentences within a text. See
`LITERATURE.md` §4 for the full table and citations.

### The graph attributes carry almost no text-discriminative signal

`discriminate.py` tests whether the windowed attributes can tell Kṛṣṇadāsa
Kavirāja's narrative apart from Prabhupāda's commentary — two texts four
centuries and one language apart. Balanced classes, non-overlapping windows,
group-wise CV splits:

| Condition | Accuracy | AUC |
|---|---|---|
| 5 graph attributes | 0.522–0.552 | 0.528–0.577 |
| Shuffled labels (null) | 0.499 | 0.501 |
| **TF-IDF unigrams** | **0.863** | **0.935** |

Words solve the task; the graph attributes barely beat chance. Between-līlā
effect sizes are correspondingly negligible (Cohen's *d* ≤ 0.224, mostly < 0.13).

**So the between-līlā graph differences are not findings.** The means differ
reliably only because *n* is large. Report effect sizes and a lexical baseline
alongside these metrics — `LITERATURE.md` §8 has the full tables.

### On the antagonist

Antagonists **do** exist in the corpus, and the sentiment method finds ones that
are narratively real:

| Figure | Mean compound | Sentences | Role |
|---|---|---|---|
| Rāmacandra Khān | −0.094 | 56 | Plots against Haridāsa Ṭhākura (Antya) |
| Kazi | −0.047 | 73 | Suppresses the saṅkīrtana movement (Ādi) |
| Kālā Kṛṣṇadāsa | −0.060 | 23 | Led astray by the Bhaṭṭathāris (Madhya) |
| Rāvaṇa | −0.007 | 20 | Referenced antagonist |

But the *lowest-scoring* figure is not usually the antagonist. At ~20 supporting
sentences the minimum is dominated by semantic confounds: **Yamarāja** (−0.302,
god of death) and **Anupama** (−0.297, a devotee who *dies*) score low on death
and grief vocabulary, not on antagonism. Selection therefore requires both a
negative mean **and** `MIN_ANTAGONIST_SENTENCES` (40) of support. The unfiltered
ranking is still written to `results/analysis.json`
(`most_negative_characters` / `most_positive_characters`), along with
`lowest_sentiment_character` so the filtered-out minimum stays visible.

**Caveat for the write-up:** VADER measures affective vocabulary in the
surrounding sentence, which is not the same thing as narrative antagonism. Death,
grief, and ascetic renunciation all depress the score without implying an
adversary. Treat this as a *screening* signal that needs manual confirmation, not
as automatic antagonist detection.
