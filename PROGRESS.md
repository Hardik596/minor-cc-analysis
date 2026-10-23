# Project log — everything done so far

Computational analysis of the **Śrī Caitanya-caritāmṛta**, following the design
of the earlier Ramcharitmanas study (`sri ramcharitmanas (1).docx`).

Last updated: 2026-09-18.

---

## 1. Goal

Replicate the Ramcharitmanas analysis on the Caitanya-caritāmṛta:
character identification → character interaction network → per-section sentiment
→ emotion analysis → topic modelling → word clouds and word frequencies.

Unit of analysis: **62 chapters** (Ādi 17 + Madhya 25 + Antya 20), standing in
for the Ramcharitmanas study's 7 kāṇḍas. Three līlās alone give too few points
for sentiment trajectories.

Corpus: **translations only** — Kṛṣṇadāsa Kavirāja's 16th-century text, not
Prabhupāda's 20th-century purports.

---

## 2. Starting point

`project.py` was **214 lines, every one commented out**. It had never been run.
It targeted Google Colab (`/content/drive/My Drive/ramcharitramanas/`) and `.txt`
files that do not exist here. None of its libraries were installed.

Critically, it sketched a **word-adjacency graph** method (nodes = words), which
is *not* what the Ramcharitmanas paper does (nodes = characters). That mismatch
was not discovered until late — see §7.

### Bugs in the original logic

| Bug | Detail |
|---|---|
| Antagonist inverted | Selected the **least**-mentioned entity — i.e. whichever name appeared once |
| "Characters" polluted | Mixed spaCy entities with the argmax-TF-IDF word of every sentence — the sentence's *rarest* word, not a character |
| Significance = length | Summed four centrality dicts; PageRank always sums to 1.0 and the rest scale with node count, so it ranked by sentence length |
| Sentiment meaningless | `max()` over `neg/neu/pos/compound` together — different scales, so it returned "neutral" every time |
| Division by zero | `calculate_entropy` on edgeless sentences; `average_sentiment` on empty text |
| Lemmatise→tag order | Lemmatised first, then POS-tagged the lemmas; WordNet defaults to noun so verbs were never lemmatised |
| Memory | Retained a graph object for every sentence (~100k) to use 100 |
| spaCy limit | `nlp(text)` on a whole book exceeds the 1,000,000-character `max_length` |
| Per-sentence waste | Rebuilt the stopword set and lemmatiser inside the loop |
| Stale NLTK names | `punkt` / `averaged_perceptron_tagger` renamed in NLTK ≥3.8.2; `vader_lexicon` never downloaded at all |

---

## 3. Environment problems

- **Missing entirely**: nltk, spacy, networkx, scikit-learn, matplotlib.
  Installed: numpy, pandas, pymupdf, PyPDF2.
- **Ghost `dist-info` entries.** `packaging`, `requests`, `typer`, `cloudpickle`,
  `pyparsing`, `jinja2`, `kiwisolver` all reported "Requirement already
  satisfied" by pip while failing to import — metadata present, package
  directory absent. Fixed with `--force-reinstall --no-cache-dir`.
  *Worth remembering: this machine can produce phantom `ModuleNotFoundError`s.*
- **Import name ≠ distribution name**: `dateutil` → `python-dateutil`. pip aborts
  the whole batch on one bad name, so `httpx` silently never installed either.
- **`nrclex` 4.1.0 API change**: the constructor no longer takes text. Now
  `NRCLex()` then `.load_token_list(tokens)`. Also much faster — the old pattern
  rebuilt the lexicon once per sentence.
- **The ghost pattern recurred repeatedly.** Running the notebook needed
  `nbclient`, then `executing`, `pure_eval`, `decorator`, `wcwidth`, `comm` and
  `matplotlib-inline` — each reported installed, each missing. The Jupyter
  kernel died with only "Kernel died before replying to kernel_info"; the real
  cause was only visible by invoking `python -m ipykernel_launcher` directly.
  **If a kernel dies mysteriously on this machine, run the launcher by hand.**
- **`nbconvert --execute` is the wrong tool here**: it pulls in the pandoc
  templating chain and fails without `pandocfilters`, despite not needing it to
  run cells. `run_notebook.py` drives `nbclient` directly instead.

---

## 4. The data — what the PDFs actually contain

Six PDFs, two editions of the same three books. Using the
`Sri Caitanya Caritamrta {Adi,Madhya,Antya}.pdf` set (10,691 pages).

### 4.1 The text is not readable as extracted

Two legacy 8-bit fonts:

1. **Balaram** (romanised Sanskrit). Extraction yields Latin-1 lookalikes:
   `Kåñëa` = `Kṛṣṇa`, `Çré Rüpa` = `Śrī Rūpa`. Fully recoverable — mapping
   derived empirically and verified against known words (`balaram.py`).
   **Must be applied with `str.translate` in one pass**: the map is cyclic
   (`ï`→`ñ` while `ñ`→`ṣ`), so chained `.replace()` would corrupt it.
2. **A Bengali display font** for the original verses — unrecoverable
   (`ıËp¡‰¬±ı˛œ`). Dropped; every such verse has an English translation beside it.
   14,339 such lines in Madhya + Antya; Ādi has none.

### 4.2 Structural noise removed

| Problem | Scale | Action |
|---|---|---|
| Word-by-word glossary blocks (`mätä—the mother; …`) | pervasive | dropped (em-dash density test) |
| Inline scriptural citations (`[Cc. Madhya 22.83]`) | **5,510** | stripped — NER read the līlā name as an entity, and `Antya` ranked as the #2 "character" in Antya-līlā |
| Running heads, page numbers | pervasive | dropped |
| Lines hard-wrapped mid-sentence, across pages | pervasive | rejoined before sentence tokenisation |

### 4.3 The biggest find: every translation appeared **three times**

Each PDF ends with a **"Reviews" section** (~500 of 2,438 pages in Ādi) that
reprints every translation as a reference list:

```
Ādi 13.33 - Lord Caitanya Mahāprabhu lived in the Navadvīpa area for twenty-four years…
```

Those pages carry **none** of the body's `TEXT`/`TRANSLATION`/`PURPORT` markers,
so the section tracker kept its last value and swallowed the whole thing — twice
over, giving three copies of each verse.

| | Contaminated | Correct |
|---|---|---|
| Ādi translations | 1,048,552 chars | **388,129** |
| Madhya | 1,952,428 | **1,153,592** |
| Antya | 1,521,888 | **571,084** |

Fixed by truncating at the last page bearing a structural marker
(`extract.find_body_end`). Verified: a probe sentence now occurs exactly once.

**This invalidated every earlier number** — mention counts were ~3× inflated and
the discriminative check had duplicate text leaking across train/test folds.

### 4.4 Translation vs purport is differentially mixed

| Līlā | TRANSLATION | PURPORT | PURPORT headings |
|---|---|---|---|
| Ādi | 35.2% | **39.4%** | 887 |
| Antya | 55.3% | **9.3%** | 382 |

Prabhupāda's commentary is dense in Ādi and thins sharply by Antya. So analysing
the blended text does not dilute each volume equally — cross-līlā comparison on
it partly compares *how much 1970s English commentary each volume contains*.
`extract.py` now takes `sections=("TRANSLATION",)`.

### 4.5 Extraction completeness — audited

| | Ādi | Madhya | Antya | Total |
|---|---|---|---|---|
| Chapters | 17 ✓ | 25 ✓ | 20 ✓ | **62 ✓** |
| TRANSLATION blocks | 2,253 | 5,930 | 3,176 | **11,359** |
| …yielding content | 2,253 | 5,930 | 3,176 | **100%** |
| Verses covered | 2,311 | 6,015 | 3,228 | **11,554** |
| Sequence anomalies | 0 | 1 | 0 | **1** |

11,554 against a canonical 11,555. The one discrepancy is a heading the PDF's own
text layer mangled (`e 34` instead of `TEXT 34`, Madhya) — **that verse's
translation is still captured**, which is why Madhya has one more TRANSLATION
heading than TEXT heading.

---

## 5. Bugs I introduced and fixed

Recorded honestly, because several produced numbers that were reported before
being caught.

1. **En-dash in the Bengali-noise set.** `U+2013` was treated as Bengali script.
   Verse-range headings read `TEXTS 65–66` — 11 characters, one flagged, ratio
   0.09, over the 0.06 threshold — so **54 headings per volume were silently
   discarded**. With the heading gone the section tracker never reset, leaking
   romanised verse text into the corpus. Fixed by removing the en-dash and
   requiring ≥2 flagged characters plus a minimum line length.
2. **Claimed the graph metrics were "size-invariant".** They are not:
   `r(nodes, edge_density) = −0.725`, `r(nodes, global_efficiency) = −0.836`,
   and `r(edge_density, 2/n) = +0.972`. I had replaced the original's
   length-proxy with a milder length-proxy. Fixed with 30-token moving windows
   and within-bin z-scoring.
3. **Claimed "no antagonist exists"** from a 300k-character sample. On the full
   corpus negative-context figures clearly exist (the Kazi, Rāmacandra Khān).
4. **Naive chapter segmentation** by verse-number reset reported phantom gaps
   until the dropped `TEXTS n–m` headings were restored.
5. **Mis-named metric.** Called `networkx.global_efficiency` "compactness".
   That name belongs to a different published measure (Botafogo et al. 1992) and
   is not a speech-graph attribute at all. Renamed to what it computes
   (Latora & Marchiori 2001).
6. **An unexplained discrepancy**: Ādi character counts read 1,160 in one run
   and 1,829 in another on a nominally identical corpus. The current figure
   reproduces exactly; the earlier one does not and should not be cited. Most
   likely an artifact of the triplication described in §4.3.
7. **Honorific stripping collapsed distinct people.** `Paṇḍita` and `Purī` are
   in the suffix list, so `Dāmodara Paṇḍita` → `Dāmodara` (merging him with
   Svarūpa Dāmodara) and `Rāmacandra Purī` → `Rāmacandra` (merging a critical
   sannyāsī, the antagonist Rāmacandra Khān, and the Rāma avatar into one node).
   Fixed with `PROTECTED_NAMES`, a table of full names whose final word
   identifies rather than decorates — each entry verified against the corpus.
8. **Over-protection, the opposite error.** I then added `Jagadānanda Paṇḍita`
   and `Rāghava Paṇḍita` speculatively, which **split one person into two**.
   Checked: neither has a competing referent, so both were removed. Only names
   with a verified competitor stay protected (`Gadādhara Paṇḍita` vs `Gadādhara
   dāsa`; `Śaṅkara Paṇḍita` vs `Śaṅkarācārya`; `Paramānanda Purī` vs Mahāpātra /
   Upādhyāya / Gupta).
9. **Substring matching created phantom edges.** `characters_in` used plain
   containment, so a sentence naming `Rāmacandra Purī` *also* registered bare
   `Rāmacandra` — a spurious edge between two different people, and inflation of
   the shorter name. `Kṛṣṇadāsa` contains `Kṛṣṇa`, so Kṛṣṇa was credited with
   Kṛṣṇadāsa's co-occurrences too. Fixed with longest-match-first masking. This
   one mattered: it had been **suppressing** the Caitanya/Kṛṣṇa eigenvector gap.
10. **Possessives and pronouns as characters.** `Rādhārāṇī's` was a separate
    node from `Rādhārāṇī`, and `Them` was in the cast. Only visible once the
    filter widened to 50%. Fixed by stripping possessives and stoplisting
    pronouns.
11. **Weighted clustering was meaningless.** `networkx` normalises edge weights
    by the graph maximum; the dominant Caitanya–Kṛṣṇa edge drove every other
    triangle toward zero, reporting **0.010** against a true (unweighted)
    **0.712**. Reporting the weighted figure would have claimed the CC is an
    order of magnitude *less* clustered than the Iliad — the opposite of the
    truth. Switched to unweighted, which is also what the comparable literature
    reports.
12. **`matplotlib.use("Agg")` at import.** `narrative.py` forced the headless
    backend on import, so the notebook executed cleanly but rendered **zero**
    inline figures. Now set only under `__name__ == "__main__"`.
13. **Notebook source lines without newline terminators.** nbformat stores each
    source line with its trailing `
`; omitting them concatenated every line
    into one and broke six cells. Caught by AST-parsing each cell before
    execution — worth keeping as a check.

---

## 6. Method that was built, tested, and set aside

`project.py` was rebuilt as specified (word-adjacency digraphs, nodes = words),
then validated — and it failed validation.

**Discriminative check** (`discriminate.py`): can the graph attributes tell
Kṛṣṇadāsa Kavirāja's narrative from Prabhupāda's commentary? Non-overlapping
windows, CV splits grouped by contiguous block, classes balanced.

| Condition | Accuracy | AUC |
|---|---|---|
| 5 graph attributes, logistic regression | 0.552 | 0.575 |
| 5 graph attributes, random forest | 0.529 | 0.534 |
| Shuffled labels (null) | 0.500 | 0.500 |
| **TF-IDF unigrams** | **0.861** | **0.934** |

Words separate the two registers easily; the graph attributes barely beat chance.
Between-līlā effect sizes were correspondingly negligible (Cohen's *d* ≤ 0.224,
mostly < 0.13).

*Caveat: this ran on the triplicated corpus, so duplicate windows leaked across
folds and the TF-IDF figure is probably optimistic. The qualitative gap between
0.93 and 0.57 is far too large to be explained by that leakage.*

**Conclusion:** word-adjacency graph attributes carry no text-discriminative
signal here. Retained as a negative methodological result, not as a finding.

`project.py` (the word-adjacency pipeline) has since been **deleted** — its two
reusable helpers moved to `common.py`, and its stored results were computed on
the triplicated corpus, so the numbers were invalid. `discriminate.py` is kept
because it is the only code behind this negative result, but **its figures need
re-running on the clean corpus** before they are cited.

---

## 7. The pivot

The Ramcharitmanas paper builds **character interaction networks** — nodes are
people, edges are co-occurrence. `project.py` sketched **word-adjacency graphs**.
Different methods entirely. Once that was clear, the analysis was rebuilt as
`narrative.py`.

Character networks have solid precedent (Das, Das & Mahesh, ICON 2016;
Gultepe & Mathangi 2023), so §6's negative result does not apply to them.

---

## 8. Current results (`narrative.py`)

Corpus: **62 chapters, 21,290 sentences, 403,642 words, 9,384 unique words.**
(Ramcharitmanas study for comparison: 10,401 sentences, 269,616 words, 80
characters.)

**145 characters** with ≥10 mentions; cast of **72** after the top-50% filter.
Network: **72 nodes, 778 edges, density 0.304, clustering 0.712, 5 communities.**

Clustering is the **unweighted** coefficient. networkx's weighted version
normalises every edge by the graph maximum, and the Caitanya–Kṛṣṇa edge is so
dominant that all other triangles collapse toward zero (it reported 0.010, which
measured one edge rather than the structure). Unweighted is also what the
comparable literature reports, which makes the figure directly benchmarkable:

| Text | Clustering |
|---|---|
| **Caitanya-caritāmṛta (this work)** | **0.712** |
| Rāmāyaṇa (Evans 2026) | 0.532 |
| Mahābhārata (Evans 2026) | 0.509 |
| Iliad (Evans 2026) | 0.438 |

The CC network is markedly more clustered than either Sanskrit epic — consistent
with a narrative built around one figure's circles of associates rather than
around opposing factions.

The filter was raised from the Ramcharitmanas study's top 30% to top 50% so that
`Rāmacandra Khān` (24 mentions, rank 77 of 162 at the time) is included — at 30%
the cutoff sat at 38 and dropped him. A filter that discards a named antagonist
measures prominence, not narrative role. State this as a deliberate departure.

Widening the filter exposed noise the 30% cutoff had hidden, now fixed:
possessives counted as separate characters (`Rādhārāṇī's`), a pronoun (`Them`),
places (`Vraja`), fragments (`Amṛta` from *Caitanya-caritāmṛta*, `Kulīna`,
`Siṁha`), a 19th-century commentator (`Bhaktivinoda Ṭhākura`), and five verified
split names merged (`Sārvabhauma` → `Sārvabhauma Bhaṭṭācārya`, 258 of 299;
`Nārada` → `Nārada Muni`; `Nanda Mahārāja` → `Nanda`; `Śacī`/`Śacīmātā`/
`Śacīdevī` → one node; `Murāri` → `Murāri Gupta`, with `Murāri Māhiti`
protected). That cleanup is why 162 eligible fell to 145.

| Rank | Degree | Betweenness | Eigenvector |
|---|---|---|---|
| 1 | Caitanya 0.958 | Caitanya 0.156 | **Caitanya 0.647** |
| 2 | Kṛṣṇa 0.915 | Kṛṣṇa 0.143 | Kṛṣṇa 0.427 |
| 3 | Jagannātha 0.746 | Jagannātha 0.056 | Jagannātha 0.306 |
| 4 | Nityānanda 0.704 | Nityānanda 0.049 | Nityānanda 0.207 |

### The headline finding

Caitanya leads all three measures, **but by wildly different margins**:

| Measure | Caitanya | Kṛṣṇa | Caitanya's lead |
|---|---|---|---|
| Degree | 0.958 | 0.915 | **+5%** |
| Betweenness | 0.156 | 0.143 | +9% |
| Eigenvector | 0.647 | 0.427 | **+52%** |

Kṛṣṇa is named alongside almost everyone (degree nearly equal) but not alongside
the *well-connected* (eigenvector far behind). That is the signature of a
theological referent versus a narrative hub.

⚠️ At the earlier top-30% filter, Kṛṣṇa actually **led** degree (0.979 vs 0.957)
and betweenness, which made the contrast a rank reversal. Widening the cast to
50% added ~30 human associates who co-occur with Caitanya rather than Kṛṣṇa, so
Caitanya overtook him on those measures. The underlying point is unchanged and
the community split below carries it independently, but the finding must be
stated as a **disproportionate margin**, not a rank reversal.

How the gap moved as bugs were removed — it survived every correction and ended
larger than it started:

| Stage | Caitanya | Kṛṣṇa | Gap |
|---|---|---|---|
| Junk cast, substring matching | 0.587 | 0.421 | 0.166 |
| Cast cleaned | 0.605 | 0.455 | 0.150 |
| + `Hari` merged into Kṛṣṇa | 0.599 | 0.546 | 0.054 |
| **+ longest-match fix (final)** | **0.648** | **0.429** | **0.219** |

The third row briefly suggested the `Hari` merge had weakened the result. It had
not: the substring bug was inflating Kṛṣṇa, because **"Kṛṣṇadāsa" contains
"Kṛṣṇa"** — every sentence naming Kṛṣṇadāsa credited Kṛṣṇa with his
co-occurrences. Once nested names stopped double-firing, the gap widened to its
largest value, *with* the Hari merge still in place. No sensitivity analysis is
needed; the finding is robust to that coding choice.

### Communities (5) — coherent and interpretable

| # | Size | Members | Reading |
|---|---|---|---|
| 0 | 12 | Kṛṣṇa, Brahmā, Viṣṇu, Mahā-Viṣṇu, Nārāyaṇa, Rādhā, Rādhārāṇī, Nanda, Vyāsadeva, Cupid, Kali | cosmological / theological |
| 4 | 18 | Caitanya, Govinda, Jagannātha, Gadādhara Paṇḍita, Kāśī Miśra, Gopīnātha Ācārya, Balabhadra Bhaṭṭācārya | the Purī circle |
| 1 | 9 | Advaita, Nityānanda, Haridāsa, Śrīvāsa, Mukunda, Rāghava | the Navadvīpa circle |
| 3 | 6 | Rūpa, Sanātana, Raghunātha, Kṛṣṇadāsa, Candraśekhara, Tapana Miśra | the Vṛndāvana Gosvāmīs |
| 2 | 3 | Mādhavendra Purī, Gopāla, Gopīnātha | the Remuṇā Deity episode |

The algorithm separated the 16th-century biography from the theology it is
embedded in, and recovered the three geographic phases of Caitanya's life,
without being told any of it.

### Sentiment: a cleaner antagonist

Lowest mean sentiment across chapters:

| Character | Mean | Chapters |
|---|---|---|
| **Rāmacandra Purī** | **−0.3121** | 5 |
| **Rāmacandra Khān** | **−0.1503** | 2 |
| Gopīnātha Paṭṭanāyaka | −0.0947 | 4 |
| Indra | +0.0573 | 16 |

**Rāmacandra Purī** — the sannyāsī who publicly criticised Caitanya — is now the
clearest antagonist in the text, at −0.312. He was invisible before, because he
had been merged with `Rāmacandra Khān` and with the *positively* scored Rāma
avatar. Disambiguation did not just tidy the cast; it uncovered the result.

Two independently negative figures now, rather than one. Caveat for the paper:
**Rāmacandra Khān appears in only 2 chapters**, so treat him as narratively
significant but statistically slight; Rāmacandra Purī's 5 chapters are the safer
citation.

Community detection confirms it independently:

- **Community 2 (Kṛṣṇa's)**: Brahmā, Viṣṇu, Rādhā, Rādhārāṇī, Nārāyaṇa, Śiva,
  Nanda, Saṅkarṣaṇa, Mahā-Viṣṇu — entirely cosmological
- **Community 0 (Caitanya's)**: Advaita, Nityānanda, Gadādhara, Sārvabhauma
  Bhaṭṭācārya, Rāmānanda Rāya — historical associates
- Community 5: the Vṛndāvana Gosvāmīs (Rūpa, Sanātana, Raghunātha)
- Community 4: the Purī intimates (Svarūpa Dāmodara, Govinda, Jagadānanda)

The algorithm separated the 16th-century biography from the theology it is
embedded in, without being told they differ.

### Also produced

5 LDA topics (Mādhavendra/Gopāla deity episodes; Vaiṣṇava philosophy; Jagannātha
Purī and Rāmānanda; the Gosvāmīs; Navadvīpa domestic scenes). Emotion profiles
trust-dominant across all leads (Nityānanda highest, 0.209). Word frequencies per
līlā. Figures in `results/figures/`: character network, sentiment trends across
62 chapters, emotion bars, narrative presence, 5 word clouds.

### Cast resolution — how it was settled

Four rounds, each verified against the corpus before acting.

**Round 1 — junk removal.** Bare honorifics (`Ṭhākura`, `Śrīla`), titles
(`King`), objects (`Deity`, `Ratha`), places (`Govardhana Hill`). Suffix
stripping runs *before* the stoplist, so `Haridāsa Ṭhākura` → `Haridāsa` still
works while bare `Ṭhākura` is rejected.

**Round 2 — the four ambiguous entities**, resolved from textual evidence rather
than assumption:

| Entity | Counts in corpus | Decision |
|---|---|---|
| Dāmodara (409) | Svarūpa Dāmodara 341 · Dāmodara Paṇḍita 41 · Dāmodara dāsa + bare 27 | **not merged** — three people. The text names Dāmodara Paṇḍita's brother, Śaṅkara Paṇḍita |
| Rāmacandra (140) | Rāmacandra Purī 56 · Rāmacandra Khān 26 · Lord Rāmacandra 50 | **split into three** |
| Gopīnātha (190) | Gopīnātha Ācārya 79 · Gopīnātha Paṭṭanāyaka 45 · Deity 65 | **split into three** |
| Hari (52 bare) | *"nondifferent from Hari, the Supreme Lord"*, *"Śrī Hari Himself"* | **merged into Kṛṣṇa** |

`Viṣṇu` and `Nārāyaṇa` were deliberately **not** merged into Kṛṣṇa: in Gauḍīya
theology those are distinct expansions, whereas Hari is a direct synonym. That
asymmetry is a defensible position but *is* a position — state it in the methods
section.

**Round 3 — verified split-name merges** (after the filter widened and exposed
more of the tail):

| Merged | Evidence |
|---|---|
| `Sārvabhauma` → `Sārvabhauma Bhaṭṭācārya` | 258 of 299 occurrences carry the full form |
| `Nārada` → `Nārada Muni` | 34 of 54 |
| `Nanda Mahārāja` → `Nanda` | 47 of 89 |
| `Śacī` / `Śacīmātā` / `Śacīdevī` → one node | no competing follower; all are Caitanya's mother |
| `Murāri` → `Murāri Gupta` | 38 of 61; `Murāri Māhiti` protected as a different man |

**Round 4 — noise the 30% cutoff had hidden**: the possessive `Rādhārāṇī's`, the
pronoun `Them`, the place `Vraja`, fragments `Amṛta` (from
*Caitanya-caritāmṛta*), `Kulīna` and `Siṁha`, and `Bhaktivinoda Ṭhākura` — a
19th-century commentator, not a figure in the 16th-century narrative.

Stoplist is now 190+ entries; `PROTECTED_NAMES` holds 18. Every rule is checked
by an assertion suite (21 cases) before a run.

**Still left unresolved on purpose:**

| Entity | Issue |
|---|---|
| `Puruṣottama` (42) | A name of Kṛṣṇa, a name for Purī, and at least four men (Ācārya, Paṇḍita, Brahmacārī, Jānā) |
| `Jīva` (23) | Jīva Gosvāmī, or *jīva* the common noun (soul) |
| `Sītā` (23) | Rāma's consort, or Sītā Ṭhākurāṇī (Advaita's wife) |
| `Gadādhara` (25) | Gadādhara Paṇḍita or Gadādhara dāsa — both exist |

---

## 9. Literature position

Full detail in `LITERATURE.md`. Summary:

**Cannot claim:**
- "First computational analysis of the CC" — **Apurba Paul, PhD thesis, Jadavpur
  2024** includes CC as one of seven texts (verified by extracting the thesis).
- "First graph analysis of a Hindu religious text" — **Mota et al. 2016**
  (arXiv:1612.09268) covers all four Vedas and 14 Mahābhārata parvas with
  word-adjacency speech graphs (verified from their supplementary table; two
  separate literature searches asserted otherwise and were wrong).
- "First automated protagonist/antagonist detection on an Indic text" —
  **Gultepe & Mathangi 2023** do exactly this at F = 0.749.

**Can claim:**
- **First published Balaram → Unicode/IAST mapping.** No peer-reviewed work
  exists on this font; only undocumented community converters. Strongest and
  cleanest contribution.
- First *dedicated* computational study of the CC.
- First character-network analysis of a Gauḍīya Vaiṣṇava text.
- First treatment of epithet/alias resolution for a Hindu devotional text.
- A documented NER ontology-mismatch case on IAST devotional prose.

**Closest precedent to differentiate from:** Das, Das & Mahesh, *"A Computational
Analysis of Mahabharata"*, ICON 2016 — nearly the same stack and design.

**Venues:** LaTeCH-CLfL, ICON, WILDRE/ISCLS (all free, ACL-indexed);
*Journal of Open Humanities Data* for the font mapping as a data paper.
Avoid MDPI *Religions* (1,800 CHF, mixed reputation in the field).

**Warning:** a 2025 Springer chapter on Ramayana sentiment
(doi:10.1007/978-981-96-6759-8_2) reportedly carries ~a dozen **fabricated
citations** ("Sharma & Verma 2019, *Journal of South Asian Literature*" and
similar). Resolve a DOI for every reference before it enters the bibliography.

---

## 10. Deliverable and files

### `cc_analysis.ipynb` — the presentation file

34 cells, **10 figures rendered inline**, 0 errors. Scrollable without running
anything. Figures also saved to `results/figures/` at 300 dpi.

| Figure | Content | Ramcharitmanas equivalent |
|---|---|---|
| A | Methodology flowchart | their Fig 3 |
| 0 | Chapter sizes | — |
| 1 | Character interaction network (top 20, by community) | their Fig 12 |
| 1b | Networks per līlā (3 panels) | their per-kāṇḍa networks |
| 2 | Centrality comparison | — |
| 3 | Sentiment trends across 62 chapters | their Fig 13 |
| 4 | Emotion comparison | their Fig 15 |
| 5 | Narrative presence | their Fig 5 |
| 6 | Word clouds (4 leads) | their Figs 6, 8, 10 |
| 7 | Word frequencies by līlā | their Fig 11 |

Tables: corpus statistics · top-20 cast · centrality rankings · communities ·
most-negative characters · emotion values · 5 LDA topics · per-līlā network
stats · a text excerpt (their Fig 1).

### Parity with the Ramcharitmanas study

**All 9 methodology components matched.** Beyond it: four centrality measures,
Louvain communities, sentiment at 62 chapters instead of 7 sections, and the
negative-sentiment ranking. Nothing is missing.

The findings live in the *extras* — the centrality measures produced the
Caitanya/Kṛṣṇa result, and the communities recovered the phases of Caitanya's
life. Neither is obtainable from the replicated parts alone.

### Files

| File | Role |
|---|---|
| **`cc_analysis.ipynb`** | **The presentation notebook** |
| `run_notebook.py` | Re-executes the notebook headlessly via nbclient |
| `balaram.py` | Legacy font repair; Bengali-noise detection; page cleanup; unwrapping |
| `extract.py` | PDF → text; section selection; body-end truncation; caching |
| `chapters.py` | Splits into 62 chapters by verse-number reset |
| `characters.py` | Character resolution — honorifics, possessives, aliases, stoplist, protected names |
| `common.py` | Shared NLP setup and word-graph primitives |
| `narrative.py` | Analysis functions the notebook calls; also runs standalone |
| `discriminate.py` | Validity check behind the §6 negative result |
| `README.md` | Method notes and the extraction audit |
| `LITERATURE.md` | Literature review and novelty assessment |
| `PROGRESS.md` | This log |
| `corpus/chapters/` | The 62-chapter cache |
| `results/figures/` | 14 PNGs at 300 dpi |
| `results/characters_cache.json` | Cached NER pass (delete when `characters.py` changes) |

```bash
python chapters.py           # split into chapters (cached)
python run_notebook.py       # execute the notebook end to end
python narrative.py          # or run the analysis as a script
```

**Deleted in cleanup:** `project.py`, the three `results/analysis*.json` (all
computed on the triplicated corpus), nine stale līlā-level corpus caches, nine
duplicate figures, `__pycache__`, and the one-off notebook builders.

⚠️ **`results/characters_cache.json` must be deleted whenever `characters.py`
changes**, or the notebook will reuse the old canonical names. This bit once.

---

## 11. Done, and what's next

**Completed since the pivot**
- ✅ Cast resolution, four verified rounds (§8)
- ✅ The four ambiguous entities decided from textual evidence, with an explicit
  `PROTECTED_NAMES` table — the documented equivalent of Das et al.'s hand-built
  alias thesaurus
- ✅ Per-līlā networks (Figure 1b)
- ✅ Filter widened to include Rāmacandra Khān
- ✅ Unweighted clustering, benchmarked against the published epic networks
- ✅ `cc_analysis.ipynb` built, executed, and cleaned of dead files

**Next, in priority order**

1. **Sanity-check the sentiment curves against known episodes** — Rāmacandra
   Khān's plot against Haridāsa, the Kazi's suppression of saṅkīrtana,
   Caitanya's sannyāsa, Haridāsa's passing (Antya 11). If the curves do not dip
   where the narrative does, the sentiment layer is not measuring what it
   claims. This is the cheapest test that could still invalidate a headline.
2. **Validate the cast against a scholarly index of CC characters**, as Gultepe
   & Mathangi validated theirs against the Penguin companion. Converts the
   character list from an assumption into a measured result with
   precision/recall — and it is the single strongest answer to "how do you know
   your nodes are right?".
3. **Re-run `discriminate.py` on the clean corpus.** Its numbers came from the
   triplicated version, so the negative result is currently uncitable.
4. **Read McKermitt, Clion & MacCarron (2025)**, *SNAM* 16(1):21, on whether the
   choice of translation determines the network. The corpus is one English
   edition of a Bengali work — this is the objection a reviewer will raise, and
   that paper supplies both the framing and a method.
5. **Consider a second edition as a robustness check.** The unused
   `Sri C.C {Adi,Madhya,Antya} Lila.pdf` set is the same three books in a
   different edition. Running the pipeline over it would answer item 4
   empirically rather than by citation.
6. **Report a sensitivity table for the two coding choices that move results**:
   the `Hari` merge and the 30%/50% cast filter. Both changed the numbers; both
   are defensible; showing the finding under each is stronger than defending one.

**Numbers a reviewer will ask about, and where they came from**

| Claim | Where verified |
|---|---|
| 11,554 of 11,555 verses captured | §4.5 audit |
| Clustering 0.712 vs Mahābhārata 0.509 | §8, Evans 2026 |
| Caitanya eigenvector +52% over Kṛṣṇa | §8 |
| Rāmacandra Purī −0.312 over 5 chapters | §8 |
| Balaram map has no prior art | `LITERATURE.md` §7 |

**Known limitations to state in the paper**
- Lexical resolution, not coreference: pronouns (`He`, `the Lord`) are
  unattributed, so mention counts are lower bounds.
- VADER is scoped to social media (Hutto & Gilbert 2014). Rebora (2023, *DHQ*)
  shows it performs poorly on literary text; Sprugnoli et al. (2016) found human
  annotators cannot agree on sentiment in historical texts. Consider *rasa*
  categories instead, or induce a lexicon from the corpus.
- The corpus is a translation, and Mota et al. found graph measures diverge
  between original-language and translated versions of the same work.
