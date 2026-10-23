# Literature review — prior work and novelty assessment

Compiled for a planned publication on graph-theoretic and sentiment analysis of
the Śrī Caitanya-caritāmṛta. Citations marked **[verified]** were checked
directly against the source in this session; the rest come from a literature
search and should be confirmed before citing.

---

## 1. The headline: prior work on this text exists

**You cannot claim "first computational analysis of the Caitanya-caritāmṛta."**

### Apurba Paul (2024) — the closest prior work **[verified]**

> *Extracting and Visualizing Ontologies of Characters, Topics and Relations
> from Texts.* PhD thesis, Dept. of CSE, Jadavpur University, Kolkata, 2024.
> 190 pp. Supervisor: Dipankar Das.
> <https://irju.jdvu.ac.in/jspui/bitstream/123456789/10196/1/PhD%20Thesis%20%28Computer%20Science%20%26%20Engineering%29%20Apurba%20Paul.pdf>

I downloaded and text-extracted this thesis. Confirmed directly:

- CC is **one of seven texts** analysed (with Rāmāyaṇa, Mahābhārata, Devī
  Bhāgavata, Harivaṃśa, Śrīmad-Bhāgavatam, and *Kṛṣṇa, the Supreme Personality
  of Godhead*), abbreviated "CC" throughout.
- **Source is the English translation** — `vedabase.io/en/library` and
  `wisdomlib.org` are cited in footnotes. Same source family as ours.
- CC statistics, Table 3.1: **20.1k sentences, 311.6k unigrams**, avg. sentence
  length 15.5 words, **20% of sentences considered**.
- It **does** do per-character sentiment (Table 5.10, "Top 5 characters with
  highest positive/negative sentiments") and per-character TF-IDF (Table 5.11).
  This overlaps our sentiment and TF-IDF work directly.
- It uses **spaCy and NLTK** — the same toolchain.

**But** — searched the full 349,803-character extraction:

| Term | Hits in thesis |
|---|---|
| `edge density` / `global efficien` / `compactness` | **0** |
| `betweenness` | **0** |
| `PageRank` | **0** |
| `word-adjacency` / `co-occurrence network` | **0** |
| `speech graph` | **0** |
| `centrality` | 2 — both describing *other people's* work in related work |

So the graph-theoretic layer is genuinely absent from the closest prior work.
The overlap is on **sentiment + character identification + TF-IDF**, and it is
substantial.

Related outputs from the same group, to cite together:
- Paul, Seal & Das (2024), "Transformer-based Pouranic topic classification in
  Indian mythology," *Sādhanā* 49:263. doi:10.1007/s12046-024-02598-6
- Paul & Das, "MythoBERT 1.0," *Sādhanā*. CC is in the pretraining corpus.
- Paul et al. (2023), "Mytho-Annotator," *ICON 2023*, pp. 574–578.
  <https://aclanthology.org/2023.icon-1.54/>

### Other work touching the CC

- **Bose (2026)**, "From Vajrayana Tara to Bengali Baul," arXiv:2606.26803 —
  the only work touching the **Bengali-language** CC, as one of 75 texts, via
  raw archive.org OCR. Non-peer-reviewed preprint; flag as such.
- **Sandhan et al. (2023/2025)**, "Aesthetics of Sanskrit Poetry … Śikṣāṣṭaka,"
  WSC-CSDH 2025. arXiv:2308.07081. Computational analysis of **Caitanya
  Mahāprabhu's own eight verses**, using *rasa* rather than Western sentiment
  categories. Thematically the nearest neighbour, and evidence that WSC-CSDH is
  a receptive venue.

### What is genuinely untouched

- No study treats the CC as its **own object** rather than a row in a table.
- No computational work on the **original Bengali/Sanskrit** CC (all existing
  work uses translation or dirty OCR).
- No stylometry, character-network, or topic model of the CC as such.
- No comparative study of CC vs. Caitanya Bhāgavata / Caitanya Maṅgala.

---

## 2. The method is not novel in kind — and this is the biggest threat

**Mota et al. (2016), "The ontogeny of discourse structure mimics the
development of literature," arXiv:1612.09268** already applied speech-graph
attributes to **676 literary and religious texts spanning ~5,000 years**,
including the **four Vedas, Mahābhārata I Ādi Parva**, Gilgamesh, Enuma Elish,
the Egyptian Book of the Dead, the Avesta, and Old Testament books.

They used SpeechGraphs with **30-token moving windows at 50% overlap** and
normalised against word-shuffled surrogates. Their finding singles out Hinduist
literature: it "evolved across 2,750 years from a primitive pattern of
near-random long-range recurrence to its opposite."

This must be cited and differentiated. Residual novelty has to come from
something narrower than "graph metrics on a religious text."

Their own caveat is worth quoting in your limitations section, because it
describes our situation exactly:

> "The mathematical analysis of ancient texts is inherently impacted by a
> plethora of confounds, such as imprecise dating, variable physical support,
> multiple authorship and versions, editing, censorship, standardization,
> **translation**…"

They also found **LSC behaves differently in original-language versions vs.
English translations** — directly relevant, since our corpus is translation.

---

### Verified corpus contents **[verified]**

I extracted the Mota et al. 2016 PDF directly, because a second search produced
the opposite claim ("no word-adjacency network of any Indian epic or Hindu
religious text"). That claim is **false**. Supplementary Table 2 contains:

- A named **"Hinduist" tradition, N=37**
- **Rigveda, Sāma veda, Yajur veda, Atharva veda** (Vedic-Sanskrit)
- **Mahābhārata parvas I–XIV as separate texts** — Ādi, Sabhā, Vana, Virāṭa,
  Udyoga, Bhīṣma, Droṇa, Karṇa, Śalya, Sauptika, Strī, Śānti, Aśvamedhika…

Method confirmed verbatim: *"Graph analysis was performed using the free software
SpeechGraphs"*, with *"moving windows of 30 words with 50% of overlap"* — nodes
are words, not characters. Sources include the Internet Sacred Text Archive.

**Consequence for your framing:** word-adjacency graph analysis of Hindu
religious texts is well established. What survives is narrower and still
defensible: Mota's Hinduist corpus is Vedic and epic, and contains **no Gauḍīya
Vaiṣṇava material at all**. "First speech-graph/word-adjacency analysis of a
Gauḍīya Vaiṣṇava text" holds; "first on a Hindu religious text" does not.

Convenient convergence: our `measure_windowed()` default (30 tokens, 50%
overlap) is exactly their protocol, so the numbers are directly comparable.

---

## 3. The metrics were mis-attributed

The original code's metric names pointed at the wrong provenance. Corrected:

| Metric as implemented | Correct citation |
|---|---|
| directed word-adjacency graph | Ferrer i Cancho & Solé (2001), *Proc. R. Soc. B* 268:2261, doi:10.1098/rspb.2001.1800 — the true ancestor |
| "speech graph" framing, edge density, LCC/LSC, ASP | Mota et al. (2012) *PLoS ONE* 7:e34928; Mota et al. (2014) *Sci. Rep.* 4:3691 |
| lemmatised, stopword-free co-occurrence networks | Amancio (2015), *PLoS ONE* 10:e0118394 — **our actual preprocessing lineage**, not Mota's |
| ~~"compactness"~~ → **global efficiency** | Latora & Marchiori (2001), *PRL* 87:198701 |
| PageRank on a word graph | Mihalcea & Tarau (2004), TextRank, EMNLP — **if you rank words by PageRank, that is TextRank** |
| degree/betweenness/closeness on word graphs of religious text | Czachesz, *J. Cognitive Historiography* 3(1–2):43–67 — precedent for exactly this |
| entropy of degree distribution | No speech-graph precedent. Cite network-entropy literature separately. |

Two naming errors were live in the code and are now fixed:

1. **"Compactness" is not a speech-graph attribute at all.** The name belongs to
   a different published measure (Botafogo, Rivlin & Shneiderman 1992). The code
   computed `networkx.global_efficiency`, which is Latora & Marchiori. Renamed.
2. Mota's preprocessing keeps **all** words; ours removes stopwords and
   lemmatises. That makes our graphs Amancio-style, **so Mota's normative values
   and effect directions do not transfer.**

---

## 4. The length confound — measured on our own corpus

The central known validity problem in this literature, and it was live in our
code. Measured on 4,854 Ādi-līlā sentences:

```
r(nodes, edge_density)      = -0.725
r(nodes, global_efficiency) = -0.836
r(nodes, entropy)           = +0.491
r(edge_density, 2/n)        = +0.972   <- chain-graph prediction
```

A word-adjacency graph over *n* distinct tokens is nearly a chain: *n*−1 edges,
so density ≈ 2/*n*. **Per-sentence edge density is very nearly a reciprocal
length measure.** Any cross-text comparison built on it is comparing sentence
lengths.

Fixes now implemented:
- `measure_windowed()` — fixed 30-token windows, 50% overlap (Mota et al. 2016).
  Use this for **all cross-līlā comparison**.
- `length_controlled_scores()` — z-scoring within node-count bins for the
  within-text sentence ranking.

### The confound changes the answer, not just the error bars

Running both measurements over the same corpus:

| Metric | Ādi | Madhya | Antya | Spread | Rank order |
|---|---|---|---|---|---|
| **Per-sentence (uncontrolled)** ||||||
| edge density | 0.2885 | 0.3078 | 0.3057 | 6.42% | Ādi < Antya < Madhya |
| global efficiency | 0.5103 | 0.5282 | 0.5256 | 3.43% | Ādi < Antya < Madhya |
| entropy | 0.8991 | 0.8845 | 0.8808 | 2.06% | Antya < Madhya < Ādi |
| **Windowed, 30 tokens (controlled)** ||||||
| edge density | 0.0917 | 0.0928 | 0.0888 | 4.39% | **Antya < Ādi < Madhya** |
| global efficiency | 0.3048 | 0.3070 | 0.2974 | 3.17% | **Antya < Ādi < Madhya** |
| entropy | 0.9759 | 0.9759 | 0.9781 | **0.23%** | **Ādi < Madhya < Antya** |

Two things to note:

1. **The rank order changes.** Uncontrolled, Ādi-līlā has the lowest edge density
   and global efficiency of the three. Controlled, it is in the middle and Antya
   is lowest. The uncontrolled result was an artifact of Ādi simply having longer
   sentences (9.64 mean nodes vs. 8.90 and 8.89).
2. **The entropy difference all but disappears** — spread collapses from 2.06% to
   0.23%, a ninefold reduction, and the order fully reverses. Essentially all of
   the apparent between-līlā entropy variation was sentence length.

Any cross-līlā claim built on the uncontrolled numbers would have been wrong in
direction, not merely imprecise. This is worth reporting in the paper as a
methodological result in its own right.

Supporting citations:
- Mota et al. (2014) replaced whole-text normalisation with moving windows for
  precisely this reason.
- **Amancio (2015)** quantifies stability: most such measures are unstable below
  a few hundred tokens — far above sentence scale.
- A 2024 scoping review of NLP in schizophrenia found plain **word count
  out-predicted most graph metrics**. Any claim of signal must survive a length
  control.

---

## 5. Sentiment analysis validity — the weakest part of the method

- **Rebora (2023), "Sentiment Analysis in Literary Studies: A Critical Survey,"
  *DHQ* 17(2)** — cite this first. Names VADER specifically as performing poorly
  on literary text; notes most SA in DH is **unvalidated** with "frail or
  disputable" links to literary theory.
- **Sprugnoli et al. (2016), *DSH* 31(4):762–772** — the killer finding:
  **inter-annotator agreement among humans** on positive/negative sentiment in
  historical texts fell below acceptable thresholds. If humans cannot agree,
  delegating to a machine is hard to justify.
- **Hutto & Gilbert (2014)**, the VADER paper itself, scopes it to *social
  media*. Using it on 16th-century devotional prose in translation is a scope
  violation you should name before a reviewer does.
- **Alternative worth engaging**: Sandhan et al. use **rasa** (Nāṭyaśāstra)
  rather than Western valence. For this corpus that is the more defensible
  emotional ontology.
- **Constructive option**: Hamilton et al. (2016), EMNLP — induce a sentiment
  lexicon from the target corpus instead of importing one.

Precedent to differentiate from:
- **Das, Das & Mahesh (2016), "A Computational Analysis of Mahabharata," ICON
  2016, pp. 219–228.** <https://aclanthology.org/W16-6328/> — NLP + sentiment +
  **social network analysis** on an Indian epic, tracking sentiment across the
  eighteen parvas alongside character centrality. **This is the closest
  precedent to the paper you are planning.** Ganguli translation, 2.86M words,
  sentence-window character co-occurrence; betweenness/closeness/degree/
  eigenvector centrality, Louvain communities, SentiWordNet + NRC emotion
  lexicon. Argues **betweenness** best discriminates true protagonists.

---

## 5b. Character networks of Indian epics — the parallel literature

Distinct from §2: these build networks whose **nodes are characters**, not words.
Ours is a word network, so this is related work rather than direct competition —
but the protagonist/antagonist claim collides with it.

- **Gultepe & Mathangi (2023)**, "A Quantitative Social Network Analysis of the
  Character Relationships in the Mahabharata," *Heritage* 6(11):7009–7030,
  doi:10.3390/heritage6110366. **Separates Pandavas (protagonists) from Kauravas
  (antagonists) at F = 0.749.** ⚠️ **Your main competitor on automated
  protagonist/antagonist detection — it is already published for an Indian epic.**
- **Evans (2026)**, "A Structural Comparison of Indian Epics," arXiv:2607.10702 —
  manually built character networks for Mahābhārata (N=537) and Rāmāyaṇa (N=247)
  vs the Iliad; full complex-network battery.
- **Evans (2026)**, "Gendered Structural Roles in Indian Epics," arXiv:2608.00534.
- **Janickyj, MacCarron, Yose & Kenna (2024)**, "Female representation across
  mythologies," *Condensed Matter Physics* 27(3):33802,
  doi:10.5488/CMP.27.33802 — the Mahābhārata is the **only** Indian text in the
  Mac Carron/Kenna comparative corpus of 21 narratives.
- **Mac Carron & Kenna (2012)**, "Universal properties of mythological networks,"
  *EPL* 99(2):28002 — the founding comparative paper (Beowulf, Iliad, Táin).
  **Contains no Indian epic.**
- **Labatut & Bost (2019)**, "Extraction and Analysis of Fictional Character
  Networks: A Survey," *ACM Computing Surveys* 52(5):89, doi:10.1145/3344548 —
  the survey to anchor a method section, including role detection.
- **Massey (2016)**, "Social network analysis of the biblical Moses," *Applied
  Network Science* 1:13 — argues that unusually highly connected deity/prophet
  nodes may be **characteristic of networks from religious texts**. A useful
  comparator given how dominant Caitanya and Kṛṣṇa are in our counts.

- **Elwert & Sellmer (2017)**, "Modeling Structure and Content: Socio-Semantic
  Network Analysis of the Mahābhārata," *Leonardo* 50(5):501,
  doi:10.1162/leon_a_01277 — character nodes carrying **automatically derived
  topic distributions** (Nubbi) on both actors and relations. The earliest
  automatic semantically-enriched character network of an Indian epic, and the
  closest anyone has come to joining word/topic-level information to character
  nodes. Cite it before claiming novelty for anything semantically enriched.
- **McKermitt, Clion & MacCarron (2025)**, "Comparing social networks of
  translations: a case study on the medieval Irish epic Táin Bó Cúailnge,"
  *Social Network Analysis and Mining* 16(1):21, doi:10.1007/s13278-025-01547-3.
  ⚠️ **Read this one.** It asks whether the choice of translation/edition
  determines the resulting network — which is precisely the objection a reviewer
  will raise against our corpus, built entirely from one English edition. It
  gives you both the framing and a method for answering it.
- **Rajput, Ahuja & Riyal (2019)**, "A statistical probe into the word frequency
  and length distributions prevalent in the translations of Bhagavad Gita,"
  *Pramana* 92(4):60, doi:10.1007/s12043-018-1709-8 — power law in Sanskrit vs
  Zipf–Mandelbrot in translation, with KL divergence and Shannon entropy. **The
  nearest existing word-level statistical-physics treatment of a Hindu text**;
  word distributions, not a word network. Cite as the closest neighbour.
- **MacCarron (2024)**, *Condensed Matter Physics* 27(3):33803 — models lost
  material via random edge removal; a precedent for arguing about recensions.

Reported gaps in this sub-literature (from search, **not** independently
verified): no character network of the Ramcharitmanas (a systematic Shodhganga
sweep reportedly confirms this — relevant because the original script targeted
Ramcharitmanas), the Bhagavad Gītā, or the Purāṇas.

**Corrected during review:** eigenvector centrality *is* used in this literature
(Yose et al. 2018; Das et al. 2016). The safe formulation is narrower: **no prior
study of an Indian epic reports PageRank.**

**Unverified risk to check before submission:** Wadhwa et al. (2025),
"Computational Analysis on Shrimad Bhagwat Geeta," LNNS 1269:139–157,
doi:10.1007/978-981-96-4319-6_12 — its abstract claims network analysis and its
bibliography is dominated by character-network papers. Paywalled; nobody in this
review could read its methods. This is the single biggest open risk to a
Gītā-network novelty claim.

---

## 5c. ⚠️ Fabricated citations are circulating in this exact area

The search flagged that **Nayan & Patnaik (2025)**, "Changing Emotions in Epics"
(Springer, doi:10.1007/978-981-96-6759-8_2) carries a reference list containing
entries that could not be found in Crossref or any index, including:

> Sharma, A. & Verma, R. (2019). "Analyzing emotional constructs in Valmiki's
> Ramayana using sentiment analysis." *Journal of South Asian Literature*,
> 10(2), 50–68.

…plus ~a dozen more with the same signature (generic two-surname authors,
plausible journal names, no DOI): Desai & Kulkarni 2020; Smith & Johnson 2018;
Brown & Lee 2020; Evans & Clark 2016; Garcia & Green 2019; Martinez & Wang 2017;
Patel & O'Connor 2021; Rezaei & Mohammadi 2017; Tanaka & Suzuki 2018; Williams &
Ahmed 2017; Allen & Scott 2018.

**I could not independently verify this** — the session's web-search budget was
exhausted before I could check Crossref myself. Treat it as an unconfirmed
warning, but act on it anyway: **resolve a DOI for every citation before it
enters your bibliography**, and be especially careful with anything you find via
an AI summariser. Citing a hallucinated reference is among the fastest ways to
lose a reviewer's confidence.

---

## 6. NER failure — publishable as a finding, with support

Our measured failure (Caitanya → GPE, Kṛṣṇa split PERSON/ORG, Vedānta → PERSON)
is not an anecdote; there is literature to anchor it.

- **Padārtha** (2026), arXiv:2608.29324 — ontology-grounded fine-grained Sanskrit
  NER benchmark, >108k mentions over >73.6k Mahābhārata verses, 18 categories
  from Nyāya-Vaiśeṣika ontology, **explicitly rejecting "modern journalistic
  conventions"** (the PERSON/ORG/GPE schema). Strongest citation for "CoNLL-style
  NER is the wrong ontology for devotional Sanskrit."
- **Naamapadam** (ACL 2023), <https://aclanthology.org/2023.acl-long.582/> —
  largest Indic NER dataset, includes Bengali; releases IndicNER.
- **MuRIL** (arXiv:2103.10730) — trained on Indian languages **and their
  transliterated counterparts**. The strongest published justification for
  preferring MuRIL over English spaCy on IAST text.
- **Ehrmann et al.**, "NER on Historical Documents: A Survey," arXiv:2109.11406 —
  canonical domain-mismatch citation.
- Historical-NER baselines report spaCy F1 ≈ 0.50 with ORG recall 0.12.

- **Mahānāma**, "A Unique Testbed for Literary Entity Discovery and Linking,"
  arXiv:2509.19844 — entity linking over the Mahābhārata. Directly relevant:
  alias resolution is the hardest part of any Indian-epic entity pipeline, and
  this is the nearest existing testbed for it.

**Gap found: no published work on epithet/alias resolution in Hindu devotional
texts**, nor on the many-names-for-one-deity problem in any Indic tradition.
Our `characters.py` addresses exactly this. Nearest analogues are Homeric-epithet
work in Classics (arXiv:2208.10795) and generic literary alias/coreference work
(arXiv:2412.11560).

---

## 7. Where the real novelty is: the Balaram encoding recovery

**No peer-reviewed publication exists on the Balaram font or its encoding.**
Everything that exists is undocumented community tooling:

- A 2004 pseudonymous JavaScript converter (bvmlu.org)
- A bare "Convert" button on vedabase.io
- A Pratyatosa utility; a Hare Krishna Society page

**None publishes a mapping table.** Our `balaram.py` map — derived empirically
and validated against known words — is a genuine, citable resource contribution,
and the most clearly novel thing in the project.

Supporting literature to position it against:
- **Mahi & Verma (2015)**, "Wrecked Indian Fonts" — legacy Indic font migration
  as a scholarly problem; notes **over 1,000 mapping tables** exist for Indian
  fonts.
- **Rahaman & Mahmud (2013)**, ICIEV/IEEE — Bengali Unicode font converter.
- **Hill & Hengchen (2019), *DSH* 34(4):825–843** — the canonical "dirty OCR
  degrades downstream analysis" paper; also proof DSH publishes this genre.
- **van Strien et al. (2020)** — OCR quality effects on NER, parsing, topic
  modelling specifically.

**Framing that makes this land:** legacy-font mojibake is *not* OCR error. It is
a deterministic, fully recoverable encoding mismatch. That distinction is itself
the contribution — the dirty-OCR literature supplies the rhetorical frame, the
Indic font-conversion literature the technical frame, and **nobody has joined
them for a romanised-Sanskrit devotional corpus.**

---

## 8. Corpus validity issue to disclose

~40% of the extracted text is **Bhaktivedanta Swami Prabhupāda's 1970s English
commentary (PURPORT)**, not Kṛṣṇadāsa Kavirāja's 16th-century text. In Ādi-līlā:
TRANSLATION 35.2%, PURPORT 39.4%, other 25.5%.

Any claim about "the Caitanya-caritāmṛta" computed over the default mix is
measuring a 16th-century Bengali work and a 20th-century commentary on it
together. `extract.py` now takes a `sections` argument;
`sections=("TRANSLATION",)` isolates the text itself.

Note this also explains the size discrepancy with Paul's thesis (20.1k sentences
vs. our 67k): they appear to have scraped translations only.

### The contamination is differential, not uniform

Measured from section headings:

| Līlā | TRANSLATION | PURPORT | PURPORT headings |
|---|---|---|---|
| Ādi | 35.2% | **39.4%** | 887 |
| Antya | 55.3% | **9.3%** | 382 |

Prabhupāda's purports are dense in Ādi-līlā and thin out sharply by Antya — a
real feature of the edition, confirmed by heading counts. So the blended corpus
does not dilute every volume equally: **cross-līlā comparison on it is partly
comparing how much 1970s English commentary each volume contains.**

### Result of the translation-only run **[verified]**

Both corpora analysed; blended results preserved at `results/analysis.blended.json`,
translation-only at `results/analysis.translation.json`.

**Sentiment — the trend survives and strengthens:**

| Līlā | Blended | Translation-only | Δ | Purport share |
|---|---|---|---|---|
| Ādi | 0.2194 | 0.2375 | +0.0181 | 39.4% |
| Madhya | 0.2078 | 0.2270 | +0.0192 | ~26% |
| Antya | 0.1810 | 0.1860 | +0.0050 | 9.3% |

The monotonic decline Ādi → Madhya → Antya holds in both, and the **range widens**
from 0.0384 to 0.0515 once commentary is removed. The purports are slightly more
negative than the translations, and because they are concentrated in Ādi, they
were *masking* the decline rather than creating it. The trend is a property of
Kṛṣṇadāsa Kavirāja's narrative, not of Prabhupāda's commentary. This is now a
defensible finding.

**Graph metrics — essentially unchanged:**

| Metric (windowed) | Blended | Translation-only |
|---|---|---|
| edge density | 0.0917 / 0.0928 / 0.0888 | 0.0916 / 0.0935 / 0.0893 |
| global efficiency | 0.3048 / 0.3070 / 0.2974 | 0.3057 / 0.3099 / 0.2996 |
| entropy | 0.9759 / 0.9759 / 0.9781 | 0.9761 / 0.9758 / 0.9779 |

Rank order is preserved exactly for edge density and global efficiency
(Antya < Ādi < Madhya); the entropy order swaps Ādi/Madhya, but on a difference
in the fourth decimal — noise.

⚠️ **This robustness is a warning, not a reassurance** — and the discriminative
check below confirms it.

### The discriminative check: the graph attributes carry almost no signal **[verified]**

`discriminate.py` classifies a 30-token window as TRANSLATION (Kṛṣṇadāsa
Kavirāja, 16th c., in translation) or PURPORT (Prabhupāda, 1970s English).
Non-overlapping windows, CV splits grouped by contiguous block so no test window
neighbours a training window, classes balanced by downsampling. 15,576 windows,
7,788 per class.

| Condition | Accuracy | AUC |
|---|---|---|
| 5 graph attributes, logistic regression | 0.552 (±0.004) | 0.577 |
| 5 graph attributes, random forest | 0.522 (±0.002) | 0.528 |
| 3 normalised attributes only, random forest | 0.522 (±0.003) | 0.527 |
| **Shuffled labels (empirical null)** | **0.499 (±0.010)** | **0.501** |
| **TF-IDF unigrams (lexical baseline)** | **0.863 (±0.013)** | **0.935** |

The task is easily solvable — words separate these registers at AUC 0.935. The
graph attributes reach AUC 0.53–0.58 against a 0.50 null. There is a *detectable*
signal, but it is negligible.

Per-attribute differences between the two registers:

| Attribute | Translation | Purport | Δ |
|---|---|---|---|
| edge density | 0.0916 | 0.0909 | −0.0007 |
| global efficiency | 0.3053 | 0.3000 | −0.0053 |
| entropy | 0.9766 | 0.9762 | −0.0005 |

### Effect sizes between līlās are negligible too **[verified]**

Cohen's *d* over non-overlapping 30-token windows, translation-only corpus:

| Metric | Ādi–Madhya | Ādi–Antya | Madhya–Antya |
|---|---|---|---|
| edge density | +0.106 | −0.124 | −0.224 |
| global efficiency | +0.093 | −0.104 | −0.197 |
| entropy | −0.032 | +0.129 | +0.157 |

All negligible by convention (|d| < 0.2) except one small. The between-līlā mean
differences reported earlier are stable only because *n* is large (3,018–5,677
windows per līlā) — they are statistically detectable and substantively trivial.

### What this means for the paper

**Do not report the between-līlā graph differences as findings.** They are real
in the sense that the means differ reliably, and meaningless in the sense that
the effect sizes are negligible and the attributes cannot distinguish texts four
centuries and one language apart.

The honest framing turns this into the paper's strongest methodological
contribution: *word-adjacency graph attributes, applied to a devotional corpus at
the scale the literature uses them, do not carry text-discriminative signal, and
a trivial lexical baseline outperforms them by 30 accuracy points.* Given how
much of the speech-graph literature rests on these attributes — and that Mota et
al. 2016 drew historical conclusions about Hinduist literature from exactly this
measurement family — a careful negative result with a proper null and a lexical
baseline is worth publishing.

Always report **effect sizes and a lexical baseline** alongside these metrics.
Neither is standard practice in this literature, which is part of the problem.

**Character counts shift substantially in Ādi:** Caitanya mentions drop
3,787 → 1,160 there, versus 4,818 → 4,513 in Antya. Ādi's character statistics
were largely driven by commentary, which cites the principals constantly.

**Antagonists change:** translation-only, Ādi and Madhya have none clearing the
support threshold (the Kazi material is substantially in purports), while Antya
retains **Rāmacandra Khān at −0.1178 over 44 sentences** — a stronger signal than
the blended run's −0.0944. He is the one antagonist robust to the corpus choice.

---

## 9. Recommended venues

Three separable contributions; splitting is better than cramming.

| Contribution | Venue | Cost |
|---|---|---|
| **Balaram encoding recovery** (strongest) | *Journal of Open Humanities Data* short data paper; or **WILDRE** @ LREC / **ISCLS** | JOHD £760 (waivers available); workshops free |
| **Graph + sentiment + NER failure analysis** | **LaTeCH-CLfL** (SIGHUM); or **ICON** (ICON-2026, Gauhati, Dec 2026) | Free, ACL Anthology |
| **Methodological critique** (VADER/NER invalid here; extraction errors propagate) | *DHQ* or *Journal of Cultural Analytics* | Free, diamond OA |

- **CLRel** ("New Horizons in Computational Linguistics for Religious Texts"),
  first edition at COLING 2025, <https://aclanthology.org/2025.clrel-1.0/> — the
  only dedicated ACL venue for religious-text computational linguistics. Ten
  papers, of which **only one touches a Hindu text**. That statistic is itself a
  citable argument that the subfield is thin. No confirmed 2026 edition, so
  monitor rather than plan around it.

- **LaTeCH-CLfL** scope explicitly lists "automatic error detection and cleaning
  of textual data," "social networks of literary characters," and "emotion
  analysis" — all three contributions. 11th edition is the realistic target.
- **WSC-CSDH / ISCLS** is the community that will understand a Gauḍīya Vaiṣṇava
  corpus and care about the encoding work. 8th ISCLS was March 2026, IIT Roorkee.
- **DSH** warning: explicitly does *not* publish "descriptions of projects, tools,
  techniques or methodologies" — needs a substantive humanities finding.

**Caution — MDPI *Religions*:** 1,800 CHF APC, and reputationally mixed within
religious studies despite being indexed (AHCI, Scopus). Every venue above is
free. Prefer *Religions of South Asia* or *MTSR* for a religious-studies
audience.

**Avoid** the "International Journal of …" titles from AIRCC, SSRG, Science
Publishing Group that surface on Indic font-conversion searches. Cross-check any
candidate against DOAJ and COPE.

---

## 10. Honest summary of the novelty position

**Defensible claims:**
1. First published **Balaram → Unicode/IAST mapping** with validation — the
   strongest and cleanest contribution, and the one with no prior art at all.
2. First **dedicated** computational study of the Caitanya-caritāmṛta (qualify
   carefully against Paul 2024, which includes CC as one of seven texts).
3. First **word-adjacency / speech-graph analysis of a Gauḍīya Vaiṣṇava text**
   (Mota et al. 2016 covers Vedic and epic Hindu texts, but no Gauḍīya material).
4. First treatment of **epithet/alias resolution for a Hindu devotional text**.
5. A documented **NER ontology-mismatch failure case** on IAST devotional prose.
6. A **demonstration that the length confound reverses conclusions**, not merely
   widens error bars (§4) — a methodological result in its own right.

**Claims to avoid:**
- ✗ "First computational analysis of the CC" — Paul 2024 precedes you.
- ✗ "First application of graph metrics to religious text" — Mota et al. 2016.
- ✗ "First word-adjacency network of a Hindu religious text" — **Mota et al. 2016
  covers all four Vedas and 14 Mahābhārata parvas. [verified directly]** A search
  result asserting otherwise is wrong; I checked the supplementary table.
- ✗ "First automated protagonist/antagonist detection on an Indic text" —
  Gultepe & Mathangi (2023) do exactly this at F = 0.749.
- ✗ "First sentiment analysis of an Indic religious text" — Das et al. 2016,
  Chandra & Kulkarni 2022.
- ✗ Anything resting on uncontrolled per-sentence graph metrics — they are
  largely a length proxy (§4).

**Process note.** Two independent searches returned directly contradictory claims
about whether word-adjacency methods had been applied to Hindu religious texts. I
resolved it by downloading and extracting the primary source. Do the same for any
load-bearing claim in your related-work section — and note that both the Paul
thesis and the Mota PDF defeated an automated fetcher and had to be extracted
with the same tooling built for this project.

**Unverified items** flagged by the search that you should check before citing:
the exact density formula in Mota 2012 Table 1; whether arXiv:1612.09268 reached
peer review; the precise year of the Czachesz article; exact author lists for two
ACL Anthology entries that failed to fetch.
