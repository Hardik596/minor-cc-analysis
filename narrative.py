"""Ramcharitmanas-style computational analysis of the Caitanya-caritāmṛta.

Replicates the design of the Ramcharitmanas study on the CC translations, with
chapters (62) standing in for kāṇḍas (7):

  1. corpus statistics
  2. character identification (frequency threshold + top-30% filter)
  3. character interaction network -- nodes are CHARACTERS, edges weighted by
     sentence-level co-occurrence
  4. centralities (degree, betweenness, closeness, eigenvector) + communities
  5. per-character sentiment trajectory across chapters
  6. per-character emotion profile (NRC lexicon)
  7. topic modelling (LDA, 5 topics)
  8. word frequencies per līlā
  9. word clouds per character

Note this is a CHARACTER network, not the word-adjacency graph the original
project.py sketched. The two are unrelated methods; the word-adjacency
attributes were shown to carry no discriminative signal (see discriminate.py),
whereas character networks have solid precedent — Das, Das & Mahesh (ICON 2016)
on the Mahābhārata being the closest analogue to this design.

Run:  python narrative.py
"""

import itertools
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

import matplotlib
if __name__ == "__main__":
    # Headless file output when run as a script. NOT set on import: doing so
    # forces Agg inside a Jupyter kernel too, and plt.show() then renders
    # nothing inline.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

import chapters as chapters_mod
import characters as char_mod
import common

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(RESULTS, "figures")

MIN_MENTIONS = 10       # as in the Ramcharitmanas study
# The Ramcharitmanas study used the top 30%. Raised to 50% here because at 30%
# the cutoff falls at 38 mentions, which excludes Rāmacandra Khān (24 mentions,
# rank 77 of 162) -- the figure who plots against Haridāsa Ṭhākura and the
# clearest human antagonist in Antya-līlā. A filter that drops a named
# antagonist is measuring prominence, not narrative role. 50% admits him with
# margin (cutoff 22) and takes the cast from 48 to 81; the network figures draw
# only the top 20, so legibility is unaffected.
TOP_PERCENT = 0.50
TOP_NETWORK = 20        # characters drawn in the interaction network
N_TOPICS = 5
LILA_ORDER = ["Adi", "Madhya", "Antya"]


# --------------------------------------------------------------------------
# characters
# --------------------------------------------------------------------------

def find_characters(nlp, chapter_texts):
    """Mention counts per canonical character, and their per-sentence spans.

    Entity candidates are pooled across spaCy labels and resolved by name,
    because label assignment is unreliable on this corpus (Caitanya is tagged
    GPE, Kṛṣṇa splits between PERSON and ORG). See characters.py.
    """
    mentions = Counter()
    surface_to_canon = {}
    for text in chapter_texts:
        for doc in nlp.pipe(_chunk(text), batch_size=4):
            for ent in doc.ents:
                if ent.label_ not in char_mod.USEFUL_LABELS:
                    continue
                canon = char_mod.canonicalise(ent.text)
                if canon:
                    mentions[canon] += 1
                    surface_to_canon[ent.text.strip()] = canon
    return mentions, surface_to_canon


def _chunk(text, limit=400_000):
    buf, size = [], 0
    for para in text.split("\n\n"):
        if size + len(para) > limit and buf:
            yield "\n\n".join(buf)
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 2
    if buf:
        yield "\n\n".join(buf)


def select_cast(mentions):
    """Frequency threshold, then the top 30% — the Ramcharitmanas filter."""
    eligible = [(c, n) for c, n in mentions.most_common() if n >= MIN_MENTIONS]
    keep = max(1, int(len(eligible) * TOP_PERCENT))
    return [c for c, _ in eligible[:keep]], eligible


def mention_index(cast, surface_to_canon):
    """Map every observed surface form to its canonical name, for matching."""
    index = defaultdict(set)
    for surface, canon in surface_to_canon.items():
        if canon in cast:
            index[canon].add(surface)
    for canon in cast:
        index[canon].add(canon)
    return index


def characters_in(sentence, index, _cache={}):
    """Which cast members are named in this sentence — longest match wins.

    Plain substring containment double-counts nested names: a sentence naming
    "Rāmacandra Purī" also contains "Rāmacandra", which registered BOTH, created
    a spurious edge between them, and inflated the shorter name. Three distinct
    Gopīnāthas clustered the same way.

    Each matched span is therefore masked before shorter forms are tested, so
    "Rāmacandra Purī" consumes its text and bare "Rāmacandra" does not also fire.
    """
    key = id(index)
    ordered = _cache.get(key)
    if ordered is None:
        # Longest forms first, so specific names claim their span before
        # general ones get a chance to match inside it.
        ordered = sorted(
            ((form, canon) for canon, forms in index.items() for form in forms),
            key=lambda pair: -len(pair[0]))
        _cache[key] = ordered

    remaining = sentence
    present = set()
    for form, canon in ordered:
        if canon in present:
            continue
        at = remaining.find(form)
        if at != -1:
            present.add(canon)
            remaining = remaining[:at] + "\x00" * len(form) + remaining[at + len(form):]
    return present


# --------------------------------------------------------------------------
# interaction network
# --------------------------------------------------------------------------

def build_interaction_network(chapter_records, index):
    """Characters are nodes; an edge weight counts shared sentences.

    Sentence-level co-occurrence follows Das, Das & Mahesh (2016). A narrower
    window than theirs is not needed here because CC sentences are short.
    """
    G = nx.Graph()
    for record in chapter_records:
        for sentence in record["sentences"]:
            present = characters_in(sentence, index)
            for a in present:
                G.add_node(a)
            for a, b in itertools.combinations(sorted(present), 2):
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)
    return G


def network_metrics(G):
    if G.number_of_nodes() == 0:
        return {}
    try:
        eigen = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except nx.PowerIterationFailedConvergence:
        eigen = {n: float("nan") for n in G}
    communities = nx.community.louvain_communities(G, weight="weight", seed=0)
    membership = {n: i for i, group in enumerate(communities) for n in group}
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "degree": nx.degree_centrality(G),
        "betweenness": nx.betweenness_centrality(G, weight=None),
        "closeness": nx.closeness_centrality(G),
        "eigenvector": eigen,
        "communities": [sorted(c) for c in communities],
        "membership": membership,
        # Unweighted. networkx's weighted clustering normalises every edge by the
        # maximum weight in the graph, and the Caitanya-Kṛṣṇa edge here is so
        # dominant that all other triangles collapse toward zero -- the weighted
        # figure came out at 0.010, which says nothing about the structure. The
        # unweighted coefficient is also what the comparable literature reports
        # (Das et al. 2016; Mac Carron & Kenna 2012).
        "clustering": nx.average_clustering(G),
        "clustering_weighted": nx.average_clustering(G, weight="weight"),
        "assortativity": (nx.degree_assortativity_coefficient(G)
                          if G.number_of_edges() > 1 else float("nan")),
    }


# --------------------------------------------------------------------------
# sentiment and emotion
# --------------------------------------------------------------------------

def sentiment_trajectories(chapter_records, index, analyzer, cast):
    """Mean compound sentiment per character per chapter."""
    traj = {c: [] for c in cast}
    for record in chapter_records:
        scores = defaultdict(list)
        for sentence in record["sentences"]:
            present = characters_in(sentence, index)
            if not present:
                continue
            compound = analyzer.polarity_scores(sentence)["compound"]
            for c in present:
                scores[c].append(compound)
        for c in cast:
            traj[c].append(round(statistics.fmean(scores[c]), 4)
                           if scores.get(c) else None)
    return traj


def emotion_profiles(chapter_records, index, cast, stop_words):
    """NRC emotion counts over each character's sentences.

    nrclex 4.x builds the lexicon per instance and exposes `load_token_list`, so
    one instance is reused and each character's tokens are scored in a single
    call. Scoring sentence by sentence would rebuild the lexicon tens of
    thousands of times.
    """
    from nrclex import NRCLex
    lexicon = NRCLex()

    bucket = defaultdict(list)
    for record in chapter_records:
        for sentence in record["sentences"]:
            for c in characters_in(sentence, index):
                bucket[c].append(sentence)

    profiles = {}
    for c in cast:
        if not bucket.get(c):
            continue
        tokens = [w.lower() for s in bucket[c] for w in word_tokenize(s)
                  if w.isalpha() and w.lower() not in stop_words]
        lexicon.load_token_list(tokens)
        totals = Counter(lexicon.raw_emotion_scores)
        total = sum(totals.values()) or 1
        profiles[c] = {k: round(v / total, 4) for k, v in totals.most_common()}
    return profiles, bucket


# --------------------------------------------------------------------------
# topics and word frequencies
# --------------------------------------------------------------------------

def topic_model(texts, stop_words, n_topics=N_TOPICS, top_words=12):
    vec = CountVectorizer(max_df=0.9, min_df=3, max_features=20000,
                          stop_words=list(stop_words))
    matrix = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0,
                                    learning_method="batch")
    lda.fit(matrix)
    names = vec.get_feature_names_out()
    return [[names[i] for i in comp.argsort()[:-top_words - 1:-1]]
            for comp in lda.components_]


def word_frequencies(text, stop_words, top=15):
    counts = Counter(w.lower() for w in word_tokenize(text)
                     if w.isalpha() and w.lower() not in stop_words
                     and len(w) > 2)
    return counts.most_common(top)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def plot_network(G, metrics, path, top=TOP_NETWORK):
    degree = metrics["degree"]
    keep = sorted(degree, key=degree.get, reverse=True)[:top]
    H = G.subgraph(keep)
    pos = nx.spring_layout(H, seed=1, weight="weight", k=0.7)
    sizes = [3000 * degree[n] + 260 for n in H]
    colours = [metrics["membership"].get(n, 0) for n in H]
    widths = [0.4 + 2.6 * (H[u][v]["weight"] / max(
        d["weight"] for *_, d in H.edges(data=True))) for u, v in H.edges()]
    plt.figure(figsize=(13, 10))
    nx.draw_networkx_edges(H, pos, width=widths, alpha=0.3)
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=colours,
                           cmap="tab10", alpha=0.9)
    nx.draw_networkx_labels(H, pos, font_size=9)
    plt.title(f"Character interaction network — top {top} characters")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_sentiment(traj, labels, path, top=6):
    plt.figure(figsize=(15, 7))
    for c in list(traj)[:top]:
        xs = [i for i, v in enumerate(traj[c]) if v is not None]
        ys = [v for v in traj[c] if v is not None]
        if xs:
            plt.plot(xs, ys, marker="o", ms=3, label=c, linewidth=1.4)
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.xticks(range(0, len(labels), 3), [labels[i] for i in
               range(0, len(labels), 3)], rotation=60, ha="right", fontsize=7)
    plt.ylabel("mean compound sentiment")
    plt.title("Sentiment trajectory by chapter")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_emotions(profiles, path, top=6):
    emotions = ["trust", "joy", "anticipation", "surprise",
                "fear", "sadness", "anger", "disgust"]
    names = list(profiles)[:top]
    width = 0.8 / max(len(names), 1)
    plt.figure(figsize=(13, 6))
    for i, name in enumerate(names):
        vals = [profiles[name].get(e, 0) for e in emotions]
        plt.bar([x + i * width for x in range(len(emotions))], vals,
                width=width, label=name)
    plt.xticks([x + 0.4 for x in range(len(emotions))], emotions)
    plt.ylabel("share of emotion words")
    plt.title("Emotion profile by character (NRC lexicon)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_mentions(chapter_records, index, cast, labels, path, top=5):
    plt.figure(figsize=(15, 6))
    for c in cast[:top]:
        counts = [sum(1 for s in r["sentences"] if c in characters_in(s, index))
                  for r in chapter_records]
        plt.plot(range(len(counts)), counts, marker="o", ms=3, label=c)
    plt.xticks(range(0, len(labels), 3), [labels[i] for i in
               range(0, len(labels), 3)], rotation=60, ha="right", fontsize=7)
    plt.ylabel("sentences mentioning character")
    plt.title("Narrative presence by chapter")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_wordclouds(bucket, cast, stop_words, top=5):
    for c in cast[:top]:
        if not bucket.get(c):
            continue
        text = " ".join(bucket[c])
        cloud = WordCloud(width=1100, height=650, background_color="white",
                          stopwords=set(stop_words) | {c.lower()},
                          collocations=False).generate(text)
        safe = c.replace(" ", "_").replace("/", "-")
        cloud.to_file(os.path.join(FIGURES, f"wordcloud_{safe}.png"))


# --------------------------------------------------------------------------

def main():
    common.ensure_nltk()
    os.makedirs(FIGURES, exist_ok=True)
    nlp = common.load_spacy()
    analyzer = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words("english"))

    print("Loading chapters...", file=sys.stderr)
    chapters = chapters_mod.load_chapters()
    records = [{"label": c["label"], "lila": c["lila"], "chapter": c["chapter"],
                "text": c["text"], "sentences": sent_tokenize(c["text"])}
               for c in chapters]
    labels = [r["label"] for r in records]

    stats = {
        "chapters": len(records),
        "lilas": len(LILA_ORDER),
        "characters_total": None,
        "sentences": sum(len(r["sentences"]) for r in records),
        "words": sum(len(word_tokenize(r["text"])) for r in records),
        "chars": sum(len(r["text"]) for r in records),
    }
    stats["unique_words"] = len({w.lower() for r in records
                                 for w in word_tokenize(r["text"]) if w.isalpha()})

    print("Identifying characters...", file=sys.stderr)
    mentions, surface = find_characters(nlp, [r["text"] for r in records])
    cast, eligible = select_cast(mentions)
    stats["characters_total"] = len(eligible)
    stats["characters_kept"] = len(cast)
    index = mention_index(cast, surface)

    print(f"  {len(eligible)} characters with >={MIN_MENTIONS} mentions; "
          f"kept top {len(cast)}", file=sys.stderr)

    print("Building interaction network...", file=sys.stderr)
    G = build_interaction_network(records, index)
    metrics = network_metrics(G)

    print("Sentiment trajectories...", file=sys.stderr)
    traj = sentiment_trajectories(records, index, analyzer, cast)

    print("Emotion profiles...", file=sys.stderr)
    profiles, bucket = emotion_profiles(records, index, cast, stop_words)

    print("Topic model...", file=sys.stderr)
    topics = topic_model([r["text"] for r in records], stop_words)

    freq_by_lila = {}
    for lila in LILA_ORDER:
        joined = "\n".join(r["text"] for r in records if r["lila"] == lila)
        freq_by_lila[lila] = word_frequencies(joined, stop_words)

    print("Figures...", file=sys.stderr)
    plot_network(G, metrics, os.path.join(FIGURES, "character_network.png"))
    plot_sentiment(traj, labels, os.path.join(FIGURES, "sentiment_trends.png"))
    plot_emotions(profiles, os.path.join(FIGURES, "emotions.png"))
    plot_mentions(records, index, cast, labels,
                  os.path.join(FIGURES, "narrative_presence.png"))
    save_wordclouds(bucket, cast, stop_words)

    def rank(metric, n=15):
        d = metrics.get(metric, {})
        return [(k, round(v, 4)) for k, v in
                sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]]

    out = {
        "corpus_stats": stats,
        "top_characters": eligible[:80],
        "cast": cast,
        "network": {
            "nodes": metrics.get("nodes"),
            "edges": metrics.get("edges"),
            "density": round(metrics.get("density", 0), 4),
            "clustering": round(metrics.get("clustering", 0), 4),
            "assortativity": round(metrics.get("assortativity", 0), 4),
            "communities": metrics.get("communities"),
            "degree_top": rank("degree"),
            "betweenness_top": rank("betweenness"),
            "closeness_top": rank("closeness"),
            "eigenvector_top": rank("eigenvector"),
        },
        "sentiment_by_chapter": traj,
        "chapter_labels": labels,
        "emotions": profiles,
        "topics": topics,
        "word_frequencies_by_lila": freq_by_lila,
    }
    path = os.path.join(RESULTS, "narrative.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f"\nCorpus: {stats['chapters']} chapters, {stats['sentences']:,} sentences, "
          f"{stats['words']:,} words, {stats['unique_words']:,} unique")
    print(f"Characters: {stats['characters_total']} with >={MIN_MENTIONS} mentions, "
          f"cast of {stats['characters_kept']}")
    print(f"Network: {metrics['nodes']} nodes, {metrics['edges']} edges, "
          f"density {metrics['density']:.4f}, "
          f"{len(metrics['communities'])} communities")
    print("\nTop by degree:      ", rank("degree", 8))
    print("Top by betweenness: ", rank("betweenness", 8))
    print("Top by eigenvector: ", rank("eigenvector", 8))
    print(f"\nWrote {path} and figures to {FIGURES}")


if __name__ == "__main__":
    main()
