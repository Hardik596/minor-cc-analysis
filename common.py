"""Shared NLP setup and word-graph primitives.

Extracted from the superseded `project.py` so the character-network analysis and
the word-graph validity check do not depend on that pipeline.
"""

import math
import sys

import networkx as nx
import nltk

_WORDNET_POS = {"J": "a", "V": "v", "N": "n", "R": "r"}


def wordnet_pos(treebank_tag):
    """Map a Treebank POS tag to the WordNet tag the lemmatiser expects.

    Without this, WordNetLemmatizer defaults to noun and never lemmatises verbs.
    """
    return _WORDNET_POS.get(treebank_tag[:1], "n")


def ensure_nltk():
    """Download NLTK data, tolerating the pre/post-3.8.2 resource renames."""
    wanted = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/stopwords", "stopwords"),
        ("sentiment/vader_lexicon", "vader_lexicon"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for path, pkg in wanted:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def load_spacy():
    """Load en_core_web_sm without the components this project does not use."""
    import spacy
    try:
        # The parser and lemmatizer are unused here; skipping them roughly
        # triples throughput over the corpus.
        return spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer"])
    except OSError:
        sys.exit("spaCy model missing. Run:\n"
                 "  python -m spacy download en_core_web_sm")


# --------------------------------------------------------------------------
# word-adjacency primitives, used only by the validity check
# --------------------------------------------------------------------------

def build_graph(tagged_words):
    """Word-adjacency digraph: an edge from each token to the next."""
    G = nx.DiGraph()
    for (word1, tag1), (word2, tag2) in zip(tagged_words, tagged_words[1:]):
        G.add_node(word1, pos=tag1)
        G.add_node(word2, pos=tag2)
        G.add_edge(word1, word2)
    return G


def edge_density(G):
    n = G.number_of_nodes()
    return 0.0 if n <= 1 else 2 * G.number_of_edges() / (n * (n - 1))


def global_efficiency(G):
    """Latora & Marchiori (2001) global efficiency of the undirected projection."""
    if G.number_of_nodes() < 2:
        return 0.0
    return nx.global_efficiency(G.to_undirected())


def degree_entropy(G, normalise=True):
    """Shannon entropy of the out-degree distribution, guarded and normalised."""
    out_degrees = dict(G.out_degree())
    total = sum(out_degrees.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for degree in out_degrees.values():
        p = degree / total
        if p > 0:
            entropy -= p * math.log2(p)
    if normalise:
        n = G.number_of_nodes()
        if n > 1:
            entropy /= math.log2(n)
    return entropy


def measure(G):
    """Graph-level measures.

    These are NOT size-invariant: measured on this corpus,
    r(nodes, edge_density) = -0.725 and r(edge_density, 2/n) = +0.972, because a
    word-adjacency graph over n tokens is close to a chain. Compare texts only
    via fixed-length windows.
    """
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "edge_density": edge_density(G),
        "global_efficiency": global_efficiency(G),
        "entropy": degree_entropy(G),
    }
