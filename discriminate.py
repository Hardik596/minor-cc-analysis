"""Can the windowed graph attributes tell the two registers apart?

The translation-only run showed that Kṛṣṇadāsa Kavirāja's 16th-century narrative
(in translation) and Prabhupāda's 1970s English commentary produce near-identical
windowed graph attributes. That is either robustness or a sign the attributes
carry no text-specific signal. This decides which.

Task: classify a 30-token window as TRANSLATION or PURPORT.

  * If the graph attributes score near 50%, they do not distinguish two texts
    four centuries and one language apart, and any between-text claim resting on
    them is unsupported.
  * A lexical (TF-IDF) baseline on the same windows shows how much signal is
    actually there to find, so a low graph score cannot be blamed on the task
    being impossible.
  * A shuffled-label run gives the empirical chance level.

Design choices that protect the result:
  * Windows do NOT overlap (step = window), so no token appears in two windows.
  * Splits are grouped by contiguous block, so a test window is never adjacent to
    a training window -- random k-fold here would leak local vocabulary.
  * Classes are balanced by downsampling, so accuracy is comparable to 50%.

Run:  python discriminate.py
"""

import random
import statistics
import sys

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import extract
import common

WINDOW = 30
BLOCK = 40          # windows per CV group (~1,200 tokens)
SEED = 0

GRAPH_FEATURES = ["nodes", "edges", "edge_density", "global_efficiency", "entropy"]
NORMALISED_ONLY = ["edge_density", "global_efficiency", "entropy"]


def tokenise(text, stop_words, lemmatizer):
    """Same preprocessing as the main pipeline, as a flat token stream."""
    tokens = []
    for sentence in sent_tokenize(text):
        filtered = [w.lower() for w in word_tokenize(sentence)
                    if w.isalpha() and w.lower() not in stop_words]
        tokens.extend((lemmatizer.lemmatize(w, common.wordnet_pos(t)), t)
                      for w, t in pos_tag(filtered))
    return tokens


def windows(tokens):
    """Non-overlapping windows: no token is shared between two samples."""
    for start in range(0, len(tokens) - WINDOW + 1, WINDOW):
        chunk = tokens[start:start + WINDOW]
        G = common.build_graph(chunk)
        if G.number_of_nodes() < 2:
            continue
        yield common.measure(G), " ".join(w for w, _ in chunk)


def build_dataset():
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    rows, texts, labels, groups = [], [], [], []
    group_id = 0
    for label, sections in ((0, ("TRANSLATION",)), (1, ("PURPORT",))):
        corpus = extract.load_corpus(sections=sections)
        for name, text in corpus.items():
            print(f"  windowing {name} "
                  f"{'TRANSLATION' if label == 0 else 'PURPORT'}...",
                  file=sys.stderr)
            tokens = tokenise(text, stop_words, lemmatizer)
            for i, (metrics, surface) in enumerate(windows(tokens)):
                rows.append([metrics[k] for k in GRAPH_FEATURES])
                texts.append(surface)
                labels.append(label)
                groups.append(group_id + i // BLOCK)
            group_id += len(tokens) // (WINDOW * BLOCK) + 2
    return (np.array(rows, dtype=float), np.array(texts, dtype=object),
            np.array(labels), np.array(groups))


def balance(X, texts, y, groups, rng):
    """Downsample the majority class so 50% is the meaningful baseline."""
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    keep = np.concatenate([rng.choice(idx0, n, replace=False),
                           rng.choice(idx1, n, replace=False)])
    rng.shuffle(keep)
    return X[keep], texts[keep], y[keep], groups[keep]


def evaluate(name, make_model, X, y, groups, n_splits=5):
    accs, aucs = [], []
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = make_model()
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        accs.append(accuracy_score(y[test], pred))
        try:
            prob = model.predict_proba(X[test])[:, 1]
            aucs.append(roc_auc_score(y[test], prob))
        except (AttributeError, ValueError):
            pass
    acc = statistics.fmean(accs)
    sd = statistics.pstdev(accs)
    auc = statistics.fmean(aucs) if aucs else float("nan")
    print(f"  {name:<44s} acc {acc:.3f} (±{sd:.3f})   AUC {auc:.3f}")
    return acc, auc


def main():
    common.ensure_nltk()
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    print("Building windows...", file=sys.stderr)
    X, texts, y, groups = build_dataset()
    print(f"\nTotal windows: {len(y):,}  "
          f"(translation {int((y == 0).sum()):,}, purport {int((y == 1).sum()):,})")

    X, texts, y, groups = balance(X, texts, y, groups, rng)
    print(f"Balanced to {len(y):,} windows "
          f"({int((y == 0).sum()):,} each); chance = 0.500\n")

    logreg = lambda: make_pipeline(StandardScaler(),
                                   LogisticRegression(max_iter=2000))
    forest = lambda: RandomForestClassifier(n_estimators=300, random_state=SEED,
                                            n_jobs=-1)

    print("GRAPH ATTRIBUTES (the question)")
    evaluate("all 5 graph attributes, logistic regression",
             logreg, X, y, groups)
    evaluate("all 5 graph attributes, random forest",
             forest, X, y, groups)
    idx = [GRAPH_FEATURES.index(k) for k in NORMALISED_ONLY]
    evaluate("3 normalised attributes only, random forest",
             forest, X[:, idx], y, groups)

    print("\nNULL (labels shuffled within the same design)")
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)
    evaluate("all 5 graph attributes, shuffled labels",
             forest, X, y_shuffled, groups)

    print("\nLEXICAL BASELINE (how much signal actually exists)")
    tfidf = lambda: make_pipeline(
        TfidfVectorizer(max_features=20000, ngram_range=(1, 1)),
        LogisticRegression(max_iter=2000))
    evaluate("TF-IDF unigrams, logistic regression", tfidf, texts, y, groups)

    print("\nPer-attribute means by class")
    print(f"  {'attribute':<20s} {'translation':>12s} {'purport':>10s} {'delta':>9s}")
    for i, k in enumerate(GRAPH_FEATURES):
        a, b = X[y == 0, i].mean(), X[y == 1, i].mean()
        print(f"  {k:<20s} {a:12.4f} {b:10.4f} {b - a:+9.4f}")


if __name__ == "__main__":
    main()
