"""
BERTopic topic modeling over time on the filtered VADER comments.
Only verbs, adjectives, and adverbs are kept (via spaCy POS tagging) so that
topic keywords reflect sentiment-bearing language rather than subject-area nouns.
Produces a monthly topic share stacked bar chart (top N topics).
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spacy
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load already-filtered comments ─────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "vader_sentiment_results.csv")
df = pd.read_csv(csv_path)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["month"] = df["timestamp"].dt.to_period("M")

docs = df["Comment"].astype(str).tolist()
print(f"Loaded {len(docs)} comments.\n")

# ── 2. POS-filter: keep only verbs, adjectives, and adverbs ───────────────────
# This surfaces sentiment-bearing vocabulary and suppresses subject-area nouns
# (e.g. "ai", "therapist") that dominate but carry no sentiment signal.
print("Running spaCy POS tagging (this may take a minute) ...")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

KEEP_POS = {"VERB", "ADJ", "ADV"}

# Words to explicitly exclude even if they pass the POS filter
SENTIMENT_BLOCKLIST = {
    "ai", "therapist", "therapists", "therapy",
    "like", "just", "really", "actually", "literally",
    "basically", "totally", "probably", "definitely",
    "absolutely", "clearly", "obviously", "seriously",
    "maybe", "perhaps", "also", "even", "still",
    "get", "got", "go", "going", "know", "think",
    "say", "said", "see", "look", "make", "made",
    "come", "want", "need", "use", "let", "try",
    "seem", "feel", "mean", "tell", "take", "give",
    "watch", "subscribe", "click", "share", "thank",
    "good", "bad", "great", "nice", "new", "old",
    "yes", "yeah", "nope", "ok", "okay", "sure",
    "lol", "lmao", "omg", "wow",
}

def pos_filter(text):
    doc = nlp(text.lower()[:500])  # cap at 500 chars for speed
    tokens = [
        t.lemma_ for t in doc
        if t.pos_ in KEEP_POS
        and not t.is_stop
        and not t.is_punct
        and len(t.lemma_) > 2
        and t.lemma_ not in SENTIMENT_BLOCKLIST
    ]
    return " ".join(tokens) if tokens else text  # fallback to original if empty

filtered_docs = [pos_filter(d) for d in docs]
print("POS filtering done.\n")

# ── 3. Stopwords for the vectorizer (lightweight since POS already filters) ───
EXTRA_STOPWORDS = list(SENTIMENT_BLOCKLIST) + [
    "i","me","my","we","our","you","your","he","him","his","she","her",
    "it","its","they","them","their","what","which","who","this","that",
    "these","those","am","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","a","an","the","and","but","if",
    "or","as","of","at","by","for","with","to","from","in","on","not",
    "very","s","t","d","ll","m","re","ve","could","would","may","might",
    "shall","can","will","should","much","many","well","every","another",
]

vectorizer = CountVectorizer(
    stop_words=EXTRA_STOPWORDS,
    min_df=5,          # word must appear in at least 5 docs
    ngram_range=(1, 2) # allow bigrams for richer labels
)

# ── 4. Fit BERTopic ───────────────────────────────────────────────────────────
print("Fitting BERTopic (this may take a few minutes) ...")
topic_model = BERTopic(
    vectorizer_model=vectorizer,
    calculate_probabilities=False,
    verbose=True,
    min_topic_size=30,
    nr_topics="auto",
)
# BERTopic uses original docs for embeddings; filtered_docs drive the vocabulary
topics, _ = topic_model.fit_transform(filtered_docs)
df["topic"] = topics

topic_info = topic_model.get_topic_info()
print("\nTop topics found:")
print(topic_info[["Topic", "Count", "Name"]].head(20).to_string(index=False))

# ── 5. Label topics with their top keywords ───────────────────────────────────
def topic_label(tid):
    if tid == -1:
        return "Outliers"
    words = [w for w, _ in topic_model.get_topic(tid)[:4]]
    return " | ".join(words)

df["topic_label"] = df["topic"].apply(topic_label)

# ── 6. Monthly aggregation — keep top N topics (by total count) ───────────────
TOP_N = 8
topic_counts = df[df["topic"] != -1]["topic_label"].value_counts()
top_topics = topic_counts.head(TOP_N).index.tolist()

df_filtered = df[df["topic_label"].isin(top_topics)].copy()

monthly = (
    df_filtered.groupby(["month", "topic_label"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)
# Keep only top topics columns, in order of total size
monthly = monthly[[t for t in top_topics if t in monthly.columns]]
monthly_pct = monthly.div(monthly.sum(axis=1), axis=0).mul(100).round(2)

print("\n" + "=" * 60)
print("MONTHLY TOPIC SHARE (%) — top topics only")
print("=" * 60)
print(monthly_pct.to_string())

# ── 7. Save enriched CSV ──────────────────────────────────────────────────────
out_csv = os.path.join(OUT_DIR, "vader_with_topics.csv")
df.to_csv(out_csv, index=False)
print(f"\nCSV saved → {out_csv}")

# ── 8. Plot ───────────────────────────────────────────────────────────────────
PALETTE = [
    "#4CAF50", "#2196F3", "#F44336", "#FF9800",
    "#9C27B0", "#00BCD4", "#795548", "#607D8B",
    "#E91E63", "#8BC34A",
]
colors = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(top_topics)}
x_lbls = [str(p) for p in monthly_pct.index]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
fig.suptitle("YouTube Comment Topic Modeling Over Time (BERTopic)\n"
             f"Top {TOP_N} Topics from {len(docs):,} Comments",
             fontsize=14, fontweight="bold")

# Line chart — raw counts
for topic in top_topics:
    if topic in monthly.columns:
        ax1.plot(x_lbls, monthly[topic], marker="o", markersize=3,
                 linewidth=2, color=colors[topic], label=topic)
ax1.set_ylabel("Comments")
ax1.set_title("Monthly Comment Volume by Topic", fontweight="bold")
ax1.legend(fontsize=7, framealpha=0.9, loc="upper left",
           bbox_to_anchor=(1.01, 1), borderaxespad=0)
ax1.tick_params(axis="x", rotation=45)

# Stacked bar — % share
bottom = np.zeros(len(monthly_pct))
for topic in top_topics:
    if topic in monthly_pct.columns:
        ax2.bar(x_lbls, monthly_pct[topic], bottom=bottom,
                color=colors[topic], label=topic, alpha=0.85)
        bottom += monthly_pct[topic].values
ax2.set_ylabel("Share (%)")
ax2.set_title("Monthly Topic Share (%) — Top Topics", fontweight="bold")
ax2.set_ylim(0, 100)
ax2.legend(fontsize=7, framealpha=0.9, loc="upper left",
           bbox_to_anchor=(1.01, 1), borderaxespad=0)
ax2.tick_params(axis="x", rotation=45)

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, "topic_modeling_over_time.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"Chart saved → {chart_path}\n")
print("All done.")
