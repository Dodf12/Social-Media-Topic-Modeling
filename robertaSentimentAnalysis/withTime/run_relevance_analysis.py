"""
1. Fetch all 10k rows from Supabase (YoutubeCommentswTimestamp)
2. Clean HTML / junk
3. Zero-shot relevance filter — keep comments about "ChatGPT as a therapist"
4. RoBERTa sentiment on the relevant subset
5. Monthly sentiment share (%) table + chart
6. Save CSV
"""
import re, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from supabase import create_client, Client
from transformers import pipeline

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = 0
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = -1
print(f"Device: {DEVICE}\n")

# ── 1. Fetch all rows (paginated) ─────────────────────────────────────────────
SUPABASE_URL = "https://yxfkplhpjgdwgfabrsrf.supabase.co"
SUPABASE_KEY = "sb_publishable_DTSUBBYxEoDpsH5e9HP40g_Q1YiQKeK"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

PAGE_SIZE = 1000
all_rows  = []
page      = 0

print("Fetching rows from Supabase ...")
while True:
    start = page * PAGE_SIZE
    end   = start + PAGE_SIZE - 1
    resp  = (
        supabase.table("YoutubeCommentswTimestamp")
        .select("id, video_id, Video_title, Comment_Timestamp, Comment")
        .range(start, end)
        .execute()
    )
    batch = resp.data
    if not batch:
        break
    all_rows.extend(batch)
    print(f"  page {page+1}: {len(batch)} rows  (total: {len(all_rows)})")
    if len(batch) < PAGE_SIZE:
        break
    page += 1

raw_df = pd.DataFrame(all_rows)
print(f"Total fetched: {len(raw_df)}\n")

# ── 2. Clean & filter useless comments ───────────────────────────────────────
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_ENT_RE = re.compile(r"&#?\w+;")
EMOJI_RE    = re.compile(
    "[\U0001F600-\U0001FFFF\U00002700-\U000027BF"
    "\U0001F300-\U0001F5FF\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F\U00002600-\U000026FF]+",
    flags=re.UNICODE,
)
URL_RE    = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HANDLE_RE = re.compile(r"@\w+")

def clean_text(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = HTML_ENT_RE.sub("'", text)
    text = text.strip()
    return text

def is_useless(text: str) -> bool:
    if not isinstance(text, str):
        return True
    if len(text.strip()) < 10:
        return True
    core = EMOJI_RE.sub("", text)
    core = URL_RE.sub("", core)
    core = HANDLE_RE.sub("", core)
    core = re.sub(r"[^\w\s]", "", core).strip()
    if len(core) < 5:
        return True
    if re.fullmatch(r"[\d\s]+", core):
        return True
    return False

df = raw_df.copy()
df["Comment"] = df["Comment"].astype(str).apply(clean_text)

before = len(df)
df = df[~df["Comment"].apply(is_useless)]
df = df.drop_duplicates(subset=["Comment"]).reset_index(drop=True)
after = len(df)
print(f"After cleaning: {before} → {after} comments ({before - after} removed)\n")

# ── 3. Zero-shot relevance classifier ────────────────────────────────────────
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
print(f"Loading zero-shot model: {ZERO_SHOT_MODEL} ...")
try:
    zs_pipe = pipeline(
        "zero-shot-classification",
        model=ZERO_SHOT_MODEL,
        device=DEVICE,
    )
except Exception:
    print("  MPS load failed — falling back to CPU")
    zs_pipe = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL, device=-1)
print("Zero-shot model ready.\n")

CANDIDATE_LABELS    = ["AI or ChatGPT as a therapist or mental health tool", "unrelated to AI therapy"]
HYPOTHESIS_TEMPLATE = "This YouTube comment is about {}"
RELEVANCE_THRESHOLD = 0.5   # keep if P(relevant label) > this

texts      = df["Comment"].tolist()
is_relevant = []

# Zero-shot pipeline doesn't batch via a simple loop well, so we call it directly in chunks
BATCH_SIZE = 16
print(f"Running relevance classifier on {len(texts)} comments ...")
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Relevance"):
    chunk   = texts[i : i + BATCH_SIZE]
    results = zs_pipe(chunk, candidate_labels=CANDIDATE_LABELS,
                      hypothesis_template=HYPOTHESIS_TEMPLATE)
    if isinstance(results, dict):
        results = [results]
    for r in results:
        # Score for the first (relevant) label
        idx   = r["labels"].index(CANDIDATE_LABELS[0])
        score = r["scores"][idx]
        is_relevant.append(score >= RELEVANCE_THRESHOLD)

df["relevant"]       = is_relevant
relevant_df          = df[df["relevant"]].reset_index(drop=True)
print(f"\nRelevant comments: {len(relevant_df)} / {len(df)} "
      f"({len(relevant_df)/len(df)*100:.1f}% of cleaned set, "
      f"{len(relevant_df)/len(raw_df)*100:.1f}% of all 10k)\n")

# ── 4. RoBERTa sentiment on relevant subset ───────────────────────────────────
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
print(f"Loading sentiment model: {SENTIMENT_MODEL} ...")
try:
    sent_pipe = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        truncation=True,
        max_length=512,
        device=DEVICE,
    )
except Exception:
    print("  MPS load failed — falling back to CPU")
    sent_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL,
                         truncation=True, max_length=512, device=-1)
print("Sentiment model ready.\n")

sent_texts = relevant_df["Comment"].tolist()
labels, scores = [], []

SENT_BATCH = 32
print(f"Running sentiment on {len(sent_texts)} relevant comments ...")
for i in tqdm(range(0, len(sent_texts), SENT_BATCH), desc="Sentiment"):
    results = sent_pipe(sent_texts[i : i + SENT_BATCH])
    for r in results:
        labels.append(r["label"])
        scores.append(round(r["score"], 6))

relevant_df = relevant_df.copy()
relevant_df["Sentiment_Label"] = labels
relevant_df["Sentiment_Score"] = scores
print(f"\nSentiment done.\n")

# ── 5. Monthly aggregation ────────────────────────────────────────────────────
relevant_df["timestamp"] = pd.to_datetime(relevant_df["Comment_Timestamp"], utc=True)
relevant_df["month"]     = relevant_df["timestamp"].dt.to_period("M")

monthly = (
    relevant_df.groupby(["month", "Sentiment_Label"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)
ordered = [l for l in ["positive", "neutral", "negative"] if l in monthly.columns]
monthly = monthly[ordered]
monthly_pct = monthly.div(monthly.sum(axis=1), axis=0).mul(100).round(2)

# ── 6. Print results ──────────────────────────────────────────────────────────
print("=" * 60)
print(f"RELEVANCE SUMMARY  (topic: ChatGPT as a therapist)")
print("=" * 60)
print(f"  Total Supabase rows : {len(raw_df)}")
print(f"  After junk filter   : {after}")
print(f"  Relevant comments   : {len(relevant_df)}")
print()

print("=" * 60)
print("MONTHLY SENTIMENT SHARE (%)")
print("=" * 60)
print(monthly_pct.to_string())
print()

print("=" * 60)
print("MONTHLY COMMENT COUNTS")
print("=" * 60)
print(monthly.to_string())
print()

# ── 7. Save CSV ────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "relevant_sentiment_with_time.csv")
relevant_df.to_csv(csv_path, index=False)
print(f"CSV saved  → {csv_path}")

# ── 8. Chart ──────────────────────────────────────────────────────────────────
COLORS  = {"positive": "#4CAF50", "neutral": "#2196F3", "negative": "#F44336"}
x_lbls  = [str(p) for p in monthly_pct.index]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle(
    'Sentiment Over Time — Comments About "ChatGPT as a Therapist"',
    fontsize=14, fontweight="bold",
)

# Line chart — raw counts
for label in ordered:
    ax1.plot(x_lbls, monthly[label], marker="o", markersize=4,
             linewidth=2, color=COLORS[label], label=label.capitalize())
ax1.set_ylabel("Comments")
ax1.set_title("Monthly Comment Volume by Sentiment", fontweight="bold")
ax1.legend(framealpha=0.9)

# Stacked bar — % share
bottom = np.zeros(len(monthly_pct))
for label in ordered:
    ax2.bar(x_lbls, monthly_pct[label], bottom=bottom,
            color=COLORS[label], label=label.capitalize(), alpha=0.85)
    bottom += monthly_pct[label].values
ax2.set_ylabel("Share (%)")
ax2.set_title("Monthly Sentiment Share (%) — Relevant Comments Only", fontweight="bold")
ax2.set_ylim(0, 100)
ax2.legend(framealpha=0.9)
ax2.tick_params(axis="x", rotation=45)

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, "relevant_sentiment_over_time.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"Chart saved → {chart_path}")
print("\nAll done.")
