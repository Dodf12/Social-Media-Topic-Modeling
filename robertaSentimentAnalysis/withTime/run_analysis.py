"""
Fetch all YoutubeCommentswTimestamp rows from Supabase, filter, run RoBERTa,
save sentiment_with_time.csv, print monthly sentiment share, and save chart.
"""
import re, os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from supabase import create_client, Client
from transformers import pipeline

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Fetch all rows (paginated) ─────────────────────────────────────────────

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

# ── 2. Filter useless comments ────────────────────────────────────────────────
EMOJI_RE  = re.compile(
    "[\U0001F600-\U0001FFFF\U00002700-\U000027BF"
    "\U0001F300-\U0001F5FF\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F\U00002600-\U000026FF]+",
    flags=re.UNICODE,
)
URL_RE    = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HANDLE_RE = re.compile(r"@\w+")
HTML_RE   = re.compile(r"<[^>]+>")       # strip HTML tags (e.g. <a href=...>)
HTML_ENT  = re.compile(r"&#?\w+;")       # decode HTML entities (&amp; &#39; etc.)

def clean_text(text: str) -> str:
    text = HTML_RE.sub(" ", text)
    text = HTML_ENT.sub(" ", text)
    return text.strip()

def is_useless(text: str) -> bool:
    if not isinstance(text, str):
        return True
    stripped = text.strip()
    if len(stripped) < 10:
        return True
    core = EMOJI_RE.sub("", stripped)
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
df = df.drop_duplicates(subset=["Comment"])
df = df.reset_index(drop=True)
after = len(df)
print(f"Rows before filter: {before}")
print(f"Rows after  filter: {after}  ({before - after} removed)\n")

# ── 3. RoBERTa inference ──────────────────────────────────────────────────────
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
print(f"Loading model {MODEL} ...")
pipe = pipeline(
    "sentiment-analysis",
    model=MODEL,
    truncation=True,
    max_length=512,
    device=-1,
)
print("Model ready. Running inference ...\n")

BATCH_SIZE = 32
texts  = df["Comment"].tolist()
labels = []
scores = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="RoBERTa"):
    results = pipe(texts[i : i + BATCH_SIZE])
    for r in results:
        labels.append(r["label"])
        scores.append(round(r["score"], 6))

df["Sentiment_Label"] = labels
df["Sentiment_Score"] = scores
print(f"\nDone. {len(df)} comments labelled.\n")

# ── 4. Parse timestamps & monthly aggregation ─────────────────────────────────
df["timestamp"] = pd.to_datetime(df["Comment_Timestamp"], utc=True)
df["month"]     = df["timestamp"].dt.to_period("M")

monthly = (
    df.groupby(["month", "Sentiment_Label"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)
ordered = [l for l in ["positive", "neutral", "negative"] if l in monthly.columns]
monthly = monthly[ordered]

monthly_pct = monthly.div(monthly.sum(axis=1), axis=0).mul(100).round(2)

# ── 5. Print monthly sentiment share ─────────────────────────────────────────
print("=" * 55)
print("MONTHLY SENTIMENT SHARE (%)")
print("=" * 55)
print(monthly_pct.to_string())
print()

print("=" * 55)
print("MONTHLY COMMENT COUNTS")
print("=" * 55)
print(monthly.to_string())
print()

# ── 6. Save CSV ────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "sentiment_with_time.csv")
df.to_csv(csv_path, index=False)
print(f"CSV saved  → {csv_path}")

# ── 7. Chart: monthly sentiment share % ──────────────────────────────────────
COLORS  = {"positive": "#4CAF50", "neutral": "#2196F3", "negative": "#F44336"}
x_lbls  = [str(p) for p in monthly_pct.index]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
fig.suptitle("YouTube Comment Sentiment Over Time", fontsize=15, fontweight="bold")

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
ax2.set_title("Monthly Sentiment Share (%)", fontweight="bold")
ax2.set_ylim(0, 100)
ax2.legend(framealpha=0.9)
ax2.tick_params(axis="x", rotation=45)

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, "sentiment_over_time.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"Chart saved → {chart_path}\n")
print("All done.")
