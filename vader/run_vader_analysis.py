"""
Fetch all YoutubeCommentswTimestamp rows from Supabase (now 15k), filter,
run VADER sentiment analysis, save CSV, and generate charts.
"""
import re, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from supabase import create_client, Client
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Fetch all rows (paginated) ─────────────────────────────────────────────

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

PAGE_SIZE = 1000
all_rows  = []
page      = 0

print("Fetching all rows from Supabase ...")
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
    print(f"  page {page+1}: fetched {len(batch)} rows  (total so far: {len(all_rows)})")
    if len(batch) < PAGE_SIZE:
        break
    page += 1

raw_df = pd.DataFrame(all_rows)
print(f"\nTotal rows fetched: {len(raw_df)}\n")

# ── 2. Clean & filter ─────────────────────────────────────────────────────────
EMOJI_RE  = re.compile(
    "[\U0001F600-\U0001FFFF\U00002700-\U000027BF"
    "\U0001F300-\U0001F5FF\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F\U00002600-\U000026FF]+",
    flags=re.UNICODE,
)
URL_RE    = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HANDLE_RE = re.compile(r"@\w+")
HTML_RE   = re.compile(r"<[^>]+>")       # strip HTML tags e.g. <a href=...>
HTML_ENT  = re.compile(r"&#?\w+;")       # decode HTML entities &amp; &#39; etc.

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
print(f"Rows before filtering : {before}")
print(f"Rows after  filtering : {after}  ({before - after} removed)\n")

# ── 3. VADER sentiment analysis ───────────────────────────────────────────────
print("Running VADER sentiment analysis ...")
sid = SentimentIntensityAnalyzer()

compound_scores = []
for text in tqdm(df["Comment"], desc="VADER"):
    compound_scores.append(sid.polarity_scores(text)["compound"])

df["VADER_Score"] = compound_scores

def classify_sentiment(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df["VADER_Sentiment"] = df["VADER_Score"].apply(classify_sentiment)

counts = df["VADER_Sentiment"].value_counts()
print(f"\nNegative : {counts.get('Negative', 0)}")
print(f"Neutral  : {counts.get('Neutral',  0)}")
print(f"Positive : {counts.get('Positive', 0)}")
print()

# ── 4. Parse timestamps & monthly aggregation ─────────────────────────────────
df["timestamp"] = pd.to_datetime(df["Comment_Timestamp"], utc=True)
df["month"]     = df["timestamp"].dt.to_period("M")

monthly = (
    df.groupby(["month", "VADER_Sentiment"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)
ordered = [l for l in ["Positive", "Neutral", "Negative"] if l in monthly.columns]
monthly = monthly[ordered]
monthly_pct = monthly.div(monthly.sum(axis=1), axis=0).mul(100).round(2)

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

# ── 5. Save CSV ────────────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "vader_sentiment_results.csv")
df.to_csv(csv_path, index=False)
print(f"CSV saved  → {csv_path}\n")

# ── 6. Charts ─────────────────────────────────────────────────────────────────
COLORS = {"Positive": "#4CAF50", "Neutral": "#2196F3", "Negative": "#F44336"}
x_lbls = [str(p) for p in monthly_pct.index]

fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle("YouTube Comment VADER Sentiment Analysis\n(15,000 Comments)", fontsize=15, fontweight="bold")

# ── Panel 1: histogram of compound scores
ax0 = axes[0]
ax0.hist(df["VADER_Score"], bins=40, color="#5C6BC0", edgecolor="white", alpha=0.85)
ax0.axvline(0, color="black", linewidth=1.2, linestyle="--", label="Neutral boundary")
ax0.set_xlabel("VADER Compound Score")
ax0.set_ylabel("Number of Comments")
ax0.set_title("Distribution of VADER Compound Scores", fontweight="bold")
ax0.legend()

# ── Panel 2: monthly comment volume by sentiment (line)
ax1 = axes[1]
for label in ordered:
    ax1.plot(x_lbls, monthly[label], marker="o", markersize=4,
             linewidth=2, color=COLORS[label], label=label)
ax1.set_ylabel("Comments")
ax1.set_title("Monthly Comment Volume by Sentiment", fontweight="bold")
ax1.legend(framealpha=0.9)
ax1.tick_params(axis="x", rotation=45)

# ── Panel 3: monthly sentiment share % (stacked bar)
ax2 = axes[2]
bottom = np.zeros(len(monthly_pct))
for label in ordered:
    ax2.bar(x_lbls, monthly_pct[label], bottom=bottom,
            color=COLORS[label], label=label, alpha=0.85)
    bottom += monthly_pct[label].values
ax2.set_ylabel("Share (%)")
ax2.set_title("Monthly Sentiment Share (%)", fontweight="bold")
ax2.set_ylim(0, 100)
ax2.legend(framealpha=0.9)
ax2.tick_params(axis="x", rotation=45)

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, "vader_sentiment_over_time.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"Chart saved → {chart_path}\n")
print("All done.")
