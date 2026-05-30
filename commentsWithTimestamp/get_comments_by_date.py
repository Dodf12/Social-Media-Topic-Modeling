import requests
from googleapiclient.discovery import build
from datetime import datetime, timezone
import csv
import os


prompts = [
    "My experience with ChatGPT as my therapist",
    "How ChatGPT helped me through a tough time",
    "The emotional support I found in ChatGPT",
    "Why your human therapist will never be replacaed by ChatGPT",
    "The surprising benefits of using ChatGPT for mental health",
    "How ChatGPT became my go-to for emotional support",
    "How human therapy can be better than ChatGPT for mental health",
]

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)


def search_videos(query, max_results=10):
    print(f"Searching: {query}")
    request = youtube.search().list(q=query, part="snippet", type="video", maxResults=max_results)
    response = request.execute()
    return [
        {"video_id": item["id"]["videoId"], "title": item["snippet"]["title"]}
        for item in response.get("items", [])
    ]


def get_all_comments(video_id, video_title):
    comments = []
    next_page_token = None
    while True:
        url = (
            f"https://www.googleapis.com/youtube/v3/commentThreads"
            f"?part=snippet&videoId={video_id}&key={YOUTUBE_API_KEY}&maxResults=100"
        )
        if next_page_token:
            url += f"&pageToken={next_page_token}"
        resp = requests.get(url)
        data = resp.json()
        if "error" in data:
            print(f"  API error for {video_id}: {data['error']['message']}")
            break
        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "video_title": video_title,
                "timestamp": snippet["publishedAt"],
                "date": snippet["publishedAt"][:10],  # YYYY-MM-DD
            })
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break
    return comments


def main():
    all_comments = []
    seen_video_ids = set()

    for prompt in prompts:
        videos = search_videos(prompt, max_results=10)
        for video in videos:
            vid = video["video_id"]
            if vid in seen_video_ids:
                continue
            seen_video_ids.add(vid)
            comments = get_all_comments(vid, video["title"])
            print(f"  {len(comments)} comments from '{video['title']}'")
            all_comments.extend(comments)

    # Sort by full ISO timestamp, earliest first
    all_comments.sort(key=lambda c: c["timestamp"])

    print(f"\nTotal comments: {len(all_comments)}")
    if all_comments:
        print(f"Earliest: {all_comments[0]['timestamp']}")
        print(f"Latest:   {all_comments[-1]['timestamp']}")

    # Write sorted dates to CSV
    out_path = os.path.join(os.path.dirname(__file__), "comments_by_date.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "timestamp", "video_id", "video_title"])
        writer.writeheader()
        writer.writerows(all_comments)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
