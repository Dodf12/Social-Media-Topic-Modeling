

prompts = [
    "My experience with ChatGPT as my therapist",
    "How ChatGPT helped me through a tough time",
    "The emotional support I found in ChatGPT",
    "Why your human therapist will never be replacaed by ChatGPT",
    "The surprising benefits of using ChatGPT for mental health",
    "How ChatGPT became my go-to for emotional support",
    "How human therapy can be better than ChatGPT for mental health",
]
import requests
from supabase import create_client, Client
from googleapiclient.discovery import build

# --- CONFIGURATION ---
SUPABASE_URL = "https://yxfkplhpjgdwgfabrsrf.supabase.co"
SUPABASE_KEY = "sb_publishable_DTSUBBYxEoDpsH5e9HP40g_Q1YiQKeK"
YOUTUBE_API_KEY = "AIzaSyDwXkKkBKqaus41XH9CmD3hA5PeE_gthI8"

# --- SUPABASE SETUP ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def search_videos(query, max_results=10):
    print(f"Searching videos for prompt: {query}")
    request = youtube.search().list(q=query, part="snippet", type="video", maxResults=max_results)
    response = request.execute()
    videos = []
    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        videos.append({"video_id": video_id, "title": title})
    print(f"Found {len(videos)} videos for prompt '{query}'")
    return videos

def get_comments(video_id, video_title, max_total=1000):
    print(f"Fetching comments for video: {video_title} ({video_id})")
    comments = []
    next_page_token = None
    while len(comments) < max_total:
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={YOUTUBE_API_KEY}&maxResults=100"
        if next_page_token:
            url += f"&pageToken={next_page_token}"
        resp = requests.get(url)
        data = resp.json()
        before_filter = len(data.get("items", []))
        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = snippet["textDisplay"]
            timestamp = snippet["publishedAt"]
            word_count = len(text.split())
            if 3 <= word_count <= 100:
                comments.append({
                    "video_id": video_id,
                    "Video_title": video_title,
                    "Comment_Timestamp": timestamp,
                    "Comment": text
                })
            if len(comments) >= max_total:
                break
        print(f"Fetched {before_filter} comments, {len(comments)} valid so far.")
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break
    return comments

def insert_comments(comments):
    for chunk in [comments[i:i+100] for i in range(0, len(comments), 100)]:
        print(f"Inserting chunk of {len(chunk)} comments...")
        supabase.table("YoutubeCommentswTimestamp").insert(chunk).execute()

def main():
    all_comments = []
    seen_comments = set()
    for prompt in prompts:
        videos = search_videos(prompt, max_results=10)
        for video in videos:
            video_id = video["video_id"]
            video_title = video["title"]
            comments = get_comments(video_id, video_title, max_total=1000)
            new_comments = []
            for c in comments:
                if c["Comment"] not in seen_comments:
                    seen_comments.add(c["Comment"])
                    new_comments.append(c)
            print(f"Adding {len(new_comments)} unique comments from video '{video_title}' ({len(comments) - len(new_comments)} duplicates skipped)")
            all_comments.extend(new_comments)
            if len(all_comments) >= 5000:
                break
        if len(all_comments) >= 5000:
            break
    all_comments = all_comments[:5000]
    print(f"Total comments to insert: {len(all_comments)}")
    insert_comments(all_comments)
    print(f"Inserted {len(all_comments)} comments into Supabase.")

if __name__ == "__main__":
    main()