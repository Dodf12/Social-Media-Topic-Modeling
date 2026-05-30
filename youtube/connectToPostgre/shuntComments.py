import os
from supabase import create_client, Client

import json

YOUR_PASSWORD = "bT+4g9mWgV+Kd-Z "
SUPABASE_URL = f"https://yxfkplhpjgdwgfabrsrf.supabase.co"
SUPABASE_KEY = "sb_publishable_DTSUBBYxEoDpsH5e9HP40g_Q1YiQKeK"

def shuntComments(supabase: Client):
    with open('videoComments/commentsData3.json', 'r') as f:
        comments_data = json.load(f)

    for video_id, comment_list in comments_data.items():
        video_title = comment_list[0]['title']
        for comment in comment_list[1:]:
            comment_text = comment['comment']

            try:
              response = (
                  supabase.table('YoutubeComments')
                  .insert({"Video_Id": video_id, "Video_title": video_title, "Comment_Text": comment_text})
                  .execute()
              )
            except Exception as e:
              print(f"Error inserting comment for video ID {video_id}: {e}")
              continue
            print(response)

def main():
  supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
  shuntComments(supabase)

if __name__ == "__main__":
    main()