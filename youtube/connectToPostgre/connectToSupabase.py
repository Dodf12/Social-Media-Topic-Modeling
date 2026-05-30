import os
from supabase import create_client, Client
import json




def connectDBTest(supabase: Client):
  response = (
    supabase.table('YoutubeComments')
    .insert({"id": 1, "Video_Id": "3BgdhHXJ-CU", "Video_title": "Test Video", "Comment_Text": "This is a test comment"})
    .execute()
  )
  return response


def shuntComments(supabase: Client):
    with open('videoComments/commentsData.json', 'r') as f:
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


def main():
  supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
  #print(connectDBTest(supabase))
  shuntComments(supabase)

if __name__ == "__main__":
    main()