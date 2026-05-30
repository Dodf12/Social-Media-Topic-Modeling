import json
from googleapiclient.discovery import build


youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

def getVideoComments(filePath):
    videoIds = filePath
    with open(videoIds, 'r') as f:
        video_data = json.load(f)
    
    comments = {}
    for video in video_data:
        commentObj = []
        commentObj.append({'title': video['title']})
        video_id = video['id']

        try: 
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=1000,
                textFormat="plainText"
            ).execute()
        except Exception as e:
            print(f"Error fetching comments for video ID {video_id}: {e}")
            continue

        counter = 0
        for r in request.get('items', []):
            comment = r['snippet']['topLevelComment']['snippet']['textDisplay']
            commentObj.append({'comment': comment})

        comments[video_id] = commentObj
    
    outputFile = 'videoComments/commentsData3.json'
    with open(outputFile, 'w') as f:
        json.dump(comments, f, indent=4)


def main():
    # getVideoComments("titleDump/videoData.json")
    # getVideoComments("titleDump/videoData2.json")
    getVideoComments("titleDump/videoData3.json")

if __name__ == "__main__":
    main()
