import googleapiclient.discovery
import json
import os

api_key = os.environ["YOUTUBE_API_KEY"]

youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

# Make a GET request to retrieve video information
request = youtube.videos().list(part="snippet", id="videoId123").execute()

searchTest = youtube.search().list(
  q="Using LLMs as therapist",
  part='snippet',
  maxResults=1,
  type='video'
).execute()

outputFile = 'scrapingData/output.json'
with open(outputFile, 'w') as f:
    json.dump(searchTest, f, indent=4)

# Print the video title
print(searchTest)