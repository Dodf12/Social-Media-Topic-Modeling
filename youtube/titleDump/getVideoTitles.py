import googleapiclient.discovery
import json

api_key = "AIzaSyDwXkKkBKqaus41XH9CmD3hA5PeE_gthI8"
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

def get_video_titles(query, max_results=500):
    # Make a search request to retrieve video information
    search_response = youtube.search().list(
        q=query,
        part='snippet',
        maxResults=max_results,
        type='video'
    ).execute()


    videoObj = []
    for item in search_response.get('items', []):
        title = item['snippet']['title']
        videoObj.append({'title': title, 'id': item['id']['videoId']})

    return videoObj

def main():
    query = "Using LLMs as therapist"
    titles = get_video_titles(query)
    
    outputFile = 'titleDump/videoData.json'
    with open(outputFile, 'w') as f:
        json.dump(titles, f, indent=4)
    
    query2 = "Is ChatGPT a good therapist?"
    titles2 = get_video_titles(query2)
    with open('titleDump/videoData2.json', 'w') as f:
        json.dump(titles2, f, indent=4)

    query3 = "AI Therapist"
    titles3 = get_video_titles(query3)
    with open('titleDump/videoData3.json', 'w') as f:
        json.dump(titles3, f, indent=4)

    print(titles)

if __name__ == "__main__":
    main()
