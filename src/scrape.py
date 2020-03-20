import os

import googleapiclient.discovery
import googleapiclient.errors
from getdevkey import getdevkey
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import numpy as np

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
DEVELOPER_KEY = getdevkey()                                                         

def main():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "YOUR_CLIENT_SECRET_FILE.json"

    # Get credentials and create an API client
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY
    )

    for year in range(2010, 2020):
        for month in range(2, 13):
            request = youtube.search().list(
                part="snippet",
                maxResults=50,
                q="news",
                videoCategoryId="25",
                type="video",
                publishedAfter="{0}-{1}-01T00:00:00Z".format(year, month - 1),
                publishedBefore="{0}-{1}-01T00:00:00Z".format(year, month),
                regionCode="CA",
                relevanceLanguage="en"
            )

            response = request.execute()
            # print(response)

            ids = [str(vid['id']['videoId']) for vid in response['items'] if 'videoId' in vid['id']]
            names = [str(vid['snippet']['title']) for vid in response['items']]
            print(len(ids), len(names))
            print(names)
            filtered_ids = []
            for video_id in ids:
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    filtered_ids.append(video_id)
                    print('Successfully added {0}'.format(video_id))
                except:
                    print('Error')
                # Format transcript into one string
                # total_transcript = ""
                # for item in transcript:
                #     total_transcript = total_transcript + " " + item["text"]
                # print(total_transcript) 
            
            np.savetxt('news_ids_{0}_{1}.csv'.format(year, month - 1), np.asarray(filtered_ids), fmt='%s')


if __name__ == "__main__":
    main()
