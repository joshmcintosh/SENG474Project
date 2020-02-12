# -*- coding: utf-8 -*-

# Sample Python code for youtube.videos.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python

import os

import googleapiclient.discovery
import googleapiclient.errors
from src.getdevkey import getdevkey

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

    # request = youtube.videos().list(
    #     part="snippet,contentDetails,statistics",
    #     id="uynBcLr8fEc"
    # )

    request = youtube.captions().list(part="id,snippet", videoId="YbJOTdZBX1g")

    response = request.execute()

    print(response)


if __name__ == "__main__":
    main()
