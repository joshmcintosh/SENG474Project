# SENG474Project
Project repository for UVic's SENG 474 Data Mining class. The project focuses on using video transcripts to predict what tags, features, or ratings YouTube would give it. 

## Dev Key 
If you need to use a developer key from Google, add it in a file called `secret.json` and put it in the `src` folder. 
Note that you do not need an API key if you are just using the `youtube_transcript_api`, but can be useful if you need to build your own query.

### Getting a Dev Key
1. Go to <https://console.developers.google.com/apis/dashboard>
2. Click on library
3. Select the YouTube API v3
4. Click Enable
5. Click Manage
6. Create Credential
7. Tick Restricted
8. Select the YouTube API v3 from the drop down
9. Copy the key and paste it into the `secret.json` file in your src folder
