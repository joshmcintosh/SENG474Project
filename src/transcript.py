# code to get transcripts from dataset
from youtube_transcript_api import YouTubeTranscriptApi as T_API
import numpy as np
from dataset import load_dataset
import time
import re

def process_transcript(transcript):
    # extract words from transcript
    extracted_transcript = []
    for line in transcript:
        text = line['text']
        text = text.lower()
        text = text.split()
        text = [re.sub('[^A-Za-z0-9]+', '', word) for word in text]
        extracted_transcript = extracted_transcript + text

    return extracted_transcript

def get_transcripts(video_ids):
    transcripts = []
    success_count = 0
    count = 0
    for video_id in video_ids:
        # get transcripts from api
        try:
            transcript = T_API.get_transcript(video_id)
            transcript = process_transcript(transcript)
            transcripts.append(transcript)
            print('SUCCESS: retrieved transcript for: ', video_id)
            success_count += 1
            
        except:
            print('Error retrieving transcript for: ', video_id)
            transcripts.append(-1)
        count += 1

    print('Success count: ', success_count)
    print('Total count: ', count)

    return np.asarray(transcripts)


def main():

    # load video_ids from dataset
    path_to_dataset = '../../data/custom_dataset.csv'
    dataset = load_dataset(path_to_dataset, ['video_id'])

    video_ids = dataset['video_id'].tolist()

    # measure time
    start = time.time()
    transcripts = get_transcripts(video_ids) # get transcripts as numpy array
    end = time.time()
    print('Completed in: ', end - start)

    # remove any empty values
    filtered = transcripts[transcripts != -1]

    # save to NPY file
    np.save('transcripts.npy', filtered)

    print('Output shape: ', filtered.shape)


if __name__ == "__main__":
    main()