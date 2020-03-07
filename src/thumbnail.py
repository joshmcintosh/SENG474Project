import os

from PIL import Image
import requests
from io import BytesIO
import time
import numpy as np
from dataset import load_dataset

EMPTY_THUMBNAIL = np.load('../data/empty_thumbnail.npy') # the thumbnail that is shown when a video has been deleted/removed

def get_thumbnails(video_ids):
    thumbnails = []
    missing = []
    count = 0
    for video_id in video_ids:
        # load image from url
        try:
            print('Loading thumbnail for video {0} ({1})'.format(video_id, count))
        
            url = 'https://i.ytimg.com/vi/' + video_id + '/default.jpg'

            response = requests.get(url)
            img = Image.open(BytesIO(response.content))

            # convert to numpy array
            np_img = np.array(img)

            # check to see if thumbnail is empty
            if np.array_equal(np_img, EMPTY_THUMBNAIL):
                thumbnails.append(-1)
            else:
                thumbnails.append((video_id, np_img))
        except:
            print('Error retrieving thumbnail for: ', video_id)
            thumbnails.append(-1)

        count += 1
    return np.asarray(thumbnails)

def main():

    # load video_ids from dataset
    path_to_dataset = '../data/custom_dataset.csv'
    dataset = load_dataset(path_to_dataset, ['video_id'])

    video_ids = dataset['video_id'].tolist()

    # measure time
    start = time.time()
    thumbnails = get_thumbnails(video_ids) # get thumbnails as numpy array
    end = time.time()
    print('Completed in: ', end - start)

    # remove any empty values
    filtered = thumbnails[thumbnails != -1]

    # save to NPY file
    np.save('thumbnails.npy', filtered)

    print('Output shape: ', filtered.shape)




if __name__ == "__main__":
    main()
