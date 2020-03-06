import os

from PIL import Image
import requests
from io import BytesIO
import time
import numpy as np
from dataset import load_dataset

def get_thumbnails(video_ids):
    thumbnails = []
    missing = []
    count = 0
    for video_id in video_ids:
        url = 'https://i.ytimg.com/vi/' + video_id + '/default.jpg'
        print('Loading thumbnail for video {0} ({1})'.format(video_id, count))
        # load image from url
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))

            # convert to numpy array
            np_img = np.array(img)
            thumbnails.append(np_img)
        except:
            print('Error retrieving thumbnail for: ', video_id)
            thumbnails.append(-1)

        count += 1
    return np.asarray(thumbnails)

def main():

    # load video_ids from dataset
    path_to_dataset = '../data/CAvideos.csv'
    dataset = load_dataset(path_to_dataset, ['video_id'])

    video_ids = dataset['video_id'].tolist()

    # measure time
    start = time.time()
    thumbnails = get_thumbnails(video_ids) # get thumbnails as numpy array
    end = time.time()
    print('Completed in: ', end - start)

    # save to NPY file
    np.save('thumbnails.npy', thumbnails)

    print('thumbnails.shape: ', thumbnails.shape)




if __name__ == "__main__":
    main()
