import os

from PIL import Image
import requests
from io import BytesIO
import numpy as np

def get_thumbnails(video_ids):
    thumbnails = []
    for video_id in video_ids:
        url = 'https://i.ytimg.com/vi/' + video_id + '/default.jpg'

        # load image from url
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))

        # perform relevant image processing
        #TODO

        # convert to numpy array
        np_img = np.array(img)
        thumbnails.append(np_img)
    return np.asarray(thumbnails)

def main():

    video_ids = ['uynBcLr8fEc', 'c0KYU2j0TM4']

    thumbnails = get_thumbnails(video_ids) # get thumbnails as numpy array
    print(thumbnails.shape)




if __name__ == "__main__":
    main()
