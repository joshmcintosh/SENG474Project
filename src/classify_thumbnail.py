import os

from PIL import Image
import requests
from io import BytesIO
import numpy as np
import pandas as pd
import cv2
from thumbnail import get_thumbnails
from dataset import load_dataset

DATASET_PATH = "./thumbnails.npy"
CASC_PATH = "./haarcascade_frontalface_default.xml"


def calculate_features(image_data, face_scale_factor=1.001):
    thumbnail_features = []

    for i in range(len(image_data)):
        image = image_data[0][i][1]

        img = Image.fromarray(image)
        img.show()

        # Get average RGB values
        image_mean = np.mean(image, axis=(0, 1))

        # Convert to grayscale
        image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calculate gradient
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        edges_x = cv2.filter2D(image_g, cv2.CV_8U, kernelx)
        edges_y = cv2.filter2D(image_g, cv2.CV_8U, kernely)
        image_grad = np.add(edges_x, edges_y)
        # Get the mean of the gradient
        image_grad_mean = np.mean(image_grad)

        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(CASC_PATH)
        # Convert to gray-scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=face_scale_factor,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # print("Found {0} faces!".format(len(faces)))

        # Add to dictionary
        thumbnail_features.append([image_mean, image_grad_mean, len(faces)])

    for x in thumbnail_features:
        print(x)

    np_thumbnail_features = np.asarray(thumbnail_features)
    return np_thumbnail_features


def main():
    # Get features using the default thumbnail
    # image_data = np.load(DATASET_PATH, allow_pickle=True)
    image_data = np.load("./thumbnails_small.npy", allow_pickle=True)
    print(image_data.shape)
    np_thumbnail_features = calculate_features(image_data)
    # save to NPY file
    np.save("thumbnail_features.npy", np_thumbnail_features)
    print("thumbnail_features.shape: ", np_thumbnail_features.shape)

    # Get features using the hq thumbnails for comparison
    # load video_ids from dataset
    path_to_dataset = "../data/custom_dataset.csv"
    dataset = load_dataset(path_to_dataset, ["video_id"])
    video_ids = dataset["video_id"].tolist()
    np_thumbnail_hq_features = None

    for id in video_ids:
        image_data = get_thumbnails([id], "hqdefault")
        features = calculate_features(image_data)
        if not np_thumbnail_hq_features:
            np_thumbnail_hq_features = features
        else:
            np_thumbnail_hq_features = np.concatenate(
                (np_thumbnail_hq_features, features), axis=0
            )

    # save to NPY file
    np.save("thumbnail_hq_features.npy", np_thumbnail_hq_features)
    print("thumbnail_hq_features.shape: ", np_thumbnail_hq_features.shape)


if __name__ == "__main__":
    main()
