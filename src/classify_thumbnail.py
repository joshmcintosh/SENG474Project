import re

import cv2
import numpy as np
import pytesseract
from dataset import load_dataset
from thumbnail import get_thumbnails

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)
DATASET_PATH = "./dataset.csv"
CASC_PATH = "./haarcascade_frontalface_default.xml"
# Use debug ONLY with a small dataset (<30)
DEBUG = False


def calculate_features(image_data, face_scale_factor=1.001):
    thumbnail_features = []

    for i in range(len(image_data)):
        if type(image_data[i]) is int and image_data[i] == -1:
            continue
        image = image_data[i][1]
        # Convert to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get average RGB values
        image_mean = np.mean(image, axis=(0, 1))

        # Calculate gradient
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        edges_x = cv2.filter2D(image_gray, cv2.CV_8U, kernelx)
        edges_y = cv2.filter2D(image_gray, cv2.CV_8U, kernely)
        image_grad = np.add(edges_x, edges_y)
        # Get the mean of the gradient
        image_grad_mean = np.mean(image_grad)

        # FACE RECOGNITION
        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(CASC_PATH)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            image_gray,
            scaleFactor=face_scale_factor,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # TEXT RECOGNITION
        # Use a bilateral filter to try and make it easier for pytesseract
        image_filt = cv2.bilateralFilter(image, d=7, sigmaColor=150, sigmaSpace=50)
        # pytesseract expects a rgb image, not a bgr as cv2 makes
        image_filt = cv2.cvtColor(image_filt, cv2.COLOR_BGR2RGB)
        # Run the text detection and return the found text
        text = pytesseract.image_to_string(image_filt)
        # Strip all non-alphanumeric characters to reduce false positives
        text = re.sub("[^A-Za-z]+", "", text)

        # Add to dictionary
        thumbnail_features.append([image_mean, image_grad_mean, len(faces), text])
        if DEBUG:
            print(
                "id:{:2d}   Mean: {:10f} {:10f} {:10f}   Grad: {:10f}   Faces: {:2d}   Text: {}".format(
                    i,
                    image_mean[0],
                    image_mean[1],
                    image_mean[2],
                    image_grad_mean,
                    len(faces),
                    text,
                )
            )
            cv2.imwrite("outputFileBLF{0:02d}".format(i) + ".png", image_filt)

    np_thumbnail_features = np.asarray(thumbnail_features)
    return np_thumbnail_features


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main():
    # Get features using the hq thumbnails
    # load video_ids from dataset
    dataset = load_dataset(DATASET_PATH, ["video_id"])
    video_ids = dataset["video_id"].tolist()
    np_thumbnail_hq_features = []

    # Split into groups of 10 so we don't flood the RAM... shouldn't affect performance much
    split_video_ids = list(chunks(video_ids, 50))
    for i, ids in enumerate(split_video_ids):
        i += 1
        image_data = get_thumbnails(ids, "hqdefault")
        np_thumbnail_hq_features.extend(
            calculate_features(image_data, face_scale_factor=1.02)
        )
        print("Completed {} of {} sublists".format(i, len(split_video_ids)))

    np_thumbnail_hq_features = np.asarray(np_thumbnail_hq_features)
    # save to NPY file
    np.save("thumbnail_features.npy", np_thumbnail_hq_features)
    print("thumbnail_features.shape: ", np_thumbnail_hq_features.shape)


if __name__ == "__main__":
    main()
