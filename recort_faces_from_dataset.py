import csv
import cv2
from typing import List

from pandas.core.computation.pytables import ast
from inference import show_all_keypoints
import os
import numpy

def read_annotations(annotation_file:str) -> List[str]:
    annotations = []

    with open(annotation_file) as csv_file:
        csv_file.readline()
        reader = csv.reader(csv_file)
        for row in reader:
            annotations.append((row[0], numpy.array(row[1:], numpy.float32).astype(numpy.int32)))

    return annotations


if __name__=="__main__":
    data_folder = "/data/ssd1/Datasets/Faces/training"
    annotation_file = "/data/ssd1/Datasets/Faces/training_frames_keypoints.csv"

    annotations = read_annotations(annotation_file)

    face_cascade = cv2.CascadeClassifier('/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

    for file, face_keypoints in annotations:
        print(file)
        print(face_keypoints)
        image = cv2.imread(os.path.join(data_folder,file))
        faces = face_cascade.detectMultiScale(image, 1.2, 2)

        cv2.rectangle(image, faces[0], (0,255,0))




        show_all_keypoints(image, face_keypoints.reshape(-1,2))
        cv2.imshow("face", image)
        key = cv2.waitKey()
        if key == 27:
            break
