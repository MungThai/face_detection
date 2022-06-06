import imp
import os
import sys
from turtle import width
import numpy
import cv2

haar_file = 'haarcascade_frontalface_default.xml'
size = 4
datasets = 'datasets'

print('Recognizing Face Please be in sufficient lights...')

(images, labels, names, id) = ([], [], {}, 0)

for(subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subject_path = os.path.join(datasets, subdir)
        for file_name in os.listdir(subject_path):
            path = subject_path + '/' + file_name
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (640, 480)

# Create a Numpy array from the two lists above
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

# OpenCV trains a model from the images
# Note for OpenCV2: remove '.face'
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Use fisherRecognizer on camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    (_, img) = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y: y + h, x: x + w]
        face_size = cv2.resize(face, (width, height))

        # Try to recognize the face
        prediction = model.predict(face_size)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if prediction[1] < 500:
            cv2.putText(img, '% s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        else:
            cv2.putText(img, 'not recognized', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        cv2.imshow('OpenCV', img)

        key = cv2.waitKey(10)
        if key == 27:
            break


