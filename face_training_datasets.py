import cv2
import numpy as np
import sys
import os

# All the faces data will be present this folder
datasets = 'datasets'

# These are sub data sets of folder, 
# for my face 
sub_data = 'Luc'

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

# defining the size of images
(width, height) = (640, 480)

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # captureDevice = webcam OpenCV >= 3.4
# webcam.set(3, 640) # set Width
# webcam.set(4, 480) # set Height

count = 1 
while count < 30:
    # Capture frame by frame
    (_, img) = webcam.read()

   #  img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the face
    faces = face.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        face = gray[y: y + h, x: x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('% s/% s.png' % (path, count), face_resize)
    count += 1

    # Show the video
    cv2.imshow('OpenCV', img)

    key = cv2.waitKey(10)
    if key == 27 & 0xff == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()


