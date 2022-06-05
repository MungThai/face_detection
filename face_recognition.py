import cv2
import numpy as np

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)
webcam.set(3, 640) # set Width
webcam.set(4, 480) # set Height

while(True):
    # Capture frame by frame
    ret, img = webcam.read()

    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the face
    faces = face.detectMultiScale(gray, 1.3, 5)


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  

    # Show the video
    cv2.imshow('video', img)

    if cv2.waitKey(30) & 0xff == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()


