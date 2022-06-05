import cv2

# Load some pre-trained data on face frontal for opencv (haar cascade algorithm)
# https://github.com/opencv/opencv
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Choose an image to detect face in
img = cv2.imread('Mung.png')

# Must convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Clever Programmer Face Detector', gray)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# print(face_coordinates)

# (x, y, w, h) = faces[0]
# cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

for(x, y, w, h) in faces:
  cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
  roi_gray = gray[y:y+h, x:x+w]
  roi_color = img[y:y+h, x:x+w]
  eyes = eye_cascade.detectMultiScale(roi_gray)
  for(ex, ey, ew, eh) in eyes:
     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
