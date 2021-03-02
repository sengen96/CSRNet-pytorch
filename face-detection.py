import numpy as np
import cv2
import time
import numpy as np

faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(0)
dir_helper = "../Y4S2/EE4208 - Intelligent System Design/Recorded Lecture/GMT20210121-052313_EE4208_gallery_1920x1080.mp4"
cap = cv2.VideoCapture(dir_helper)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
fps = cap.get(cv2.CAP_PROP_FPS)
# cap.set(5, fps ) # set fps
time_total = 0.0
elapsed = 0.0
while True:
    time_total +=elapsed
    start = time.time()
    cap.set(1, int(time_total*fps))
    # print(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, img = cap.read()
    img = cv2.resize(img, (480, 320))
    # print(img.shape)
    # input("?")
    # img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]  
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    elapsed = time.time() - start
cap.release()
cv2.destroyAllWindows()