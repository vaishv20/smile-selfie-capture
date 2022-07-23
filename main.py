from tkinter import Frame
from turtle import color
import cv2
import datetime
capture = cv2.VideoCapture(0)
face_casket = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
while True:
    grab,Frame = capture.read()
    original_frame = Frame.copy()
    gray = cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)
    face = face_casket.detectMultiScale(gray,1.3,5)
    for x,y,w,h in face:
        cv2.rectangle(Frame,(x,y),(x+w,y+h),(0,255,255),2)
        face_roi = Frame[y:y + h,x:x + w]
        gray_roi = gray[y:y + h,x:x + w]
        smile = smile_cascade.detectMultiScale(gray_roi,1.3,25)
        for x1,y1, w1, h1 in smile:
            cv2.rectangle(face_roi,(x1,y1),(x1 + w1,y1 + h1),(0,0,255),2)
            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            file_name = f'selfie-{time_stamp}.png'
            cv2.imwrite(file_name,original_frame)
        cv2.imshow('face',Frame)
        if cv2.waitKey(10) == ord('q'):
            break
