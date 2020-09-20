import cv2
import time
from datetime import *
import pandas

first_frame=None
status_list=[None,None]
times=[]
df=pandas.DataFrame(columns=["start","end"])

video=cv2.VideoCapture(0)
while(True):
    check, frame = video.read()  # check is boolean and frame is numpy
    status=0
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)#removes noise and improves accuracy
    if first_frame is None:
        first_frame=gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)

    thresh_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]

    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    #creating contours
    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,None)
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status=1
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y),(x+w, y+h),(0,255,0),3)

    status_list.append(status)
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    cv2.imshow("gray", gray)
    cv2.imshow("delta frame",delta_frame)
    cv2.imshow("threshold frame",thresh_frame)
    cv2.imshow("color frame",frame)

    key=cv2.waitKey(1)
    if(key==ord('q')):
        if status==1:
            times.append(datetime.now())
        break

print(status_list)
print(times)

for i in range(0,len(times),2):
    df=df.append({"start":times[i],"end":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()