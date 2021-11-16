from typing import Counter
import cv2
import numpy as np
from numpy.lib.index_tricks import c_
# Web Camera
cap=cv2.VideoCapture('video.mp4')
min_width_rect=80 #rectangle width
min_height_rect=80 #rectangle height
count_line_position=550
# Initilize substractor
algo =cv2.createBackgroundSubtractorMOG2()

def center_handle(x,y,ew,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy
Detect=[]
offset=6 #allowable error between pixel
counter=0
while True:
    ret,frame1=cap.read()
    grey=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(grey,(3,3),5)

    #Applying on each frame
    img_sub=algo.apply(blur)
    dilat=cv2.dilate(img_sub,np.ones((5,5)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada=cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada=cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    Countershape,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3)
    for (i,c) in enumerate(Countershape):
        (x,y,w,h)=cv2.boundingRect(c)
        validate_counter=(w>=min_width_rect) and (h>=min_height_rect)
        if not validate_counter:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        center=center_handle(x,y,w,h)
        Detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)
        for (x,y) in Detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
                cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
                Detect.remove((x,y))
                print("Veichle Counter:"+str(counter))

    cv2.putText(frame1,"VEHICLE COUNTER :"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)







    # cv2.imshow('detecter',dilatada)

    cv2.imshow('Vedio Original',frame1)

    if cv2.waitKey(1)==13:
        break
cv2.destroyAllWindows()
cap.release()

