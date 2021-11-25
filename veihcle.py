import numpy as np
from typing import Counter
import cv2
from numpy.lib.index_tricks import c_
# Web Camera
capture=cv2.VideoCapture('video.mp4')
minimum_width_of_rect=80 #width of rectangle 
minimum_height_of_rect=80 #height of rectangle 
count_position_of_car=550
# Initilize substractor
algorithm =cv2.createBackgroundSubtractorMOG2()

def center_(a,b,w,h):
    x1=int(w/2)
    y1=int(h/2)
    ca=a+x1
    cb=b+y1
    return ca,cb
Detector=[]
offset=6 #permissible error between pixel
counter_of_car=0
while True:
    ret,my_frame=capture.read()
    grey_color=cv2.cvtColor(my_frame,cv2.COLOR_BGR2GRAY)
    blur_to_see=cv2.GaussianBlur(grey_color,(3,3),5)
    #Applying on each frame
    img_sub=algorithm.apply(blur_to_see)
    to_dilat=cv2.dilate(img_sub,np.ones((5,5)))
    kerneling=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada=cv2.morphologyEx(to_dilat,cv2.MORPH_CLOSE,kerneling)
    dilatada=cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kerneling)
    Countershape,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(my_frame,(25,count_position_of_car),(1200,count_position_of_car),(255,127,0),3)
    for (i,c) in enumerate(Countershape):
        (a,b,w,h)=cv2.boundingRect(c)
        validation_of_counter=(w>=minimum_width_of_rect) and (h>=minimum_height_of_rect)
        if not validation_of_counter:
            continue
        cv2.rectangle(my_frame,(a,b),(a+w,b+h),(0,255,0),2)
        center_points=center_(a,b,w,h)
        Detector.append(center_points)
        cv2.circle(my_frame,center_points,4,(0,0,255),-1)
        for (a,b) in Detector:
            if a<(count_position_of_car+offset) and b>(count_position_of_car-offset):
                counter_of_car+=1
                cv2.line(my_frame,(25,count_position_of_car),(1200,count_position_of_car),(0,127,255),3)
                Detector.remove((a,b))
                print("Veichle Counter:"+str(counter_of_car))

    cv2.putText(my_frame,"VEHICLE COUNTER :"+str(counter_of_car),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

    # cv2.imshow('detecter',dilatada)

    cv2.imshow('Vedio Original',my_frame)

    if cv2.waitKey(1)==13:
        break
cv2.destroyAllWindows()
capture.release()

