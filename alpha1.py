import cv2
import numpy as np

import time 
import datetime
from twilio.rest import Client


fin = 0

def twilioman():
    global fin 
    account_sid="ACeea55aea688a7f9ffefbf097f844f695"
    auth_token="1707bde97e99e222af889038b181e0ec"
    client=Client(account_sid,auth_token)
    message=client.messages.create(body=" ALERT: You left your red ball behind !!",from_="+13464722678",to="+917892000892")
    print("ALERT : RED BALL LEFT BEHIND !!")
    fin = 1


def countdown(h, m, s,cx_face,cy_face):
 
    # Calculate the total number of seconds
    total_seconds = h * 3600 + m * 60 + s
 
    # While loop that checks if total_seconds reaches zero
    # If not zero, decrement total time by one second
    while (total_seconds > 0):
 
        # Timer represents time left on countdown
        timer = datetime.timedelta(seconds = total_seconds)
        
        # Prints the time left on the timer
        print(timer, end="\r")
 
        # Delays the program one second
        time.sleep(1)
 
 
        # Reduces total time by one second
        total_seconds -= 1

        if(cx_face or cy_face != 0):
            break
        
    print("Countdown finished")
    twilioman()

from twilio.rest import Client
################################# LIBS ABOVE #############
frame = cv2.VideoCapture(1)
tripped = 0

def calc_dist(cx1,cy1,cx2,cy2):

    if(cx1,cx2,cx2,cy2 != 0):
        return ((cy2 - cy1)**2 + (cx2 - cx1)**2)**0.5

while(True):    
    alerted = 0
    cx_red = 0
    cy_red = 0
    cx_face = 0
    cy_face = 0

    ret,feed = frame.read()
    if ret == False:
        break

    hsv1=cv2.cvtColor(feed,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(feed,cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=9)

    for (x,y,w,h) in faces:
        cv2.rectangle(feed,(x,y),(x+w,y+h),(200,0,0),2)
        cx_face = (x + x + w) // 2
        cy_face = (y + y + h) // 2

        #print("Face Coordinates:",cx_face,cy_face)
        roi_color = feed[y:y+h, x:x+w]

    red_ub = np.array([179,255,255],np.uint8)
    red_lb = np.array([91,115,94],np.uint8)
    red_mask= cv2.inRange(hsv1,red_lb,red_ub)	#ub- upper bound, lb- lower bound

    kernel=np.ones((15,15),"uint8")

    ret,thresh_red = cv2.threshold(red_mask,127,255,0) # binary thresh
  
    #dilation effects go below- 
    #red
    red_mask = cv2.dilate(red_mask,kernel)

    #contour to track below- 
    contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):

        area = cv2.contourArea(contour)
        M_red = cv2.moments(thresh_red)
        if(area > 5000):
            
            if M_red["m00"] != 0 :
                cx_red = int(M_red["m10"] / M_red["m00"])
                cy_red = int(M_red["m01"] / M_red["m00"])

                cv2.circle(feed, (cx_red,cy_red),5 , (255,255,255), -1)

            else:
                cx_red , cy_red = 0 , 0

            #print("Red: ",cx,cy)

            x, y, w, h = cv2.boundingRect(contour)
            feed = cv2.rectangle(feed, (x, y),(x + w, y + h),(0, 0, 255), 2)
            cv2.putText(feed, "RED", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255))
 
    euclidian_dist = calc_dist(cx_face,cy_face,cx_red,cy_red)
    print("Inter object distance:",euclidian_dist)
   
    if(euclidian_dist > 320):
        countdown(0,0,15,cx_face,cy_face)
        if(fin):
            break
       
    cv2.imshow("main window",feed)
    if cv2.waitKey(1) == ord('q'):
        frame.release()
        cv2.destroyAllWindows()
