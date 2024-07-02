import cv2
import numpy as np
import os
import string


print(os.getcwd())    



minValue = 70

cap = cv2.VideoCapture(0)
interrupt = -1  

while True:
    _, frame = cap.read()
    # Simulating mirror image                      Mirror image
    frame = cv2.flip(frame, 1)
    
   
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])   # as shape has 2 parameters (MAYBE)
    y1 = 10                             
    x2 = frame.shape[1]-10                # x1,y1,x2,y2 not used it was jst for test
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI (Region of interest, wo dabba jahna sign dikhana h)
    # The increment/decrement by 1 is to compensate for the bounding box
    #cv2.rectangle(frame, (220-1, 9), (620+1, 419), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[10:410, 220:520]
#    roi = cv2.resize(roi, (64, 64))
    

    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    #cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
    kernel = np.ones((3,3),np.uint8)   # later on
         
    # define range of skin color in HSV   (later on)
    lower_skin = np.array([0,0,0], dtype=np.uint8)  #0,20,70
    upper_skin = np.array([255,255,255], dtype=np.uint8) # 20,255,255
        
     #extract skin colur image    # later on
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
    cv2.imshow("mask",mask)
        
    #extrapolate the hand to fill dark spots within  #later on
    mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #blur the image
    mask = cv2.GaussianBlur(mask,(5,5),100)    
    blur = cv2.GaussianBlur(gray,(5,5),2)
    
    # #blur = cv2.bilateralFilter(roi,9,75,75)
    #cv2.imshow("mask",mask)
    
    th3 = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2) #mask/blur
    ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    
    #time.sleep(5)
    #cv2.imwrite("/home/rc/Downloads/soe/im1.jpg", roi)
    #test_image = func("/home/rc/Downloads/soe/im1.jpg")


    
    test_image = cv2.resize(test_image, (300,300))
    cv2.imshow("test", test_image)
        
   
    
cap.release()
cv2.destroyAllWindows()

