# Some changes made therefore again copy paste from vs code

import numpy as np
from tensorflow.keras.models import model_from_json
import operator
import cv2
import sys, os
import matplotlib.pyplot as plt
import hunspell
from string import ascii_uppercase

print("Select model")
print("1.CNN (3 conv layers) 2.VGG19 3.InceptionV3 4.Resnet50")
temp='3'
print("algorithm number selected is {}".format(temp))


#os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\model")
os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\CNN3conv2")
#os.chdir(r"E:\Anaconda\Spyder\Sign-Language-to-Text-master\Sign-Language-to-Text-master\VGG19conv")


# Loading the model
json_file = open("model-bw32.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw32.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame,1)  # 1 tha   (0 karne se img vertically flip ho ja ra h)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
 
    
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
   
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (200, 200))    #200,150
#    roi = cv2.resize(roi, (64, 64))
    

    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    #cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
    kernel = np.ones((3,3),np.uint8)   # later on
         
    # define range of skin color in HSV   (later on)
    lower_skin = np.array([20,40,40], dtype=np.uint8)  #0,20,70   (20,40,50)-Good
    upper_skin = np.array([218,231,250], dtype=np.uint8) # 20,255,255
        
     #extract skin colur image    # later on
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
    #cv2.imshow("mask",mask)
        
    #extrapolate the hand to fill dark spots within  #later on
    mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #blur the image
    mask = cv2.GaussianBlur(mask,(5,5),100)    
    blur = cv2.GaussianBlur(gray,(5,5),2)
    
    #cv2.imshow("Gausmask",mask)
    
    # #blur = cv2.bilateralFilter(roi,9,75,75)
    #cv2.imshow("mask",mask)
    
    th3 = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2) #mask/blur
    
    #cv2.imshow("TH3",th3)
    
    ret, test_image = cv2.threshold(mask, 70 , 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #th3 '''
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (200, 200))         #128,128 normal  yhn jo dega cnn neuron me utna dega,
                                                                #grayscalekarega
    # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)      ######
    # _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)   #####
    
    
    ''' 
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
     
    _, test_image = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)'''
    
    
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1,200,200, 1)) #1,128,128,1 norm , (1,100,100, 3)
    prediction={}                                       #3 only for TL as there RGB
    prediction['blank'] = result[0][0]
    inde = 1
    for i in ascii_uppercase:
        prediction[i] = result[0][inde]
        inde += 1
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()
