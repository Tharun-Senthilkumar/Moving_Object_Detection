import cv2 #image
import time #delay
import imutils #resize

cam = cv2.VideoCapture(0) #cam id
time.sleep(1) #optional

firstFrame=None#initialise
mainFrame=None
area = 500

while True:
    _,img = cam.read() #read frame from camera
    text = "No Object detected" 
    img = imutils.resize(img, width=750) #resize 

    
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #color 2 Gray scale image
    
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0) #smoothened
    
    if firstFrame is None:
            firstFrame = gaussianImg #capturing 1st frame on 1st iteration
            continue
    if mainFrame is None:
        mainFrame=img

    imgDiff = cv2.absdiff(firstFrame, gaussianImg) #absolute diff b/w 1st nd current frame
    
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1] #binary
    
    threshImg = cv2.dilate(threshImg, None, iterations=2)  #to overcome leftovers
    
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)  #finding the area with variying thresholds
    
    cnts = imutils.grab_contours(cnts)  #combining all the contour area
    for c in cnts:
            if cv2.contourArea(c) < area:
                    continue
            (x, y, w, h) = cv2.boundingRect(c)  #taking the positions of the rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Moving Object detected"
    #print(text)
    cv2.putText(img, text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Moving object detection",img)
    cv2.imshow("Main frame",mainFrame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
