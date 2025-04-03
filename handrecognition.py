import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
else:
    print("Camera opened successfully")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frame (stream end?). Exiting ...")
        break
    
    color = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)   # ' cv.COLOR_BGR2GRAY' - Color is currently grayscale
   
    # Display resulting frame
    cv.imshow('frame', color)
    if cv.waitKey(1) == ord('q'):
        print("Exiting...")
        break

# Display resulting frame
cap.release()
cv.destroyAllWindows()