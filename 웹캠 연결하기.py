import cv2 as cv

cap = cv.VideoCapture(0)

while(True):
    ret, cam = cap.read()

    if(ret) :
        cv.imshow('camera', cam)
        
        
        if cv.waitKey(1) & 0xFF == 27:
            break 
                     
cap.release()
cv.destroyAllWindows()
