import cv2 as cv

cap = cv.VideoCapture(0)

while(True):
    ret,img_color = cap.read()
    
    if(ret):
        cv.imshow('camara', img_color)
        img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
        ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
        
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        
        cv.drawContours(img_color, contours, 0, (0, 255, 0), 3)
        
        
        if cv.waitKey(1) & 0xFF == 27:
            break
    
            
cap.release()
cv.destroyAllWindows()

