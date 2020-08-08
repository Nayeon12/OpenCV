import cv2 as cv
import numpy as np

def nothing(x):
    pass

face = cv.imread('face_0.png') 
crop = cv.imread('beigePT.png') 

cutImg = face[0:395, 0:329] # y 값 먼저

grayCrop = cv.cvtColor(crop, cv.COLOR_BGR2GRAY) 


cv.namedWindow('mask')
cv.createTrackbar('threshold', 'mask', 0, 255, nothing) 
cv.setTrackbarPos('threshold', 'mask', 127)


while(True): 
    low = cv.getTrackbarPos('threshold', 'mask') 
    

    ret,mask = cv.threshold(grayCrop, 100, 255, cv.THRESH_BINARY)  
    cv.imshow('mask', mask)
    

    mask_inv = cv.bitwise_not(mask)
    cv.imshow('mask_inv', mask_inv)  
    

    fg = cv.bitwise_and(crop, crop, mask=mask) 
    cv.imshow('fg', fg) 

    
    bg = cv.bitwise_and(cutImg, cutImg, mask=mask_inv) 
    cv.imshow('bg', bg)
    

    img = cv.add(fg,bg) #바이너리 이미지와 자른 부분을 합침


    face[0:395, 0:329] = img # 배경이미지 자른부분에 들어감
    
    cv.imshow('result', face)  
    
    
    
   
    if cv.waitKey(1)&0xFF == 27: # esc 누르면 닫음
        break



cv.imshow('result', face) 
cv.imshow('mask', mask) 
cv.imshow('mask_inv', mask_inv) 
cv.imshow('fg', fg) 
cv.imshow('bg', bg) 


cv.waitKey(0) 
cv.destroyAllWindows()

