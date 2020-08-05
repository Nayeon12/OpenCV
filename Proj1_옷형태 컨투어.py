import cv2 as cv  
import numpy as np


def nothing(x):
    pass

cv.namedWindow('binary')
cv.createTrackbar('threshold', 'binary', 0, 255, nothing) 
cv.setTrackbarPos('threshold', 'binary', 127)

img_color = cv.imread('grayT.png')
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)


while(True): 
    low = cv.getTrackbarPos('threshold', 'binary') 

    ret,img_binary = cv.threshold(img_gray, low, 255, cv.THRESH_BINARY_INV) # THRESH_BINARY_INV : 반전된 마스크 이미지
  
    cv.imshow('binary', img_binary)
    
    kernel = np.ones((1,1), np.uint8)
    img_mask = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel) 
    img_mask = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel)

    img_result = cv.bitwise_and(img_mask, img_mask, mask = img_mask)
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) # 컨투어 검출


    cv.drawContours(img_color, contours, 0, (0, 255, 0), 3) # 인덱스0, 초록색
    
    cv.imshow('result', img_result) 
    
    cv.imshow("color", img_color)
    
    if cv.waitKey(1)&0xFF == 27: # esc 누르면 닫음
        break

cv.destroyAllWindows()


cv.waitKey(0)
