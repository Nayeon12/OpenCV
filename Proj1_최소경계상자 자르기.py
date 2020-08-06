import cv2 as cv  
import numpy as np


def nothing(x):
    pass

cv.namedWindow('binary')
cv.createTrackbar('threshold', 'binary', 0, 255, nothing) 
cv.setTrackbarPos('threshold', 'binary', 127)

img_color = cv.imread('nerdyT.png')
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


    cv.drawContours(img_color, contours, 0, (0, 255, 0), 1) # 인덱스0, 초록색
    
    cv.imshow('result', img_result) 
    
    cv.imshow("color", img_color)       
    
    contours_xy = np.array(contours)
    contours_xy.shape
    
    
    # x의 min과 max 찾기
    x_min, x_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0]) 
            x_min = min(value)
            x_max = max(value)
#    print(x_min)
#    print(x_max)
 
    # y의 min과 max 찾기
    y_min, y_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1]) 
            y_min = min(value)
            y_max = max(value)
#    print(y_min)
#    print(y_max)


    x = x_min
    y = y_min
    w = x_max-x_min
    h = y_max-y_min
    
    
    img_crop = img_color[y:y+h, x:x+w]
    cv.imwrite('crop.png', img_crop)
    crop_image = cv.imread('crop.png')

    
    if cv.waitKey(1)&0xFF == 27: # esc 누르면 닫음
        break
        

cv.imshow('crop result', crop_image)
cv.waitKey(0)
cv.destroyAllWindows()

cv.destroyAllWindows()


cv.waitKey(0)
