import cv2 as cv
import numpy as np

def nothing(x):
    pass


img_color = cv.imread('9.jpg')
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(img_noise, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) # 컨투어 검출

kernel = np.ones((5,5), np.uint8)
img_mask = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel) 
img_mask = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel)

img_noise = cv.bitwise_and(img_mask, img_mask, mask = img_mask)

cv.drawContours(img_noise, contours, 0, (0, 255, 0), 3) # 인덱스0, 초록색


cv.imshow("noise", img_mask)
cv.imshow("binary", img_binary)
cv.imshow("color", img_color)
cv.waitKey(0)
