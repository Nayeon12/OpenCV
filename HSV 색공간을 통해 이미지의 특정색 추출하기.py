import numpy as np
import cv2

img_color = cv2.imread('hsv.png') # 이미지 파일을 컬러로 불러옴
height, width = img_color.shape[:2] # 이미지의 높이와 너비 불러옴, 가로 [0], 세로[1]

img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV) # cvtColor 함수를 이용하여 hsv 색공간으로 변환

lower_blue = (120-10, 30, 30) # hsv 이미지에서 바이너리 이미지로 생성 , 적당한 값 30
upper_blue = (120+10, 255, 255)
img_mask = cv2.inRange(img_hsv, lower_blue, upper_blue) # 범위내의 픽셀들은 흰색, 나머지 검은색

img_result = cv2.bitwise_and(img_color, img_color, mask = img_mask) # 바이너리 이미지를 마스크로 사용하여 원본이미지에서 범위값에 해당하는 영상부분을 획득

cv2.imshow('img_color', img_color)
cv2.imshow('img_mask', img_mask)
cv2.imshow('img_result', img_result)

cv2.waitKey(0)
cv2.destroyAllWindows()

