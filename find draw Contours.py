# 컨투어

# 영상이나 이미지의 윤곽선을 검출하기위해 사용
# 외곽이나 내곽의 윤곽선을 검출

# contours, hierarchy = cv.findContours(image, mode, method[, contours[, hierarchy[, offset]]])

# Contour Approximation Method 윤곽선에 포인터를 지정하는 방식
# cv.CHAIN_APPROX_NONE  모든 경계선
# cv.CHAIN_APPROX_SIMPLE  시작점과 끝점만


# Contour Retrieval Mode 컨투어의 결과를 어떤 식으로 리턴 할지 결정
# RETR_TREE  [Next, Previous, First_Child, Parent]
# RETR_LIST  모든 컨투어가 같은 계층
# RETR_EXTERNAL  가장외곽만 리턴
# RETR_CCOMP  


# 영역 크기, 근사화, 무게중심, 경계사각형, Convex Hull, Convexity Defects


import cv2 as cv

img_color = cv.imread('test1.png')
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) # 컨투어 검출

cv.drawContours(img_color, contours, 0, (0, 255, 0), 3) # 인덱스0, 파란색
cv.drawContours(img_color, contours, 1, (255, 0, 0), 3) # 인덱스1, 초록색


cv.imshow("result", img_color)
cv.waitKey(0)


###

import cv2 as cv

img_color = cv.imread('test4.png')
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) # 컨투어 검출

for cnt in contours:
    for p in cnt:
        cv.circle(img_color, (p[0][0], p[0][1]), 10, (255, 0, 0), -1) # 모든좌표마다 파란원을 그리도록 함
    

cv.imshow("result", img_color)
cv.waitKey(0)



