import cv2


def nothing(x): # 트랙바 생성시 필요, 사용 x
    pass

cv2.namedWindow('Binary') # 트랙바를 붙힐 윈도우를 설정
cv2.createTrackbar('threshold', 'Binary', 0, 255, nothing) # 1) 트랙바의 식별자
cv2.setTrackbarPos('threshold', 'Binary', 85) # 트랙바의 초기값 127로 설정


img_color = cv2.imread('ball.jpg', cv2.IMREAD_COLOR)

cv2.imshow('Color', img_color)
cv2.waitKey(0)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray', img_gray)
cv2.waitKey(0)

while(True): # 트랙바를 이동시 결과를 바로 확인할 수 있도록 
    low = cv2.getTrackbarPos('threshold', 'Binary') # 트랙바의 현재값을 가져와 임계값으로 사용할 수 있도록 함
    #이진화
    ret,img_binary = cv2.threshold(img_gray, low, 255, cv2.THRESH_BINARY_INV) # THRESH_BINARY_INV : 반전된 마스크 이미지
    # 그레이스케일 이미지 이어야함, 두번째 값을 기준으로 결과 이미지의 픽셀이 흰색 혹은 검은색이 됨,
    # threshold 보다 픽셀값이 클때, 세번째 파라미터로 픽셀값을 지정함, 작으면 0으로 지정

    cv2.imshow('Binary', img_binary)
    
    img_result = cv2.bitwise_and(img_color, img_color, mask = img_binary) # 원본이미지와 바이너리 이미지를 엔드연산하는 코드 
    cv2.imshow('Result', img_result) 
    
    if cv2.waitKey(1)&0xFF == 27: # esc 누르면 닫음
        break

cv2.destroyAllWindows()


