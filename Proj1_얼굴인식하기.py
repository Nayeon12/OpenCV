# Haar cascade 검출기는 학습데이터를 이용해서 이미지에서 특정 객체를 검출하는 역할

# haarcascade_frontface.xml 과 haarcascade_eye.xml 를 이용하였음


import cv2 as cv
import numpy as np

font = cv.FONT_HERSHEY_SIMPLEX
def faceDetect():
    eye_detect = False
    face_cascade = cv.CascadeClassifier('haarcascade_frontface.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    info = ''
    
    try:
        capture = cv.VideoCapture(0)
        
    except:
        print("Camera Is Not Found")
        return
    
    img_counter = 0
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        if eye_detect:
            info = 'Eye Detection On'
        else:
            info = 'Eye Detection Off'
         
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # 이미지 크기가 얼마나 줄어드는지 / 
        #유지해야하는 사각형의 이웃수, 값이 높을 수록 감지횟수 down, 품질 up
        
        
        cv.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)
        
        for (x, y, w, h) in faces: # 여러 faces 중 (x, y, w, h) 는 하나의 묶음
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2 ) # 얼굴에 최소 상자 입히기
            cv.putText(frame, 'Detected Face', (x - 5, y - 5), font, 0.5, (255, 255, 0), 2)
            if eye_detect:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        cv.imshow('frame', frame)
        k = cv.waitKey(30)
        
        if k == ord('i'):
            eye_detect = not eye_detect
        if k == 27:
            break
            
        elif k== 32:
        # SPACE pressed
            img_name = "face_{}.png".format(img_counter)
            cv.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            
    capture.release()
    cv.destroyAllWindows()
    
faceDetect()
