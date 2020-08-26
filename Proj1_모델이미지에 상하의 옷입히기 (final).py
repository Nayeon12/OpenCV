import cv2 as cv
import numpy as np
from PIL import Image

print("=============START=============")
print("===============================")

print("옷을 입힐 대상을 불러오세요. ")
print("1. 사진으로 불러오기")
print("2. 카메라를 통해 사진찍기")
print("3. 카메라로 실시간 인식하기")
print("===============================")

Input = input()


if Input == "1":
    print("파일명을 입력해주세요!")
    model_img = input()
    i = Image.open(model_img)
    X, Y = i.size
    

    print(X,Y)
    
elif Input == "2":
    print("스페이스바를 눌러 캡쳐해주세요!")
    cam = cv.VideoCapture(0)

    cv.namedWindow("capture")

    while True:
        ret, frame = cam.read()
        cv.imshow("capture", frame)
        
        if not ret:
            break
        k = cv.waitKey(1)

        if k%256 == 27:
            print("Closing")
            break
            
        elif k%256 == 32:
            img_name = "model_cap.png"
            cv.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            
        cam.release()
        cv.destroyAllWindows()

#else:
    
    

print("===============================")

def nothing(x):
      pass



print("옷의 이미지를 불러오세요. ")
clothes_img = input()

cv.namedWindow('binary')
cv.createTrackbar('threshold', 'binary', 0, 255, nothing) 
cv.setTrackbarPos('threshold', 'binary', 127)

img_color = cv.imread(clothes_img)
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)


print("===============================")

while(True): 

    low = cv.getTrackbarPos('threshold', 'binary') 

    ret,img_binary = cv.threshold(img_gray, low, 255, cv.THRESH_BINARY_INV)
    cv.imshow('binary', img_binary)

    kernel = np.ones((1,1), np.uint8)

    img_mask = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel) 
    img_mask = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel)
    img_result = cv.bitwise_and(img_mask, img_mask, mask = img_mask)
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) 

    cv.drawContours(img_color, contours, 0, (0, 255, 0), 1) 

    cv.imshow('result', img_result) 

    cv.imshow("color", img_color)       

    contours_xy = np.array(contours)
    contours_xy.shape


    x_min, x_max = 0,0
    value = list()

    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0]) 
            x_min = min(value)
            x_max = max(value)

    y_min, y_max = 0,0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1]) 
            y_min = min(value)
            y_max = max(value)


    x = x_min
    y = y_min
    w = x_max-x_min
    h = y_max-y_min



    #if cv.waitKey(1)%256 == 32:
    img_crop = img_color[y:y+h, x:x+w]
    img_name = "crop_0.png"
    cv.imwrite(img_name, img_crop)
    crop_image = cv.imread(img_name)

    if cv.waitKey(1)&0xFF == 27: # esc 누르면 닫음
        break
        



cv.imshow('crop result', crop_image)
cv.waitKey(0)
cv.destroyAllWindows()

cv.destroyAllWindows()


cv.waitKey(0)

print("===============================")

print("자를 옷이 남았나요? (Y/N)")
crop_input = input()


if crop_input == "Y":
    print("파일명을 입력하세요.")
    clothes1_img = input()
    cv.namedWindow('binary')
    cv.createTrackbar('threshold', 'binary', 0, 255, nothing) 
    cv.setTrackbarPos('threshold', 'binary', 127)

    img_color = cv.imread(clothes1_img)
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)



    while(True): 

        low = cv.getTrackbarPos('threshold', 'binary') 

        ret,img_binary = cv.threshold(img_gray, low, 255, cv.THRESH_BINARY_INV)
        cv.imshow('binary', img_binary)

        kernel = np.ones((1,1), np.uint8)

        img_mask = cv.morphologyEx(img_binary, cv.MORPH_OPEN, kernel) 
        img_mask = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel)
        img_result = cv.bitwise_and(img_mask, img_mask, mask = img_mask)
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_NONE) 

        cv.drawContours(img_color, contours, 0, (0, 255, 0), 1) 

        cv.imshow('result', img_result) 

        cv.imshow("color", img_color)       

        contours_xy = np.array(contours)
        contours_xy.shape


        x_min, x_max = 0,0
        value = list()

        for i in range(len(contours_xy)):
            for j in range(len(contours_xy[i])):
                value.append(contours_xy[i][j][0][0]) 
                x_min = min(value)
                x_max = max(value)

        y_min, y_max = 0,0
        value = list()
        for i in range(len(contours_xy)):
            for j in range(len(contours_xy[i])):
                value.append(contours_xy[i][j][0][1]) 
                y_min = min(value)
                y_max = max(value)


        x = x_min
        y = y_min
        w = x_max-x_min
        h = y_max-y_min



        #if cv.waitKey(1)%256 == 32:
        img_crop = img_color[y:y+h, x:x+w]
        img_name = "crop_1.png"
        cv.imwrite(img_name, img_crop)
        crop_image = cv.imread(img_name)

        if cv.waitKey(1)&0xFF == 27: # esc 누르면 닫음
            break




    cv.imshow('crop result', crop_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.destroyAllWindows()


    cv.waitKey(0)


    
# 옷이미지 크기 구하기    
    
i = Image.open("crop_0.png")

cropX, cropY = i.size

crop = cropY/cropX


i2 = Image.open("crop_1.png")

cropX2, cropY2 = i2.size

crop2 = cropY2/cropX2


    

font = cv.FONT_HERSHEY_SIMPLEX

def faceDetect():
    eye_detect = False
    face_cascade = cv.CascadeClassifier('haarcascade_frontface.xml')
    eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    info = ''
    
    if Input == "1":
        try:
            model = cv.imread(model_img)
        
        except:
            print("Not Found")
            return
    
    if Input == "2":
        try:
            model = cv.imread("model_cap.png")
        except:
            print("Not Found")
            return
    
    print("===============================")
    print("상의/하의 중 입히고 싶은 옷은 무엇인가요?")
    print("1. 상의만")
    print("2. 하의만")
    print("3. 상 하의 둘다")
    
    plus_input = input()
    print("===============================")
    
    
    while True:
        if eye_detect:
            info = 'Eye Detection On'
        else:
            info = 'Eye Detection Off'
         
        gray = cv.cvtColor(model, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        
        cv.putText(model, info, (5, 15), font, 0.5, (255, 0, 255), 1)
        
        for (x, y, w, h) in faces: 
            cv.rectangle(model, (x, y), (x + w, y + h), (255, 0, 0), 2 )
            cv.putText(model, 'Detected Face', (x - 5, y - 5), font, 0.5, (255, 255, 0), 2)
            crop_img = cv.imread('crop_0.png')  
            crop_img1 = cv.imread('crop_1.png')
            
            if plus_input == "1":
    
                resize_img = cv.resize(crop_img, dsize=(int(3 * w), int(crop * 3 * h)), interpolation=cv.INTER_AREA)

                model[y+h+(int)(h*0.5):y+h+(int)(h*0.5) + int(crop * 3*h), x - w:x - w + int(3*w)] = resize_img 
                
                result = model[0:Y,0:X]
                img_name = "result.png"
                cv.imwrite(img_name, result)
            
            
            elif plus_input == "2":
            
                resize_img1 = cv.resize(crop_img1, dsize=(int(2 * w), int(crop2 * 2 * h)), interpolation=cv.INTER_AREA)

                model[y+h+(int)(h*0.5) + int(3*h):y+h+(int)(h*0.5) + int(3*h) + int(crop2*2*h), x - int(w*0.5):x - int(w*0.5) + int(2*w)] = resize_img1
                
                result = model[0:Y,0:X]
                img_name = "result.png"
                cv.imwrite(img_name, result)
            
            elif plus_input == "3":
            
                resize_img = cv.resize(crop_img, dsize=(int(3 * w), int(crop * 3 * h)), interpolation=cv.INTER_AREA)
                resize_img1 = cv.resize(crop_img1, dsize=(int(2 * w), int(crop2 * 2 * h)), interpolation=cv.INTER_AREA)

                model[y+h+(int)(h*0.5):y+h+(int)(h*0.5) + int(crop *3*h), x - w:x - w + int(3*w)] = resize_img 
                model[y+h+(int)(h*0.5) + int(3*h):y+h+(int)(h*0.5) + int(3*h) + int(crop2*2*h), x - int(w*0.5):x - int(w*0.5) + int(2*w)] = resize_img1 
                
                result = model[0:Y,0:X]
                img_name = "result.png"
                cv.imwrite(img_name, result)

                
            
            if eye_detect:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
                    
            
        
        cv.imshow('result', model)
        k = cv.waitKey(30)
        
        if k == ord('i'):
            eye_detect = not eye_detect
        if k == 27:
            break
            

    cv.destroyAllWindows()
    
faceDetect()

print("===============================")
#img = cv.imshow('result',model)
#result = cv.imread(img)
#img_name = "result.png"
#cv.imwrite(img_name, result)



print("저장이 완료되었습니다.")

print("===============================")
