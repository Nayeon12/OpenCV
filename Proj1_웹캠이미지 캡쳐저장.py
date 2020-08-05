import cv2 as cv

cam = cv.VideoCapture(0)

cv.namedWindow("capture")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv.imshow("capture", frame)
    if not ret:
        break
    k = cv.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Closing")
        break
        
    elif k%256 == 32:
        # SPACE pressed
        img_name = "clothes_{}.png".format(img_counter)
        cv.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv.destroyAllWindows()
