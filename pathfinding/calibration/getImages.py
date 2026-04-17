import cv2
import sys

camera_name = 'usb' 

if camera_name == 'usb':
    camera_type = 1
elif camera_name == 'built-in':
    camera_type = 0
else:
    print("please select a valid camera type")
    sys.exit(1)

cap = cv2.VideoCapture(camera_type)

num = 0

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('./pathfinding/calibration/images/image' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img 1',img)