import numpy as np
import cv2 as cv
import glob
import sys

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

camera_name = 'usb' 

if camera_name == 'usb':
    camera_type = 1
elif camera_name == 'built-in':
    camera_type = 0
else:
    print("please select a valid camera type")
    sys.exit(1)

chessboardSize = (7,7)

cap = cv.VideoCapture(camera_type)
img_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
img_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
cap.release()

frameSize = (img_w, img_h)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 25
objp = objp * size_of_chessboard_squares_mm

objpoints = []
imgpoints = []

images = glob.glob('./pathfinding/calibration/images/*.png')

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

with open(f'./pathfinding/calibration/matrices/intrinsic_{camera_name}.npy', 'wb') as f:
    np.save(f, cameraMatrix)

with open(f'./pathfinding/calibration/matrices/dist_{camera_name}.npy', 'wb') as f:
    np.save(f, dist)

############## UNDISTORTION #####################################################

print(cameraMatrix)

img = cv.imread('./pathfinding/calibration/images/image1.png')
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

print(newCameraMatrix)

with open(f'./pathfinding/calibration/matrices/intrinsicNew_{camera_name}.npy', 'wb') as f:
    np.save(f, newCameraMatrix)

# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
rx, ry, rw, rh = roi
dst = dst[ry:ry+rh, rx:rx+rw]
cv.imwrite(f'caliResult1_{camera_name}.png', dst)

# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)  # w,h now still correct
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
rx, ry, rw, rh = roi
dst = dst[ry:ry+rh, rx:rx+rw]
cv.imwrite(f'caliResult2_{camera_name}.png', dst)

# Reprojection Error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error/len(objpoints)))