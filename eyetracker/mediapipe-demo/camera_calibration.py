import numpy as np
import cv2
import glob

# Dimensions of the chessboard (number of inner corners)
CHECKERBOARD = (7, 10)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# World coordinates for 3D points (0,0,0), (1,0,0)...
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)

cap = cv2.VideoCapture(64)  # Use your verified index

while len(objpoints) < 20:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow("Calibration", img)

        # Press 's' to save the points for this frame
        if cv2.waitKey(1) & 0xFF == ord("s"):
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"Captured {len(objpoints)}/20")

    cv2.imshow("Calibration", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break


#### Generate camera calibration matrix
# # Load your real calibration
# with np.load("camera_calib.npz") as data:
#     cam_matrix = data['mtx']
#     dist_coeffs = data['dist']
#
# # In your PnP call, use the real distortion coefficients instead of zeros
# _, rot_vec, trans_vec = cv2.solvePnP(
#     model_points, image_points, cam_matrix, dist_coeffs
# )
