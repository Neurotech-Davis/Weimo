import cv2
import numpy as np
import pyautogui

class CameraConfig():
    def __init__(self, height, angle):
        self.height = height
        self.angle  = angle  # degrees downward tilt from horizontal

def load_calibration(camera_name):
    base = f'./pathfinding/calibration/matrices'
    K     = np.load(f'{base}/intrinsicNew_{camera_name}.npy')
    dist  = np.load(f'{base}/dist_{camera_name}.npy')
    return K, dist

def get_cursor_pixel(img_w, img_h):
    sx, sy = pyautogui.position()
    return int(np.clip(sx, 0, img_w - 1)), int(np.clip(sy, 0, img_h - 1))

def pixel_to_point(px, py, K, cam_cfg):
    '''
    Uses intrinsic matrix K to compute accurate ray angles.
    Returns (h_angle_deg, dist_mm).
    '''
    # these are just where to appropriate values happen to be in this matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    h_angle = np.degrees(np.arctan2(px - cx, fx))   # +right
    v_angle = np.degrees(np.arctan2(py - cy, fy))   # +down in frame

    total_depression_deg = cam_cfg.angle + v_angle
    total_depression_rad = np.radians(total_depression_deg)

    if total_depression_rad <= 0:
        return h_angle, float('inf')

    dist = (cam_cfg.height / np.tan(total_depression_rad)) / np.cos(np.radians(h_angle))
    return h_angle, dist

def main():
    CAMERA_NAME = 'usb' # options are usb or built-in
    K, dist_coeffs = load_calibration(CAMERA_NAME)
    # K = the camera intrisics matrix, focal point and principal point
    # dist_coeffs = vector of variables describing how distortion happens in camera

    cam_cfg = CameraConfig(
        height=12,  # mm (1.2 m)
        angle=0,    # degrees downward tilt, 0 is straight ahead
    )

    # one for usb cam
    cap = cv2.VideoCapture(1)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    window_name = "Cursor Position"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 0, 0)
    cv2.resizeWindow(window_name, img_w, img_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # undistort using the refined matrix (no black border artifacts)
        frame = cv2.undistort(frame, K, dist_coeffs, None, K)

        px, py = get_cursor_pixel(img_w, img_h)
        h_angle, dist = pixel_to_point(px, py, K, cam_cfg)

        # this is all about drawing stuff on screen
        cv2.drawMarker(frame, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"pixel : ({px}, {py})",       (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"angle : {h_angle:+.1f} deg", (20, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"dist  : {dist:.0f} mm",      (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()