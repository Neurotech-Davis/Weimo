import cv2
import numpy as np
import pyautogui

class CameraConfig():
    def __init__(self, height, angle, h_fov, v_fov):
        '''
        height : camera height above ground (any unit — distance output matches)
        angle  : downward tilt of camera from horizontal (degrees)
        h_fov  : horizontal field of view (degrees)
        v_fov  : vertical field of view (degrees)
        '''
        self.height = height
        self.angle  = angle
        self.h_fov  = h_fov
        self.v_fov  = v_fov

def get_cursor_pixel(img_w, img_h):
    screen_x, screen_y = pyautogui.position()
    px = int(np.clip(screen_x, 0, img_w - 1))
    py = int(np.clip(screen_y, 0, img_h - 1))
    return px, py

def pixel_to_point(px, py, img_w, img_h, cam_cfg):
    '''
    Returns (h_angle, dist):
      h_angle : horizontal angle from camera centre (degrees, +right)
      dist    : ground distance from camera base (same units as cam_cfg.height)
    '''
    # --- horizontal angle ---
    # map px in [0, img_w] → angle in [-h_fov/2, +h_fov/2]
    dx = px - img_w / 2
    h_angle = (dx / (img_w / 2)) * (cam_cfg.h_fov / 2)

    # --- ground distance ---
    # map py in [0, img_h] → vertical offset angle; positive py = lower in frame = more depression
    dy = py - img_h / 2
    pixel_v_angle = (dy / (img_h / 2)) * (cam_cfg.v_fov / 2)

    total_depression_deg = cam_cfg.angle + pixel_v_angle
    total_depression_rad = np.radians(total_depression_deg)

    if total_depression_rad <= 0:
        # point is at or above the horizon — no ground intersection
        return h_angle, float('inf')

    # calculate distance with horizontal and vertical angle taken into account
    dist = (cam_cfg.height / np.tan(total_depression_rad)) / np.cos(np.radians(h_angle))
    return h_angle, dist

def main():
    cam_cfg = CameraConfig(
        height=12,   # mm — camera mounted 1.2 m high
        angle=0,      # degrees downward tilt, 0 = straight forward
        h_fov=48.8,    # horizontal FOV, how far to the sides it sees
        v_fov=34.2,    # vertical FOV, how far up and down it sees
    )

    # open camera and get dimensions of the camera
    cap = cv2.VideoCapture(1) # 0 for built in camera, 1 for usb
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

        px, py = get_cursor_pixel(img_w, img_h)
        h_angle, dist = pixel_to_point(px, py, img_w, img_h, cam_cfg)

        cv2.drawMarker(frame, (px, py), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"pixel : ({px}, {py})",          (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"angle : {h_angle:+.1f} deg",    (20, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"dist  : {dist:.0f} mm",         (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()