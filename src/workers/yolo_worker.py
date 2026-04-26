import cv2
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ── Constants ─────────────────────────────────────────────────────────────────

# Loading model
models_dir = Path(__file__).parent.parent / "models"
MODEL_NAME = str(models_dir / "yolov8n.pt")

MAX_DIST_MM = 200  # ignore objects further than this
TRACK_HALF_WIDTH_MM = 180  # half the buggy's physical track width

_MATRICES_DIR = (
    Path(__file__).parent.parent.parent / "pathfinding" / "calibration" / "matrices"
)


# ── Camera ────────────────────────────────────────────────────────────────────


def _load_K(camera_name: str) -> np.ndarray:
    """Load the refined intrinsic matrix for 'built-in' or 'usb'."""
    return np.load(_MATRICES_DIR / f"intrinsicNew_{camera_name}.npy")


def _build_path_mask(
    W: int, H: int, K: np.ndarray, mount_height: float, mount_angle: float
) -> np.ndarray:
    """
    Precompute a binary mask (H, W) where 1 = pixel maps to a ground point
    within the buggy's forward path.

    Done once at startup so the per-frame check is just a crop + .any().
    Uses the same back-projection math as screen_to_location.py.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    U, V = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32)
    )  # both (H, W)

    h_angles = np.degrees(np.arctan2(U - cx, fx))  # horizontal angle per pixel
    v_angles = np.degrees(np.arctan2(V - cy, fy))  # vertical angle per pixel

    depression = np.radians(mount_angle + v_angles)  # total downward depression

    above_horizon = depression <= 0
    safe_dep = np.where(above_horizon, 1e-6, depression)  # avoid division by zero

    dist = (mount_height / np.tan(safe_dep)) / np.cos(np.radians(h_angles))
    lateral = dist * np.tan(np.radians(h_angles))

    mask = (
        ~above_horizon
        & (dist > 0)
        & (dist < MAX_DIST_MM)
        & (np.abs(lateral) < TRACK_HALF_WIDTH_MM)
    ).astype(np.uint8)

    return mask


# ── Worker ────────────────────────────────────────────────────────────────────


def yolo_worker(shared_state):
    # 1. SETUP — runs once, initialise hardware/models/connections
    shared_state.yolo_running.value = True

    mount_height = shared_state.mount_height  # mm above ground
    mount_angle = shared_state.mount_angle  # degrees downward tilt

    W = shared_state.PATH_FRAME_W
    H = shared_state.PATH_FRAME_H

    # Frames in path_frame_buffer are already undistorted with intrinsicNew_*,
    # so use the matching matrix for back-projection.
    cam_name = "usb" if shared_state.pathcam_index.value != 0 else "built-in"
    K = _load_K(cam_name)

    # Precompute which pixels correspond to the danger zone on the ground.
    path_mask = _build_path_mask(W, H, K, mount_height, mount_angle)

    # load in yolo model, cheapest one possible it only needs to detect object
    model = YOLO(MODEL_NAME)

    # 2. LOOP — runs until shutdown signal
    try:
        while not shared_state.shutdown.is_set():
            # read from hardware or shared_state
            # process
            # write results to shared_state

            # get input from pathcam frame buffer
            if not shared_state.path_frame_ready.wait(timeout=1.0):
                continue
            shared_state.path_frame_ready.clear()

            with shared_state.path_frame_buffer.get_lock():
                frame = (
                    np.frombuffer(
                        shared_state.path_frame_buffer.get_obj(), dtype=np.uint8
                    )
                    .reshape(H, W, 3)
                    .copy()
                )

            # pass in input to yolo model
            results = model(frame, verbose=False)[0]

            # check if any detection overlaps the precomputed path mask
            obstacle = False
            for box in results.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                if path_mask[y1:y2, x1:x2].any():
                    obstacle = True
                    break

            shared_state.yolo_obstacle_detected.value = obstacle
            print(f"[yolo_worker] {'OBSTACLE' if obstacle else 'clear'}")

    except Exception as e:
        print(f"[yolo_worker] Fatal error: {e}")
    finally:
        # TEARDOWN
        # 3. TEARDOWN — release hardware, close connections
        shared_state.yolo_running.value = False


if __name__ == "__main__":
    W, H = 640, 480
    MOUNT_HEIGHT_TEST = 300.0  # mm — adjust to match your setup
    MOUNT_ANGLE_TEST = 30.0  # degrees downward — adjust to match your setup

    K = _load_K("built-in")
    path_mask = _build_path_mask(W, H, K, MOUNT_HEIGHT_TEST, MOUNT_ANGLE_TEST)
    model = YOLO(MODEL_NAME)

    # Precompute a green overlay of the danger zone for visualisation
    mask_overlay = np.zeros((H, W, 3), dtype=np.uint8)
    mask_overlay[path_mask == 1] = (0, 255, 0)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    print("Running — press Q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, verbose=False)[0]

        obstacle = False
        for box in results.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            in_path = path_mask[y1:y2, x1:x2].any()
            if in_path:
                obstacle = True

            # red box = in path, blue box = outside path
            colour = (0, 0, 255) if in_path else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            label = results.names[int(box.cls)]
            cv2.putText(
                frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1
            )

        # blend danger zone overlay onto frame
        frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.3, 0)

        status = "OBSTACLE" if obstacle else "clear"
        status_col = (0, 0, 255) if obstacle else (0, 255, 0)
        cv2.putText(
            frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_col, 2
        )

        print(f"[yolo_worker] {status}")
        cv2.imshow("yolo_worker test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
