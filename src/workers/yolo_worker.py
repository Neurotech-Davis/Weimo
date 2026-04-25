from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME          = "yolov8n.pt"   # cheapest YOLO — detection only, no segmentation needed
MAX_DIST_MM         = 1500           # ignore objects further than this
TRACK_HALF_WIDTH_MM = 350            # half the buggy's physical track width

_MATRICES_DIR = Path(__file__).parent.parent.parent / "pathfinding" / "calibration" / "matrices"


# ── Camera ────────────────────────────────────────────────────────────────────

def _load_K(camera_name: str) -> np.ndarray:
    """Load the refined intrinsic matrix for 'built-in' or 'usb'."""
    return np.load(_MATRICES_DIR / f"intrinsicNew_{camera_name}.npy")


def _build_path_mask(W: int, H: int, K: np.ndarray,
                     mount_height: float, mount_angle: float) -> np.ndarray:
    """
    Precompute a binary mask (H, W) where 1 = pixel maps to a ground point
    within the buggy's forward path.

    Done once at startup so the per-frame check is just a crop + .any().
    Uses the same back-projection math as screen_to_location.py.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    U, V = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))   # both (H, W)

    h_angles = np.degrees(np.arctan2(U - cx, fx))        # horizontal angle per pixel
    v_angles = np.degrees(np.arctan2(V - cy, fy))        # vertical angle per pixel

    depression = np.radians(mount_angle + v_angles)      # total downward depression

    above_horizon = depression <= 0
    safe_dep = np.where(above_horizon, 1e-6, depression)  # avoid division by zero

    dist = (mount_height / np.tan(safe_dep)) / np.cos(np.radians(h_angles))
    lateral = dist * np.tan(np.radians(h_angles))

    mask = (
        ~above_horizon &
        (dist > 0) &
        (dist < MAX_DIST_MM) &
        (np.abs(lateral) < TRACK_HALF_WIDTH_MM)
    ).astype(np.uint8)

    return mask


# ── Worker ────────────────────────────────────────────────────────────────────

def yolo_worker(shared_state):
    # 1. SETUP — runs once, initialise hardware/models/connections
    shared_state.yolo_running.value = True

    mount_height = shared_state.mount_height   # mm above ground
    mount_angle  = shared_state.mount_angle    # degrees downward tilt

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
                frame = np.frombuffer(
                    shared_state.path_frame_buffer.get_obj(), dtype=np.uint8
                ).reshape(H, W, 3).copy()

            # pass in input to yolo model
            results = model(frame, verbose=False)[0]

            # check if any detection overlaps the precomputed path mask
            obstacle = False
            for box in results.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                if path_mask[y1:y2, x1:x2].any():
                    obstacle = True
                    break

            shared_state.obstacle_detected.value = obstacle
            print(f"[yolo_worker] {'OBSTACLE' if obstacle else 'clear'}")

    except Exception as e:
        print(f"[yolo_worker] Fatal error: {e}")
    finally:
        # TEARDOWN
        # 3. TEARDOWN — release hardware, close connections
        shared_state.yolo_running.value = False
