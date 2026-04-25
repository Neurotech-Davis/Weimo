"""
workers/lidar_worker.py

RPLIDAR worker process.
- Connects to the sensor with retry logic
- Each scan: writes a 360-float distance array to shared_state.lidar_distances
- Sets shared_state.obstacle_detected for the front cone (345-15 deg, threshold 500 mm)
- Cardinal direction debug print uses a +/-10 deg window min-search so it works even
  when only ~80 of 360 angle slots are populated per scan
"""

from rplidar import RPLidar, RPLidarException
import time

# Windows: COM3/COM9
PORT = "/dev/ttyUSB0"
BAUD = 115200
MIN_DISTANCE = 50  # mm
MAX_DISTANCE = 2000  # mm

# Physical mount offset: 90 = rot 90deg CW.  -90 = 90deg CCW.
MOUNT_OFFSET_DEG = 90

# Front-cone obstacle threshold
OBSTACLE_THRESHOLD_MM = 500
FRONT_CONE_START = 345
FRONT_CONE_END = 15  # wraps through 0 deg


# Debug helper
def window_min(buf: list, center: int, half: int = 10) -> float:
    """
    Return the minimum non-zero distance within [center-half, center+half] (mod 360).
    Returns 0.0 if no valid reading exists in the window.
    """
    best = 0.0
    for offset in range(-half, half + 1):
        v = buf[(center + offset) % 360]
        if v > 0.0:
            best = v if best == 0.0 else min(best, v)
    return best


def attempt_lidar_connection(port, baud, max_retries=5, retry_delay=2):
    for attempt in range(1, max_retries + 1):
        lidar = None
        try:
            print(
                f"[lidar_worker] Attempting connection on {port} (Attempt {attempt})..."
            )
            lidar = RPLidar(port, baudrate=baud, timeout=1)
            lidar.stop()
            lidar.stop_motor()
            time.sleep(2.0)  # wait for motor to physically spin down
            info = lidar.get_info()
            health = lidar.get_health()
            print(f"[lidar_worker] Device info: {info}")
            print(f"[lidar_worker] Health: {health}")

            lidar.start_motor()
            time.sleep(3.0)  # wait for motor to stabilise
            print("[lidar_worker] LIDAR connected and motor spinning.")
            return lidar

        except (RPLidarException, Exception) as e:
            print(f"[lidar_worker] Attempt {attempt}/{max_retries} failed: {e}")
            if lidar:
                try:
                    lidar.stop()
                    lidar.stop_motor()
                    lidar.disconnect()
                except:
                    pass
            if attempt < max_retries:
                time.sleep(retry_delay)
    return None


def lidar_worker(shared_state):
    lidar = attempt_lidar_connection(PORT, BAUD)
    if lidar is None:
        print("[lidar_worker] Could not connect to RPLIDAR")
        shared_state.lidar_running.value = False
        return

    shared_state.lidar_running.value = True
    local_buffer = [0.0] * 360

    try:
        for scan in lidar.iter_scans(max_buf_meas=1000, min_len=50):
            if shared_state.shutdown.is_set():
                break

            local_buffer = [0.0] * 360
            for _, ang, dist in scan:
                if MIN_DISTANCE <= dist <= MAX_DISTANCE:
                    idx = (int(ang) - MOUNT_OFFSET_DEG) % 360
                    # keep the closest reading if multiple hits in the same slot
                    if local_buffer[idx] == 0.0 or dist < local_buffer[idx]:
                        local_buffer[idx] = dist

            shared_state.lidar_distances[:] = local_buffer[:]
            # obstacle detection (front cone 345-15 deg)
            front_indices = list(range(FRONT_CONE_START, 360)) + list(
                range(0, FRONT_CONE_END + 1)
            )
            collision_imminent = any(
                0 < local_buffer[i] < OBSTACLE_THRESHOLD_MM for i in front_indices
            )
            shared_state.obstacle_detected.value = collision_imminent

            # debug: cardinal directions via +/-10 deg window
            n = window_min(local_buffer, 0)
            e = window_min(local_buffer, 90)
            s = window_min(local_buffer, 180)
            w = window_min(local_buffer, 270)
            filled = sum(1 for v in local_buffer if v > 0.0)
            print(
                f"[LIDAR] pts={filled:3d} | "
                f"N:{n:5.0f}mm  E:{e:5.0f}mm  S:{s:5.0f}mm  W:{w:5.0f}mm"
                + ("  OBSTACLE" if collision_imminent else "")
            )

    except Exception as e:
        print(f"[lidar_worker] Fatal error: {e}")
    finally:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        print("[lidar_worker] Successfully shut down LIDAR...")
        shared_state.lidar_running.value = False
