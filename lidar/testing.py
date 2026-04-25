from rplidar import RPLidar, RPLidarException
import time

PORT = "COM3"
BAUD = 115200
# BAUD = 256000  # try this if 115200 fails

lidar = RPLidar(PORT, baudrate=BAUD, timeout=1)

try:
    lidar.start_motor()
    time.sleep(1.0)

    print("Reading one full scan...\n")

    for scan_index, scan in enumerate(lidar.iter_scans(max_buf_meas=2000, min_len=100)):
        print(f"Scan {scan_index}")
        print(f"Number of points: {len(scan)}\n")

        for quality, angle, distance in scan:
            print(f"Quality: {quality}\tAngle: {angle:.2f}\tDistance: {distance:.2f} mm")

        break  # only print one scan

except RPLidarException as e:
    print("RPLidarException:", e)

except KeyboardInterrupt:
    print("\nStopped")

finally:
    try:
        lidar.stop()
    except Exception:
        pass

    try:
        lidar.stop_motor()
    except Exception:
        pass

    try:
        lidar.disconnect()
    except Exception:
        pass