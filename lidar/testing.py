from rplidar import RPLidar, RPLidarException
import time

# PORT = "COM3"
PORT = "/dev/ttyUSB0"
BAUD = 115200
# BAUD = 256000  # try this if 115200 fails

lidar = RPLidar(PORT, baudrate=BAUD, timeout=3)
lidar.connect()
time.sleep(0.1)

try:
    lidar.connect()
    if hasattr(lidar, "clear_input"):
        print("i have clear input")
        lidar.clear_input()
    elif hasattr(lidar, "_serial"):
        print("i have serial reset input buffer")
        lidar._serial.reset_input_buffer()

    info = lidar.get_info()
    print("Device info:", info)

    health = lidar.get_health()
    print("Health:", health)  # should be ('Good', 0)

    lidar.start_motor()
    time.sleep(3.0)  # <-- give motor time to stabilize
    print("Reading one full scan...\n")

    for scan_index, scan in enumerate(lidar.iter_scans(max_buf_meas=2000, min_len=100)):
        print(f"Scan {scan_index}")
        print(f"Number of points: {len(scan)}\n")

        for quality, angle, distance in scan:
            print(
                f"Quality: {quality}\tAngle: {angle:.2f}\tDistance: {distance:.2f} mm"
            )

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
