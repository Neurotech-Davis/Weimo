from rplidar import RPLidar, RPLidarException
import serial
import time

PORT = "COM3"
BAUD = 115200
#BAUD = 256000
lidar = RPLidar(PORT, baudrate=BAUD, timeout=1)

try:
    lidar.start_motor()
    time.sleep(0.2)

    
    for idx, scan in enumerate(lidar.iter_scans(max_buf_meas=100, min_len=5)):
        # scan is a list of (quality, angle, distance)
        if idx == 0:
            print('index 0')
            s = scan
            break
    print(scan)
    '''
    n=0
    for new_scan, quality, angle, distance in lidar.iter_measures(max_buf_meas=500):
        if new_scan:
            print('---NEW SCAN---')
        print(f"Quality: {quality}\tAngle: {angle}\tDistance: {distance}")
        n+=1
        if n >20:
            break
    '''
except RPLidarException as e:
    print("RPLidarException:", e)

finally:
    try:
        lidar.stop()
    except Exception:
        pass
    try:
        lidar.stop_motor()
    except Exception:
        pass
    lidar.disconnect()
