import cv2

def list_avail_cams(max=10):
    avail_indicies = []
    for index in range(max):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                avail_indicies.append(index)

            cap.release()
    return avail_indicies

print("avail: ", list_avail_cams())