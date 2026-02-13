# Usage

1. Initialize a virtual environment
2. `pip install -r requirements.txt`
3. Change the `camera_index` to your system camera, zero (should be zero, but if not debug with `v4l2-ctl --list-devices`)
4. `python3 mediapipe_demo.py`

> Depending on your system, you might need some additional installs to get mouse working. On Ubuntu: `sudo apt-get install python3-tk python3-dev`

# Notes

Taking some notes while developing, ignore if just reading through

#### Docs:

- [updated solutions page](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [face landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

#### Extending to gaze estimation:

- [100 lines of code medium](https://medium.com/@amit.aflalo2/eye-gaze-estimation-using-a-webcam-in-100-lines-of-code-570d4683fe23)

- [build a mouse w eye yt vid](https://www.youtube.com/watch?v=k3PcVruvZCs)

#### Specifically head pose estimation:

- Real time head pose estimation, mediapipe + opencv:
  - [youtube vid](https://www.youtube.com/watch?v=-toNMaS4SeQ)
  - [medium](https://medium.com/@jaykumaran2217/real-time-head-pose-estimation-facemesh-with-mediapipe-and-opencv-a-comprehensive-guide-b63a2f40b7c6)

- [also the 100 lines of code](https://medium.com/@amit.aflalo2/eye-gaze-estimation-using-a-webcam-in-100-lines-of-code-570d4683fe23)

- [head tracking mouse control](https://github.com/JEOresearch/EyeTracker/tree/main/HeadTracker)
