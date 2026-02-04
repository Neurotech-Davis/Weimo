# Usage

1. Initialize a virtual environment
2. `pip install -r requirements.txt`
3. Change the `camera_index` to your system camera, zero (should be zero, but if not debug with `v4l2-ctl --list-devices`)
4. `python3 mediapipe_demo.py`

# Notes

Taking some notes while developing

#### Docs:

- [updated solutions page](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [face landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

#### Extending to gaze estimation:

- [100 lines of code medium](https://medium.com/@amit.aflalo2/eye-gaze-estimation-using-a-webcam-in-100-lines-of-code-570d4683fe23)

- [build a mouse w eye yt vid](https://www.youtube.com/watch?v=k3PcVruvZCs)
