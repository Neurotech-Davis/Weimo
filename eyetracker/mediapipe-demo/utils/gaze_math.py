import numpy as np


def relative(landmark, shape):
    return int(landmark.x * shape[1]), int(landmark.y * shape[0])


def relativeT(landmark, shape):
    return int(landmark.x * shape[1]), int(landmark.y * shape[0], 0)


def gaze(frame, points):
    """
    2D image points.
    relative takes mediapipe points that normelized to [-1, 1] and returns image points
    at (x,y) format
    """
    image_points = np.array(
        [
            relative(points.landmark[4], frame.shape),  # Nose tip
            relative(points.landmark[152], frame.shape),  # Chin
            relative(points.landmark[263], frame.shape),  # Left eye left corner
            relative(points.landmark[33], frame.shape),  # Right eye right corner
            relative(points.landmark[287], frame.shape),  # Left Mouth corner
            relative(points.landmark[57], frame.shape),  # Right mouth corner
        ],
        dtype="double",
    )

    # 3D model points.
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),  # Nose tip
            (0, -63.6, -12.5),  # Chin
            (-43.3, 32.7, -26),  # Left eye left corner
            (43.3, 32.7, -26),  # Right eye right corner
            (-28.9, -28.9, -24.1),  # Left Mouth corner
            (28.9, -28.9, -24.1),  # Right mouth corner
        ]
    )
    """
    3D model eye points
    The center of the eye ball
    """
    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])
