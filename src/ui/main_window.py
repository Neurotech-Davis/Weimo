"""
ui/main_window.py

MVP main window.
Layout: camera feed (left) | controls + readouts (right)

Right panel sections:
  - Worker Status     : running/stopped indicators per worker
  - Camera Switcher   : swap between eyetracker and pathfinding feed
  - Gaze Readout      : live X/Y numeric display
  - Eyetracker Ctrl   : camera index, smoothing factor
  - Motor Controls    : command buttons + state readout
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QGroupBox,
    QCheckBox,
    QPushButton,
    QButtonGroup,
    QRadioButton,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

FEED_EYETRACKER = "eyetracker"
FEED_PATHFINDING = "pathfinding"

# Must match COMMANDS dict in motor_worker.py
MOTOR_COMMANDS = {
    "STOP": 0,
    "▲ Forward": 1,
    "▼ Backward": 2,
    "◄ Rot Left": 3,
    "► Rot Right": 4,
    "« Strafe L": 5,
    "» Strafe R": 6,
}

MOTOR_STATE_NAMES = {v: k for k, v in MOTOR_COMMANDS.items()}


class MainWindow(QMainWindow):
    def __init__(self, shared_state):
        super().__init__()
        self.shared_state = shared_state
        self.setWindowTitle("Weimo")
        self.resize(1100, 650)
        self.mirrored = True
        self._active_feed = FEED_EYETRACKER

        self._build_ui()

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(self._build_feed_panel(), stretch=3)
        layout.addLayout(self._build_right_panel(), stretch=1)

    def _build_feed_panel(self) -> QLabel:
        self._feed_label = QLabel("Waiting for camera...")
        self._feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._feed_label.setMinimumSize(640, 480)
        self._feed_label.setStyleSheet("background: #111; color: #555;")
        return self._feed_label

    def _build_right_panel(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(10)

        layout.addWidget(self._build_status_group())
        layout.addWidget(self._build_camera_switcher_group())
        layout.addWidget(self._build_gaze_group())
        layout.addWidget(self._build_eyetracker_controls_group())
        layout.addWidget(self._build_motor_group())
        layout.addStretch()

        return layout

    # --- Worker status ---

    def _build_status_group(self) -> QGroupBox:
        box = QGroupBox("Worker Status")
        layout = QVBoxLayout(box)

        self._status_labels = {}
        for name in ("eyetracker", "pathfinding", "classifier", "motor"):
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            indicator = QLabel("●")
            indicator.setStyleSheet("color: #555;")
            self._status_labels[name] = indicator
            row.addStretch()
            row.addWidget(indicator)
            layout.addLayout(row)

        return box

    # --- Camera switcher ---

    def _build_camera_switcher_group(self) -> QGroupBox:
        box = QGroupBox("Camera Feed")
        layout = QVBoxLayout(box)

        self._feed_btn_group = QButtonGroup(box)
        self._radio_eyetracker = QRadioButton("Eyetracker")
        self._radio_pathfinding = QRadioButton("Pathfinding")
        self._radio_eyetracker.setChecked(True)

        self._feed_btn_group.addButton(self._radio_eyetracker, 0)
        self._feed_btn_group.addButton(self._radio_pathfinding, 1)

        self._radio_eyetracker.toggled.connect(
            lambda checked: self._on_feed_switched(FEED_EYETRACKER) if checked else None
        )
        self._radio_pathfinding.toggled.connect(
            lambda checked: (
                self._on_feed_switched(FEED_PATHFINDING) if checked else None
            )
        )

        self._mirror_check = QCheckBox("Mirror feed")
        self._mirror_check.setChecked(True)
        self._mirror_check.toggled.connect(lambda v: setattr(self, "mirrored", v))

        layout.addWidget(self._radio_eyetracker)
        layout.addWidget(self._radio_pathfinding)
        layout.addWidget(self._mirror_check)

        return box

    # --- Gaze readout ---

    def _build_gaze_group(self) -> QGroupBox:
        box = QGroupBox("Gaze Position")
        layout = QVBoxLayout(box)

        self._gaze_x_label = QLabel("X:  --")
        self._gaze_y_label = QLabel("Y:  --")
        self._face_label = QLabel("Face: --")

        for lbl in (self._gaze_x_label, self._gaze_y_label, self._face_label):
            lbl.setStyleSheet("font-family: monospace; font-size: 13px;")
            layout.addWidget(lbl)

        return box

    # --- Eyetracker controls ---

    def _build_eyetracker_controls_group(self) -> QGroupBox:
        box = QGroupBox("Eyetracker")
        layout = QVBoxLayout(box)

        layout.addWidget(QLabel("Eyetracker Camera Index"))
        self._cam_spin = QSpinBox()
        self._cam_spin.setRange(0, 100)
        self._cam_spin.setValue(self.shared_state.camera_index.value)
        self._cam_spin.valueChanged.connect(self._on_eyetracker_cam_changed)
        layout.addWidget(self._cam_spin)

        layout.addWidget(QLabel("Pathfinding Camera Index"))
        self._path_cam_spin = QSpinBox()
        self._path_cam_spin.setRange(0, 100)
        self._path_cam_spin.setValue(self.shared_state.pathfinding_camera_index.value)
        self._path_cam_spin.valueChanged.connect(self._on_pathfinding_cam_changed)
        layout.addWidget(self._path_cam_spin)

        layout.addWidget(QLabel("Smoothing"))
        self._smooth_label = QLabel(f"{self.shared_state.smoothing_factor.value:.2f}")
        self._smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self._smooth_slider.setRange(0, 100)
        self._smooth_slider.setValue(
            int(self.shared_state.smoothing_factor.value * 100)
        )
        self._smooth_slider.valueChanged.connect(self._on_smoothing_changed)
        layout.addWidget(self._smooth_slider)
        layout.addWidget(self._smooth_label)

        return box

    # --- Motor controls ---

    def _build_motor_group(self) -> QGroupBox:
        box = QGroupBox("Motor")
        layout = QVBoxLayout(box)

        self._motor_state_label = QLabel("State: --")
        self._motor_state_label.setStyleSheet(
            "font-family: monospace; font-size: 13px;"
        )
        layout.addWidget(self._motor_state_label)

        # row 1: stop spans full width
        self._btn_stop = QPushButton("STOP")
        self._btn_stop.setStyleSheet(
            "background: #cc3333; color: white; font-weight: bold;"
        )
        self._btn_stop.clicked.connect(lambda: self._send_motor_command(0))
        layout.addWidget(self._btn_stop)

        # row 2: forward
        row_fwd = QHBoxLayout()
        btn_fwd = QPushButton("▲")
        btn_fwd.clicked.connect(lambda: self._send_motor_command(1))
        row_fwd.addStretch()
        row_fwd.addWidget(btn_fwd)
        row_fwd.addStretch()
        layout.addLayout(row_fwd)

        # row 3: strafe left | rot left | rot right | strafe right
        row_mid = QHBoxLayout()
        for label, cmd_id in (("«", 5), ("◄", 3), ("►", 4), ("»", 6)):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, c=cmd_id: self._send_motor_command(c))
            row_mid.addWidget(btn)
        layout.addLayout(row_mid)

        # row 4: backward
        row_bwd = QHBoxLayout()
        btn_bwd = QPushButton("▼")
        btn_bwd.clicked.connect(lambda: self._send_motor_command(2))
        row_bwd.addStretch()
        row_bwd.addWidget(btn_bwd)
        row_bwd.addStretch()
        layout.addLayout(row_bwd)

        return box

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def _tick(self):
        self._update_feed()
        self._update_gaze_readout()
        self._update_worker_status()
        self._update_motor_readout()

    def _update_feed(self):
        if self._active_feed == FEED_EYETRACKER:
            self._render_eyetracker_feed()
        else:
            self._render_pathfinding_feed()

    def _render_eyetracker_feed(self):
        if not self.shared_state.frame_ready.is_set():
            return

        with self.shared_state.frame_buffer.get_lock():
            buf = np.frombuffer(
                self.shared_state.frame_buffer.get_obj(), dtype=np.uint8
            )
            frame = buf.reshape(
                (self.shared_state.FRAME_H, self.shared_state.FRAME_W, 3)
            ).copy()

        if self.mirrored:
            frame = cv2.flip(frame, 1)

        gaze_x, gaze_y = self.shared_state.get_gaze()
        if gaze_x >= 0 and gaze_y >= 0:
            h, w = frame.shape[:2]
            cx = int((1.0 - gaze_x) * w) if self.mirrored else int(gaze_x * w)
            cy = int(gaze_y * h)
            cv2.circle(frame, (cx, cy), 12, (0, 255, 100), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 100), -1)

        self._display_frame(frame)

    def _render_pathfinding_feed(self):
        if not hasattr(self.shared_state, "pathfinding_frame_ready"):
            self._feed_label.setText("Pathfinding feed not available")
            return
        if not self.shared_state.pathfinding_frame_ready.is_set():
            return

        with self.shared_state.pathfinding_frame_buffer.get_lock():
            buf = np.frombuffer(
                self.shared_state.pathfinding_frame_buffer.get_obj(), dtype=np.uint8
            )
            frame = buf.reshape(
                (self.shared_state.FRAME_H, self.shared_state.FRAME_W, 3)
            ).copy()

        if self.mirrored:
            frame = cv2.flip(frame, 1)

        self._display_frame(frame)

    def _display_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._feed_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._feed_label.setPixmap(pixmap)

    def _update_gaze_readout(self):
        gx, gy = self.shared_state.get_gaze()
        face = self.shared_state.face_detected.value
        if gx >= 0:
            self._gaze_x_label.setText(f"X:  {gx:.3f}")
            self._gaze_y_label.setText(f"Y:  {gy:.3f}")
        else:
            self._gaze_x_label.setText("X:  --")
            self._gaze_y_label.setText("Y:  --")
        self._face_label.setText(f"Face: {'detected' if face else 'not detected'}")

    def _update_worker_status(self):
        status_map = {
            "eyetracker": self.shared_state.tracker_running.value,
            "pathfinding": self.shared_state.pathfinding_running.value,
            "classifier": self.shared_state.classifier_running.value,
            "motor": self.shared_state.motor_running.value,
        }
        for name, label in self._status_labels.items():
            running = status_map.get(name, False)
            label.setStyleSheet("color: #00cc66;" if running else "color: #555;")

    def _update_motor_readout(self):
        state_id = self.shared_state.motor_state.value
        self._motor_state_label.setText(
            f"State: {MOTOR_STATE_NAMES.get(state_id, '?')}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_feed_switched(self, feed: str):
        self._active_feed = feed
        self._feed_label.setText(f"Switching to {feed} feed...")

    def _on_eyetracker_cam_changed(self, value: int):
        self.shared_state.camera_index.value = value

    def _on_pathfinding_cam_changed(self, value: int):
        self.shared_state.pathfinding_camera_index.value = value

    def _on_smoothing_changed(self, value: int):
        factor = value / 100.0
        self.shared_state.smoothing_factor.value = factor
        self._smooth_label.setText(f"{factor:.2f}")

    def _send_motor_command(self, cmd_id: int):
        self.shared_state.motor_command.value = cmd_id

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._timer.stop()
        self.shared_state.shutdown.set()
        event.accept()
