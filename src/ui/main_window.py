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
MOCK_STATE_STYLES = {
    0: ("IDLE", "#888888"),
    1: ("MOVE", "#00cc66"),
    2: ("JAW CLENCH", "#cc8800"),
}


class MainWindow(QMainWindow):
    def __init__(self, shared_state, mock_classifier=False):
        super().__init__()
        self.shared_state = shared_state
        self.mock_classifier = mock_classifier
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
        layout.setSpacing(6)  # tighter spacing

        layout.addWidget(self._build_status_group())
        layout.addWidget(self._build_camera_switcher_group())
        layout.addWidget(self._build_gaze_group())
        layout.addWidget(self._build_eyetracker_controls_group())
        layout.addWidget(self._build_motor_group())
        layout.addWidget(self._build_pathfinding_group())

        if self.mock_classifier:
            layout.addWidget(self._build_mock_classifier_group())
        else:
            layout.addWidget(self._build_classifier_group())

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
        self._mirror_check = QCheckBox("Mirror")
        self._mirror_check.setChecked(True)
        self._mirror_check.toggled.connect(lambda v: setattr(self, "mirrored", v))

        layout.addWidget(self._radio_eyetracker)
        layout.addWidget(self._radio_pathfinding)
        layout.addStretch()
        layout.addWidget(self._mirror_check)
        return box

    # --- Gaze readout ---

    def _build_gaze_group(self) -> QGroupBox:
        box = QGroupBox("Gaze")
        layout = QVBoxLayout(box)
        layout.setSpacing(2)

        self._gaze_label = QLabel("Pos:  --  |  Face: not detected")
        self._gaze_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self._gaze_label)
        return box

    # --- Eyetracker controls ---

    def _build_eyetracker_controls_group(self) -> QGroupBox:
        box = QGroupBox("Eyetracker")
        layout = QVBoxLayout(box)

        # row 1: both camera spinners on same line
        cam_row = QHBoxLayout()

        cam_row.addWidget(QLabel("Eye:"))
        self._cam_spin = QSpinBox()
        self._cam_spin.setRange(0, 100)
        self._cam_spin.setValue(self.shared_state.camera_index.value)
        self._cam_spin.setFixedWidth(55)
        self._cam_spin.valueChanged.connect(self._on_eyetracker_cam_changed)
        cam_row.addWidget(self._cam_spin)

        cam_row.addSpacing(10)

        cam_row.addWidget(QLabel("Path:"))
        self._path_cam_spin = QSpinBox()
        self._path_cam_spin.setRange(0, 100)
        self._path_cam_spin.setValue(self.shared_state.pathfinding_camera_index.value)
        self._path_cam_spin.setFixedWidth(55)
        self._path_cam_spin.valueChanged.connect(self._on_pathfinding_cam_changed)
        cam_row.addWidget(self._path_cam_spin)

        cam_row.addStretch()
        layout.addLayout(cam_row)

        # row 2: smoothing slider
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Smooth:"))
        self._smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self._smooth_slider.setRange(0, 100)
        self._smooth_slider.setValue(
            int(self.shared_state.smoothing_factor.value * 100)
        )
        self._smooth_slider.valueChanged.connect(self._on_smoothing_changed)
        smooth_row.addWidget(self._smooth_slider)
        self._smooth_label = QLabel(f"{self.shared_state.smoothing_factor.value:.2f}")
        self._smooth_label.setFixedWidth(30)
        smooth_row.addWidget(self._smooth_label)
        layout.addLayout(smooth_row)

        return box

    # -- Classifier output --
    def _build_classifier_group(self) -> QGroupBox:
        box = QGroupBox("Classifier")
        layout = QVBoxLayout(box)

        self._pred_label = QLabel("Prediction: --")
        self._pred_label.setStyleSheet(
            "font-family: monospace; font-size: 13px; font-weight: bold;"
        )
        layout.addWidget(self._pred_label)

        self._conf_label = QLabel("Confidence: --")
        self._conf_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self._conf_label)

        return box

    # -- Mock classifier --
    def _build_mock_classifier_group(self) -> QGroupBox:
        box = QGroupBox("⚠ Mock Classifier")
        box.setStyleSheet("QGroupBox { color: orange; font-weight: bold; }")
        layout = QVBoxLayout(box)

        # current state indicator
        self._mock_state_label = QLabel("Current: IDLE")
        self._mock_state_label.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: #888888;"
        )
        layout.addWidget(self._mock_state_label)

        btn_move = QPushButton("Simulate MOVE")
        btn_move.setStyleSheet("background: #00cc66; color: white; font-weight: bold;")
        btn_move.clicked.connect(self._mock_move)
        layout.addWidget(btn_move)

        btn_jaw = QPushButton("Simulate JAW CLENCH")
        btn_jaw.setStyleSheet("background: #cc8800; color: white; font-weight: bold;")
        btn_jaw.clicked.connect(self._mock_jaw_clench)
        layout.addWidget(btn_jaw)

        btn_idle = QPushButton("Simulate IDLE")
        btn_idle.setStyleSheet("background: #555555; color: white;")
        btn_idle.clicked.connect(self._mock_idle)
        layout.addWidget(btn_idle)

        return box

    def _mock_move(self):
        self.shared_state.prediction.value = 1
        self.shared_state.pred_confidence.value = 1.0
        self._update_mock_state_label(1)

    def _mock_jaw_clench(self):
        self.shared_state.prediction.value = 2
        self.shared_state.pred_confidence.value = 1.0
        self._update_mock_state_label(2)

    def _mock_idle(self):
        self.shared_state.prediction.value = 0
        self.shared_state.pred_confidence.value = 1.0
        self._update_mock_state_label(0)

    def _update_mock_state_label(self, pred: int):
        name, color = MOCK_STATE_STYLES.get(pred, ("?", "#fff"))
        self._mock_state_label.setText(f"Current: {name}")
        self._mock_state_label.setStyleSheet(
            f"font-family: monospace; font-size: 12px; "
            f"font-weight: bold; color: {color};"
        )

    # --- Motor controls ---

    def _build_motor_group(self) -> QGroupBox:
        box = QGroupBox("Motor")
        layout = QVBoxLayout(box)

        # state readout
        self._motor_state_label = QLabel("State: --")
        self._motor_state_label.setStyleSheet(
            "font-family: monospace; font-size: 13px;"
        )
        layout.addWidget(self._motor_state_label)

        # target readout
        self._target_label = QLabel("Target: --")
        self._target_label.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: #888;"
        )
        layout.addWidget(self._target_label)

        # emergency stop — only manual control that makes sense
        self._btn_stop = QPushButton("⬛ EMERGENCY STOP")
        self._btn_stop.setStyleSheet(
            "background: #cc3333; color: white; font-weight: bold; "
            "font-size: 14px; padding: 8px;"
        )
        self._btn_stop.clicked.connect(lambda: self._send_motor_command(0))
        layout.addWidget(self._btn_stop)

        return box

    # -- Pathfinding group

    def _build_pathfinding_group(self) -> QGroupBox:
        box = QGroupBox("Pathfinding")
        layout = QVBoxLayout(box)

        self._angle_dist_label = QLabel("→ --°  |  --mm")
        self._obstacle_label = QLabel("Obstacle: --")

        for lbl in (self._angle_dist_label, self._obstacle_label):
            lbl.setStyleSheet("font-family: monospace; font-size: 12px;")
            layout.addWidget(lbl)

        return box

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def _tick(self):
        self._update_feed()
        self._update_gaze_readout()
        self._update_worker_status()
        self._update_motor_readout()
        self._update_pathfinding_readout()
        if not self.mock_classifier:
            self._update_classifier_readout()

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
            cx = int(gaze_x * w)
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
        face_str = "✓" if face else "✗"
        if gx >= 0:
            self._gaze_label.setText(f"Pos:  ({gx:.3f}, {gy:.3f})  |  Face: {face_str}")
        else:
            self._gaze_label.setText(f"Pos:  --  |  Face: {face_str}")

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

    def _update_classifier_readout(self):
        pred = self.shared_state.prediction.value
        if pred < 0:
            self._pred_label.setText("Prediction: --")
            self._conf_label.setText("Confidence: --")
            return

        label = {0: "idle", 1: "move", 2: "jaw_clench"}.get(pred, "?")
        colors = {0: "#888888", 1: "#00cc66", 2: "#cc8800"}
        self._pred_label.setText(f"Prediction: {label.upper()}")
        self._pred_label.setStyleSheet(
            f"font-family: monospace; font-size: 13px; font-weight: bold; color: {colors.get(pred, '#fff')};"
        )
        conf = self.shared_state.pred_confidence.value
        if conf >= 0:
            self._conf_label.setText(f"Confidence: {conf:.0%}")

    def _update_motor_readout(self):
        state_id = self.shared_state.motor_state.value
        state_name = {0: "IDLE", 1: "DRIVING"}.get(state_id, "?")
        colors = {0: "#888888", 1: "#00cc66"}
        self._motor_state_label.setText(f"State: {state_name}")
        self._motor_state_label.setStyleSheet(
            f"font-family: monospace; font-size: 13px; "
            f"font-weight: bold; color: {colors.get(state_id, '#fff')};"
        )

        angle = self.shared_state.target_angle.value
        dist = self.shared_state.target_dist.value
        if state_id == 1:
            self._target_label.setText(f"Target: {angle:+.1f}°  {dist:.0f}mm")
        else:
            self._target_label.setText("Target: --")

    def _update_pathfinding_readout(self):
        angle = self.shared_state.target_angle.value
        dist = self.shared_state.target_dist.value
        obs = self.shared_state.obstacle_detected.value

        self._angle_dist_label.setText(f"→ {angle:+.1f}°  |  {dist:.0f}mm")
        self._obstacle_label.setText(f"Obstacle: {'⚠ YES' if obs else 'clear'}")

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
