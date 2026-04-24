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

import time
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

# constants at top of main_window.py
ZONE_WIDTH = 0.15  # 15% of frame width
ZONE_HEIGHT = 0.25  # each zone is 25% of frame height
ZONE_DWELL_SEC = 0.5  # how long gaze must dwell before activating

# zone definitions: (y_min, y_max, turn_degrees)
TURN_ZONES_LEFT = [
    (0.0, 0.25, -45),  # top left    → 45° CCW
    (0.75, 1.0, -90),  # bottom left → 90° CCW
]
TURN_ZONES_RIGHT = [
    (0.0, 0.25, 45),  # top right    → 45° CW
    (0.75, 1.0, 90),  # bottom right → 90° CW
]


class MainWindow(QMainWindow):
    def __init__(self, shared_state, mock_classifier=False, mixed_classifier=False):
        super().__init__()
        self.shared_state = shared_state
        self.mock_classifier = mock_classifier
        self.mixed_classifier = mixed_classifier
        self.setWindowTitle("Weimo")
        self.resize(1100, 650)
        self.mirrored = True
        self._active_feed = FEED_EYETRACKER

        self._path_cap = None
        self._current_path_idx = None
        self._zone_dwell_start = None
        self._zone_active_deg = None

        self._build_ui()

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_group(self, title: str) -> tuple:
        """Create a QGroupBox with a vertical layout and consistent tight margins."""
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(3)
        return box, layout

    def _make_hgroup(self, title: str) -> tuple:
        """Create a QGroupBox with a horizontal layout and tight margins."""
        box = QGroupBox(title)
        layout = QHBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)
        return box, layout

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(6, 4, 6, 4)
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
        layout.setSpacing(4)

        layout.addWidget(self._build_status_group())
        layout.addWidget(self._build_camera_switcher_group())
        layout.addWidget(self._build_gaze_group())
        layout.addWidget(self._build_eyetracker_controls_group())
        layout.addWidget(self._build_motor_group())
        layout.addWidget(self._build_pathfinding_group())

        if self.mock_classifier:
            layout.addWidget(self._build_mock_classifier_group())
        elif self.mixed_classifier:
            layout.addWidget(self._build_classifier_group())  # live readout
            layout.addWidget(self._build_mock_classifier_group())  # + override buttons
        else:
            layout.addWidget(self._build_classifier_group())

        layout.addStretch()
        return layout

    # --- Worker status ---

    def _build_status_group(self) -> QGroupBox:
        box, layout = self._make_hgroup("Worker Status")

        self._status_labels = {}
        for name in ("eyetracker", "pathfinding", "classifier", "motor"):
            label = QLabel(f"● {name[:3]}")
            label.setStyleSheet("color: #555; font-family: monospace; font-size: 11px;")
            self._status_labels[name] = label
            layout.addWidget(label)

        return box

    # --- Camera switcher ---

    def _build_camera_switcher_group(self) -> QGroupBox:
        box, layout = self._make_group("Camera Feed")

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

        # put all three on one row
        row = QHBoxLayout()
        row.addWidget(self._radio_eyetracker)
        row.addWidget(self._radio_pathfinding)
        row.addStretch()
        row.addWidget(self._mirror_check)
        layout.addLayout(row)

        return box

    # --- Gaze readout ---

    def _build_gaze_group(self) -> QGroupBox:
        box, layout = self._make_group("Gaze")

        self._gaze_label = QLabel("Pos:  --  |  Face: not detected")
        self._gaze_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self._gaze_label)
        return box

    # --- Eyetracker controls ---

    def _build_eyetracker_controls_group(self) -> QGroupBox:
        box, layout = self._make_group("Eyetracker")

        # row 1: both camera spinners on same line
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Eye:"))
        self._cam_spin = QSpinBox()
        self._cam_spin.setRange(0, 100)
        self._cam_spin.setValue(self.shared_state.camera_index.value)
        self._cam_spin.setFixedWidth(75)
        self._cam_spin.valueChanged.connect(self._on_eyetracker_cam_changed)
        cam_row.addWidget(self._cam_spin)

        cam_row.addSpacing(20)

        cam_row.addWidget(QLabel("Path:"))
        self._path_cam_spin = QSpinBox()
        self._path_cam_spin.setRange(0, 100)
        self._path_cam_spin.setValue(self.shared_state.pathfinding_camera_index.value)
        self._path_cam_spin.setFixedWidth(75)
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

    # --- Classifier output ---

    def _build_classifier_group(self) -> QGroupBox:
        box, layout = self._make_group("Classifier")

        self._pred_label = QLabel("Prediction: --")
        self._pred_label.setStyleSheet(
            "font-family: monospace; font-size: 13px; font-weight: bold;"
        )
        layout.addWidget(self._pred_label)

        self._conf_label = QLabel("Confidence: --")
        self._conf_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self._conf_label)

        return box

    # --- Mock / override classifier ---

    def _build_mock_classifier_group(self) -> QGroupBox:
        title = "⚠ Overrides" if self.mixed_classifier else "⚠ Mock Classifier"
        box, layout = self._make_group(title)
        box.setStyleSheet("QGroupBox { color: orange; font-weight: bold; }")

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
            f"font-family: monospace; font-size: 12px; font-weight: bold; color: {color};"
        )

    # --- Motor controls ---

    def _build_motor_group(self) -> QGroupBox:
        box, layout = self._make_group("Motor")

        self._motor_state_label = QLabel("State: --")
        self._motor_state_label.setStyleSheet(
            "font-family: monospace; font-size: 13px;"
        )
        layout.addWidget(self._motor_state_label)

        self._target_label = QLabel("Target: --")
        self._target_label.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: #888;"
        )
        layout.addWidget(self._target_label)

        self._btn_stop = QPushButton("⬛ EMERGENCY STOP")
        self._btn_stop.setStyleSheet(
            "background: #cc3333; color: white; font-weight: bold; "
            "font-size: 14px; padding: 8px;"
        )
        self._btn_stop.clicked.connect(lambda: self._send_motor_command(1))
        layout.addWidget(self._btn_stop)

        return box

    # --- Pathfinding group ---

    def _build_pathfinding_group(self) -> QGroupBox:
        box, layout = self._make_group("Pathfinding")

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

        gaze_x, gaze_y = self.shared_state.get_gaze()
        self._check_turn_zones(gaze_x, gaze_y)

    def _update_feed(self):
        if self._active_feed == FEED_EYETRACKER:
            if self._path_cap is not None:
                self._path_cap.release()
                self._path_cap = None
            self._render_eyetracker_feed()
        else:
            self._render_pathfinding_feed()

    def _draw_turn_zones(
        self, frame: np.ndarray, gaze_x: float, gaze_y: float
    ) -> np.ndarray:
        """Draw turn zones onto a frame. Works for both feeds."""
        h, w = frame.shape[:2]
        zone_w_px = int(ZONE_WIDTH * w)

        dwell_progress = 0.0
        if self._zone_dwell_start is not None and self._zone_active_deg is not None:
            dwell_progress = min(
                1.0, (time.time() - self._zone_dwell_start) / ZONE_DWELL_SEC
            )

        all_zones = [(0, zone_w_px, *z[:2], z[2]) for z in TURN_ZONES_LEFT] + [
            (w - zone_w_px, w, *z[:2], z[2]) for z in TURN_ZONES_RIGHT
        ]

        for x1, x2, y_min, y_max, deg in all_zones:
            y1 = int(y_min * h)
            y2 = int(y_max * h)
            is_active = self._zone_active_deg == deg

            base_color = (0, 180, 255) if is_active else (80, 80, 80)
            cv2.rectangle(frame, (x1, y1), (x2, y2), base_color, 2)

            if is_active and dwell_progress > 0:
                bar_h = int((y2 - y1) * dwell_progress)
                bar_color = (0, 255, 150) if dwell_progress < 1.0 else (0, 255, 0)
                cv2.rectangle(frame, (x1 + 2, y2 - bar_h), (x2 - 2, y2), bar_color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), base_color, 2)

            direction = "L" if deg < 0 else "R"
            text = f"{direction}{abs(deg)}°"
            cv2.putText(
                frame,
                text,
                (x1 + 4, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        return frame

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
        frame = self._draw_turn_zones(frame, gaze_x, gaze_y)

        if gaze_x >= 0 and gaze_y >= 0:
            h, w = frame.shape[:2]
            cx = int(gaze_x * w)
            cy = int(gaze_y * h)
            cv2.circle(frame, (cx, cy), 12, (0, 255, 100), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 100), -1)

        self._display_frame(frame)

    def _render_pathfinding_feed(self):
        idx = self.shared_state.pathfinding_camera_index.value
        if self._path_cap is None or idx != self._current_path_idx:
            if self._path_cap is not None:
                self._path_cap.release()
            self._path_cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            self._current_path_idx = idx

        ret, frame = self._path_cap.read()
        if not ret:
            h, w = self.shared_state.FRAME_H, self.shared_state.FRAME_W
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            text = f"Cam {idx} Offline"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            cv2.putText(
                frame,
                text,
                ((w - text_size[0]) // 2, (h + text_size[1]) // 2),
                font,
                1,
                (100, 100, 100),
                2,
            )
        elif self.mirrored:
            frame = cv2.flip(frame, 1)

        gaze_x, gaze_y = self.shared_state.get_gaze()
        frame = self._draw_turn_zones(frame, gaze_x, gaze_y)

        if gaze_x >= 0 and gaze_y >= 0:
            h, w = frame.shape[:2]
            cx = int(gaze_x * w)
            cy = int(gaze_y * h)
            cv2.circle(frame, (cx, cy), 12, (0, 255, 100), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 100), -1)

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
            color = "#00cc66" if running else "#555"
            label.setStyleSheet(
                f"color: {color}; font-family: monospace; font-size: 11px;"
            )

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
            f"font-family: monospace; font-size: 13px; font-weight: bold; "
            f"color: {colors.get(pred, '#fff')};"
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

    def _check_turn_zones(self, gaze_x: float, gaze_y: float):
        """Check if gaze is dwelling in a turn zone."""
        if gaze_x < 0 or gaze_y < 0:
            self._zone_dwell_start = None
            self._zone_active_deg = None
            return

        matched_deg = None

        if gaze_x <= ZONE_WIDTH:
            for y_min, y_max, deg in TURN_ZONES_LEFT:
                if y_min <= gaze_y <= y_max:
                    matched_deg = deg
                    break
        elif gaze_x >= (1.0 - ZONE_WIDTH):
            for y_min, y_max, deg in TURN_ZONES_RIGHT:
                if y_min <= gaze_y <= y_max:
                    matched_deg = deg
                    break

        if matched_deg is not None:
            if self._zone_active_deg != matched_deg:
                self._zone_dwell_start = time.time()
                self._zone_active_deg = matched_deg
            else:
                dwell = time.time() - self._zone_dwell_start
                if dwell >= ZONE_DWELL_SEC:
                    if (
                        self.shared_state.prediction.value == 1
                        and self.shared_state.pred_confidence.value >= 0.95
                    ):
                        print(f"[UI] zone turn triggered: {matched_deg}°")
                        self.shared_state.turn_command.value = float(matched_deg)
                        self.shared_state.prediction.value = 0  # consume
                        self._zone_dwell_start = (
                            time.time()
                        )  # reset to avoid re-trigger
        else:
            self._zone_dwell_start = None
            self._zone_active_deg = None

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
        if self._path_cap is not None:
            self._path_cap.release()
        self.shared_state.shutdown.set()
        event.accept()
