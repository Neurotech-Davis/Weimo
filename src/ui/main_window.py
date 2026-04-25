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
import math
import numpy as np
from collections import deque
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QGridLayout,
    QGroupBox,
    QCheckBox,
    QPushButton,
    QButtonGroup,
    QRadioButton,
    QSizePolicy,
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

# LIDAR minimap constants
MINIMAP_RADIUS = 80  # px — radius of the polar plot circle
MINIMAP_MARGIN = 12  # px — gap from frame edge
MINIMAP_MAX_DIST = 2000.0  # mm — distance that maps to the outer ring
MINIMAP_RINGS = 3  # number of range rings to draw
MINIMAP_ALPHA = 0.55  # overlay transparency (0=invisible, 1=opaque)

# distance thresholds for point colouring (mm)
MINIMAP_RED_THRESH = 500
MINIMAP_YELLOW_THRESH = 1200


class MainWindow(QMainWindow):
    def __init__(
        self,
        shared_state,
        mock_classifier=False,
        mixed_classifier=False,
        demo_classifier=False,
    ):
        super().__init__()
        self.shared_state = shared_state
        self.mock_classifier = mock_classifier
        self.mixed_classifier = mixed_classifier
        self.demo_classifier = demo_classifier
        self.setWindowTitle("Weimo")
        self.resize(1100, 650)
        self.mirrored = True
        self._active_feed = FEED_EYETRACKER

        self._zone_dwell_start = None
        self._zone_active_deg = None

        self._lidar_history: deque = deque(maxlen=5)

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

        if self.demo_classifier:
            layout.addWidget(self._build_classifier_group())  # live readout
            layout.addWidget(self._build_demo_classifier_group())
        elif self.mock_classifier:
            layout.addWidget(self._build_mock_classifier_group())
        elif self.mixed_classifier:
            layout.addWidget(self._build_classifier_group())  # live readout
            layout.addWidget(self._build_mock_classifier_group())  # + override buttons
        else:
            layout.addWidget(self._build_classifier_group())

        layout.addWidget(self._build_lidar_minimap_widget())
        return layout

    # --- Worker status ---

    def _build_status_group(self) -> QGroupBox:
        box = QGroupBox("Worker Status")
        layout = QGridLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(4)

        workers = [
            "eyetracker",
            "pathfinding",
            "classifier",
            "motor",
            "pathcam",
            "lidar",
        ]

        self._status_labels = {}
        for i, name in enumerate(workers):
            label = QLabel(f"● {name[:4]}")
            label.setStyleSheet("color: #555; font-family: monospace; font-size: 11px;")
            self._status_labels[name] = label

            row = i // 3
            col = i % 3
            layout.addWidget(label, row, col)

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

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Eye:"))
        self._cam_spin = QSpinBox()
        self._cam_spin.setRange(0, 100)
        self._cam_spin.setValue(self.shared_state.eye_camera_index.value)
        self._cam_spin.setFixedWidth(75)
        self._cam_spin.valueChanged.connect(self._on_eyetracker_cam_changed)
        cam_row.addWidget(self._cam_spin)

        cam_row.addSpacing(20)

        cam_row.addWidget(QLabel("Path:"))
        self._path_cam_spin = QSpinBox()
        self._path_cam_spin.setRange(0, 100)
        self._path_cam_spin.setValue(self.shared_state.pathcam_index.value)
        self._path_cam_spin.setFixedWidth(75)
        self._path_cam_spin.valueChanged.connect(self._on_pathfinding_cam_changed)
        cam_row.addWidget(self._path_cam_spin)
        cam_row.addStretch()
        layout.addLayout(cam_row)

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

    # Replace _build_classifier_group entirely

def _build_classifier_group(self) -> QGroupBox:
    box, layout = self._make_group("Classifier")

    # ── Existing prediction/confidence labels ──────────────────────
    self._pred_label = QLabel("Prediction: --")
    self._pred_label.setStyleSheet(
        "font-family: monospace; font-size: 13px; font-weight: bold;"
    )
    layout.addWidget(self._pred_label)

    self._conf_label = QLabel("Confidence: --")
    self._conf_label.setStyleSheet("font-family: monospace; font-size: 12px;")
    layout.addWidget(self._conf_label)

    # ── NEW: feedback buttons ──────────────────────────────────────
    feedback_row = QHBoxLayout()

    self._btn_correct = QPushButton("✓ Correct")
    self._btn_correct.setStyleSheet(
        "background: #1a6b3a; color: white; font-weight: bold; padding: 4px;"
    )
    self._btn_correct.clicked.connect(self._on_feedback_correct)

    self._btn_wrong = QPushButton("✗ Wrong")
    self._btn_wrong.setStyleSheet(
        "background: #6b1a1a; color: white; font-weight: bold; padding: 4px;"
    )
    self._btn_wrong.clicked.connect(self._on_feedback_wrong)

    feedback_row.addWidget(self._btn_correct)
    feedback_row.addWidget(self._btn_wrong)
    layout.addLayout(feedback_row)

    # ── NEW: session recording status ─────────────────────────────
    self._recording_label = QLabel("● Recording: waiting...")
    self._recording_label.setStyleSheet(
        "font-family: monospace; font-size: 11px; color: #555;"
    )
    layout.addWidget(self._recording_label)

    # ── NEW: annotation counter ───────────────────────────────────
    self._annotation_label = QLabel("Annotations: 0")
    self._annotation_label.setStyleSheet(
        "font-family: monospace; font-size: 11px; color: #888;"
    )
    layout.addWidget(self._annotation_label)

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

    # --- Demo classifier ---

    def _build_demo_classifier_group(self) -> QGroupBox:
        box, layout = self._make_group("Demo Classifier")
        box.setStyleSheet("QGroupBox { color: #5599ff; font-weight: bold; }")

        self._demo_mode_btn = QPushButton("EEG ACTIVE  [m to override]")
        self._demo_mode_btn.setCheckable(True)
        self._demo_mode_btn.setChecked(False)
        self._demo_mode_btn.setStyleSheet(
            "background: #1a472a; color: #00cc66; font-weight: bold; "
            "padding: 6px; border: 1px solid #00cc66;"
        )
        self._demo_mode_btn.clicked.connect(self._toggle_demo_override)
        layout.addWidget(self._demo_mode_btn)

        self._demo_state_label = QLabel("Current: IDLE")
        self._demo_state_label.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: #888888;"
        )
        layout.addWidget(self._demo_state_label)

        btn_row = QHBoxLayout()

        self._btn_demo_move = QPushButton("MOVE")
        self._btn_demo_move.setToolTip(", key")
        self._btn_demo_move.setStyleSheet(
            "background: #00cc66; color: white; font-weight: bold; padding: 6px;"
        )
        self._btn_demo_move.clicked.connect(self._demo_move)
        self._btn_demo_move.setEnabled(False)

        self._btn_demo_idle = QPushButton("IDLE")
        self._btn_demo_idle.setToolTip(". key")
        self._btn_demo_idle.setStyleSheet(
            "background: #555555; color: white; padding: 6px;"
        )
        self._btn_demo_idle.clicked.connect(self._demo_idle)
        self._btn_demo_idle.setEnabled(False)

        self._btn_demo_jaw = QPushButton("JAW")
        self._btn_demo_jaw.setToolTip("/ key")
        self._btn_demo_jaw.setStyleSheet(
            "background: #cc8800; color: white; font-weight: bold; padding: 6px;"
        )
        self._btn_demo_jaw.clicked.connect(self._demo_jaw_clench)
        self._btn_demo_jaw.setEnabled(False)

        btn_row.addWidget(self._btn_demo_move)
        btn_row.addWidget(self._btn_demo_idle)
        btn_row.addWidget(self._btn_demo_jaw)
        layout.addLayout(btn_row)

        hint = QLabel(", move  |  . idle  |  / jaw clench  |  m toggle")
        hint.setStyleSheet("font-size: 10px; color: #666; font-family: monospace;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        return box

    def _toggle_demo_override(self):
        override = self._demo_mode_btn.isChecked()
        self.shared_state.demo_override.value = override
        if override:
            self._demo_mode_btn.setText("SAFETY KEYS ACTIVE  [m for EEG]")
            self._demo_mode_btn.setStyleSheet(
                "background: #472a1a; color: #cc8800; font-weight: bold; "
                "padding: 6px; border: 1px solid #cc8800;"
            )
        else:
            self._demo_mode_btn.setText("EEG ACTIVE  [m to override]")
            self._demo_mode_btn.setStyleSheet(
                "background: #1a472a; color: #00cc66; font-weight: bold; "
                "padding: 6px; border: 1px solid #00cc66;"
            )
        for btn in (self._btn_demo_move, self._btn_demo_idle, self._btn_demo_jaw):
            btn.setEnabled(override)

    def _demo_move(self):
        self.shared_state.prediction.value = 1
        self.shared_state.pred_confidence.value = 1.0
        self._update_demo_state_label(1)

    def _demo_jaw_clench(self):
        self.shared_state.prediction.value = 2
        self.shared_state.pred_confidence.value = 1.0
        self._update_demo_state_label(2)

    def _demo_idle(self):
        self.shared_state.prediction.value = 0
        self.shared_state.pred_confidence.value = 1.0
        self._update_demo_state_label(0)

    def _update_demo_state_label(self, pred: int):
        name, color = MOCK_STATE_STYLES.get(pred, ("?", "#fff"))
        self._demo_state_label.setText(f"Current: {name}")
        self._demo_state_label.setStyleSheet(
            f"font-family: monospace; font-size: 12px; font-weight: bold; color: {color};"
        )

    def keyPressEvent(self, event):
        if self.demo_classifier:
            key = event.key()
            if key == Qt.Key.Key_M:
                self._demo_mode_btn.setChecked(not self._demo_mode_btn.isChecked())
                self._toggle_demo_override()
            elif self.shared_state.demo_override.value:
                if key == Qt.Key.Key_Comma:
                    self._demo_move()
                elif key == Qt.Key.Key_Period:
                    self._demo_idle()
                elif key == Qt.Key.Key_Slash:
                    self._demo_jaw_clench()
        super().keyPressEvent(event)

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

        self._vlm_label = QLabel("VLM: Waiting...")
        self._vlm_label.setWordWrap(True)
        self._vlm_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #aaa;"
        )
        layout.addWidget(self._vlm_label)

        self._angle_dist_label = QLabel("→ --°  |  --mm")
        self._obstacle_label = QLabel("Obstacle: --")

        for lbl in (self._angle_dist_label, self._obstacle_label):
            lbl.setStyleSheet("font-family: monospace; font-size: 12px;")
            layout.addWidget(lbl)

        return box

    # -- LIDAR group --
    def _build_lidar_minimap_widget(self) -> QLabel:
        self._minimap_label = QLabel()
        self._minimap_label.setMinimumSize(160, 160)
        self._minimap_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._minimap_label.setStyleSheet("background: #111; border: 1px solid #333;")
        self._minimap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return self._minimap_label

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------
    def _tick(self):
        self._snapshot_lidar()
        self._update_feed()
        self._update_gaze_readout()
        self._update_worker_status()
        self._update_motor_readout()
        self._update_pathfinding_readout()
        self._update_lidar_minimap()
        if not self.mock_classifier:
            self._update_classifier_readout()
        gaze_x, gaze_y = self.shared_state.get_gaze()
        self._check_turn_zones(gaze_x, gaze_y) #added 2 lines
        self._update_recording_status() 

    def _update_feed(self):
        if self._active_feed == FEED_EYETRACKER:
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

    def _snapshot_lidar(self):
        """Copy the current shared lidar_distances into the rolling history."""
        if not self.shared_state.lidar_running.value:
            return
        self._lidar_history.append(list(self.shared_state.lidar_distances[:]))

    def _update_lidar_minimap(self):
        """Render the latest lidar snapshot onto the Qt minimap label."""
        if not self._lidar_history:
            return
        size = self._minimap_label.size()
        w, h = size.width(), size.height()
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        self._draw_lidar_minimap(canvas, w // 2, h // 2, min(w, h) // 2 - 8)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data, w, h, w * 3, QImage.Format.Format_RGB888
        ).copy()  # use copy to prevent pointer degredation

        self._minimap_label.setPixmap(QPixmap.fromImage(qimg))

    def _draw_lidar_minimap(self, canvas: np.ndarray, cx: int, cy: int, r: int):
        """
        Draw a polar LIDAR map onto canvas (BGR, in-place).
        cx, cy : centre pixel of the minimap circle
        r      : radius in pixels
        """
        if not self._lidar_history:
            return
        latest = self._lidar_history[-1]

        # background disc
        cv2.circle(canvas, (cx, cy), r, (20, 20, 20), -1)
        cv2.circle(canvas, (cx, cy), r, (60, 60, 60), 1)

        # range rings + outermost distance label
        for ring in range(1, MINIMAP_RINGS + 1):
            ring_r = int(r * ring / MINIMAP_RINGS)
            cv2.circle(canvas, (cx, cy), ring_r, (45, 45, 45), 1)
        cv2.putText(
            canvas,
            f"{int(MINIMAP_MAX_DIST / 1000)}m",
            (cx + r - 18, cy - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (70, 70, 70),
            1,
        )

        # front-cone highlight (345-15 deg)
        obstacle = self.shared_state.obstacle_detected.value
        cone_fill = (0, 0, 60) if obstacle else (0, 40, 80)
        cone_border = (0, 0, 200) if obstacle else (0, 100, 200)

        cone_pts = [(cx, cy)]
        for deg in range(345, 376):  # 345-375 wraps to cover 345-15
            angle_rad = math.radians((deg % 360) - 90)
            cone_pts.append(
                (
                    int(cx + r * math.cos(angle_rad)),
                    int(cy + r * math.sin(angle_rad)),
                )
            )
        cv2.fillPoly(canvas, [np.array(cone_pts, dtype=np.int32)], cone_fill)
        cv2.ellipse(canvas, (cx, cy), (r, r), 90, -15, 15, cone_border, 1)

        # scan points — distance-coloured with history gradation
        for i, scan in enumerate(self._lidar_history):
            # Scale intensity from dim (oldest) to bright (newest)
            intensity = (i + 1) / len(self._lidar_history)

            for deg, dist in enumerate(scan):
                if dist <= 0.0:
                    continue
                plot_r = int(r * min(dist, MINIMAP_MAX_DIST) / MINIMAP_MAX_DIST)
                angle_rad = math.radians(deg - 90)  # 0 deg = up (forward)
                px = int(cx + plot_r * math.cos(angle_rad))
                py = int(cy + plot_r * math.sin(angle_rad))

                # Multiply base BGR values by intensity for alpha-like fade
                if dist < MINIMAP_RED_THRESH:
                    color = (0, 0, int(220 * intensity))  # red
                elif dist < MINIMAP_YELLOW_THRESH:
                    color = (0, int(200 * intensity), int(220 * intensity))  # yellow
                else:
                    color = (0, int(200 * intensity), int(80 * intensity))  # green

                cv2.circle(canvas, (px, py), 2, color, -1)

        # vehicle centre dot
        cv2.circle(canvas, (cx, cy), 3, (255, 255, 255), -1)

        # obstacle alert glyph inside cone
        if obstacle:
            cv2.putText(
                canvas,
                "!",
                (cx - 4, cy - r // 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        # closest front reading printed at bottom of circle
        front_idx = list(range(345, 360)) + list(range(0, 16))
        front_dists = [latest[i] for i in front_idx if latest[i] > 0.0]
        if front_dists:
            closest = min(front_dists)
            color_c = (0, 0, 220) if closest < MINIMAP_RED_THRESH else (180, 180, 180)
            cv2.putText(
                canvas,
                f"{int(closest)}mm",
                (cx - 22, cy + r - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color_c,
                1,
            )

    def _render_eyetracker_feed(self):
        if not self.shared_state.eye_frame_ready.is_set():
            return

        with self.shared_state.eye_frame_buffer.get_lock():
            buf = np.frombuffer(
                self.shared_state.eye_frame_buffer.get_obj(), dtype=np.uint8
            )
            frame = buf.reshape(
                (self.shared_state.EYE_FRAME_H, self.shared_state.EYE_FRAME_W, 3)
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
        if not self.shared_state.path_frame_ready.is_set():
            return

        with self.shared_state.path_frame_buffer.get_lock():
            buf = np.frombuffer(
                self.shared_state.path_frame_buffer.get_obj(), dtype=np.uint8
            )
            frame = buf.reshape(
                (self.shared_state.PATH_FRAME_H, self.shared_state.PATH_FRAME_W, 3)
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
            "pathcam": self.shared_state.pathcam_running.value,
            "lidar": self.shared_state.lidar_running.value,
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

        verdict = self.shared_state.vlm_last_verdict.value.decode("utf-8")
        if verdict:
            self._vlm_label.setText(f"VLM: {verdict}")
            if "OBSTACLE" in verdict.upper():
                self._vlm_label.setStyleSheet("color: #ff4444; font-weight: bold;")
            else:
                self._vlm_label.setStyleSheet("color: #00cc66;")

        if self.shared_state.vlm_is_busy.value:
            self._vlm_label.setText("VLM: Analyzing path...")

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
        self.shared_state.eye_camera_index.value = value

    def _on_pathfinding_cam_changed(self, value: int):
        self.shared_state.pathcam_index.value = value

    def _on_smoothing_changed(self, value: int):
        factor = value / 100.0
        self.shared_state.smoothing_factor.value = factor
        self._smooth_label.setText(f"{factor:.2f}")

    def _send_motor_command(self, cmd_id: int):
        self.shared_state.motor_command.value = cmd_id

    # Add these two callback methods anywhere in the class callbacks section

    def _on_feedback_correct(self):
        """Mark the current prediction as correct in the recording."""
        if hasattr(self.shared_state, 'feedback_correct'):
            self.shared_state.feedback_correct.set()
        # Flash green briefly so user knows it registered
        self._btn_correct.setStyleSheet(
            "background: #00ff66; color: black; font-weight: bold; padding: 4px;"
        )
        QTimer.singleShot(300, lambda: self._btn_correct.setStyleSheet(
            "background: #1a6b3a; color: white; font-weight: bold; padding: 4px;"
        ))

    def _on_feedback_wrong(self):
        """Mark the current prediction as wrong in the recording."""
        if hasattr(self.shared_state, 'feedback_wrong'):
            self.shared_state.feedback_wrong.set()
        self._btn_wrong.setStyleSheet(
            "background: #ff4444; color: white; font-weight: bold; padding: 4px;"
        )
        QTimer.singleShot(300, lambda: self._btn_wrong.setStyleSheet(
            "background: #6b1a1a; color: white; font-weight: bold; padding: 4px;"
        ))


    


    # Add this new method:
    def _update_recording_status(self):
        """Polls the recording state from shared_state and updates the label."""
        if not hasattr(self.shared_state, 'recording_active'):
            return

        active     = self.shared_state.recording_active.value
        n_annots   = getattr(self.shared_state, 'annotation_count', None)
        count      = n_annots.value if n_annots is not None else 0

        if active:
            self._recording_label.setText("● Recording: active")
            self._recording_label.setStyleSheet(
                "font-family: monospace; font-size: 11px; color: #00cc66;"
            )
        else:
            self._recording_label.setText("● Recording: stopped")
            self._recording_label.setStyleSheet(
                "font-family: monospace; font-size: 11px; color: #cc3333;"
            )

        self._annotation_label.setText(f"Annotations: {count}")     

    

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._timer.stop()
        self.shared_state.shutdown.set()
        event.accept()
