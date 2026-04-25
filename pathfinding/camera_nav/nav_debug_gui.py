"""
Camera-Nav Debug GUI (Fully Integrated)
"""

import sys
import os
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QDoubleSpinBox,
    QGroupBox,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

# Ensure the parent directory is in the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pathfinding.camera_nav.screen_to_location import CameraConfig, pixel_to_point

# ── Calibration ────────────────────────────────────────────────────────────────

CALIB_DIR = os.path.join(os.path.dirname(__file__), "..", "calibration", "matrices")


def load_calibration():
    # Keep the distinction between the raw source matrix and the optimal new matrix
    K_new = np.load(os.path.join(CALIB_DIR, "intrinsicNew_usb.npy"))
    K_raw = np.load(os.path.join(CALIB_DIR, "intrinsic_usb.npy"))
    dist = np.load(os.path.join(CALIB_DIR, "dist_usb.npy"))
    return K_new, K_raw, dist


# ── Interactive Video Label ────────────────────────────────────────────────────


class InteractiveVideoLabel(QLabel):
    """Handles video display and calculates true frame coordinates from mouse position."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #111;")
        self.setMouseTracking(True)

        self.frame_w = 640
        self.frame_h = 480
        self.cursor_px = None  # (x, y) relative to true frame resolution

    def mouseMoveEvent(self, event):
        self._update_cursor(event.position().x(), event.position().y())

    def mousePressEvent(self, event):
        self._update_cursor(event.position().x(), event.position().y())

    def leaveEvent(self, event):
        self.cursor_px = None
        if self.window() and hasattr(self.window(), "on_pixel_selected"):
            self.window().on_pixel_selected(None, None)

    def _update_cursor(self, lx, ly):
        if self.pixmap() is None:
            return

        # Label dimensions vs Scaled Pixmap dimensions
        lw, lh = self.width(), self.height()
        pw, ph = self.pixmap().width(), self.pixmap().height()

        # Calculate offsets caused by Qt.KeepAspectRatio centering
        off_x = (lw - pw) / 2.0
        off_y = (lh - ph) / 2.0

        # Mouse position relative to the actual image pixels
        px_pixmap = lx - off_x
        py_pixmap = ly - off_y

        # If hovering inside the image bounds, scale back to original camera resolution
        if 0 <= px_pixmap <= pw and 0 <= py_pixmap <= ph:
            true_x = int((px_pixmap / pw) * self.frame_w)
            true_y = int((py_pixmap / ph) * self.frame_h)
            self.cursor_px = (true_x, true_y)
            if self.window() and hasattr(self.window(), "on_pixel_selected"):
                self.window().on_pixel_selected(true_x, true_y)
        else:
            self.cursor_px = None
            if self.window() and hasattr(self.window(), "on_pixel_selected"):
                self.window().on_pixel_selected(None, None)


# ── Main window ────────────────────────────────────────────────────────────────


class NavDebugWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera-Nav Debug")

        # Load math components
        self.K_new, self.K_raw, self.dist = load_calibration()
        self.cam_cfg = CameraConfig(height=110.0, angle=-10.0)

        # Initialize hardware
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("FATAL ERROR: Could not open /dev/video64.")
            sys.exit(1)

        self._build_ui()

        # Start render loop
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(33)  # ~30 fps

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # left: camera feed
        self.feed_label = InteractiveVideoLabel()
        layout.addWidget(self.feed_label, stretch=3)

        # right: controls + output
        sidebar = QWidget()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet("background: #1e1e1e; color: #ddd;")
        vbox = QVBoxLayout(sidebar)
        vbox.setContentsMargins(12, 12, 12, 12)
        vbox.setSpacing(16)
        layout.addWidget(sidebar)

        # ── Camera config ──
        cfg_group = QGroupBox("Camera mount")
        cfg_group.setStyleSheet(self._group_style())
        cfg_layout = QGridLayout(cfg_group)
        cfg_layout.setSpacing(6)

        cfg_layout.addWidget(self._label("Height (mm)"), 0, 0)
        self.height_spin = self._spin(1, 5000, self.cam_cfg.height, 1)
        self.height_spin.valueChanged.connect(self._on_cfg_changed)
        cfg_layout.addWidget(self.height_spin, 0, 1)

        cfg_layout.addWidget(self._label("Angle (°)"), 1, 0)
        self.angle_spin = self._spin(-90, 90, self.cam_cfg.angle, 0.5)
        self.angle_spin.valueChanged.connect(self._on_cfg_changed)
        cfg_layout.addWidget(self.angle_spin, 1, 1)

        vbox.addWidget(cfg_group)

        # ── Output ──
        out_group = QGroupBox("Output")
        out_group.setStyleSheet(self._group_style())
        out_layout = QGridLayout(out_group)
        out_layout.setSpacing(6)

        out_layout.addWidget(self._label("Pixel"), 0, 0)
        self.lbl_pixel = self._value_label("--")
        out_layout.addWidget(self.lbl_pixel, 0, 1)

        out_layout.addWidget(self._label("Undistorted"), 1, 0)
        self.lbl_undist = self._value_label("--")
        out_layout.addWidget(self.lbl_undist, 1, 1)

        out_layout.addWidget(self._label("H angle"), 2, 0)
        self.lbl_angle = self._value_label("--")
        out_layout.addWidget(self.lbl_angle, 2, 1)

        out_layout.addWidget(self._label("Distance"), 3, 0)
        self.lbl_dist = self._value_label("--")
        out_layout.addWidget(self.lbl_dist, 3, 1)

        vbox.addWidget(out_group)
        vbox.addStretch()

        hint = QLabel("Move mouse over feed\nto compute location")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(hint)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        return lbl

    def _value_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #0f0; font-family: monospace; font-size: 13px;")
        return lbl

    def _spin(self, lo, hi, val, step):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        s.setDecimals(1)
        s.setStyleSheet(
            "background: #2a2a2a; color: #eee; border: 1px solid #444;"
            "padding: 2px 4px; font-size: 12px;"
        )
        return s

    def _group_style(self):
        return (
            "QGroupBox { color: #aaa; font-size: 12px; border: 1px solid #444;"
            "border-radius: 4px; margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )

    # ── Logic & Slots ──────────────────────────────────────────────────────────

    def _on_cfg_changed(self):
        self.cam_cfg = CameraConfig(
            height=self.height_spin.value(),
            angle=self.angle_spin.value(),
        )
        # Re-trigger math on current cursor position
        if self.feed_label.cursor_px:
            self.on_pixel_selected(*self.feed_label.cursor_px)

    def on_pixel_selected(self, px, py):
        """Executes your pathfinding math engine dynamically."""
        if px is None or py is None:
            self.lbl_pixel.setText("--")
            self.lbl_undist.setText("--")
            self.lbl_angle.setText("--")
            self.lbl_dist.setText("--")
            self.lbl_dist.setStyleSheet(
                "color: #0f0; font-family: monospace; font-size: 13px;"
            )
            return

        self.lbl_pixel.setText(f"({px}, {py})")

        # 1. Optical correction mapping
        pt = np.array([[[px, py]]], dtype=np.float32)
        undist = cv2.undistortPoints(pt, self.K_raw, self.dist, P=self.K_new)
        ux = int(undist[0, 0, 0])
        uy = int(undist[0, 0, 1])
        self.lbl_undist.setText(f"({ux}, {uy})")

        # 2. Physical geometry mapping
        h_angle, dist = pixel_to_point(ux, uy, self.K_new, self.cam_cfg)
        self.lbl_angle.setText(f"{h_angle:+.2f}°")

        if dist == float("inf"):
            self.lbl_dist.setText("∞  (horizon)")
            self.lbl_dist.setStyleSheet(
                "color: #f80; font-family: monospace; font-size: 13px;"
            )
        else:
            self.lbl_dist.setText(f"{dist:.1f} mm")
            self.lbl_dist.setStyleSheet(
                "color: #0f0; font-family: monospace; font-size: 13px;"
            )

    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self.feed_label.frame_h, self.feed_label.frame_w = frame.shape[:2]

        # Draw crosshair directly on the raw cv2 matrix
        if self.feed_label.cursor_px:
            cx, cy = self.feed_label.cursor_px
            arm = 14
            cv2.line(frame, (cx - arm, cy), (cx + arm, cy), (0, 255, 0), 2)
            cv2.line(frame, (cx, cy - arm), (cx, cy + arm), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), 2)

        # Convert and render with strict memory retention (.copy)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        q_img = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        ).copy()

        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.feed_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.feed_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        event.accept()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = NavDebugWindow()
    win.resize(960, 560)
    win.show()
    sys.exit(app.exec())
