"""
Camera-nav debug GUI.

Shows the USB camera feed. Click or hover to select a pixel; the same
undistort → pixel_to_point pipeline used by pathfinding_worker runs in
real time and the computed angle / distance are shown alongside controls
to tweak camera height and mount angle.

Run from repo root:
    python pathfinding/camera_nav/nav_debug_gui.py
"""

import sys
import os
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QDoubleSpinBox, QGroupBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pathfinding.camera_nav.screen_to_location import CameraConfig, pixel_to_point

# ── Calibration paths ──────────────────────────────────────────────────────────

CALIB_DIR = os.path.join(os.path.dirname(__file__), "..", "calibration", "matrices")

def load_calibration():
    K     = np.load(os.path.join(CALIB_DIR, "intrinsicNew_usb.npy"))
    dist  = np.load(os.path.join(CALIB_DIR, "dist_usb.npy"))
    K_raw = np.load(os.path.join(CALIB_DIR, "intrinsic_usb.npy"))
    return K, K_raw, dist


# ── Camera feed widget ─────────────────────────────────────────────────────────

class CameraFeed(QLabel):
    """Displays the camera frame and emits the pixel under the mouse."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #111;")
        self.setMouseTracking(True)

        self._frame_w = 640
        self._frame_h = 480
        self._cursor_px = None   # (x, y) in camera pixels, or None
        self._pixmap = None

    def set_frame(self, bgr_frame: np.ndarray):
        """Update the displayed frame (BGR numpy array)."""
        self._frame_h, self._frame_w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(img)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap is None or self._cursor_px is None:
            return

        # figure out where the pixmap is drawn inside the label
        lw, lh = self.width(), self.height()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scale  = min(lw / pw, lh / ph)
        draw_w = int(pw * scale)
        draw_h = int(ph * scale)
        off_x  = (lw - draw_w) // 2
        off_y  = (lh - draw_h) // 2

        cx, cy = self._cursor_px
        sx = int(off_x + cx * scale)
        sy = int(off_y + cy * scale)

        p = QPainter(self)
        pen = QPen(QColor(0, 255, 0), 2)
        p.setPen(pen)
        arm = 14
        p.drawLine(sx - arm, sy, sx + arm, sy)
        p.drawLine(sx, sy - arm, sx, sy + arm)
        p.drawEllipse(sx - 5, sy - 5, 10, 10)
        p.end()

    def _label_to_camera(self, lx, ly):
        """Convert label-space coordinates to camera pixel coordinates."""
        if self._pixmap is None:
            return None
        lw, lh = self.width(), self.height()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scale  = min(lw / pw, lh / ph)
        draw_w = int(pw * scale)
        draw_h = int(ph * scale)
        off_x  = (lw - draw_w) // 2
        off_y  = (lh - draw_h) // 2
        cx = (lx - off_x) / scale
        cy = (ly - off_y) / scale
        if 0 <= cx < pw and 0 <= cy < ph:
            return int(cx), int(cy)
        return None

    def mouseMoveEvent(self, event):
        pt = self._label_to_camera(event.position().x(), event.position().y())
        self._cursor_px = pt
        self.update()
        if pt is not None:
            self.window().on_pixel_selected(*pt)

    def mousePressEvent(self, event):
        pt = self._label_to_camera(event.position().x(), event.position().y())
        self._cursor_px = pt
        self.update()
        if pt is not None:
            self.window().on_pixel_selected(*pt)

    def leaveEvent(self, event):
        self._cursor_px = None
        self.update()


# ── Main window ────────────────────────────────────────────────────────────────

class NavDebugWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera-Nav Debug")

        self.K, self.K_raw, self.dist = load_calibration()
        self.cam_cfg = CameraConfig(height=110, angle=-10)

        self.cap = cv2.VideoCapture(1)   # USB camera
        if not self.cap.isOpened():
            print("Warning: could not open camera index 1, trying 0")
            self.cap = cv2.VideoCapture(0)

        self._build_ui()

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)   # ~30 fps

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # left: camera feed
        self.feed = CameraFeed()
        layout.addWidget(self.feed, stretch=3)

        # right: controls + output
        sidebar = QWidget()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet("background: #1e1e1e; color: #ddd;")
        vbox = QVBoxLayout(sidebar)
        vbox.setContentsMargins(12, 12, 12, 12)
        vbox.setSpacing(16)
        layout.addWidget(sidebar)

        # ── Camera config ──────────────────────────────────────────────────────
        cfg_group = QGroupBox("Camera mount")
        cfg_group.setStyleSheet(self._group_style())
        cfg_layout = QGridLayout(cfg_group)
        cfg_layout.setSpacing(6)

        cfg_layout.addWidget(self._label("Height (mm)"), 0, 0)
        self.height_spin = self._spin(1, 5000, 110, 1)
        self.height_spin.valueChanged.connect(self._on_cfg_changed)
        cfg_layout.addWidget(self.height_spin, 0, 1)

        cfg_layout.addWidget(self._label("Angle (°)"), 1, 0)
        self.angle_spin = self._spin(-90, 90, -10, 0.5)
        self.angle_spin.valueChanged.connect(self._on_cfg_changed)
        cfg_layout.addWidget(self.angle_spin, 1, 1)

        vbox.addWidget(cfg_group)

        # ── Output ─────────────────────────────────────────────────────────────
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

        # hint
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

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_cfg_changed(self):
        self.cam_cfg = CameraConfig(
            height=self.height_spin.value(),
            angle=self.angle_spin.value(),
        )

    def on_pixel_selected(self, px: int, py: int):
        """Called whenever the mouse is over a valid camera pixel."""
        self.lbl_pixel.setText(f"({px}, {py})")

        # same undistort step as pathfinding_worker.py
        pt = np.array([[[px, py]]], dtype=np.float32)
        undist = cv2.undistortPoints(pt, self.K_raw, self.dist, P=self.K)
        ux = int(undist[0, 0, 0])
        uy = int(undist[0, 0, 1])
        self.lbl_undist.setText(f"({ux}, {uy})")

        h_angle, dist = pixel_to_point(ux, uy, self.K, self.cam_cfg)
        self.lbl_angle.setText(f"{h_angle:+.2f}°")
        if dist == float("inf"):
            self.lbl_dist.setText("∞  (above horizon)")
            self.lbl_dist.setStyleSheet("color: #f80; font-family: monospace; font-size: 13px;")
        else:
            self.lbl_dist.setText(f"{dist:.1f} mm")
            self.lbl_dist.setStyleSheet("color: #0f0; font-family: monospace; font-size: 13px;")

    def _tick(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        # undistort the display frame so the visual matches the math
        frame = cv2.undistort(frame, self.K_raw, self.dist, None, self.K)
        self.feed.set_frame(frame)

    def closeEvent(self, event):
        self._timer.stop()
        self.cap.release()
        super().closeEvent(event)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = NavDebugWindow()
    win.resize(960, 560)
    win.show()
    sys.exit(app.exec())
