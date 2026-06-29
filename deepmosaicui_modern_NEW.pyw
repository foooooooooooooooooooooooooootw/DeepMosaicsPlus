"""
DeepMosaicsPlus — Modern 3-panel workstation UI
Layout: Left sidebar (config) · Centre (frame viewport) · Bottom (run bar + log)

Run with:  python deepmosaicui_qt2.py
"""

import os, sys, glob, re
from pathlib import Path

from PyQt6.QtCore import (
    Qt, QTimer, QProcess, QProcessEnvironment,
    QThread, pyqtSignal, QSize, QRect, QPoint,
)
from PyQt6.QtGui import (
    QColor, QFont, QFontDatabase, QPainter, QPen, QBrush,
    QPixmap, QImage, QIcon, QPalette, QDragEnterEvent, QDropEvent,
    QLinearGradient,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout, QStackedLayout,
    QLabel, QLineEdit, QCheckBox, QComboBox, QSlider,
    QPushButton, QPlainTextEdit, QScrollArea, QFrame,
    QGroupBox, QSizePolicy, QFileDialog,
    QButtonGroup, QAbstractButton, QMessageBox, QDialog,
    QDialogButtonBox, QSizeGrip, QSpinBox,
)

# ── Palette tokens ─────────────────────────────────────────────────────────────
C = {
    "bg":          "#0a0c0e",
    "surface":     "#111417",
    "surface2":    "#161b21",
    "border":      "#1e2530",
    "border2":     "#252d38",
    "accent":      "#2d6fd4",
    "accent_dim":  "#1a3a6a",
    "text":        "#c8d0db",
    "text_dim":    "#6b7a8d",
    "text_bright": "#e8edf4",
    "green":       "#4caf76",
    "green_dim":   "#1a3a26",
    "green_bd":    "#2a5a36",
    "amber":       "#c8943a",
    "red":         "#c85050",
    "red_dim":     "#3a1a1a",
    "red_bd":      "#5a2a2a",
    "purple":      "#9a70e0",
    "purple_dim":  "#1e1230",
    "purple_bd":   "#3a2260",
    "mono":        "#3ccc6e",   # monospace green
    "mask_tint":   "#ff3366",   # overlay tint for detected mosaic regions
}

QSS = f"""
* {{ box-sizing: border-box; }}

QMainWindow, QWidget {{
    background: {C['bg']};
    color: {C['text']};
    font-family: "Inter", "Segoe UI", "SF Pro Text", system-ui, sans-serif;
    font-size: 13px;
    border: none;
    outline: none;
}}

/* ── Sidebar scroll ── */
QScrollArea {{ background: {C['surface']}; border: none; }}
QScrollBar:vertical {{
    background: transparent; width: 4px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C['border2']}; border-radius: 2px; min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

/* ── Section headers ── */
QLabel#section_head {{
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: {C['text_dim']};
    padding: 14px 0 6px 0;
    border-bottom: 1px solid {C['border']};
    margin-bottom: 8px;
}}

/* ── Row labels ── */
QLabel#row_label {{
    font-size: 12px; color: {C['text_dim']};
    min-width: 108px; max-width: 108px;
    padding-top: 7px;
}}

/* ── Inputs ── */
QLineEdit, QComboBox {{
    background: {C['bg']};
    border: 1px solid {C['border2']};
    border-radius: 5px;
    padding: 5px 9px;
    color: {C['text']};
    font-size: 12px;
    selection-background-color: {C['accent']};
}}
QLineEdit:focus {{ border-color: {C['accent']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox::down-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 4px solid {C['text_dim']};
    width: 0; height: 0; margin-right: 7px;
}}
QComboBox QAbstractItemView {{
    background: {C['surface2']}; border: 1px solid {C['border2']};
    color: {C['text']}; selection-background-color: {C['accent']};
    border-radius: 5px; padding: 2px;
}}

/* ── Checkboxes ── */
QCheckBox {{
    color: {C['text']}; spacing: 7px; font-size: 12px;
}}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {C['border2']}; border-radius: 3px;
    background: {C['bg']};
}}
QCheckBox::indicator:checked {{
    background: {C['accent']}; border-color: {C['accent']};
    image: none;
}}

/* ── Sliders ── */
QSlider::groove:horizontal {{
    height: 3px; background: {C['border2']}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C['accent']};
    width: 13px; height: 13px; margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {C['accent']}; border-radius: 2px;
}}

/* ── Bottom bar ── */
QWidget#bottom_bar {{
    background: {C['surface']};
    border-top: 1px solid {C['border']};
}}

/* ── Log ── */
QPlainTextEdit#log {{
    background: {C['bg']};
    color: #7a9ab0;
    border: none;
    font-size: 11px;
    padding: 6px 10px;
}}

/* ── Cmd preview ── */
QLabel#cmd_preview {{
    background: {C['bg']};
    color: {C['mono']};
    border-top: 1px solid {C['border']};
    font-size: 11px;
    font-family: "JetBrains Mono", "Consolas", monospace;
    padding: 5px 12px;
}}

/* ── Buttons ── */
QPushButton {{
    border-radius: 5px; padding: 6px 14px;
    font-size: 12px; font-weight: 500;
    border: 1px solid {C['border2']};
    background: {C['surface2']};
    color: {C['text']};
}}
QPushButton:hover {{ background: {C['surface']}; border-color: {C['accent_dim']}; }}

QPushButton#btn_run {{
    background: {C['green_dim']}; color: {C['green']};
    border-color: {C['green_bd']};
    padding: 8px 28px; font-size: 13px; font-weight: 600;
}}
QPushButton#btn_run:hover {{ background: #1f4a2e; }}
QPushButton#btn_run:disabled {{
    background: #111a13; color: #2a4a30; border-color: #1a2a1f;
}}
QPushButton#btn_cancel {{
    background: {C['red_dim']}; color: {C['red']};
    border-color: {C['red_bd']};
}}
QPushButton#btn_cancel:hover {{ background: #4a1f1f; }}
QPushButton#btn_browse {{
    padding: 4px 10px; font-size: 11px;
    background: {C['surface2']}; color: {C['text_dim']};
    border-color: {C['border']};
    min-width: 56px; max-width: 56px;
}}
QPushButton#btn_browse:hover {{ color: {C['text']}; }}

/* ── Hint labels ── */
QLabel#hint {{
    color: {C['text_dim']}; font-size: 11px;
    padding: 1px 0 6px 0;
    line-height: 1.5;
}}

/* ── Value labels ── */
QLabel#val_lbl {{
    color: {C['text_dim']}; font-size: 11px;
    font-family: "JetBrains Mono", monospace;
    min-width: 34px; max-width: 34px;
    qproperty-alignment: AlignRight;
}}

/* ── Viewport area ── */
QWidget#viewport_bg {{
    background: #050607;
    border-right: 1px solid {C['border']};
}}

/* ── Sidebar ── */
QWidget#sidebar {{
    background: {C['surface']};
    border-right: 1px solid {C['border']};
}}

/* ── Overlay toggle buttons ── */
QPushButton#overlay_btn {{
    padding: 4px 10px; font-size: 11px;
    background: {C['surface2']}; color: {C['text_dim']};
    border: 1px solid {C['border2']};
    border-radius: 4px;
}}
QPushButton#overlay_btn:checked {{
    background: {C['accent_dim']}; color: {C['text_bright']};
    border-color: {C['accent']};
}}

/* ── Progress ── */
QProgressBar {{
    background: {C['border']}; border: none;
    border-radius: 2px; height: 3px; text-align: center;
}}
QProgressBar::chunk {{
    background: {C['accent']}; border-radius: 2px;
}}

/* ── Exp badge ── */
QLabel#exp_badge {{
    color: {C['purple']}; background: {C['purple_dim']};
    border: 1px solid {C['purple_bd']};
    border-radius: 3px; padding: 1px 6px;
    font-size: 10px; font-weight: 700;
    letter-spacing: 0.06em;
}}
"""


# ── Custom widgets ─────────────────────────────────────────────────────────────

class FrameViewport(QWidget):
    """Centre panel: shows a video frame with optional mask overlay."""
    file_dropped = pyqtSignal(str)
    clicked      = pyqtSignal()   # emitted when empty zone is clicked

    MODES = ["frame", "mask", "overlay"]

    def __init__(self):
        super().__init__()
        self.setObjectName("viewport_bg")
        self.setAcceptDrops(True)
        self._frame_pix  = None
        self._mask_pix   = None
        self._tint_pix   = None   # pre-built tinted QPixmap at mask resolution
        self._mode       = "overlay"
        self._opacity    = 0.55
        self._empty      = True

    # ── Public API ──────────────────────────────────────────────────────────
    def set_frame(self, path: str):
        p = QPixmap(path)
        if not p.isNull():
            self._frame_pix = p
            self._empty = False
            self.update()

    def set_mask(self, path: str):
        p = QPixmap(path)
        if not p.isNull():
            self._mask_pix = p
            self._rebuild_tint()
            self.update()

    def _rebuild_tint(self):
        """Build a tinted QPixmap from the current mask and opacity."""
        if self._mask_pix is None:
            self._tint_pix = None
            return
        try:
            import numpy as np
            import cv2 as _cv2

            # Work at mask native resolution — paintEvent scales to dst
            img = self._mask_pix.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
            ptr = img.bits()
            ptr.setsize(img.sizeInBytes())
            gray = np.frombuffer(ptr, dtype=np.uint8).reshape(img.height(), img.width()).copy()

            tc = QColor(C["mask_tint"])
            bgra = np.zeros((img.height(), img.width(), 4), dtype=np.uint8)
            bgra[..., 0] = tc.blue()
            bgra[..., 1] = tc.green()
            bgra[..., 2] = tc.red()
            bgra[..., 3] = (gray.astype(np.float32) * self._opacity).clip(0, 255).astype(np.uint8)

            ok, buf = _cv2.imencode('.png', bgra)
            if ok:
                qi = QImage()
                qi.loadFromData(bytes(buf))
                self._tint_pix = QPixmap.fromImage(qi)
            else:
                self._tint_pix = None
        except Exception:
            self._tint_pix = None

    def clear(self):
        self._frame_pix = None
        self._mask_pix  = None
        self._tint_pix  = None
        self._empty     = True
        self.update()

    def set_mode(self, mode: str):
        self._mode = mode
        self.update()

    def set_opacity(self, v: float):
        self._opacity = v
        self._rebuild_tint()   # rebuild with new opacity
        self.update()

    # ── Drag & drop ────────────────────────────────────────────────────────
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        urls = e.mimeData().urls()
        if urls:
            self.file_dropped.emit(urls[0].toLocalFile())

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._empty:
            self.clicked.emit()

    # ── Paint ──────────────────────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self.rect()

        # Background
        p.fillRect(rect, QColor(C["bg"]))

        if self._empty or self._frame_pix is None:
            self._draw_empty(p, rect)
            return

        # Compute centred, aspect-fitted destination rect
        def fit(pix: QPixmap) -> QRect:
            sw, sh = pix.width(), pix.height()
            ww, wh = rect.width(), rect.height()
            scale = min(ww / sw, wh / sh)
            dw, dh = int(sw * scale), int(sh * scale)
            dx, dy = (ww - dw) // 2, (wh - dh) // 2
            return QRect(dx, dy, dw, dh)

        dst = fit(self._frame_pix)

        if self._mode == "frame":
            p.drawPixmap(dst, self._frame_pix)

        elif self._mode == "mask":
            if self._mask_pix:
                p.drawPixmap(dst, self._mask_pix)
            else:
                p.drawPixmap(dst, self._frame_pix)

        elif self._mode == "overlay":
            p.drawPixmap(dst, self._frame_pix)
            if self._mask_pix and self._tint_pix is not None:
                p.drawPixmap(dst, self._tint_pix)

        p.end()

    def _draw_empty(self, p: QPainter, rect: QRect):
        cx, cy = rect.center().x(), rect.center().y()

        # Dashed drop-zone box
        pen = QPen(QColor(C["border2"]))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(1)
        p.setPen(pen)
        margin = 60
        p.drawRect(margin, margin, rect.width()-margin*2, rect.height()-margin*2)

        # Icon-ish chevron
        p.setPen(QPen(QColor(C["border2"]), 2))
        for i, dy in enumerate([-14, 0, 14]):
            x0, x1 = cx - 22, cx
            x2 = cx + 22
            y  = cy - 30 + dy
            p.drawLine(x0, y, x1, y + 10)
            p.drawLine(x1, y + 10, x2, y)

        p.setPen(QColor(C["text_dim"]))
        f = p.font()
        f.setPixelSize(13)
        p.setFont(f)
        p.drawText(QRect(0, cy + 20, rect.width(), 22),
                   Qt.AlignmentFlag.AlignHCenter, "Click or drop a video / image here")
        f.setPixelSize(11)
        p.setFont(f)
        p.setPen(QColor(C["border2"]))
        p.drawText(QRect(0, cy + 42, rect.width(), 20),
                   Qt.AlignmentFlag.AlignHCenter, "or use the Browse button in the sidebar")


class OverlayBar(QWidget):
    """Toolbar above viewport: source select + view mode + tint opacity."""
    mode_changed    = pyqtSignal(str)   # "frame" | "mask" | "overlay"
    source_changed  = pyqtSignal(str)   # "detection" | "cleaned"
    opacity_changed = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(6)

        # Source toggle
        src_lbl = QLabel("Source:")
        src_lbl.setStyleSheet(f"color: {C['text_dim']}; font-size: 11px;")
        h.addWidget(src_lbl)

        self._src_btns = {}
        for label, src in [("Detection", "detection"), ("Cleaned", "cleaned")]:
            b = QPushButton(label)
            b.setObjectName("overlay_btn")
            b.setCheckable(True)
            b.clicked.connect(lambda _, s=src: self._select_src(s))
            self._src_btns[src] = b
            h.addWidget(b)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {C['border2']};")
        sep.setFixedWidth(1)
        h.addSpacing(6)
        h.addWidget(sep)
        h.addSpacing(6)

        # View mode (only relevant for detection source)
        view_lbl = QLabel("View:")
        view_lbl.setStyleSheet(f"color: {C['text_dim']}; font-size: 11px;")
        h.addWidget(view_lbl)

        self._mode_btns = {}
        for label, mode in [("Frame", "frame"), ("Mask", "mask"), ("Overlay", "overlay")]:
            b = QPushButton(label)
            b.setObjectName("overlay_btn")
            b.setCheckable(True)
            b.clicked.connect(lambda _, m=mode: self._select_mode(m))
            self._mode_btns[mode] = b
            h.addWidget(b)

        h.addSpacing(10)
        tint_lbl = QLabel("Tint")
        tint_lbl.setStyleSheet(f"color: {C['text_dim']}; font-size: 11px;")
        h.addWidget(tint_lbl)

        self._opacity_sl = QSlider(Qt.Orientation.Horizontal)
        self._opacity_sl.setRange(10, 80)
        self._opacity_sl.setValue(55)
        self._opacity_sl.setFixedWidth(80)
        self._opacity_sl.valueChanged.connect(lambda v: self.opacity_changed.emit(v / 100))
        h.addWidget(self._opacity_sl)

        h.addStretch()
        self._select_src("detection")
        self._select_mode("overlay")

    def _select_src(self, src: str):
        for s, b in self._src_btns.items():
            b.setChecked(s == src)
        self.source_changed.emit(src)

    def _select_mode(self, mode: str):
        for m, b in self._mode_btns.items():
            b.setChecked(m == mode)
        self.mode_changed.emit(mode)


# ── Sidebar helpers ────────────────────────────────────────────────────────────
def _section(title: str) -> QLabel:
    l = QLabel(title)
    l.setObjectName("section_head")
    return l

def _rl(text: str) -> QLabel:
    l = QLabel(text)
    l.setObjectName("row_label")
    return l

def _hint(text: str) -> QLabel:
    l = QLabel(text)
    l.setObjectName("hint")
    l.setWordWrap(True)
    return l

def _inp(placeholder="", value="", w=None) -> QLineEdit:
    e = QLineEdit()
    e.setPlaceholderText(placeholder)
    e.setText(value)
    if w: e.setFixedWidth(w)
    return e

def _browse() -> QPushButton:
    b = QPushButton("Browse")
    b.setObjectName("btn_browse")
    return b

def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {C['border']}; background: {C['border']}; max-height: 1px;")
    return f

def _row(*widgets, spacing=8) -> QHBoxLayout:
    h = QHBoxLayout()
    h.setSpacing(spacing)
    h.setContentsMargins(0, 0, 0, 0)
    for w in widgets:
        if w is None: h.addStretch()
        elif isinstance(w, int): h.addSpacing(w)
        else: h.addWidget(w)
    return h

def _sl_row(layout, label, hint_txt, default, lo, hi, steps):
    cb = QCheckBox(label)
    layout.addWidget(cb)
    layout.addWidget(_hint(hint_txt))

    sl = QSlider(Qt.Orientation.Horizontal)
    step_int = max(1, int((hi - lo) / steps * 100))
    sl.setRange(int(lo * 100), int(hi * 100))
    sl.setSingleStep(step_int)
    sl.setValue(int(default * 100))

    val = QLabel(f"{default:.2f}")
    val.setObjectName("val_lbl")
    sl.valueChanged.connect(lambda v, l=val: l.setText(f"{v/100:.2f}"))

    r = QHBoxLayout()
    r.setContentsMargins(4, 0, 0, 8)
    r.setSpacing(8)
    amt = QLabel("Amount")
    amt.setStyleSheet(f"color: {C['text_dim']}; font-size: 11px;")
    r.addWidget(amt)
    r.addWidget(sl)
    r.addWidget(val)
    layout.addLayout(r)
    return cb, sl, val


# ── Main Window ────────────────────────────────────────────────────────────────

class TricolorScrubber(QWidget):
    """Seek bar with three colour zones: extracted / detected / cleaned."""
    valueChanged   = pyqtSignal(int)
    sliderPressed  = pyqtSignal()
    sliderReleased = pyqtSignal()

    C_EXTRACTED = "#2d5080"
    C_DETECTED  = "#6a3d8f"
    C_CLEANED   = "#1e6b3a"
    C_TRACK     = "#1a2030"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._value       = 0
        self._maximum     = 0
        self._n_extracted = 0
        self._n_detected  = 0
        self._n_cleaned   = 0
        self._pressed     = False

    def setRange(self, lo: int, hi: int):
        self._maximum = max(0, hi)
        self._value   = min(self._value, self._maximum)
        self.update()

    def setValue(self, v: int):
        v = max(0, min(v, self._maximum))
        if v != self._value:
            self._value = v
            self.valueChanged.emit(v)
        self.update()

    def value(self) -> int:
        return self._value

    def setProgress(self, n_extracted: int, n_detected: int, n_cleaned: int):
        self._n_extracted = n_extracted
        self._n_detected  = n_detected
        self._n_cleaned   = n_cleaned
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h    = self.width(), self.height()
        pw      = 12   # playhead width
        margin  = pw // 2 + 1   # keep playhead fully inside widget
        track_w = w - margin * 2
        track_h = 5
        tx      = margin
        ty      = (h - track_h) // 2

        # Track background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(self.C_TRACK)))
        p.drawRoundedRect(tx, ty, track_w, track_h, 3, 3)

        if self._maximum > 0:
            def px(n): return int(track_w * min(n, self._maximum) / self._maximum)
            if self._n_extracted > 0:
                p.setBrush(QBrush(QColor(self.C_EXTRACTED)))
                p.drawRoundedRect(tx, ty, px(self._n_extracted), track_h, 3, 3)
            if self._n_detected > 0:
                p.setBrush(QBrush(QColor(self.C_DETECTED)))
                p.drawRoundedRect(tx, ty, px(self._n_detected), track_h, 3, 3)
            if self._n_cleaned > 0:
                p.setBrush(QBrush(QColor(self.C_CLEANED)))
                p.drawRoundedRect(tx, ty, px(self._n_cleaned), track_h, 3, 3)

        # Playhead — clamped within margins
        if self._maximum > 0:
            hx = margin + int(track_w * self._value / self._maximum)
        else:
            hx = margin
        ph  = 14
        py_ = (h - ph) // 2
        p.setBrush(QBrush(QColor("#d0dae8")))
        p.setPen(QPen(QColor("#3a4a60"), 1))
        p.drawRoundedRect(hx - pw // 2, py_, pw, ph, 5, 5)
        p.end()

    def _pos_to_value(self, x: int) -> int:
        pw     = 12
        margin = pw // 2 + 1
        track_w = self.width() - margin * 2
        if self._maximum == 0 or track_w <= 0:
            return 0
        rel = x - margin
        return max(0, min(self._maximum, round(self._maximum * rel / track_w)))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            self.sliderPressed.emit()
            self.setValue(self._pos_to_value(e.position().x()))

    def mouseMoveEvent(self, e):
        if self._pressed:
            v = self._pos_to_value(e.position().x())
            self._value = max(0, min(v, self._maximum))
            self.valueChanged.emit(self._value)
            self.update()

    def mouseReleaseEvent(self, e):
        if self._pressed:
            self._pressed = False
            self.setValue(self._pos_to_value(e.position().x()))
            self.sliderReleased.emit()


class MainWindow(QMainWindow):
    _log_sig = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepMosaicsPlus")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setMinimumSize(1100, 700)
        self.resize(1280, 820)
        self._drag_pos = None
        self._process   = None
        self._resume_choice = "fresh"
        self._cached_frames: list = []
        self._cached_cleaned: list = []
        self._scrubbing: bool = False
        self._user_scrubbing: bool = False
        self._frame_dir: str | None = None
        self._mask_dir:  str | None = None
        self._replace_dir: str | None = None
        self._max_frame_count: int = 0
        self._ever_cleaned: bool = False
        self._phase: str = "detecting"
        self._n_detected_last: int = 0
        self._extraction_done: bool = False
        self._max_detected: int = 0
        self._log_detected_frame: int = 0
        self._log_cleaned_frame: int = 0
        self._log_total: int = 0
        self._frame_dir = None   # temp/video2image
        self._mask_dir  = None   # temp/mosaic_mask
        self._result_dir = None
        self._watcher_timer = QTimer(self)
        self._watcher_timer.timeout.connect(self._poll_frames)
        self._last_frame_idx = 0
        self._log_sig.connect(self._append_log)

        self._build()

    # ── Build ──────────────────────────────────────────────────────────────────
    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Custom title bar (spans full window width) ─────────────────────────
        title_bar = QWidget()
        title_bar.setFixedHeight(42)
        title_bar.setStyleSheet(
            f"background: {C['surface2']}; border-bottom: 1px solid {C['border']};"
        )
        tb = QHBoxLayout(title_bar)
        tb.setContentsMargins(16, 0, 10, 0)
        tb.setSpacing(0)

        accent = C["accent"]
        app_lbl = QLabel(
            f"DeepMosaics<span style='color:{accent}'>+</span>"
        )
        app_lbl.setTextFormat(Qt.TextFormat.RichText)
        app_lbl.setStyleSheet(
            f"font-size: 15px; font-weight: 700; color: {C['text_bright']};"
            f" letter-spacing: -0.02em; background: transparent;"
        )
        tb.addWidget(app_lbl)
        tb.addStretch()

        for symbol, tip, slot in [
            ("─", "Minimise", self.showMinimized),
            ("□", "Maximise", self._toggle_maximise),
            ("✕", "Close",    self.close),
        ]:
            btn = QPushButton(symbol)
            btn.setToolTip(tip)
            btn.setFixedSize(36, 36)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; border: none;
                    color: {C['text_dim']}; font-size: 13px;
                    border-radius: 4px;
                }}
                QPushButton:hover {{ background: {'#c0392b' if symbol == '✕' else C['border2']};
                                     color: {C['text_bright']}; }}
            """)
            btn.clicked.connect(slot)
            tb.addWidget(btn)

        # Make title bar draggable
        title_bar.mousePressEvent   = self._tb_press
        title_bar.mouseMoveEvent    = self._tb_move
        title_bar.mouseReleaseEvent = self._tb_release
        title_bar.mouseDoubleClickEvent = lambda e: self._toggle_maximise()

        outer.addWidget(title_bar)

        # ── Top: sidebar + viewport splitter ──────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {C['border']}; }}")
        outer.addWidget(splitter, 1)

        # Left sidebar
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(200)
        # No setMaximumWidth — fully draggable
        sb_outer = QVBoxLayout(sidebar)
        sb_outer.setContentsMargins(0, 0, 0, 0)
        sb_outer.setSpacing(0)

        # Scroll area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_w = QWidget()
        self._sb = QVBoxLayout(scroll_w)
        self._sb.setContentsMargins(16, 4, 16, 20)
        self._sb.setSpacing(4)
        scroll.setWidget(scroll_w)
        sb_outer.addWidget(scroll, 1)
        splitter.addWidget(sidebar)

        # Centre: viewport + overlay bar
        centre = QWidget()
        centre.setObjectName("viewport_bg")
        cv = QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)

        self._overlay_bar = OverlayBar()
        self._overlay_bar.setStyleSheet(
            f"background: {C['surface2']}; border-bottom: 1px solid {C['border']};"
        )
        cv.addWidget(self._overlay_bar)

        self._viewport = FrameViewport()
        cv.addWidget(self._viewport, 1)

        # Frame scrubber strip
        scrub = QWidget()
        scrub.setStyleSheet(f"background: {C['surface2']}; border-top: 1px solid {C['border']};")
        sl = QHBoxLayout(scrub)
        sl.setContentsMargins(12, 7, 12, 7)
        sl.setSpacing(10)

        self._frame_label = QLabel("No frames")
        self._frame_label.setFixedWidth(120)
        self._frame_label.setStyleSheet(f"color: {C['text_dim']}; font-size: 11px; font-family: monospace;")
        sl.addWidget(self._frame_label)

        self._scrubber = TricolorScrubber()
        self._scrubber.setRange(0, 0)
        self._scrubber.valueChanged.connect(self._scrub_frame)
        self._scrubber.sliderPressed.connect(self._on_scrub_press)
        self._scrubber.sliderReleased.connect(self._on_scrub_release)
        sl.addWidget(self._scrubber, 1)

        self._live_btn = QPushButton("● LIVE")
        self._live_btn.setObjectName("overlay_btn")
        self._live_btn.setCheckable(True)
        self._live_btn.setChecked(True)
        self._live_btn.setFixedWidth(64)
        self._live_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 3px 8px; font-size: 10px; font-weight: 700;
                border-radius: 4px; border: 1px solid {C['border2']};
                background: {C['surface2']}; color: {C['text_dim']};
                letter-spacing: 0.05em;
            }}
            QPushButton:checked {{
                background: #1a3a18; color: #5ecf5a;
                border-color: #2a6a26;
            }}
        """)
        self._live_btn.clicked.connect(self._on_live_btn)
        sl.addWidget(self._live_btn)

        for dot_col, dot_lbl in [
            (TricolorScrubber.C_EXTRACTED, "Extracted"),
            (TricolorScrubber.C_DETECTED,  "Detected"),
            (TricolorScrubber.C_CLEANED,   "Cleaned"),
        ]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {dot_col}; font-size: 11px; padding: 0 0 0 6px;")
            lbl2 = QLabel(dot_lbl)
            lbl2.setStyleSheet(f"color: {C['text_dim']}; font-size: 10px;")
            sl.addWidget(dot)
            sl.addWidget(lbl2)

        cv.addWidget(scrub)
        splitter.addWidget(centre)
        splitter.setSizes([270, 270])

        # ── Build sidebar sections ─────────────────────────────────────────────
        self._build_sidebar()

        # ── Bottom bar ─────────────────────────────────────────────────────────
        bottom = QWidget()
        bottom.setObjectName("bottom_bar")
        bv = QVBoxLayout(bottom)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(0)

        # Command preview
        self._cmd_lbl = QLabel("python deepmosaic.py")
        self._cmd_lbl.setObjectName("cmd_preview")
        self._cmd_lbl.setWordWrap(False)
        bv.addWidget(self._cmd_lbl)

        # Action row + log
        action_row = QWidget()
        action_row.setStyleSheet(f"background: {C['surface']};")
        ar = QHBoxLayout(action_row)
        ar.setContentsMargins(14, 10, 14, 10)
        ar.setSpacing(10)

        self._btn_run    = QPushButton("▶  Run")
        self._btn_cancel = QPushButton("■  Cancel")
        self._btn_run.setObjectName("btn_run")
        self._btn_cancel.setObjectName("btn_cancel")
        self._btn_run.clicked.connect(self._run)
        self._btn_cancel.clicked.connect(self._cancel)
        ar.addWidget(self._btn_run)
        ar.addWidget(self._btn_cancel)
        ar.addSpacing(8)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet(f"color: {C['text_dim']}; font-size: 12px;")
        ar.addWidget(self._status_lbl)
        ar.addStretch()

        self._log = QPlainTextEdit()
        self._log.setObjectName("log")
        mono = QFont("JetBrains Mono, Consolas, Courier New")
        mono.setPointSize(10)
        self._log.setFont(mono)
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(110)
        self._log.setMinimumHeight(110)
        ar.addWidget(self._log, 1)

        bv.addWidget(action_row)
        outer.addWidget(bottom)

        # Resize grip — overlaid at bottom-right, not in a layout
        self._grip = QSizeGrip(self)
        self._grip.setStyleSheet("background: transparent;")
        self._grip.resize(16, 16)

        # Wire overlay controls
        self._overlay_bar.mode_changed.connect(self._viewport.set_mode)
        self._overlay_bar.opacity_changed.connect(self._viewport.set_opacity)
        self._overlay_bar.source_changed.connect(self._on_source_changed)
        self._viewport_source = "detection"
        self._viewport.file_dropped.connect(self._on_file_dropped)
        self._viewport.clicked.connect(self._on_viewport_clicked)

        # Wire all inputs → refresh command
        for w in self.findChildren(QLineEdit):
            w.textChanged.connect(self._refresh)
        for w in self.findChildren(QCheckBox):
            w.toggled.connect(self._refresh)
        for w in self.findChildren(QComboBox):
            w.currentTextChanged.connect(self._refresh)
        for w in self.findChildren(QSpinBox):
            w.valueChanged.connect(self._refresh)
        for w in self.findChildren(QSlider):
            w.valueChanged.connect(self._refresh)

        self._refresh()

    def _build_sidebar(self):
        sb = self._sb

        # ── Input ─────────────────────────────────────────────────────────────
        sb.addWidget(_section("Input"))

        # Media
        self.media_in = _inp("Video or image…")
        mb = _browse()
        mb.clicked.connect(lambda: self._pick_file(self.media_in, "Select media"))
        sb.addLayout(_row(_rl("Media"), self.media_in, mb))

        # Model
        self.model_in = _inp("Path to .pth…", "./clean_youknow_video.pth")
        mlb = _browse()
        mlb.clicked.connect(lambda: self._pick_file(self.model_in, "Select model", "Models (*.pth)"))
        sb.addLayout(_row(_rl("Model"), self.model_in, mlb))

        # Result
        self.result_in = _inp("Output folder…", "./result")
        rlb = _browse()
        rlb.clicked.connect(lambda: self._pick_folder(self.result_in))
        sb.addLayout(_row(_rl("Result folder"), self.result_in, rlb))

        # ── Config ────────────────────────────────────────────────────────────
        sb.addWidget(_section("Config"))

        self.gpu_in  = _inp("-1 for CPU", "0", 58)
        self.fps_in  = _inp("0 = src", "0", 58)
        sb.addLayout(_row(_rl("GPU / FPS"), self.gpu_in, 4, self.fps_in, None))

        self.netg_sel = QComboBox()
        self.netg_sel.addItems(["auto","unet_128","unet_256","resnet_9blocks","HD","video"])
        sb.addLayout(_row(_rl("Network G"), self.netg_sel, None))

        self.start_in = _inp("HH:MM:SS", "00:00:00", 82)
        self.dur_in   = _inp("0 = full",  "00:00:00", 82)
        sb.addLayout(_row(_rl("Start / Dur"), self.start_in, 4, self.dur_in, None))

        self.outsize_in = _inp("0 = src", "0", 58)
        self.mask_thr   = _inp("0–255",  "48", 48)
        sb.addLayout(_row(_rl("Size / Mask ⊕"), self.outsize_in, 4, self.mask_thr, None))
        sb.addWidget(_hint("⊕ Mask threshold — lower = more sensitive detection"))

        self.encode_crf = QSpinBox()
        self.encode_crf.setRange(0, 51)
        self.encode_crf.setValue(18)
        self.encode_crf.setFixedWidth(58)
        self.encode_crf.setStyleSheet(f"""
            QSpinBox {{
                background: {C['bg']}; border: 1px solid {C['border2']};
                border-radius: 5px; padding: 5px 6px; color: {C['text']}; font-size: 12px;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; }}
        """)

        self.decode_qv = QSpinBox()
        self.decode_qv.setRange(1, 31)
        self.decode_qv.setValue(1)
        self.decode_qv.setFixedWidth(58)
        self.decode_qv.setStyleSheet(self.encode_crf.styleSheet())

        sb.addLayout(_row(_rl("Encode CRF"), self.encode_crf, 12, _rl("Decode QV"), self.decode_qv, None))
        sb.addWidget(_hint("Encode CRF: 0=lossless · 18=default · 28=smaller  |  Decode QV: 1=best · 31=smallest"))

        # ── Detection ─────────────────────────────────────────────────────────
        sb.addWidget(_section("Detection"))

        self.min_area_in = _inp("px²",       "300", 62)
        self.min_size_in = _inp("px half-w", "100", 62)
        sb.addLayout(_row(_rl("Area / Size"), self.min_area_in, 4, self.min_size_in, None))
        sb.addWidget(_hint("Min area: blob filter after thresholding  ·  Min size: gate before network runs"))

        self.medfilt_in = _inp("", "5", 48)
        self.exmult_in  = _inp("", "auto", 52)
        sb.addLayout(_row(_rl("MedFilt / ExMult"), self.medfilt_in, 4, self.exmult_in, None))

        self.no_feather_cb  = QCheckBox("No edge feathering")
        self.all_mosaic_cb  = QCheckBox("All mosaic areas (slower)")
        self.debug_cb       = QCheckBox("Debug mode")
        sb.addWidget(self.no_feather_cb)
        sb.addWidget(self.all_mosaic_cb)
        sb.addWidget(self.debug_cb)

        # Traditional
        self.traditional_cb = QCheckBox("Traditional method")
        sb.addWidget(self.traditional_cb)
        self._trad_w = QWidget()
        tw = QHBoxLayout(self._trad_w)
        tw.setContentsMargins(14, 0, 0, 0)
        tw.setSpacing(6)
        self.tr_blur_in = _inp("", "10", 52)
        self.tr_down_in = _inp("", "10", 52)
        tw.addWidget(QLabel("Blur"))
        tw.addWidget(self.tr_blur_in)
        tw.addWidget(QLabel("Down"))
        tw.addWidget(self.tr_down_in)
        tw.addStretch()
        self._trad_w.setVisible(False)
        self.traditional_cb.toggled.connect(self._trad_w.setVisible)
        sb.addWidget(self._trad_w)

        # ── Experimental ──────────────────────────────────────────────────────
        sb.addWidget(_section("Post-Processing"))
        badge = QLabel("EXPERIMENTAL")
        badge.setObjectName("exp_badge")
        sb.addWidget(badge)
        sb.addWidget(_hint("Applied to cleaned patch before compositing. Order: freq → bilateral → luma USM."))

        self.freq_cb, self.freq_sl, self.freq_val = _sl_row(
            sb, "Frequency injection",
            "Borrows edges from original mosaic. Start here. Keep ≤ 0.6.",
            0.30, 0.05, 0.80, 15)

        self.bilateral_cb, self.bilateral_sl, self.bilateral_val = _sl_row(
            sb, "Bilateral sharpen",
            "Edge-preserving sharpen, less haloing than USM.",
            0.50, 0.10, 2.00, 19)

        self.luma_cb, self.luma_sl, self.luma_val = _sl_row(
            sb, "Luma sharpen (USM)",
            "Unsharp mask on Y channel. Apply last.",
            1.00, 0.10, 3.00, 29)

        sb.addStretch()

    # ── Command generation ─────────────────────────────────────────────────────
    def _generate_command(self):
        cmd = ["python", "deepmosaic.py"]

        if self.debug_cb.isChecked():       cmd.append("--debug")
        gpu = self.gpu_in.text().strip()
        if gpu:                             cmd.extend(["--gpu_id", gpu])
        m = self.media_in.text().strip()
        if m:                               cmd.extend(["--media_path", m])
        s = self.start_in.text().strip()
        if s and s != "00:00:00":           cmd.extend(["-ss", s])
        d = self.dur_in.text().strip()
        if d and d != "00:00:00":           cmd.extend(["-t", d])
        mdl = self.model_in.text().strip()
        if mdl:                             cmd.extend(["--model_path", mdl])
        r = self.result_in.text().strip()
        if r:                               cmd.extend(["--result_dir", r])
        ng = self.netg_sel.currentText()
        if ng != "auto":                    cmd.extend(["--netG", ng])
        fps = self.fps_in.text().strip()
        if fps and fps != "0":              cmd.extend(["--fps", fps])
        cmd.append("--no_preview")   # always: we show frames in our own viewport
        cmd.append("--keep_frames")  # always: prevent frame deletion so seek bar stays stable
        crf = self.encode_crf.value()
        if crf != 18:
            cmd.extend(["--encode_crf", str(crf)])
        qv = self.decode_qv.value()
        if qv != 1:
            cmd.extend(["--decode_qv", str(qv)])
        os_ = self.outsize_in.text().strip()
        if os_ and os_ != "0":             cmd.extend(["--output_size", os_])
        mt = self.mask_thr.text().strip()
        if mt and mt != "48":              cmd.extend(["--mask_threshold", mt])
        if self.traditional_cb.isChecked():
            cmd.append("--traditional")
            bl = self.tr_blur_in.text().strip()
            if bl and bl != "10":          cmd.extend(["--tr_blur", bl])
            dn = self.tr_down_in.text().strip()
            if dn and dn != "10":          cmd.extend(["--tr_down", dn])
        if self.no_feather_cb.isChecked(): cmd.append("--no_feather")
        if self.all_mosaic_cb.isChecked(): cmd.append("--all_mosaic_area")
        ma = self.min_area_in.text().strip()
        if ma and ma != "300":             cmd.extend(["--min_mosaic_area", ma])
        ms = self.min_size_in.text().strip()
        if ms and ms != "100":             cmd.extend(["--min_mosaic_size", ms])
        mf = self.medfilt_in.text().strip()
        if mf and mf != "5":              cmd.extend(["--medfilt_num", mf])
        ex = self.exmult_in.text().strip()
        if ex and ex != "auto":            cmd.extend(["--ex_mult", ex])

        # Experimental — always emit amount (no conditional guard)
        if self.freq_cb.isChecked():
            cmd += ["--freq_inject", "--freq_inject_amount", f"{self.freq_sl.value()/100:.2f}"]
        if self.bilateral_cb.isChecked():
            cmd += ["--bilateral_sharpen", "--bilateral_sharpen_amount", f"{self.bilateral_sl.value()/100:.2f}"]
        if self.luma_cb.isChecked():
            cmd += ["--luma_sharpen", "--luma_sharpen_amount", f"{self.luma_sl.value()/100:.2f}"]

        return cmd

    def _refresh(self, *_):
        try:
            cmd = self._generate_command()
            self._cmd_lbl.setText(" ".join(cmd))
        except Exception:
            pass

    # ── File pickers & drop ────────────────────────────────────────────────────
    def _pick_file(self, target, title, filt="All files (*)"):
        p, _ = QFileDialog.getOpenFileName(self, title, "", filt)
        if p:
            target.setText(p)

    def _pick_folder(self, target):
        p = QFileDialog.getExistingDirectory(self, "Select folder", "")
        if p:
            target.setText(p)

    def _on_file_dropped(self, path: str):
        self.media_in.setText(path)
        # Try to show first frame immediately
        self._try_show_source_thumb(path)

    def _on_viewport_clicked(self):
        """Empty drop zone clicked — open file dialog."""
        self._pick_file(self.media_in, "Select media file")

    def _try_show_source_thumb(self, path: str):
        """Show a quick preview of the source file if it's an image."""
        ext = Path(path).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            self._viewport.set_frame(path)

    # ── Custom title bar drag ──────────────────────────────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        g = self._grip
        g.move(self.width() - g.width(), self.height() - g.height())

    def _tb_press(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def _tb_move(self, e):
        if self._drag_pos and e.buttons() == Qt.MouseButton.LeftButton:
            if self.isMaximized():
                self.showNormal()
            self.move(e.globalPosition().toPoint() - self._drag_pos)

    def _tb_release(self, e):
        self._drag_pos = None

    def _toggle_maximise(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()

    # ── Frame watcher ──────────────────────────────────────────────────────────
    def _start_watcher(self):
        base = "./tmp/DeepMosaics_temp"
        self._frame_dir   = os.path.join(base, "video2image")
        self._mask_dir    = os.path.join(base, "mosaic_mask")
        self._replace_dir = os.path.join(base, "replace_mosaic")
        self._cached_frames  = []
        self._cached_cleaned = []
        self._scrubbing = False
        self._max_frame_count = 0
        self._ever_cleaned = False
        self._phase = "detecting"
        self._n_detected_last = 0
        self._extraction_done = False
        self._max_detected = 0
        self._max_cleaned = 0
        self._log_detected_frame = 0
        self._log_cleaned_frame = 0
        self._log_total = 0
        self._watcher_timer.start(500)

    def _stop_watcher(self):
        self._watcher_timer.stop()

    def _poll_frames(self):
        if not self._frame_dir:
            return

        # Detection frames
        if os.path.isdir(self._frame_dir):
            frames = sorted(
                glob.glob(os.path.join(self._frame_dir, "*.jpg")) +
                glob.glob(os.path.join(self._frame_dir, "*.png"))
            )
            if frames:
                self._cached_frames = frames

        # Cleaned frames
        if self._replace_dir and os.path.isdir(self._replace_dir):
            cleaned = sorted(
                glob.glob(os.path.join(self._replace_dir, "*.jpg")) +
                glob.glob(os.path.join(self._replace_dir, "*.png"))
            )
            if cleaned:
                self._cached_cleaned = cleaned

        # Use whichever cache is non-empty for the scrubber
        if not self._cached_frames and not self._cached_cleaned:
            return

        n_extracted = len(self._cached_frames)
        n_cleaned   = len(self._cached_cleaned)
        # Use log-parsed counts for detection progress — file count lags and misses blank frames
        log_detected = self._log_detected_frame
        log_total    = self._log_total if self._log_total > 0 else n_extracted

        # Never shrink the scrubber
        n = max(self._max_frame_count, n_extracted, n_cleaned, log_total)
        self._max_frame_count = n

        # Detection band: latched log count, and snap to full once cleaning begins
        if self._phase == "cleaning":
            self._max_detected = max(self._max_detected, n if n > 0 else n_extracted)
        else:
            self._max_detected = max(self._max_detected, log_detected)

        # Cleaning band: use latched log count (file glob lags async writer and freezes during mux)
        self._max_cleaned = max(getattr(self, '_max_cleaned', 0), self._log_cleaned_frame, n_cleaned)

        self._scrubber.blockSignals(True)
        self._scrubber.setRange(0, max(0, n - 1))
        self._scrubber.setProgress(n_extracted, self._max_detected, self._max_cleaned)
        running = self._process and self._process.state() != QProcess.ProcessState.NotRunning

        # Phase latch (file-based fallback — _on_stdout handles it faster via log)
        if n_cleaned > 0:
            self._ever_cleaned = True
        if self._ever_cleaned and self._phase == "detecting":
            self._phase = "cleaning"

        # _extraction_done is latched by _on_stdout when Step:2 appears in the log

        if running and self._live_btn.isChecked() and not self._user_scrubbing:
            if self._phase == "cleaning" and self._cached_cleaned:
                # Use log-parsed cleaning frame for live advance — ahead of file count
                log_cl = self._log_cleaned_frame
                live_idx = min(
                    (log_cl - 1) if log_cl > 0 else (len(self._cached_cleaned) - 1),
                    n_extracted - 1 if n_extracted > 0 else 0
                )
            elif self._extraction_done and n_extracted > 0:
                if log_detected > self._n_detected_last:
                    # New detection progress from log — advance to that frame
                    live_idx = min(log_detected - 1, n_extracted - 1)
                    self._n_detected_last = log_detected
                elif self._n_detected_last > 0:
                    # Log hasn't updated yet this poll cycle — hold at last known frame
                    live_idx = min(self._n_detected_last - 1, n_extracted - 1)
                else:
                    live_idx = None
            elif not self._extraction_done and n_extracted > 0:
                # During extraction: don't chase the end (parallel threads fill non-linearly)
                # Instead show the median frame so it doesn't jump to end immediately
                live_idx = n_extracted // 2
            else:
                live_idx = None
            if live_idx is not None:
                self._scrubber.setValue(live_idx)
                self._scrubber.blockSignals(False)
                self._load_frame(live_idx)
            else:
                self._scrubber.blockSignals(False)
        else:
            self._scrubber.blockSignals(False)

        # Status — detection uses log (fast, catches blank frames), cleaning uses file count (complete)
        if running:
            denom = self._max_frame_count if self._max_frame_count > 0 else 1
            if self._phase == "cleaning":
                pct = min(100, int(100 * n_cleaned / denom))
                self._status_lbl.setText(f"Cleaning… {pct}%  ({n_cleaned}/{self._max_frame_count})")
            else:
                pct = min(100, int(100 * log_detected / denom))
                self._status_lbl.setText(f"Detecting… {pct}%  ({log_detected}/{self._max_frame_count})")

    def _on_scrub_press(self):
        """User grabbed the slider — pause live auto-advance."""
        self._user_scrubbing = True
        self._live_btn.setChecked(False)

    def _on_scrub_release(self):
        """User released the slider — load the chosen frame, stay paused."""
        self._user_scrubbing = False
        self._load_frame(self._scrubber.value())

    def _on_live_btn(self, checked: bool):
        """Clicking LIVE re-attaches to the running process feed."""
        if checked:
            self._user_scrubbing = False
            # Jump immediately to latest frame
            n = len(self._cached_frames)
            if n:
                self._scrubber.blockSignals(True)
                self._scrubber.setValue(n - 1)
                self._scrubber.blockSignals(False)
                self._load_frame(n - 1)

    def _scrub_frame(self, idx: int):
        # valueChanged fires during programmatic setValue too — only act on
        # user drags (slider is pressed) to give real-time frame preview
        if self._scrubbing:
            return
        if self._user_scrubbing:
            self._load_frame(idx)

    def _load_frame(self, idx: int):
        if self._viewport_source == "cleaned":
            frames = self._cached_cleaned
        else:
            frames = self._cached_frames

        n_total = max(self._max_frame_count, len(frames))
        if not frames or idx < 0:
            return
        actual_idx = min(idx, len(frames) - 1)

        self._scrubbing = True
        try:
            frame_path = frames[actual_idx]
            self._viewport.set_frame(frame_path)
            self._frame_label.setText(f"Frame {idx+1} / {n_total}")

            if self._viewport_source == "detection":
                fname = os.path.basename(frame_path)
                mask_path = os.path.join(self._mask_dir, fname) if self._mask_dir else None
                if mask_path and os.path.isfile(mask_path):
                    self._viewport.set_mask(mask_path)
                else:
                    self._viewport._mask_pix = None
                    self._viewport._tint_pix = None
                    self._viewport.update()
            else:
                # Cleaned — pair mask by index so overlay works on cleaned too
                if self._cached_frames and actual_idx < len(self._cached_frames):
                    fname = os.path.basename(self._cached_frames[actual_idx])
                    mask_path = os.path.join(self._mask_dir, fname) if self._mask_dir else None
                    if mask_path and os.path.isfile(mask_path):
                        self._viewport.set_mask(mask_path)
                    else:
                        self._viewport._mask_pix = None
                        self._viewport._tint_pix = None
                        self._viewport.update()
                else:
                    self._viewport._mask_pix = None
                    self._viewport._tint_pix = None
                    self._viewport.update()
        finally:
            self._scrubbing = False

    def _on_source_changed(self, source: str):
        self._viewport_source = source
        self._viewport._mask_pix = None
        self._viewport._tint_pix = None
        self._viewport.update()
        self._load_frame(self._scrubber.value())

    # ── Process ────────────────────────────────────────────────────────────────
    def _run(self):
        if not self.media_in.text().strip():
            self._log_sig.emit("⚠  No media file selected.\n"); return
        if not self.model_in.text().strip():
            self._log_sig.emit("⚠  No model file selected.\n"); return
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._log_sig.emit("⚠  Already running.\n"); return

        cmd = self._generate_command()
        self._log.clear()
        self._viewport.clear()
        self._scrubber.setRange(0, 0)
        self._frame_label.setText("No frames")
        self._user_scrubbing = False
        self._live_btn.setChecked(True)
        self._max_frame_count = 0
        self._ever_cleaned = False
        self._phase = "detecting"
        self._cached_frames = []
        self._cached_cleaned = []
        self._n_detected_last = 0
        self._extraction_done = False
        self._max_detected = 0
        self._max_cleaned = 0
        self._log_detected_frame = 0
        self._log_cleaned_frame = 0
        self._log_total = 0
        self._viewport_source = "detection"
        self._overlay_bar._select_src("detection")
        self._status_lbl.setText("Running…")
        self._status_lbl.setStyleSheet(f"color: {C['amber']}; font-size: 12px;")

        # Check for leftover temp files from a previous run
        import shutil
        base = os.path.join(str(Path(__file__).parent), "tmp", "DeepMosaics_temp")
        has_temp = os.path.isfile(os.path.join(base, "step.json"))
        if has_temp:
            dlg = QDialog(self)
            dlg.setWindowTitle("Previous run detected")
            dlg.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
            dlg.setStyleSheet(f"""
                QDialog {{
                    background: {C['surface2']};
                    border: 1px solid {C['border2']};
                    border-radius: 8px;
                }}
                QLabel {{ color: {C['text']}; background: transparent; }}
                QPushButton {{
                    border-radius: 5px; padding: 7px 18px;
                    font-size: 12px; font-weight: 500;
                    border: 1px solid {C['border2']};
                    background: {C['surface2']}; color: {C['text']};
                    min-width: 110px;
                }}
                QPushButton:hover {{ background: {C['border2']}; }}
                QPushButton#btn_fresh {{
                    background: {C['accent_dim']}; color: {C['text_bright']};
                    border-color: {C['accent']};
                }}
                QPushButton#btn_fresh:hover {{ background: #224488; }}
            """)
            v = QVBoxLayout(dlg)
            v.setContentsMargins(24, 20, 24, 20)
            v.setSpacing(14)

            title_lbl = QLabel("Unfinished work found")
            title_lbl.setStyleSheet(f"font-size: 14px; font-weight: 600; color: {C['text_bright']};")
            v.addWidget(title_lbl)

            body_lbl = QLabel(
                "Temp files from a previous run exist.\n"
                "Resume will continue from where it left off.\n"
                "Fresh start will delete the temp files and begin again."
            )
            body_lbl.setStyleSheet(f"color: {C['text_dim']}; font-size: 12px; line-height: 1.5;")
            body_lbl.setWordWrap(True)
            v.addWidget(body_lbl)

            btn_row = QHBoxLayout()
            btn_row.setSpacing(8)
            btn_cancel_dlg = QPushButton("Cancel")
            btn_resume     = QPushButton("Resume")
            btn_fresh      = QPushButton("Fresh start")
            btn_fresh.setObjectName("btn_fresh")
            btn_row.addWidget(btn_cancel_dlg)
            btn_row.addStretch()
            btn_row.addWidget(btn_resume)
            btn_row.addWidget(btn_fresh)
            v.addLayout(btn_row)

            choice = {"v": None}
            btn_cancel_dlg.clicked.connect(lambda: (choice.update({"v": "cancel"}), dlg.accept()))
            btn_resume.clicked.connect(    lambda: (choice.update({"v": "resume"}), dlg.accept()))
            btn_fresh.clicked.connect(     lambda: (choice.update({"v": "fresh"}),  dlg.accept()))

            dlg.exec()

            if choice["v"] == "cancel" or choice["v"] is None:
                self._status_lbl.setText("Ready")
                self._status_lbl.setStyleSheet(f"color: {C['text_dim']}; font-size: 12px;")
                return
            elif choice["v"] == "fresh":
                shutil.rmtree(base, ignore_errors=True)
                self._resume_choice = "fresh"
                self._log_sig.emit("↺  Temp files cleared — starting fresh.\n")
            else:
                self._resume_choice = "resume"
                self._log_sig.emit("⏩  Resuming from previous run.\n")

        self._process = QProcess(self)
        self._process.setWorkingDirectory(str(Path(__file__).parent))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self._process.setProcessEnvironment(env)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.finished.connect(self._on_finished)

        self._log_sig.emit(f"▶  {' '.join(cmd)}\n{'─'*60}\n")
        self._process.start(cmd[0], cmd[1:])
        self._start_watcher()

    def _on_stdout(self):
        if not self._process: return
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._log_sig.emit(data)

        # Auto-answer the resume prompt
        if "unfinished" in data.lower() and "y/n" in data.lower():
            answer = b"y\n" if self._resume_choice == "resume" else b"n\n"
            self._process.write(answer)
            self._log_sig.emit(f"→ Auto-answered: {'y (resume)' if answer == b'y\\n' else 'n (fresh start)'}\n")

        # Parse frame progress from \rN/TOTAL progress lines.
        # Exclude extraction tqdm lines ("Extracting frames: 42%|...") and step lines (N/4).
        # Only count lines where total > 10 and the line doesn't come from tqdm extraction output.
        frame_matches = []
        for line in re.split(r'[\r\n]', data):
            if 'extracting' in line.lower() or 'extract' in line.lower():
                continue  # skip tqdm extraction lines
            for a, b in re.findall(r'(\d+)/(\d+)', line):
                ia, ib = int(a), int(b)
                if ib > 10 and ia <= ib:
                    frame_matches.append((ia, ib))
        if frame_matches:
            cur, total = frame_matches[-1]
            if self._phase == "detecting":
                self._log_detected_frame = cur
                self._log_total = total
            elif self._phase == "cleaning":
                self._log_cleaned_frame = cur
                self._log_total = total

        # Use log lines to latch phase earlier than file polling can detect
        low = data.lower()
        if self._phase == "detecting" and any(x in low for x in ["clean mosaic", "replace mosaic", "step:3"]):
            self._ever_cleaned = True
            self._phase = "cleaning"
            if self._live_btn.isChecked():
                self._overlay_bar._select_src("cleaned")
                self._viewport_source = "cleaned"

        # Reset _n_detected_last when detection begins so extraction frame count doesn't block it
        if not self._extraction_done and any(x in low for x in ["step:2", "find mosaic", "mosaic location"]):
            self._extraction_done = True
            self._n_detected_last = 0
            self._log_detected_frame = 0

    def _on_finished(self, exit_code, _):
        self._stop_watcher()
        self._poll_frames()   # final update
        self._live_btn.setChecked(False)  # stop live tracking — seekbar is now fully interactive
        ok = exit_code == 0
        self._status_lbl.setText("Done" if ok else f"Error (code {exit_code})")
        self._status_lbl.setStyleSheet(
            f"color: {C['green'] if ok else C['red']}; font-size: 12px;")
        self._log_sig.emit(f"\n{'─'*60}\n{'✅  Done!' if ok else f'❌  Exited with code {exit_code}'}\n")

    def _cancel(self):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate()
            if not self._process.waitForFinished(3000):
                self._process.kill()
            self._stop_watcher()
            self._live_btn.setChecked(False)
            self._status_lbl.setText("Cancelled")
            self._status_lbl.setStyleSheet(f"color: {C['red']}; font-size: 12px;")
            self._log_sig.emit("\n■  Cancelled.\n")
        else:
            self._log_sig.emit("ℹ  No running process.\n")

    def _append_log(self, text: str):
        self._log.moveCursor(self._log.textCursor().MoveOperation.End)
        self._log.insertPlainText(text)
        self._log.moveCursor(self._log.textCursor().MoveOperation.End)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DeepMosaicsPlus")
    app.setStyleSheet(QSS)
    for family in ("Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue"):
        if family in QFontDatabase.families():
            app.setFont(QFont(family, 12))
            break

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
