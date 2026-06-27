# app.py
import sys
import os
import shutil
import logging
import subprocess
import traceback
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Optional, List

import numpy as np
import soundfile as sf
import sounddevice as sd

try:
    from pedalboard import Pedalboard, HighpassFilter, Compressor, Limiter
    _PEDALBOARD_OK = True
except ImportError:
    _PEDALBOARD_OK = False

try:
    import pyloudnorm as pyln
    _PYLOUDNORM_OK = True
except ImportError:
    _PYLOUDNORM_OK = False

try:
    import torch
    import torchaudio
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from models_backends import (
    EnhancementRequest,
    detect_active_backends,
)
import reuse             # noqa: F401  (registers ReuseBackend)
import audiosr_backend   # noqa: F401  (registers AudioSRBackend)
import deepfilternet     # noqa: F401  (registers DeepFilterNetBackend)
import flashsr_backend   # noqa: F401  (registers FlashSRBackend)
import universr_backend  # noqa: F401  (registers UniverSRBackend)
import lavasr_backend    # noqa: F401  (registers LavaSRBackend)
import novasr_backend    # noqa: F401  (registers NovaSRBackend)
import clearervoice_backend  # noqa: F401  (registers ClearerVoiceBackend)
import flowhigh_backend  # noqa: F401  (registers FlowHighBackend)
import voicefixer_backend  # noqa: F401  (registers VoiceFixerBackend)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QFileDialog, QProgressBar,
    QFrame, QSizePolicy, QMessageBox, QComboBox, QCheckBox,
    QSlider, QStatusBar, QSplitter, QScrollArea, QSpinBox, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QFont, QColor, QPainter, QIcon, QPixmap,
    QDragEnterEvent, QDropEvent, QPen, QPalette,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

APP_DIR    = Path(__file__).parent.parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "demucs_tmp").mkdir(exist_ok=True)

_ACTIVE_BACKEND_CLASSES = detect_active_backends()


def model_dir_for(model_id: str) -> Path:
    return APP_DIR / "models" / model_id

C = {
    "bg":         "#141414",
    "surface":    "#1e1e1e",
    "panel":      "#252525",
    "border":     "#333333",
    "border2":    "#444444",
    "accent":     "#2a6aaa",
    "accent2":    "#1a4a7a",
    "accent_dim": "#2a6aaa22",
    "text":       "#cccccc",
    "text2":      "#888888",
    "text3":      "#555555",
    "success":    "#4a9aff",
    "warning":    "#ffb300",
    "error":      "#ff5555",
    "player":     "#141414",
    "green":      "#3ab870",
    "green_dim":  "#3ab87022",
}

STYLE = f"""
QWidget {{
    background-color: {C["bg"]};
    color: {C["text"]};
    font-family: "Segoe UI", "Ubuntu", sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    border: 1px solid {C["border"]};
    border-radius: 5px;
    margin-top: 18px;
    padding: 12px 10px 10px 10px;
    background-color: {C["panel"]};
    font-weight: bold;
    color: #aaaaaa;
    font-size: 11px;
    letter-spacing: 1px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px; top: 4px;
    padding: 0 6px;
    background-color: {C["panel"]};
    color: #888888;
    font-size: 11px;
}}
QPushButton {{
    background-color: {C["surface"]};
    border: 1px solid {C["border2"]};
    border-radius: 4px;
    color: {C["text"]};
    padding: 5px 14px;
    font-size: 12px;
}}
QPushButton:hover  {{ background-color: #333333; border-color: #666666; color: white; }}
QPushButton:pressed {{ background-color: #1a1a1a; }}
QPushButton:disabled {{ color: {C["text3"]}; border-color: {C["border"]}; }}
QCheckBox {{
    color: {C["text2"]};
    font-size: 12px;
    spacing: 8px;
    background-color: transparent;
}}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 2px solid #555555;
    border-radius: 3px;
    background-color: {C["surface"]};
}}
QCheckBox::indicator:hover   {{ border-color: {C["accent"]}; background-color: #252535; }}
QCheckBox::indicator:checked {{ background-color: {C["accent2"]}; border-color: {C["accent"]}; }}
QCheckBox::indicator:checked:hover {{ background-color: #2a5a9a; }}
QCheckBox:disabled {{ color: {C["text3"]}; }}
QCheckBox::indicator:disabled {{ border-color: {C["border"]}; background-color: {C["surface"]}; }}
QSlider::groove:horizontal  {{ height: 4px; background-color: {C["border2"]}; border-radius: 2px; }}
QSlider::handle:horizontal  {{
    background-color: {C["accent"]}; border: none;
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background-color: {C["accent"]}; border-radius: 2px; }}
QComboBox, QSpinBox, QLineEdit {{
    background-color: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 3px;
    color: {C["text"]};
    padding: 4px 8px;
    min-height: 24px;
}}
QComboBox:focus, QSpinBox:focus, QLineEdit:focus {{ border-color: {C["accent"]}; }}
QComboBox::drop-down {{ border: none; padding-right: 8px; }}
QComboBox QAbstractItemView {{
    background-color: {C["panel"]};
    color: {C["text"]};
    selection-background-color: {C["accent2"]};
    border: 1px solid {C["border2"]};
}}
QScrollBar:vertical {{ background-color: {C["surface"]}; width: 8px; border-radius: 4px; }}
QScrollBar::handle:vertical {{ background-color: {C["border2"]}; border-radius: 4px; min-height: 20px; }}
QScrollBar::handle:vertical:hover {{ background-color: {C["accent"]}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 0; }}
QProgressBar {{
    background-color: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 3px;
    text-align: center;
    color: {C["text2"]};
    height: 16px;
    font-size: 10px;
}}
QProgressBar::chunk {{ background-color: {C["accent"]}; border-radius: 2px; }}
QStatusBar {{
    background-color: {C["surface"]};
    border-top: 1px solid {C["border"]};
    color: {C["text2"]};
    font-size: 11px;
}}
QLabel {{ color: {C["text"]}; background-color: transparent; }}
QSplitter::handle {{ background-color: {C["border"]}; width: 2px; }}
QScrollArea {{ border: none; background-color: transparent; }}
"""

ENHANCE_BTN = f"""
QPushButton {{
    background-color: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #1e5fa8,stop:1 #133f74);
    border: 1px solid #2a6aaa; border-radius: 5px; color: white;
    font-size: 14px; font-weight: 700; padding: 11px 32px;
}}
QPushButton:hover {{
    background-color: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #2a72c8,stop:1 #1a4e8c);
    border-color: #4a8acc;
}}
QPushButton:pressed {{ background-color: #0e3060; }}
QPushButton:disabled {{ background-color: #1a2535; color: #556070; border-color: #2a3545; }}
"""

DOWNLOAD_BTN = f"""
QPushButton {{
    background-color: #2a2200; border: 1px solid {C["warning"]};
    border-radius: 4px; color: {C["warning"]};
    font-size: 12px; font-weight: 600; padding: 6px 16px;
}}
QPushButton:hover {{ background-color: #3a3200; color: white; border-color: #ffcc44; }}
QPushButton:disabled {{ color: {C["text3"]}; border-color: {C["border"]}; background-color: {C["surface"]}; }}
"""

def _coloured_btn(color: str) -> str:
    return f"""
QPushButton {{
    background-color: #1e1e1e; border: 1px solid {color};
    border-radius: 4px; color: {color}; font-size: 12px; padding: 5px 12px;
}}
QPushButton:hover {{ background-color: #2a2a2a; color: white; }}
QPushButton:disabled {{ color: {C["text3"]}; border-color: {C["border"]}; }}
"""


def _fmt(s: float) -> str:
    s = int(max(0, s))
    h = s // 3600; m = (s % 3600) // 60; sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _open_path(p: str) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(p)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])
    except Exception as e:
        logger.error(f"open_path error: {e}")


def _ffmpeg_ok() -> bool:
    try:
        return subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma", ".aiff"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".mts", ".m2ts", ".wmv", ".m4v"}

# ════════════════════════════════════════════════════════════════════════════
#  WaveformWidget
# ════════════════════════════════════════════════════════════════════════════
class WaveformWidget(QWidget):
    seeked = pyqtSignal(float)

    def __init__(self, h: int = 60, label: str = "No audio", parent=None):
        super().__init__(parent)
        self.setFixedHeight(h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._peaks: List[float] = []
        self._pos:   float       = 0.0
        self._label              = label
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAutoFillBackground(False)

    def set_audio(self, audio: np.ndarray):
        n     = max(1, self.width() // 3)
        chunk = max(1, len(audio) // n)
        peaks = [float(np.max(np.abs(audio[i * chunk:(i + 1) * chunk])))
                 for i in range(n) if len(audio[i * chunk:(i + 1) * chunk])]
        mx = max(peaks) if peaks and max(peaks) > 0 else 1.0
        self._peaks = [p / mx for p in peaks]
        self.update()

    def set_position(self, p: float):
        self._pos = max(0.0, min(1.0, p))
        self.update()

    def clear(self):
        self._peaks = []; self._pos = 0.0; self.update()

    def mousePressEvent(self, e):
        if self._peaks and e.button() == Qt.MouseButton.LeftButton:
            self.seeked.emit(max(0.0, min(1.0, e.position().x() / max(1, self.width()))))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(C["player"]))

        if not self._peaks:
            p.setPen(QColor(C["border"]))
            p.drawLine(0, h // 2, w, h // 2)
            f = QFont(); f.setPointSize(9); p.setFont(f)
            p.setPen(QColor(C["text3"]))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._label)
            return

        mid  = h / 2.0
        ma   = mid - 4
        n    = len(self._peaks)
        bw   = max(1.0, w / n)
        px_x = int(self._pos * w)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#141414"))
        p.drawRect(0, 0, px_x, h)

        for i, pk in enumerate(self._peaks):
            x  = i * bw
            bh = max(2, pk * ma)
            cx = int(x + bw / 2)
            color = C["accent"] if x < px_x else C["border2"]
            p.setPen(QPen(QColor(color), max(1.0, bw * 0.6)))
            p.drawLine(cx, int(mid - bh), cx, int(mid + bh))

        p.setPen(QPen(QColor("#e0e0e0"), 1.5))
        p.drawLine(px_x, 0, px_x, h)


# ════════════════════════════════════════════════════════════════════════════
#  MiniPlayer
# ════════════════════════════════════════════════════════════════════════════
class MiniPlayer(QWidget):
    def __init__(self, header: str = "", accent: str = None, parent=None):
        super().__init__(parent)
        self._accent = accent or C["accent"]
        self._header = header
        self._path:          Optional[str]        = None
        self._data:          Optional[np.ndarray] = None
        self._sr             = 44100
        self._playing        = False
        self._cursor         = 0
        self._vol_value      = 1.0
        self._stream         = None
        self._info_channels  = 1
        self._info_subtype   = ""
        self._timer          = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(4)

        if self._header:
            hdr = QHBoxLayout(); hdr.setSpacing(8)
            lbl = QLabel(self._header)
            lbl.setStyleSheet(f"color:{C['text3']};font-size:9px;letter-spacing:2px;"
                              f"background-color:transparent;border:none;font-weight:600;")
            sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet(f"background-color:{C['border']};")
            hdr.addWidget(lbl); hdr.addWidget(sep, 1)
            lay.addLayout(hdr)

        self._wave = WaveformWidget(h=62)
        self._wave.seeked.connect(self._seek)
        lay.addWidget(self._wave)

        ctrl = QHBoxLayout(); ctrl.setSpacing(6)

        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setFixedHeight(28)
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._toggle)
        self._play_btn.setStyleSheet(f"""
QPushButton {{
    background-color:#2a2a2a; border:1px solid {self._accent};
    border-radius:4px; color:{self._accent};
    font-size:12px; font-weight:600; padding:4px 14px;
}}
QPushButton:hover  {{ background-color:#333; color:white; border-color:{self._accent}; }}
QPushButton:disabled {{ color:{C["text3"]}; border-color:{C["border"]}; background-color:{C["surface"]}; }}
""")

        self._time_lbl = QLabel("—")
        self._time_lbl.setStyleSheet(
            f"color:{C['text3']};font-size:10px;font-family:'Consolas',monospace;background-color:transparent;")
        self._file_lbl = QLabel("")
        self._file_lbl.setStyleSheet(f"color:{C['text2']};font-size:10px;background-color:transparent;")
        self._file_lbl.setMaximumWidth(220)

        vol_icon = QLabel("🔊")
        vol_icon.setStyleSheet("font-size:11px;background-color:transparent;border:none;")
        self._vol = QSlider(Qt.Orientation.Horizontal)
        self._vol.setRange(0, 150); self._vol.setValue(100); self._vol.setFixedWidth(80)
        self._vol.valueChanged.connect(lambda v: setattr(self, "_vol_value", v / 100.0))

        ctrl.addWidget(self._play_btn)
        ctrl.addWidget(self._time_lbl)
        ctrl.addStretch()
        ctrl.addWidget(self._file_lbl)
        ctrl.addWidget(vol_icon)
        ctrl.addWidget(self._vol)
        lay.addLayout(ctrl)

    def load(self, path: str):
        self._stop_now()
        self._info_channels = 1
        self._info_subtype  = ""
        try:
            fi = sf.info(path)
            self._info_channels = fi.channels
            self._info_subtype  = fi.subtype
        except Exception:
            pass
        try:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self._data = audio; self._sr = sr; self._path = path; self._cursor = 0
            self._wave.set_audio(audio)
            dur = len(audio) / max(1, sr)
            self._time_lbl.setText(f"0:00 / {_fmt(dur)}")
            self._time_lbl.setStyleSheet(
                f"color:{C['text2']};font-size:10px;font-family:'Consolas',monospace;background-color:transparent;")
            self._file_lbl.setText(Path(path).name)
            self._play_btn.setEnabled(True)
        except Exception as ex:
            self._time_lbl.setText(f"Error: {ex}")
            logger.error(f"MiniPlayer.load error: {traceback.format_exc()}")

    def clear(self):
        self._stop_now()
        self._data = None; self._path = None
        self._wave.clear()
        self._play_btn.setEnabled(False)
        self._time_lbl.setText("—")
        self._time_lbl.setStyleSheet(
            f"color:{C['text3']};font-size:10px;font-family:'Consolas',monospace;background-color:transparent;")
        self._file_lbl.setText("")

    def get_path(self) -> Optional[str]:
        return self._path

    def get_info(self) -> Optional[str]:
        if self._data is None or self._path is None:
            return None
        _bits = {
            "PCM_16": "16-bit", "PCM_24": "24-bit", "PCM_32": "32-bit",
            "FLOAT": "32-bit float", "DOUBLE": "64-bit float",
            "PCM_S8": "8-bit", "PCM_U8": "8-bit",
        }
        dur      = len(self._data) / max(1, self._sr)
        size_kb  = Path(self._path).stat().st_size // 1024 if Path(self._path).exists() else 0
        ch_str   = "Mono" if self._info_channels == 1 else "Stereo" if self._info_channels == 2 else f"{self._info_channels}ch"
        bits_str = _bits.get(self._info_subtype, "")
        parts    = [Path(self._path).suffix.upper().lstrip("."), f"{self._sr / 1000:.1f} kHz", ch_str]
        if bits_str:
            parts.append(bits_str)
        parts += [_fmt(dur), f"{size_kb} KB"]
        return "  ·  ".join(parts)

    def _toggle(self):
        if self._playing: self._pause()
        else:             self._play()

    def _play(self):
        if self._data is None: return
        self._playing = True; self._play_btn.setText("■  Pause")
        audio = self._data.astype(np.float32)
        chunk = audio[self._cursor:] if self._cursor < len(audio) else audio
        if not len(chunk): self._cursor = 0; chunk = audio

        def cb(out, frames, _ti, _st):
            nonlocal chunk
            if not self._playing: raise sd.CallbackStop()
            n = min(frames, len(chunk))
            if n == 0: out[:] = 0; raise sd.CallbackStop()
            out[:n, 0] = chunk[:n] * self._vol_value
            if frames > n: out[n:] = 0
            chunk = chunk[n:]; self._cursor += n

        try:
            self._stream = sd.OutputStream(
                samplerate=self._sr, channels=1, dtype="float32",
                callback=cb, finished_callback=self._on_end)
            self._stream.start(); self._timer.start()
        except Exception:
            self._playing = False; self._play_btn.setText("▶  Play")

    def _pause(self):
        def _safe():
            self._playing = False; self._play_btn.setText("▶  Play"); self._timer.stop()
            if self._stream:
                try: self._stream.stop()
                except Exception: pass
        QTimer.singleShot(0, _safe)

    def _stop_now(self):
        def _safe():
            self._playing = False
            if hasattr(self, "_play_btn"): self._play_btn.setText("▶  Play")
            if hasattr(self, "_timer"):    self._timer.stop()
            if self._stream:
                try: self._stream.stop(); self._stream.close()
                except Exception: pass
                self._stream = None
            self._cursor = 0
            if hasattr(self, "_wave"): self._wave.set_position(0.0)
        QTimer.singleShot(0, _safe)

    def _on_end(self):
        def _safe():
            self._playing = False; self._play_btn.setText("▶  Play"); self._timer.stop()
            self._cursor = 0; self._wave.set_position(0.0)
            if self._data is not None:
                self._time_lbl.setText(f"0:00 / {_fmt(len(self._data) / max(1, self._sr))}")
        QTimer.singleShot(0, _safe)

    def _tick(self):
        if self._data is None: return
        frac = self._cursor / max(1, len(self._data))
        self._wave.set_position(frac)
        self._time_lbl.setText(
            f"{_fmt(self._cursor / max(1, self._sr))} / "
            f"{_fmt(len(self._data) / max(1, self._sr))}")

    def _seek(self, frac: float):
        if self._data is None: return
        was = self._playing; self._pause()
        self._cursor = int(frac * len(self._data))
        self._wave.set_position(frac)
        if was: QTimer.singleShot(80, self._play)


# ════════════════════════════════════════════════════════════════════════════
#  DropZone
# ════════════════════════════════════════════════════════════════════════════
class DropZone(QFrame):
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(88)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._file:  Optional[str] = None
        self._hover: bool          = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QVBoxLayout(self); lay.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.setSpacing(3)
        self._icon_lbl = QLabel("🎵")
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_lbl.setStyleSheet("font-size:22px;background-color:transparent;border:none;")
        self._main_lbl = QLabel("Drop audio / video here  or  click to browse")
        self._main_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main_lbl.setStyleSheet(
            f"color:{C['text3']};font-size:12px;background-color:transparent;border:none;")
        self._sub_lbl = QLabel("WAV · MP3 · FLAC · OGG · MP4 · MKV · MOV …")
        self._sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub_lbl.setStyleSheet(
            f"color:{C['text3']};font-size:10px;background-color:transparent;border:none;letter-spacing:1px;")
        lay.addWidget(self._icon_lbl)
        lay.addWidget(self._main_lbl)
        lay.addWidget(self._sub_lbl)
        self._refresh_style()

    def _refresh_style(self):
        if self._file:
            b, bg = C["accent"], C["bg"]
        elif self._hover:
            b, bg = C["accent"], "#0a0a14"
        else:
            b, bg = C["border"], C["surface"]
        self.setStyleSheet(f"QFrame{{background-color:{bg};border:2px dashed {b};border-radius:7px;}}")

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._hover = True; self._refresh_style()

    def dragLeaveEvent(self, _):
        self._hover = False; self._refresh_style()

    def dropEvent(self, e: QDropEvent):
        self._hover = False
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).suffix.lower() in AUDIO_EXTS | VIDEO_EXTS:
                self._set_file(p); break
        self._refresh_style()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._browse()

    def _browse(self):
        exts = " ".join(f"*{x}" for x in sorted(AUDIO_EXTS | VIDEO_EXTS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio / Video", str(Path.home()),
            f"Audio / Video ({exts});;All Files (*.*)")
        if path: self._set_file(path)

    def _set_file(self, path: str):
        ext = Path(path).suffix.lower()
        if ext not in AUDIO_EXTS and ext not in VIDEO_EXTS: return
        self._file = path
        is_video   = ext in VIDEO_EXTS
        icon       = "🎬" if is_video else "🎵"
        try:
            size_str = f"{Path(path).stat().st_size / 1024 / 1024:.1f} MB"
        except Exception:
            size_str = ""
        self._icon_lbl.setText(icon)
        self._main_lbl.setText(f"{Path(path).name}")
        self._main_lbl.setStyleSheet(
            f"color:{C['text']};font-size:12px;background-color:transparent;border:none;font-weight:600;")
        self._sub_lbl.setText(f"{ext.upper().lstrip('.')}  ·  {size_str}")
        self._refresh_style()
        self.file_dropped.emit(path)

    def clear_file(self):
        self._file = None
        self._icon_lbl.setText("🎵")
        self._main_lbl.setText("Drop audio / video here  or  click to browse")
        self._main_lbl.setStyleSheet(
            f"color:{C['text3']};font-size:12px;background-color:transparent;border:none;")
        self._sub_lbl.setText("WAV · MP3 · FLAC · OGG · MP4 · MKV · MOV …")
        self._sub_lbl.setStyleSheet(
            f"color:{C['text3']};font-size:10px;background-color:transparent;border:none;letter-spacing:1px;")
        self._refresh_style()


# ════════════════════════════════════════════════════════════════════════════
#  Workers
# ════════════════════════════════════════════════════════════════════════════
class BaseWorker(QThread):
    status = pyqtSignal(str)
    error  = pyqtSignal(str)


class DownloadModelWorker(BaseWorker):
    finished = pyqtSignal()

    def __init__(self, backend, model_dir: Path):
        super().__init__()
        self.backend   = backend
        self.model_dir = model_dir

    def run(self):
        try:
            self.backend.download(self.model_dir, progress_cb=self.status.emit)
            self.finished.emit()
        except Exception:
            self.error.emit(traceback.format_exc())


class VideoExtractWorker(BaseWorker):
    finished = pyqtSignal(str)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path

    def run(self):
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="reuse_vid_")
            os.close(fd)
            self.status.emit(f"Extracting audio from: {Path(self.video_path).name}…")
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", self.video_path,
                 "-ac", "1",
                 "-vn", tmp],
                capture_output=True, timeout=900,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed:\n{r.stderr.decode(errors='replace')[-500:]}")
            self.status.emit("✅ Audio extracted from video.")
            self.finished.emit(tmp)
        except Exception:
            if tmp and Path(tmp).exists():
                try:
                    Path(tmp).unlink()
                except Exception:
                    pass
            self.error.emit(traceback.format_exc())


class DemucsWorker(BaseWorker):
    finished_2stem = pyqtSignal(str)
    finished_4stem = pyqtSignal(str, str)

    def __init__(self, input_path: str, out_dir: str,
                 model: str = "htdemucs_ft", four_stem: bool = False):
        super().__init__()
        self.input_path = input_path
        self.out_dir    = out_dir
        self.model      = model
        self.four_stem  = four_stem

    def run(self):
        try:
            cmd = [sys.executable, "-m", "demucs"]
            if not self.four_stem:
                cmd += ["--two-stems=vocals"]
            cmd += ["-n", self.model, "--out", self.out_dir, self.input_path]

            self.status.emit(
                f"Running Demucs ({self.model}, {'4-stem' if self.four_stem else '2-stem'})…"
                "  (first run downloads ~320 MB model)")
            r = subprocess.run(cmd, capture_output=True, timeout=7200)
            if r.returncode != 0:
                raise RuntimeError(
                    f"Demucs failed:\n{r.stderr.decode(errors='replace')[-700:]}")

            stem = Path(self.input_path).stem
            base = Path(self.out_dir) / self.model / stem

            vocals = base / "vocals.wav"
            if not vocals.exists():
                found = list(Path(self.out_dir).rglob("vocals.wav"))
                if not found:
                    raise RuntimeError("vocals.wav not found after Demucs run.")
                vocals = found[0]
                base = vocals.parent

            if not self.four_stem:
                self.status.emit("✅ Demucs complete (2-stem)")
                self.finished_2stem.emit(str(vocals))
                return

            stems_to_sum = ["drums", "bass", "other"]
            arrays = []
            out_sr = None
            for s in stems_to_sum:
                p = base / f"{s}.wav"
                if p.exists():
                    data, sr = sf.read(str(p), dtype="float32")
                    arrays.append(data)
                    out_sr = sr

            if not arrays:
                raise RuntimeError("Background stems (drums/bass/other) not found after 4-stem Demucs run.")

            background = np.sum(arrays, axis=0)
            bg_path = str(Path(self.out_dir) / f"background_{stem}.wav")
            sf.write(bg_path, background, out_sr)

            self.status.emit("✅ Demucs complete (4-stem)")
            self.finished_4stem.emit(str(vocals), bg_path)

        except Exception:
            self.error.emit(traceback.format_exc())


class BackgroundWorker(BaseWorker):
    finished = pyqtSignal(str)

    def __init__(self, bg_path: str, highpass_hz: float = 80.0,
                 threshold_db: float = -18.0, ratio: float = 2.5):
        super().__init__()
        self.bg_path      = bg_path
        self.highpass_hz  = highpass_hz
        self.threshold_db = threshold_db
        self.ratio        = ratio

    def run(self):
        try:
            if not _PEDALBOARD_OK:
                raise RuntimeError(
                    "pedalboard is not installed.\nRun: pip install pedalboard")
            self.status.emit("Processing background (EQ + Compression)…")
            audio, sr = sf.read(self.bg_path, dtype="float32")

            mono = audio.ndim == 1
            if mono:
                audio_pb = audio[np.newaxis, :]
            else:
                audio_pb = audio.T

            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=self.highpass_hz),
                Compressor(
                    threshold_db=self.threshold_db,
                    ratio=self.ratio,
                    attack_ms=10.0,
                    release_ms=200.0,
                ),
            ])

            processed_pb = board(audio_pb, sr)

            if mono:
                processed = processed_pb[0]
            else:
                processed = processed_pb.T

            stem     = Path(self.bg_path).stem
            out_path = str(OUTPUT_DIR / f"bg_processed_{stem}.wav")
            sf.write(out_path, processed, sr)

            self.status.emit("✅ Background processing complete.")
            self.finished.emit(out_path)

        except Exception:
            self.error.emit(traceback.format_exc())


class MixWorker(BaseWorker):
    finished = pyqtSignal(str)

    def __init__(self, vocals_path: str, bg_path: str,
                 vocal_lufs: float = -14.0, bg_lufs: float = -19.0,
                 stem_name: str = "output", use_lufs: bool = True):
        super().__init__()
        self.vocals_path = vocals_path
        self.bg_path     = bg_path
        self.vocal_lufs  = vocal_lufs
        self.bg_lufs     = bg_lufs
        self.stem_name   = stem_name
        self.use_lufs    = use_lufs

    @staticmethod
    def _to_stereo(a: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            return np.stack([a, a], axis=1)
        if a.shape[1] == 1:
            return np.concatenate([a, a], axis=1)
        return a

    @staticmethod
    def _normalize_lufs(audio: np.ndarray, target: float, sr: int) -> np.ndarray:
        if not _PYLOUDNORM_OK:
            return audio
        try:
            meter    = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            if not (np.isinf(loudness) or np.isnan(loudness)):
                gain = 10 ** ((target - loudness) / 20)
                return audio * gain
        except Exception:
            pass
        return audio

    def run(self):
        try:
            if not _PEDALBOARD_OK:
                raise RuntimeError(
                    "pedalboard is not installed.\nRun: pip install pedalboard")
            self.status.emit("Creating final mix…")

            vocals, sr_v = sf.read(self.vocals_path, dtype="float32")
            bg,     sr_b = sf.read(self.bg_path,     dtype="float32")

            if sr_v != sr_b:
                if _TORCH_OK:
                    bg_t = torch.from_numpy(bg.T if bg.ndim > 1 else bg[np.newaxis]).float()
                    bg_t = torchaudio.functional.resample(bg_t, sr_b, sr_v)
                    bg   = bg_t.numpy().T if bg.ndim > 1 else bg_t.numpy()[0]
                    sr_b = sr_v
                else:
                    raise RuntimeError(
                        f"Sample rate mismatch: vocals {sr_v} Hz, background {sr_b} Hz.\n"
                        "Install torch for automatic resampling: pip install torch torchaudio")

            sr = sr_v

            vocals = self._to_stereo(vocals)
            bg     = self._to_stereo(bg)

            if self.use_lufs:
                vocals = self._normalize_lufs(vocals, self.vocal_lufs, sr)
                bg     = self._normalize_lufs(bg,     self.bg_lufs,    sr)

            min_len = min(len(vocals), len(bg))
            mix = vocals[:min_len] + bg[:min_len]

            board = Pedalboard([Limiter(threshold_db=-1.0, release_ms=50.0)])
            mix = board(mix.T, sr).T

            out_path = str(OUTPUT_DIR / f"final_mix_{self.stem_name}.wav")
            sf.write(out_path, mix, sr, subtype="PCM_24")

            self.status.emit("✅ Final mix complete.")
            self.finished.emit(out_path)

        except Exception:
            self.error.emit(traceback.format_exc())


class BackendWorker(BaseWorker):
    finished   = pyqtSignal(str)
    log_output = pyqtSignal(str)

    def __init__(self, backend, input_path: str, output_dir: Path, params: dict):
        super().__init__()
        self.backend    = backend
        self.input_path = input_path
        self.output_dir = output_dir
        self.params     = params

    def run(self):
        try:
            request = EnhancementRequest(
                input_path=self.input_path,
                output_dir=str(self.output_dir),
                params=self.params,
            )
            result = self.backend.process(
                request,
                progress_cb=self.status.emit,
                log_cb=self.log_output.emit,
            )
            self.finished.emit(result.output_path)
        except Exception:
            self.error.emit(traceback.format_exc())


class BatchFileItem(QWidget):
    removed       = pyqtSignal(str)
    check_changed = pyqtSignal()
    clicked       = pyqtSignal(str)
    shift_clicked = pyqtSignal(str)

    _STATUS_COLOR = {
        "pending":    C["text3"],
        "processing": C["accent"],
        "done":       C["green"],
        "error":      C["error"],
    }
    _STATUS_ICON = {
        "pending":    "○",
        "processing": "◉",
        "done":       "✓",
        "error":      "✗",
    }

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path     = path
        self._status   = "pending"
        self._was_done = False
        self._selected = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(6)

        self._chk = QCheckBox()
        self._chk.setChecked(True)
        self._chk.setFixedWidth(20)
        self._chk.stateChanged.connect(self._on_check_state_changed)
        self._chk.installEventFilter(self)

        self._icon_lbl = QLabel("○")
        self._icon_lbl.setFixedWidth(14)
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_lbl.setStyleSheet(
            f"color:{C['text3']};font-size:12px;background-color:transparent;")
        self._icon_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self._icon_lbl.installEventFilter(self)

        self._name_lbl = QLabel(Path(path).name)
        self._name_lbl.setStyleSheet(
            f"color:{C['text']};font-size:11px;background-color:transparent;")
        self._name_lbl.setToolTip(path)
        self._name_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        self._name_lbl.installEventFilter(self)

        btn = QPushButton("✕")
        btn.setFixedSize(18, 18)
        btn.setStyleSheet(
            f"QPushButton{{background-color:transparent;border:none;"
            f"color:{C['text3']};font-size:10px;padding:0;}}"
            f"QPushButton:hover{{color:{C['error']};}}")
        btn.clicked.connect(lambda: self.removed.emit(self._path))

        lay.addWidget(self._chk)
        lay.addWidget(self._icon_lbl)
        lay.addWidget(self._name_lbl, 1)
        lay.addWidget(btn)

        self._refresh_bg()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.shift_clicked.emit(self._path)
                return
        super().mousePressEvent(e)

    def eventFilter(self, obj, e):
        if obj is self._chk:
            if e.type() == e.Type.MouseButtonPress and e.button() == Qt.MouseButton.LeftButton:
                if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.shift_clicked.emit(self._path)
                    return True
            return False
        if obj in (self._name_lbl, self._icon_lbl):
            if e.type() == e.Type.MouseButtonPress and e.button() == Qt.MouseButton.LeftButton:
                if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.shift_clicked.emit(self._path)
                else:
                    self.clicked.emit(self._path)
                return True
        return super().eventFilter(obj, e)

    def set_selected(self, selected: bool):
        self._selected = selected
        self._refresh_bg()

    def _refresh_bg(self):
        if self._selected:
            self.setStyleSheet(
                f"background-color:{C['accent_dim']};border-radius:3px;")
        else:
            self.setStyleSheet("background-color:transparent;")

    def _on_check_state_changed(self, state):
        if state:
            if self._status == "done":
                self._status = "pending"
                self._icon_lbl.setText(self._STATUS_ICON["pending"])
                self._icon_lbl.setStyleSheet(
                    f"color:{self._STATUS_COLOR['pending']};font-size:12px;background-color:transparent;")
                self._name_lbl.setStyleSheet(
                    f"color:{C['text2']};font-size:11px;background-color:transparent;")
        else:
            if self._status == "pending" and self._was_done:
                self._status = "done"
                self._icon_lbl.setText(self._STATUS_ICON["done"])
                self._icon_lbl.setStyleSheet(
                    f"color:{self._STATUS_COLOR['done']};font-size:12px;background-color:transparent;")
                self._name_lbl.setStyleSheet(
                    f"color:{C['green']};font-size:11px;background-color:transparent;")
        self.check_changed.emit()

    def set_status(self, status: str):
        self._status = status
        if status == "done":
            self._was_done = True
        color = self._STATUS_COLOR.get(status, C["text3"])
        icon  = self._STATUS_ICON.get(status, "○")
        self._icon_lbl.setText(icon)
        self._icon_lbl.setStyleSheet(
            f"color:{color};font-size:12px;background-color:transparent;")
        if status == "done":
            nc = C["green"]
            self._chk.blockSignals(True)
            self._chk.setChecked(False)
            self._chk.blockSignals(False)
        elif status == "error":
            nc = C["error"]
        else:
            nc = C["text2"]
        self._name_lbl.setStyleSheet(
            f"color:{nc};font-size:11px;background-color:transparent;")

    def is_checked(self) -> bool:
        return self._chk.isChecked()

    def set_checked(self, checked: bool):
        self._chk.setChecked(checked)

    @property
    def path(self) -> str:
        return self._path

    @property
    def status(self) -> str:
        return self._status
        
        
class BatchPanel(QWidget):
    files_changed   = pyqtSignal()
    paths_requested = pyqtSignal(list)
    file_selected   = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items:             dict          = {}
        self._session_name:      str           = ""
        self._selected_path:     Optional[str] = None
        self._last_clicked_path: Optional[str] = None
        self._drop_hover:        bool          = False
        self.setAcceptDrops(True)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(6)

        title = QLabel("FILE QUEUE")
        title.setStyleSheet(
            f"color:{C['text2']};font-size:10px;letter-spacing:3px;"
            f"font-weight:600;background-color:transparent;")
        lay.addWidget(title)

        btn_row = QHBoxLayout(); btn_row.setSpacing(4)
        btn_files = QPushButton("+ Files")
        btn_files.setFixedHeight(26)
        btn_files.setToolTip("Add individual audio files to the queue")
        btn_files.clicked.connect(self._add_files_dialog)
        btn_folder = QPushButton("📁 Folder")
        btn_folder.setFixedHeight(26)
        btn_folder.setToolTip("Add all audio files from a folder (recursive)")
        btn_folder.clicked.connect(self._add_folder_dialog)
        btn_clear = QPushButton("Clear")
        btn_clear.setFixedHeight(26)
        btn_clear.clicked.connect(self._clear_all)
        btn_row.addWidget(btn_files)
        btn_row.addWidget(btn_folder)
        btn_row.addWidget(btn_clear)
        lay.addLayout(btn_row)

        sel_row = QHBoxLayout(); sel_row.setSpacing(4)
        btn_sel_all = QPushButton("Select All")
        btn_sel_all.setFixedHeight(22)
        btn_sel_all.setStyleSheet(
            f"QPushButton{{background-color:{C['surface']};border:1px solid {C['border']};"
            f"border-radius:3px;color:{C['text2']};font-size:11px;padding:2px 8px;}}"
            f"QPushButton:hover{{background-color:#2a2a2a;color:{C['text']};}}")
        btn_sel_all.clicked.connect(self.select_all)
        btn_sel_unprocessed = QPushButton("Select Unprocessed")
        btn_sel_unprocessed.setFixedHeight(22)
        btn_sel_unprocessed.setStyleSheet(
            f"QPushButton{{background-color:{C['surface']};border:1px solid {C['border']};"
            f"border-radius:3px;color:{C['text2']};font-size:11px;padding:2px 8px;}}"
            f"QPushButton:hover{{background-color:#2a2a2a;color:{C['text']};}}")
        btn_sel_unprocessed.clicked.connect(self.select_unprocessed)
        btn_desel_all = QPushButton("Deselect All")
        btn_desel_all.setFixedHeight(22)
        btn_desel_all.setStyleSheet(
            f"QPushButton{{background-color:{C['surface']};border:1px solid {C['border']};"
            f"border-radius:3px;color:{C['text2']};font-size:11px;padding:2px 8px;}}"
            f"QPushButton:hover{{background-color:#2a2a2a;color:{C['text']};}}")
        btn_desel_all.clicked.connect(self.deselect_all)
        sel_row.addWidget(btn_sel_all)
        sel_row.addWidget(btn_sel_unprocessed)
        sel_row.addWidget(btn_desel_all)
        lay.addLayout(sel_row)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setAcceptDrops(True)
        self._scroll.setStyleSheet(
            f"QScrollArea{{background-color:{C['surface']};"
            f"border:1px solid {C['border']};border-radius:4px;}}")

        self._list_w   = QWidget()
        self._list_w.setAcceptDrops(True)
        self._list_w.installEventFilter(self)
        self._list_lay = QVBoxLayout(self._list_w)
        self._list_lay.setContentsMargins(2, 2, 2, 2)
        self._list_lay.setSpacing(1)
        self._list_lay.addStretch()
        self._scroll.setWidget(self._list_w)
        lay.addWidget(self._scroll, 1)

        self._drop_overlay = QLabel("Drop files here")
        self._drop_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_overlay.setStyleSheet(
            f"color:{C['accent']};font-size:13px;font-weight:600;"
            f"background-color:{C['accent_dim']};"
            f"border:2px dashed {C['accent']};border-radius:4px;")
        self._drop_overlay.setVisible(False)
        lay.addWidget(self._drop_overlay)

        self._summary = QLabel("No files")
        self._summary.setStyleSheet(
            f"color:{C['text3']};font-size:10px;background-color:transparent;")
        lay.addWidget(self._summary)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in e.mimeData().urls()]
            if any(Path(p).suffix.lower() in AUDIO_EXTS | VIDEO_EXTS for p in paths):
                e.acceptProposedAction()
                self._drop_hover = True
                self._scroll.setStyleSheet(
                    f"QScrollArea{{background-color:{C['surface']};"
                    f"border:2px dashed {C['accent']};border-radius:4px;}}")
                self._drop_overlay.setVisible(True)
                return
        e.ignore()

    def dragLeaveEvent(self, _):
        self._drop_hover = False
        self._scroll.setStyleSheet(
            f"QScrollArea{{background-color:{C['surface']};"
            f"border:1px solid {C['border']};border-radius:4px;}}")
        self._drop_overlay.setVisible(False)

    def dropEvent(self, e: QDropEvent):
        self._drop_hover = False
        self._scroll.setStyleSheet(
            f"QScrollArea{{background-color:{C['surface']};"
            f"border:1px solid {C['border']};border-radius:4px;}}")
        self._drop_overlay.setVisible(False)
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        valid = [p for p in paths if Path(p).suffix.lower() in AUDIO_EXTS | VIDEO_EXTS]
        if valid:
            e.acceptProposedAction()
            self.paths_requested.emit(valid)

    def eventFilter(self, obj, e):
        if obj is self._list_w:
            if e.type() == e.Type.DragEnter:
                self.dragEnterEvent(e)
                return True
            if e.type() == e.Type.DragLeave:
                self.dragLeaveEvent(e)
                return True
            if e.type() == e.Type.Drop:
                self.dropEvent(e)
                return True
        return super().eventFilter(obj, e)

    def _add_files_dialog(self):
        exts = " ".join(f"*{x}" for x in sorted(AUDIO_EXTS))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Audio Files", str(Path.home()),
            f"Audio Files ({exts});;All Files (*.*)")
        if paths:
            self.paths_requested.emit(paths)

    def _add_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Audio Folder", str(Path.home()))
        if not folder:
            return
        fp    = Path(folder)
        paths = sorted(
            str(p) for p in fp.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
        if not paths:
            QMessageBox.information(self, "No audio files",
                f"No audio files found in:\n{folder}")
            return
        if not self._session_name:
            self._session_name = fp.name
        self.paths_requested.emit(paths)

    def add_files(self, paths: list):
        added = False
        for path in paths:
            if path not in self._items:
                item = BatchFileItem(path)
                item.removed.connect(self._remove_file)
                item.check_changed.connect(self.files_changed.emit)
                item.clicked.connect(self._on_item_clicked)
                item.shift_clicked.connect(self._on_item_shift_clicked)
                self._items[path] = item
                self._list_lay.insertWidget(self._list_lay.count() - 1, item)
                added = True
        if added:
            if not self._session_name and paths:
                self._session_name = Path(paths[0]).parent.name or "batch"
            self._update_summary()
            self.files_changed.emit()

    def _on_item_clicked(self, path: str):
        if self._selected_path and self._selected_path in self._items:
            self._items[self._selected_path].set_selected(False)
        self._selected_path = path
        if path in self._items:
            self._items[path].set_selected(True)
        self.file_selected.emit(path)
        
    def _on_item_shift_clicked(self, path: str):
        all_paths = list(self._items.keys())
        try:
            idx_b = all_paths.index(path)
        except ValueError:
            self._items[path].set_checked(True)
            self.files_changed.emit()
            return

        last_checked_idx = None
        for i, p in enumerate(all_paths):
            if i != idx_b and self._items[p].is_checked():
                last_checked_idx = i

        if last_checked_idx is None:
            self._items[path].set_checked(True)
            self.files_changed.emit()
            return

        lo, hi = min(last_checked_idx, idx_b), max(last_checked_idx, idx_b)
        for p in all_paths[lo:hi + 1]:
            self._items[p].set_checked(True)
        self.files_changed.emit()
    
    def set_selected_path(self, path: Optional[str]):
        if self._selected_path and self._selected_path in self._items:
            self._items[self._selected_path].set_selected(False)
        self._selected_path = path
        if path and path in self._items:
            self._items[path].set_selected(True)

    def get_selected_path(self) -> Optional[str]:
        return self._selected_path

    def get_checked_paths(self) -> List[str]:
        return [p for p, it in self._items.items() if it.is_checked()]

    def _remove_file(self, path: str):
        if path in self._items:
            item = self._items.pop(path)
            if self._selected_path == path:
                self._selected_path = None
            if self._last_clicked_path == path:
                self._last_clicked_path = None
            self._list_lay.removeWidget(item)
            item.deleteLater()
            self._update_summary()
            self.files_changed.emit()

    def _clear_all(self):
        if not self._items:
            return
        selected = self._selected_path
        checked  = [p for p, it in self._items.items() if it.is_checked()]

        msg = QMessageBox(self)
        msg.setWindowTitle("Clear Queue")

        if selected and selected in self._items and checked:
            msg.setText("What would you like to remove?")
            btn_one      = msg.addButton("Selected file only",       QMessageBox.ButtonRole.AcceptRole)
            btn_checked  = msg.addButton(f"Checked ({len(checked)})", QMessageBox.ButtonRole.DestructiveRole)
            btn_all      = msg.addButton("Everything",               QMessageBox.ButtonRole.DestructiveRole)
            btn_cancel   = msg.addButton("Cancel",                   QMessageBox.ButtonRole.RejectRole)
        elif checked:
            msg.setText("What would you like to remove?")
            btn_one      = None
            btn_checked  = msg.addButton(f"Checked ({len(checked)})", QMessageBox.ButtonRole.DestructiveRole)
            btn_all      = msg.addButton("Everything",               QMessageBox.ButtonRole.DestructiveRole)
            btn_cancel   = msg.addButton("Cancel",                   QMessageBox.ButtonRole.RejectRole)
        else:
            msg.setText("Remove all files from the queue?")
            btn_one      = None
            btn_checked  = None
            btn_all      = msg.addButton("Remove All",               QMessageBox.ButtonRole.DestructiveRole)
            btn_cancel   = msg.addButton("Cancel",                   QMessageBox.ButtonRole.RejectRole)

        msg.exec()
        clicked = msg.clickedButton()

        if clicked == btn_cancel or clicked is None:
            return

        if btn_one is not None and clicked == btn_one:
            item = self._items.pop(selected)
            self._list_lay.removeWidget(item)
            item.deleteLater()
            self._selected_path = None
            if self._last_clicked_path == selected:
                self._last_clicked_path = None
            self._update_summary()
            self.files_changed.emit()
            return

        if btn_checked is not None and clicked == btn_checked:
            for p in list(checked):
                if p in self._items:
                    item = self._items.pop(p)
                    if self._last_clicked_path == p:
                        self._last_clicked_path = None
                    self._list_lay.removeWidget(item)
                    item.deleteLater()
            if self._selected_path not in self._items:
                self._selected_path = None
            self._update_summary()
            self.files_changed.emit()
            return

        if clicked == btn_all:
            for item in list(self._items.values()):
                self._list_lay.removeWidget(item)
                item.deleteLater()
            self._items.clear()
            self._selected_path     = None
            self._last_clicked_path = None
            self._session_name      = ""
            self._update_summary()
            self.files_changed.emit()
        
    def select_all(self):
        for item in self._items.values():
            item.set_checked(True)
        self.files_changed.emit()

    def select_unprocessed(self):
        for item in self._items.values():
            item.set_checked(item.status not in ("done",))
        self.files_changed.emit()

    def deselect_all(self):
        for item in self._items.values():
            item.set_checked(False)
        self.files_changed.emit()

    def _update_summary(self):
        total = len(self._items)
        done  = sum(1 for it in self._items.values() if it.status == "done")
        if total == 0:
            self._summary.setStyleSheet(
                f"color:{C['text3']};font-size:10px;background-color:transparent;")
            self._summary.setText("No files")
        else:
            color = C["green"] if done == total else C["text3"]
            self._summary.setStyleSheet(
                f"color:{color};font-size:10px;background-color:transparent;")
            self._summary.setText(f"{done} / {total} processed")

    def update_eta(self, eta_str: str):
        total = len(self._items)
        done  = sum(1 for it in self._items.values() if it.status == "done")
        if total == 0:
            return
        color = C["green"] if done == total else C["text3"]
        self._summary.setStyleSheet(
            f"color:{color};font-size:10px;background-color:transparent;")
        label = f"{done} / {total} processed"
        if eta_str:
            label += f"  ·  ETA {eta_str}"
        self._summary.setText(label)

    def mark_processing(self, path: str):
        if path in self._items:
            self._items[path].set_status("processing")

    def mark_done(self, path: str):
        if path in self._items:
            self._items[path].set_status("done")
            self._update_summary()
            self.files_changed.emit()

    def mark_error(self, path: str):
        if path in self._items:
            self._items[path].set_status("error")
            self._update_summary()

    def get_pending_files(self) -> List[str]:
        return [p for p, it in self._items.items()
                if it.is_checked() and it.status in ("pending", "error")]

    def has_pending_files(self) -> bool:
        return any(it.is_checked() and it.status in ("pending", "error")
                   for it in self._items.values())

    def has_checked_files(self) -> bool:
        return any(it.is_checked() for it in self._items.values())

    def has_files(self) -> bool:
        return bool(self._items)

    @property
    def session_name(self) -> str:
        return self._session_name or "batch"
        
        
# ════════════════════════════════════════════════════════════════════════════
#  InfoBadge
# ════════════════════════════════════════════════════════════════════════════
class InfoBadge(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self.setStyleSheet(
            f"background-color:{C['panel']}; border:1px solid {C['border']};"
            f"border-radius:3px; color:{C['text2']}; font-size:10px;"
            f"padding:2px 8px; letter-spacing:0.5px;")

    def set_info(self, text: str):
        self.setText(text)
        self.setVisible(bool(text))

    def clear_info(self):
        self.setText(""); self.setVisible(False)


# ════════════════════════════════════════════════════════════════════════════
#  MainWindow
# ════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._backends = {cls().model_id: cls() for cls in _ACTIVE_BACKEND_CLASSES}
        first_backend = next(iter(self._backends.values()))

        self.setWindowTitle(first_backend.header_title)
        self.resize(1300, 840)
        self.setMinimumSize(920, 660)
        self.setStyleSheet(f"background-color: {C['bg']};")
        _ico = APP_DIR / "icon.ico"
        if _ico.exists():
            self.setWindowIcon(QIcon(str(_ico)))

        self._batch_panel = BatchPanel()
        self._batch_panel.files_changed.connect(self._update_enhance_btn)
        self._batch_panel.paths_requested.connect(self._add_paths)
        self._batch_panel.file_selected.connect(self._on_batch_file_selected)

        self._current_model_id       = first_backend.model_id
        self._current_path           = None
        self._worker                 = None
        self._output_dir             = OUTPUT_DIR
        self._batch_start_time       = None
        self._batch_file_count       = 0
        self._batch_done_count       = 0
        self._demucs_bg_path         = None
        self._demucs_enhanced_vocals = None
        self._processed_map: dict = {}

        self._build_ui()
        self._refresh_model_status()

    @property
    def _backend(self):
        return self._backends[self._current_model_id]

    # ── UI build ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(10)

        title_row = QHBoxLayout()
        title_lbl = QLabel(self._backend.header_title)
        self._title_lbl = title_lbl
        title_lbl.setStyleSheet(f"color:{C['text']};font-size:16px;font-weight:700;letter-spacing:1px;background-color:transparent;")
        subtitle = QLabel("by Mubumbutu")
        subtitle.setStyleSheet(f"color:{C['text3']};font-size:11px;letter-spacing:1px;padding-left:6px;background-color:transparent;")
        self._model_status_lbl = QLabel("Model: not downloaded")
        self._model_status_lbl.setStyleSheet(f"color:{C['warning']};font-size:11px;background-color:transparent;")
        self._btn_download = QPushButton("⬇ Download Model")
        self._btn_download.setStyleSheet(DOWNLOAD_BTN)
        self._btn_download.clicked.connect(self._download_model)
        title_row.addWidget(title_lbl)
        title_row.addWidget(subtitle)
        title_row.addStretch()
        title_row.addWidget(self._model_status_lbl)
        title_row.addWidget(self._btn_download)
        root.addLayout(title_row)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background-color:{C['border']};")
        root.addWidget(sep)

        content_lay = QHBoxLayout()
        content_lay.setSpacing(8)

        batch_widget = QWidget()
        batch_lay = QVBoxLayout(batch_widget)
        batch_lay.setContentsMargins(0, 0, 0, 0)
        batch_lay.addWidget(self._batch_panel)
        content_lay.addWidget(batch_widget, 0)

        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(2)

        in_panel = QWidget()
        in_lay = QVBoxLayout(in_panel)
        in_lay.setContentsMargins(0, 0, 6, 0); in_lay.setSpacing(8)
        self._in_player = MiniPlayer("ORIGINAL AUDIO", accent=C["border2"])
        in_lay.addWidget(self._in_player)
        self._in_badge = InfoBadge()
        in_lay.addWidget(self._in_badge)
        in_lay.addStretch()
        h_split.addWidget(in_panel)

        out_panel = QWidget()
        out_lay = QVBoxLayout(out_panel)
        out_lay.setContentsMargins(6, 0, 0, 0); out_lay.setSpacing(8)
        out_title = QLabel("LAST PROCESSED")
        out_title.setStyleSheet(f"color:{C['text2']};font-size:10px;letter-spacing:3px;font-weight:600;background-color:transparent;")
        out_lay.addWidget(out_title)
        self._out_player = MiniPlayer("ENHANCED AUDIO", accent=C["success"])
        out_lay.addWidget(self._out_player)
        self._out_badge = InfoBadge()
        out_lay.addWidget(self._out_badge)
        btns_row = QHBoxLayout(); btns_row.setSpacing(8)
        self._btn_open_dir = QPushButton("📂 Open Output Folder")
        self._btn_open_dir.clicked.connect(lambda: _open_path(str(self._output_dir)))
        btns_row.addWidget(self._btn_open_dir)
        btns_row.addStretch()
        out_lay.addLayout(btns_row)
        out_lay.addStretch()
        h_split.addWidget(out_panel)
        h_split.setSizes([400, 400])

        content_lay.addWidget(h_split, 1)
        root.addLayout(content_lay, 1)

        opts_grp = QGroupBox("PROCESSING OPTIONS")
        opts_lay = QVBoxLayout(opts_grp)
        opts_lay.setSpacing(8)

        self._build_dynamic_params_widget(opts_lay)

        out_settings_row = QHBoxLayout()
        out_settings_row.setSpacing(10)
        fmt_lbl = QLabel("Output Format:")
        fmt_lbl.setStyleSheet(f"color:{C['text2']};font-size:12px;background-color:transparent;")
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItem("WAV", "wav")
        self._fmt_combo.addItem("FLAC", "flac")
        self._fmt_combo.addItem("MP3", "mp3")
        self._fmt_combo.addItem("OGG", "ogg")
        self._fmt_combo.setFixedWidth(90)
        out_folder_lbl = QLabel("Save to:")
        out_folder_lbl.setStyleSheet(f"color:{C['text2']};font-size:12px;background-color:transparent;")
        self._out_dir_display = QLabel(str(self._output_dir))
        self._out_dir_display.setStyleSheet(
            f"color:{C['text2']};font-size:11px;background-color:transparent;")
        self._out_dir_display.setToolTip(str(self._output_dir))
        self._btn_browse_out = QPushButton("📁 Change Folder")
        self._btn_browse_out.setFixedHeight(26)
        self._btn_browse_out.clicked.connect(self._choose_output_dir)
        out_settings_row.addWidget(fmt_lbl)
        out_settings_row.addWidget(self._fmt_combo)
        out_settings_row.addSpacing(12)
        out_settings_row.addWidget(out_folder_lbl)
        out_settings_row.addWidget(self._out_dir_display, 1)
        out_settings_row.addWidget(self._btn_browse_out)
        opts_lay.addLayout(out_settings_row)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet(f"background-color:{C['border']};")
        opts_lay.addWidget(sep1)

        row2 = QHBoxLayout(); row2.setSpacing(14)
        self._chk_demucs = QCheckBox("🎙 Vocal isolation (Demucs · htdemucs_ft)")
        self._chk_demucs.setToolTip("Separate vocals from background before enhancement.")
        self._chk_demucs.toggled.connect(self._on_demucs_toggled)
        row2.addWidget(self._chk_demucs)
        self._stems_lbl = QLabel("Stems:")
        self._stems_lbl.setStyleSheet(f"color:{C['text2']};font-size:12px;background-color:transparent;")
        self._stems_combo = QComboBox()
        self._stems_combo.addItem("2-stem (vocals / rest)", 2)
        self._stems_combo.addItem("4-stem (vocals / drums / bass / other)", 4)
        self._stems_combo.setFixedWidth(240)
        self._stems_combo.currentIndexChanged.connect(self._on_stems_changed)
        row2.addWidget(self._stems_lbl)
        row2.addWidget(self._stems_combo)
        row2.addStretch()
        opts_lay.addLayout(row2)

        self._row3_widget = QWidget()
        row3_vlay = QVBoxLayout(self._row3_widget)
        row3_vlay.setContentsMargins(0, 0, 0, 0)
        row3_vlay.setSpacing(6)
        row3a = QHBoxLayout(); row3a.setSpacing(14)
        self._chk_bg_process = QCheckBox("🎚 Process background (EQ + Compression)")
        self._chk_bg_process.toggled.connect(self._on_bg_process_toggled)
        row3a.addWidget(self._chk_bg_process)
        self._chk_final_mix = QCheckBox("🎛 Create final mix (vocals + background)")
        self._chk_final_mix.toggled.connect(self._on_final_mix_toggled)
        row3a.addWidget(self._chk_final_mix)
        row3a.addStretch()
        row3_vlay.addLayout(row3a)

        self._row3b_widget = QWidget()
        row3b = QHBoxLayout(self._row3b_widget)
        row3b.setContentsMargins(24, 0, 0, 0); row3b.setSpacing(14)
        hp_lbl = QLabel("Highpass:"); hp_lbl.setStyleSheet(f"color:{C['text2']};font-size:12px;background-color:transparent;")
        self._spinbox_highpass = QSpinBox(); self._spinbox_highpass.setRange(20, 500); self._spinbox_highpass.setValue(80); self._spinbox_highpass.setSuffix(" Hz"); self._spinbox_highpass.setFixedWidth(115)
        thr_lbl = QLabel("Threshold:"); thr_lbl.setStyleSheet(f"color:{C['text2']};font-size:12px;background-color:transparent;")
        self._spinbox_threshold = QSpinBox(); self._spinbox_threshold.setRange(-60, 0); self._spinbox_threshold.setValue(-18); self._spinbox_threshold.setSuffix(" dB"); self._spinbox_threshold.setFixedWidth(115)
        ratio_lbl = QLabel("Ratio:"); ratio_lbl.setStyleSheet(f"color:{C['text2']};font-size:12px;background-color:transparent;")
        self._spinbox_ratio = QDoubleSpinBox(); self._spinbox_ratio.setRange(1.0, 20.0); self._spinbox_ratio.setValue(2.5); self._spinbox_ratio.setSingleStep(0.5); self._spinbox_ratio.setDecimals(1); self._spinbox_ratio.setSuffix(":1"); self._spinbox_ratio.setFixedWidth(100)
        row3b.addWidget(hp_lbl); row3b.addWidget(self._spinbox_highpass)
        row3b.addWidget(thr_lbl); row3b.addWidget(self._spinbox_threshold)
        row3b.addWidget(ratio_lbl); row3b.addWidget(self._spinbox_ratio)
        row3b.addStretch()
        self._row3b_widget.setVisible(False)
        row3_vlay.addWidget(self._row3b_widget)
        self._row3_widget.setVisible(False)
        opts_lay.addWidget(self._row3_widget)

        self._row4_widget = QWidget()
        row4 = QHBoxLayout(self._row4_widget)
        row4.setContentsMargins(20, 0, 0, 0); row4.setSpacing(20)
        self._chk_lufs = QCheckBox("LUFS normalization")
        self._chk_lufs.setChecked(True)
        self._chk_lufs.toggled.connect(self._on_lufs_toggled)
        row4.addWidget(self._chk_lufs)
        vocal_lbl = QLabel("Vocals:"); vocal_lbl.setStyleSheet(f"color:{C['text2']};font-size:12px;background-color:transparent;")
        self._vocal_lufs_slider = QSlider(Qt.Orientation.Horizontal); self._vocal_lufs_slider.setRange(10, 30); self._vocal_lufs_slider.setValue(14); self._vocal_lufs_slider.setFixedWidth(130)
        self._vocal_lufs_lbl = QLabel("-14 LUFS"); self._vocal_lufs_lbl.setStyleSheet(f"color:{C['accent']};font-size:11px;font-family:'Consolas',monospace;background-color:transparent;min-width:70px;")
        self._vocal_lufs_slider.valueChanged.connect(lambda v: self._vocal_lufs_lbl.setText(f"-{v} LUFS"))
        bg_lbl = QLabel("Background:"); bg_lbl.setStyleSheet(f"color:{C['text2']};font-size:12px;background-color:transparent;")
        self._bg_lufs_slider = QSlider(Qt.Orientation.Horizontal); self._bg_lufs_slider.setRange(14, 34); self._bg_lufs_slider.setValue(19); self._bg_lufs_slider.setFixedWidth(130)
        self._bg_lufs_lbl = QLabel("-19 LUFS"); self._bg_lufs_lbl.setStyleSheet(f"color:{C['text2']};font-size:11px;font-family:'Consolas',monospace;background-color:transparent;min-width:70px;")
        self._bg_lufs_slider.valueChanged.connect(lambda v: self._bg_lufs_lbl.setText(f"-{v} LUFS"))
        row4.addWidget(vocal_lbl); row4.addWidget(self._vocal_lufs_slider); row4.addWidget(self._vocal_lufs_lbl)
        row4.addWidget(bg_lbl); row4.addWidget(self._bg_lufs_slider); row4.addWidget(self._bg_lufs_lbl)
        row4.addStretch()
        self._row4_widget.setVisible(False)
        opts_lay.addWidget(self._row4_widget)

        root.addWidget(opts_grp)

        self._btn_enhance = QPushButton("⚡ Process Queue")
        self._btn_enhance.setStyleSheet(ENHANCE_BTN)
        self._btn_enhance.setEnabled(False)
        self._btn_enhance.clicked.connect(self._run_enhancement)
        root.addWidget(self._btn_enhance)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(16)
        root.addWidget(self._progress)

        self._sb = QStatusBar(); self.setStatusBar(self._sb)
        self._set_status("Ready.")

    # ── Signals & helpers ────────────────────────────────────────────────────

    def _on_demucs_toggled(self, checked):
        self._stems_combo.setEnabled(checked)
        four = checked and self._stems_combo.currentData() == 4
        self._row3_widget.setVisible(four)
        self._row4_widget.setVisible(four and self._chk_final_mix.isChecked())

    def _build_dynamic_params_widget(self, opts_lay) -> None:
        self._dynamic_widgets: dict = {}
        for p in self._backend.processing_params:
            key      = p["key"]
            ptype    = p["type"]
            label    = p.get("label", key)
            tooltip  = p.get("tooltip", "")
            default  = p.get("default")

            row_w = QWidget()
            row   = QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(12)

            if ptype == "choice":
                lbl = QLabel(f"{label}:")
                lbl.setStyleSheet(f"color:{C['text2']};font-size:12px;background-color:transparent;")
                combo = QComboBox()
                for val, display in p.get("options", []):
                    combo.addItem(display, val)
                idx = next((i for i, (v, _) in enumerate(p.get("options", [])) if v == default), 0)
                combo.setCurrentIndex(idx)
                combo.setFixedWidth(240)
                if tooltip:
                    combo.setToolTip(tooltip)
                row.addWidget(lbl)
                row.addWidget(combo)
                row.addStretch()
                self._dynamic_widgets[key] = combo

                if key == "chunk_mode":
                    def _on_chunk_mode_changed(text, rw=row_w):
                        sec_w = self._dynamic_widgets.get("chunk_sec_row")
                        if sec_w:
                            sec_w.setVisible(combo.currentData() != "none")
                    combo.currentIndexChanged.connect(_on_chunk_mode_changed)

            elif ptype in ("spin", "double_spin"):
                lbl = QLabel(f"{label}:")
                lbl.setStyleSheet(f"color:{C['text2']};font-size:12px;background-color:transparent;")
                if ptype == "double_spin":
                    spin = QDoubleSpinBox()
                    spin.setDecimals(p.get("decimals", 2))
                    spin.setSingleStep(p.get("step", 0.1))
                    spin.setValue(float(default) if default is not None else 0.0)
                else:
                    spin = QSpinBox()
                    spin.setSingleStep(p.get("step", 1))
                    spin.setValue(int(default) if default is not None else 0)
                spin.setRange(p.get("min", 0), p.get("max", 100))
                if p.get("suffix"):
                    spin.setSuffix(p["suffix"])
                spin.setFixedWidth(110)
                if tooltip:
                    spin.setToolTip(tooltip)
                row.addWidget(lbl)
                row.addWidget(spin)
                row.addStretch()
                self._dynamic_widgets[key] = spin

                if key == "chunk_sec":
                    self._dynamic_widgets["chunk_sec_row"] = row_w
                    chunk_mode_combo = self._dynamic_widgets.get("chunk_mode")
                    if chunk_mode_combo:
                        row_w.setVisible(chunk_mode_combo.currentData() != "none")

            elif ptype == "bool":
                chk = QCheckBox(label)
                chk.setChecked(bool(default))
                if tooltip:
                    chk.setToolTip(tooltip)
                row.addWidget(chk)
                row.addStretch()
                self._dynamic_widgets[key] = chk

            opts_lay.addWidget(row_w)

    def _build_backend_params(self) -> dict:
        backend = self._backend
        params = {"model_dir": str(model_dir_for(backend.model_id))}
        for p in backend.processing_params:
            key   = p["key"]
            ptype = p["type"]
            w     = self._dynamic_widgets.get(key)
            if w is None:
                continue
            if ptype == "choice":
                params[key] = w.currentData()
            elif ptype == "spin":
                params[key] = w.value()
            elif ptype == "double_spin":
                params[key] = w.value()
            elif ptype == "bool":
                params[key] = w.isChecked()
        return params

    def _start_backend_worker(self, input_path: str, on_finished) -> None:
        backend = self._backend
        self._worker = BackendWorker(
            backend, input_path, self._output_dir, self._build_backend_params())
        self._worker.status.connect(self._set_status)
        self._worker.log_output.connect(lambda line: print(line, end="", flush=True))
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(on_finished)
        self._worker.start()

    def _on_demucs_2stem_done(self, vocals_path: str):
        self._start_backend_worker(vocals_path, self._on_file_processed)

    def _on_demucs_4stem_done(self, vocals_path: str, bg_path: str):
        self._demucs_bg_path = bg_path
        self._start_backend_worker(vocals_path, self._on_enhanced_vocals_4stem_done)

    def _on_enhanced_vocals_4stem_done(self, enhanced_vocals: str):
        self._demucs_enhanced_vocals = enhanced_vocals
        if self._chk_bg_process.isChecked() and _PEDALBOARD_OK:
            hp  = float(self._spinbox_highpass.value())
            thr = float(self._spinbox_threshold.value())
            rat = self._spinbox_ratio.value()
            self._worker = BackgroundWorker(self._demucs_bg_path, hp, thr, rat)
            self._worker.status.connect(self._set_status)
            self._worker.error.connect(self._on_worker_error)
            self._worker.finished.connect(self._on_bg_processed_done)
            self._worker.start()
        else:
            self._on_bg_processed_done(self._demucs_bg_path)

    def _on_bg_processed_done(self, processed_bg: str):
        if self._chk_final_mix.isChecked() and _PEDALBOARD_OK:
            stem       = Path(self._current_path).stem
            vocal_lufs = -float(self._vocal_lufs_slider.value())
            bg_lufs    = -float(self._bg_lufs_slider.value())
            use_lufs   = self._chk_lufs.isChecked() and _PYLOUDNORM_OK
            self._worker = MixWorker(
                self._demucs_enhanced_vocals, processed_bg,
                vocal_lufs, bg_lufs, stem, use_lufs)
            self._worker.status.connect(self._set_status)
            self._worker.error.connect(self._on_worker_error)
            self._worker.finished.connect(self._on_file_processed)
            self._worker.start()
        else:
            self._on_file_processed(self._demucs_enhanced_vocals)

    def _on_stems_changed(self, _):
        if not self._chk_demucs.isChecked(): return
        four = self._stems_combo.currentData() == 4
        self._row3_widget.setVisible(four)
        self._row4_widget.setVisible(four and self._chk_final_mix.isChecked())

    def _on_bg_process_toggled(self, checked):
        self._row3b_widget.setVisible(checked)

    def _on_final_mix_toggled(self, checked):
        four = self._chk_demucs.isChecked() and self._stems_combo.currentData() == 4
        self._row4_widget.setVisible(four and checked)

    def _on_lufs_toggled(self, checked):
        for w in (self._vocal_lufs_slider, self._vocal_lufs_lbl, self._bg_lufs_slider, self._bg_lufs_lbl):
            w.setEnabled(checked)

    def _on_batch_file_selected(self, path: str):
        self._in_player.load(path)
        self._in_badge.set_info(self._in_player.get_info() or "")
        self._set_status(f"Preview: {Path(path).name}")
        out_path = self._processed_map.get(path)
        if out_path and Path(out_path).exists():
            self._out_player.load(out_path)
            self._out_badge.set_info(self._out_player.get_info() or "")
        else:
            self._out_player.clear()
            self._out_badge.clear_info()

    def _choose_output_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", str(self._output_dir))
        if folder:
            self._output_dir = Path(folder)
            self._out_dir_display.setText(str(self._output_dir))
            self._out_dir_display.setToolTip(str(self._output_dir))

    def _convert_to_format(self, src: str, dest: str, fmt: str) -> None:
        if fmt in ("wav", "flac"):
            data, sr = sf.read(src, dtype="float32")
            subtype = "PCM_24"
            sf.write(dest, data, sr, subtype=subtype)
        else:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", src, dest],
                capture_output=True, timeout=300,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg conversion failed:\n{r.stderr.decode(errors='replace')[-400:]}")

    # ── Model status ─────────────────────────────────────────────────────────
    def _model_ready(self) -> bool:
        backend = self._backend
        if not backend.download_repo:
            return True
        md = model_dir_for(backend.model_id)
        return md.exists() and any(f for f in md.iterdir() if not f.name.startswith("."))

    def _refresh_model_status(self):
        backend = self._backend
        if not backend.download_repo:
            self._model_status_lbl.setText("✅ Model: ready")
            self._model_status_lbl.setStyleSheet(f"color:{C['success']};font-size:11px;background-color:transparent;")
            self._btn_download.setVisible(False)
        elif self._model_ready():
            self._model_status_lbl.setText("✅ Model: ready")
            self._model_status_lbl.setStyleSheet(f"color:{C['success']};font-size:11px;background-color:transparent;")
            self._btn_download.setVisible(True)
            self._btn_download.setText("🔄 Re-download")
            self._btn_download.setStyleSheet(_coloured_btn(C["text3"]))
        else:
            self._model_status_lbl.setText("⚠ Model: not downloaded")
            self._model_status_lbl.setStyleSheet(f"color:{C['warning']};font-size:11px;background-color:transparent;")
            self._btn_download.setVisible(True)
            self._btn_download.setText("⬇ Download Model")
            self._btn_download.setStyleSheet(DOWNLOAD_BTN)
        self._update_enhance_btn()

    def _download_model(self):
        if self._worker and self._worker.isRunning(): return
        backend = self._backend
        self._btn_download.setEnabled(False)
        self._progress.setVisible(True)
        self._set_status(f"Downloading {backend.download_repo}…")
        self._worker = DownloadModelWorker(backend, model_dir_for(backend.model_id))
        self._worker.status.connect(self._set_status)
        self._worker.finished.connect(self._on_download_done)
        self._worker.error.connect(lambda e: self._on_error("Download failed", e,
            reset=lambda: (self._btn_download.setEnabled(True), self._progress.setVisible(False))))
        self._worker.start()

    def _on_download_done(self):
        self._progress.setVisible(False)
        self._btn_download.setEnabled(True)
        self._refresh_model_status()
        self._set_status("✅ Model downloaded.", C["success"])

    # ── File drop → batch ────────────────────────────────────────────────────
    def _add_paths(self, paths: list):
        audio  = [p for p in paths if Path(p).suffix.lower() in AUDIO_EXTS]
        videos = [p for p in paths if Path(p).suffix.lower() in VIDEO_EXTS]

        if audio:
            was_empty = not self._batch_panel.has_files()
            self._batch_panel.add_files(audio)
            if was_empty or self._in_player.get_path() is None:
                self._in_player.load(audio[0])
                self._in_badge.set_info(self._in_player.get_info() or "")
            n = len(audio)
            self._set_status(f"Added {n} file{'s' if n > 1 else ''} to queue.")

        for vp in videos:
            if not _ffmpeg_ok():
                QMessageBox.warning(self, "ffmpeg not found",
                    "ffmpeg is required to extract audio from video files.")
                break
            self._extract_video_and_add(vp)

    def _on_file_dropped(self, path: str):
        self._add_paths([path])

    def _extract_video_and_add(self, path: str):
        self._set_status("Extracting audio from video…")
        self._progress.setVisible(True)
        self._worker = VideoExtractWorker(path)
        self._worker.status.connect(self._set_status)
        self._worker.finished.connect(lambda tmp: (
            self._progress.setVisible(False),
            self._batch_panel.add_files([tmp]),
            self._in_player.load(tmp),
            self._in_badge.set_info(self._in_player.get_info() or ""),
            self._set_status(f"Video audio extracted and added to queue"),
        ))
        self._worker.error.connect(lambda e: self._on_error("Video extraction failed", e,
            reset=lambda: self._progress.setVisible(False)))
        self._worker.start()

    # ── Batch processing ─────────────────────────────────────────────────────
    def _update_enhance_btn(self):
        has_checked = self._batch_panel.has_checked_files()
        model_ok    = self._model_ready()
        self._btn_enhance.setEnabled(has_checked and model_ok)
        self._btn_enhance.setText("⚡ Process Queue" if has_checked else "⚡ Enhance Audio")

    def _run_enhancement(self):
        if not self._batch_panel.has_checked_files():
            QMessageBox.warning(self, "No files", "Select files to process first.")
            return
        if not self._model_ready():
            QMessageBox.warning(self, "No model", "Download the model first.")
            return
        if self._worker and self._worker.isRunning():
            return
        self._btn_enhance.setEnabled(False)
        self._batch_start_time = time.time()
        self._batch_done_count = 0
        self._batch_file_count = len(self._batch_panel.get_pending_files())
        self._process_next_file()

    def _process_next_file(self):
        pending = self._batch_panel.get_pending_files()
        if not pending:
            self._btn_enhance.setEnabled(True)
            self._batch_panel.update_eta("")
            self._set_status("✅ Batch processing complete!", C["success"])
            return

        self._btn_enhance.setEnabled(False)
        self._current_path = pending[0]
        self._batch_panel.mark_processing(self._current_path)
        self._progress.setVisible(True)

        backend    = self._backend
        use_demucs = backend.supports_demucs_preprocessing and self._chk_demucs.isChecked()

        if use_demucs:
            four_stem  = self._stems_combo.currentData() == 4
            demucs_out = str(OUTPUT_DIR / "demucs_tmp")
            self._worker = DemucsWorker(self._current_path, demucs_out, "htdemucs_ft", four_stem)
            self._worker.status.connect(self._set_status)
            self._worker.error.connect(self._on_worker_error)
            if four_stem:
                self._worker.finished_4stem.connect(self._on_demucs_4stem_done)
            else:
                self._worker.finished_2stem.connect(self._on_demucs_2stem_done)
            self._worker.start()
        else:
            self._start_backend_worker(self._current_path, self._on_file_processed)

    def _on_file_processed(self, out_path: str):
        self._batch_panel.mark_done(self._current_path)

        fmt     = self._fmt_combo.currentData()
        out_dir = self._output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        stem      = Path(self._current_path).stem
        dest_name = f"{stem}.{fmt}"
        dest_path = out_dir / dest_name

        try:
            self._convert_to_format(out_path, str(dest_path), fmt)
            if Path(out_path).resolve() != dest_path.resolve():
                try:
                    Path(out_path).unlink()
                except Exception:
                    pass
            self._processed_map[self._current_path] = str(dest_path)
            self._out_player.load(str(dest_path))
            self._out_badge.set_info(self._out_player.get_info() or "")
        except Exception as e:
            logger.error(f"Save error: {e}")

        self._batch_done_count += 1
        eta_str = ""
        if self._batch_start_time and self._batch_done_count > 0:
            elapsed   = time.time() - self._batch_start_time
            avg       = elapsed / self._batch_done_count
            remaining = self._batch_file_count - self._batch_done_count
            if remaining > 0:
                eta_str = _fmt(avg * remaining)

        self._batch_panel.update_eta(eta_str)
        status_msg = f"✅ Saved {dest_name} ({self._batch_done_count}/{self._batch_file_count})"
        if eta_str:
            status_msg += f"  ·  ETA {eta_str}"
        self._set_status(status_msg, C["success"])

        self._progress.setVisible(False)
        QTimer.singleShot(300, self._process_next_file)

    def _on_worker_error(self, e: str):
        self._batch_panel.mark_error(self._current_path)
        self._progress.setVisible(False)
        self._set_status("Processing error", C["error"])
        QMessageBox.critical(self, "Processing Error", e)
        QTimer.singleShot(300, self._process_next_file)

    # ── Output & helpers ─────────

    def _set_status(self, msg: str, color: str = None):
        color = color or C["text2"]
        self._sb.setStyleSheet(f"background-color:{C['surface']};border-top:1px solid {C['border']};color:{color};font-size:11px;")
        self._sb.showMessage(msg)

    def _on_error(self, title: str, tb: str, reset=None):
        logger.error(f"{title}:\n{tb}")
        if reset: reset()
        self._progress.setVisible(False)
        self._set_status(f"{title}: {tb.splitlines()[-1]}", C["error"])
        QMessageBox.critical(self, title, tb)

    def closeEvent(self, e):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(3000)
        e.accept()


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════
def main():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "NVIDIA.REUSE.SpeechEnhancement.1")

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("Audio & Speech Enhancement")

    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window,  QColor(C["bg"]))
    pal.setColor(QPalette.ColorRole.Base,    QColor(C["bg"]))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)

    _ico = APP_DIR / "icon.ico"
    if _ico.exists():
        app_icon = QIcon(str(_ico))
    else:
        pm = QPixmap(64, 64); pm.fill(QColor(C["surface"]))
        painter = QPainter(pm)
        painter.setFont(QFont("Segoe UI Emoji", 30))
        painter.setPen(QColor(C["accent"]))
        painter.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "⚡")
        painter.end()
        app_icon = QIcon(pm)

    app.setWindowIcon(app_icon)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
