#!/usr/bin/env python3
"""
Drone Gaze GUI — Updated & fixed:
 - Start button launches drone_gaze_blink_control.py, extract_gaze_vector.py, elg_run.py, ablink.py
 - Left-bottom 40x60 box shows ELG processed video (via ZMQ JPEG stream) with UDP gaze arrow as fallback
 - Right-bottom 40x60 box shows Blink status or /tmp/blink_preview.jpg
 - PDF loader fixed (now a proper method on InstructionsScreen)
 - Button signals fixed to use .emit
"""
# conda activate gaze_gui
# cd '/media/hn_97/VOLUME G/GazeML-org/src'
# python drone_gui.py

from __future__ import annotations
import os, sys, re, json, socket, threading, time, subprocess
from dataclasses import dataclass
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
import zmq

try:
    import cv2
except Exception:
    cv2 = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# ------------------------------- Config ------------------------------------ #
GAZEML_SRC = "/media/hn_97/VOLUME G/GazeML-org/src"

DEFAULT_PDF_PATHS = [
    "/mnt/data/PAPER_10_GazeRace.pdf",
    str(Path(GAZEML_SRC) / "PAPER_10_GazeRace.pdf"),
]

# Tiny preview (files are optional; GUI also draws from live signals)
ELG_PREVIEW_PATH = "/tmp/elg_preview.jpg"      # optional still (fallback)
BLINK_PREVIEW_PATH = "/tmp/blink_preview.jpg"  # optional still

DRONE_VIDEO_SRC = os.environ.get("DRONE_VIDEO_SRC", "udp://0.0.0.0:11111")

# elg_run.py → UDP(JSON) {"gx":float,"gy":float,"ok":bool}  (for arrow fallback + telemetry)
ELG_UDP_HOST = "127.0.0.1"
ELG_UDP_PORT = 50555

# elg_run.py → ZMQ JPEG stream endpoint (processed frames)
ELG_ZMQ_ENDPOINT = os.environ.get("ELG_PUB", "tcp://127.0.0.1:5556")

# ------------------------ Helpers / extra config --------------------------- #
def _first_existing_path(paths: list[str]) -> str | None:
    for p in paths:
        if Path(p).exists():
            return str(Path(p))
    return None

# Blink tracker script locations (repo path first, then uploaded fallback)
ABLINK_PATH = _first_existing_path([
    "/media/hn_97/VOLUME G/gaze-tracking-main/ablink.py",
    "/mnt/data/ablink.py",
])
ABLINK_WORKDIR = str(Path(ABLINK_PATH).parent) if ABLINK_PATH else None
ABLINK_ENV = os.environ.get("ABLINK_ENV", "gaze_tracking_env")  # change if your env differs

# -------------------------- Subprocess launchers --------------------------- #
@dataclass
class ProcSpec:
    name: str
    conda_env: str
    workdir: str
    script: str
    args: list[str]

@dataclass
class ManagedProc:
    spec: ProcSpec
    popen: subprocess.Popen | None = None

    def start(self):
        if self.popen and self.popen.poll() is None:
            return
        # Use conda run to guarantee the correct env without relying on bashrc.
        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", self.spec.conda_env,
            "python", self.spec.script, *self.spec.args
        ]
        self.popen = subprocess.Popen(
            cmd,
            cwd=self.spec.workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

    def stop(self):
        if self.popen and self.popen.poll() is None:
            try:
                self.popen.terminate()
                try:
                    self.popen.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.popen.kill()
            except Exception:
                pass
        self.popen = None

    def is_running(self) -> bool:
        return self.popen is not None and self.popen.poll() is None

# Your processes
PROC_SPECS = [
    ProcSpec("drone_gaze_blink_control", "tello_env", GAZEML_SRC, "drone_gaze_blink_control.py", []),
    ProcSpec("extract_gaze_vector",       "gazeml",   GAZEML_SRC, "extract_gaze_vector.py",       []),
    ProcSpec("elg_run",                   "gazeml",   GAZEML_SRC, "elg_run.py",                   []),
]

# Optionally include ablink.py if present
if ABLINK_PATH and ABLINK_WORKDIR:
    PROC_SPECS.append(
        ProcSpec("ablink", ABLINK_ENV, ABLINK_WORKDIR, Path(ABLINK_PATH).name, [])
    )

# ------------------------------- GUI -------------------------------------- #
class HomeScreen(QtWidgets.QWidget):
    start_clicked = QtCore.pyqtSignal()
    instructions_clicked = QtCore.pyqtSignal()
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.addStretch(1)
        title = QtWidgets.QLabel("Drone Gaze — Control Center")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("font-size: 28px; font-weight: 600;")
        lay.addWidget(title)
        lay.addSpacing(30)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        start = QtWidgets.QPushButton("Start your Drone")
        start.setFixedSize(300, 80)
        start.setStyleSheet("font-size: 20px; border-radius: 12px;")
        start.clicked.connect(self.start_clicked.emit)
        btn_row.addWidget(start)
        btn_row.addSpacing(40)
        instr = QtWidgets.QPushButton("Control Instructions")
        instr.setFixedSize(300, 80)
        instr.setStyleSheet("font-size: 20px; border-radius: 12px;")
        instr.clicked.connect(self.instructions_clicked.emit)
        btn_row.addWidget(instr)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)
        lay.addStretch(2)

class InstructionsScreen(QtWidgets.QWidget):
    back_clicked = QtCore.pyqtSignal()
    def __init__(self, pdf_paths: list[str]):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Control Instructions")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        header.addWidget(title); header.addStretch(1)
        lay.addLayout(header)
        self.text = QtWidgets.QTextEdit(); self.text.setReadOnly(True)
        self.text.setStyleSheet("font-size: 15px;")
        lay.addWidget(self.text, 1)
        back = QtWidgets.QPushButton("Back")
        back.setFixedSize(120, 42)
        back.setStyleSheet("font-size: 16px; border-radius: 10px;")
        back.clicked.connect(self.back_clicked.emit)
        row = QtWidgets.QHBoxLayout(); row.addStretch(1); row.addWidget(back); lay.addLayout(row)
        self.load_pdf_text(pdf_paths)

    def load_pdf_text(self, paths: list[str]):
        built_in = [
            str(Path(__file__).with_name("Tello_Gaze_Blink_Quickstart.pdf")),
            "/mnt/data/PAPER_10_GazeRace.pdf",
        ]
        candidates = [p for p in built_in if p] + list(paths or [])

        if PyPDF2 is None:
            self.text.setPlainText("Install PyPDF2 to view instructions from the PDF.")
            return

        for cand in candidates:
            try:
                path = Path(cand)
                if not path.exists():
                    continue
                with path.open("rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = []
                    for page in reader.pages:
                        txt = page.extract_text() or ""
                        pages.append(txt)
                    content = "\n\n".join(pages).strip()
                    if not content:
                        self.text.setPlainText(
                            f"Loaded {path.name}, but it contains little/no extractable text.\n"
                            "If this is a scanned PDF, consider installing PyMuPDF (pymupdf)."
                        )
                    else:
                        self.text.setPlainText(content)
                    return
            except Exception as e:
                print(f"[PDF] Failed to read {cand}: {e}")

        self.text.setPlainText(
            "PDF not found.\n\n"
            "Place Tello_Gaze_Blink_Quickstart.pdf next to this GUI file, "
            "or use /mnt/data/PAPER_10_GazeRace.pdf, or pass a path into load_pdf_text()."
        )

class ExperimentScreen(QtWidgets.QWidget):
    back_to_home = QtCore.pyqtSignal()
    elg_jpeg_received = QtCore.pyqtSignal(bytes)   # live frames from elg_run.py (ZMQ)

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # State
        self.video_enabled = False
        self.recording = False
        self.altitude = None
        self.battery = None

        # ELG live gaze (UDP arrow fallback)
        self.gaze_ok = False
        self.gx = 0.0
        self.gy = 0.0

        # Blink status
        self.last_blink_text = ""
        self._clear_blink_timer = QtCore.QTimer(self)
        self._clear_blink_timer.setInterval(2000)
        self._clear_blink_timer.timeout.connect(self._clear_blink_text)

        # Layout
        self.setStyleSheet("background-color: black;")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)

        # Top status row
        status_row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Altitude = N/A    Battery = N/A    ")
        self.status_label.setStyleSheet("color: white; font-size: 16px;")
        self.rec_label = QtWidgets.QLabel("")
        self.rec_label.setStyleSheet("color: red; font-size: 16px; font-weight: 600;")
        status_row.addWidget(self.status_label)
        status_row.addWidget(self.rec_label)
        status_row.addStretch(1)
        lay.addLayout(status_row)

        # Center video / placeholder
        self.center_stack = QtWidgets.QStackedWidget()
        self.placeholder = QtWidgets.QLabel("press m button to start video streaming from drone")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #bbbbbb; font-size: 20px;")
        self.center_stack.addWidget(self.placeholder)
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.center_stack.addWidget(self.video_label)
        lay.addWidget(self.center_stack, 1)

        # Bottom preview row (40x60 each)
        previews = QtWidgets.QHBoxLayout()
        self.elg_preview = self._boxed_preview_label()   # left (ELG processed video)
        self.blink_preview = self._boxed_preview_label() # right (Blink)
        previews.addWidget(self.elg_preview, alignment=Qt.AlignmentFlag.AlignLeft  | Qt.AlignmentFlag.AlignBottom)
        previews.addStretch(1)
        previews.addWidget(self.blink_preview, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(previews)

        # Back button (Esc also works)
        btn_row = QtWidgets.QHBoxLayout(); btn_row.addStretch(1)
        back = QtWidgets.QPushButton("Back")
        back.setFixedSize(120, 42)
        back.clicked.connect(self.back_to_home.emit)
        btn_row.addWidget(back); lay.addLayout(btn_row)

        # Timers
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.timeout.connect(self._refresh_previews)
        self.preview_timer.start(100)  # ~10 fps
        self.telemetry_timer = QtCore.QTimer(self)
        self.telemetry_timer.timeout.connect(self._refresh_telemetry_label)
        self.telemetry_timer.start(250)

        # Video thread ctrl
        self._video_thread = None
        self._video_stop = threading.Event()

        # ZMQ frame hookup
        self.elg_jpeg_received.connect(self._set_elg_jpeg)

    def _boxed_preview_label(self) -> QtWidgets.QLabel:
        lab = QtWidgets.QLabel()
        lab.setFixedSize(40, 60)  # width x height
        lab.setStyleSheet("background-color: #222; border: 1px solid #555;")
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return lab

    # ---------- Key handling ---------- #
    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == Qt.Key.Key_M:
            self.toggle_video()
        elif e.key() == Qt.Key.Key_Escape:
            self.back_to_home.emit()
        else:
            super().keyPressEvent(e)

    # ---------- Recording state ---------- #
    @QtCore.pyqtSlot(bool)
    def set_recording(self, on: bool):
        self.recording = on
        if on:
            self.rec_label.setText("Recording…")
        else:
            self.rec_label.setText("Recording stopped")
            QtCore.QTimer.singleShot(2000, lambda: self.rec_label.setText(""))

    # ---------- Telemetry ---------- #
    @QtCore.pyqtSlot(float, float)
    def update_telemetry(self, alt: float, bat: float):
        if alt == alt:  # not NaN
            self.altitude = alt
        if bat == bat:
            self.battery = bat

    def _refresh_telemetry_label(self):
        alt = f"{self.altitude:.1f} cm" if isinstance(self.altitude, (int, float)) else "N/A"
        bat = f"{self.battery:.0f}%"  if isinstance(self.battery, (int, float)) else "N/A"
        self.status_label.setText(f"Altitude = {alt}    Battery = {bat}    ")

    # ---------- ELG gaze from UDP ---------- #
    @QtCore.pyqtSlot(float, float, bool)
    def update_gaze(self, gx: float, gy: float, ok: bool):
        self.gx, self.gy, self.gaze_ok = gx, gy, ok

    def _draw_elg_preview_pixmap(self) -> QtGui.QPixmap:
        w, h = self.elg_preview.width(), self.elg_preview.height()
        img = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32)
        img.fill(QtGui.QColor("#222"))
        p = QtGui.QPainter(img)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        pen = QtGui.QPen(QtGui.QColor("#666")); pen.setWidth(1); p.setPen(pen)
        p.drawLine(w//2, 2, w//2, h-2)
        p.drawLine(2, h//2, w-2, h//2)
        color = QtGui.QColor("#0f0") if self.gaze_ok else QtGui.QColor("#888")
        pen = QtGui.QPen(color); pen.setWidth(2); p.setPen(pen)
        cx, cy = w//2, h//2
        scale = min(w, h) * 0.45
        ex = cx + int(self.gx * scale)
        ey = cy + int(-self.gy * scale)
        p.drawLine(cx, cy, ex, ey)
        p.drawEllipse(ex-1, ey-1, 2, 2)
        p.end()
        return QtGui.QPixmap.fromImage(img)

    # ---------- ZMQ ELG video slot ---------- #
    @QtCore.pyqtSlot(bytes)
    def _set_elg_jpeg(self, jpg_bytes: bytes):
        img = QtGui.QImage.fromData(jpg_bytes, 'JPG')
        if not img.isNull():
            pix = QtGui.QPixmap.fromImage(img).scaled(
                self.elg_preview.width(), self.elg_preview.height(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.elg_preview.setPixmap(pix)

    # ---------- Blink status ---------- #
    @QtCore.pyqtSlot(str)
    def update_blink_status(self, text: str):
        self.last_blink_text = text
        self._clear_blink_timer.start()

    def _clear_blink_text(self):
        self.last_blink_text = ""
        self._clear_blink_timer.stop()

    def _draw_blink_status_pixmap(self) -> QtGui.QPixmap:
        w, h = self.blink_preview.width(), self.blink_preview.height()
        img = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32)
        img.fill(QtGui.QColor("#222"))
        if self.last_blink_text:
            p = QtGui.QPainter(img)
            p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            pen = QtGui.QPen(QtGui.QColor("#ff5252")); p.setPen(pen)
            font = QtGui.QFont(); font.setPointSize(7); font.setBold(True)
            p.setFont(font)
            rect = QtCore.QRect(1, 1, w-2, h-2)
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.last_blink_text)
            p.end()
        return QtGui.QPixmap.fromImage(img)

    # ---------- Previews ---------- #
    def _refresh_previews(self):
        # ELG: if no live pixmap yet, draw the fallback arrow
        pm = self.elg_preview.pixmap()
        if pm is None or pm.isNull():
            self.elg_preview.setPixmap(self._draw_elg_preview_pixmap())

        # Blink: show image if exists; else status text pixmap
        shown = False
        try:
            if os.path.exists(BLINK_PREVIEW_PATH):
                img = QtGui.QImage(BLINK_PREVIEW_PATH)
                if not img.isNull():
                    pix = QtGui.QPixmap.fromImage(img).scaled(
                        self.blink_preview.width(), self.blink_preview.height(),
                        Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.blink_preview.setPixmap(pix)
                    shown = True
        except Exception:
            pass
        if not shown:
            self.blink_preview.setPixmap(self._draw_blink_status_pixmap())

    # ---------- Center Video (Tello) ---------- #
    def toggle_video(self):
        if self.video_enabled:
            self.stop_video()
        else:
            self.start_video()

    def start_video(self):
        if cv2 is None:
            return
        if self._video_thread and self._video_thread.is_alive():
            return
        self.video_enabled = True
        self.center_stack.setCurrentWidget(self.video_label)
        self._video_stop.clear()

        def _run():
            cap = cv2.VideoCapture(DRONE_VIDEO_SRC)
            if not cap.isOpened():
                while not self._video_stop.is_set():
                    time.sleep(0.1)
                return
            while not self._video_stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01); continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                img = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
                pix = QtGui.QPixmap.fromImage(img)
                self.video_label.setPixmap(pix.scaled(
                    self.video_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            cap.release()

        self._video_thread = threading.Thread(target=_run, daemon=True)
        self._video_thread.start()

    def stop_video(self):
        self.video_enabled = False
        self._video_stop.set()
        self.center_stack.setCurrentWidget(self.placeholder)

# ------------------------------ Main Window ------------------------------- #
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Gaze GUI")
        self.resize(1280, 720)

        self.procs: dict[str, ManagedProc] = {spec.name: ManagedProc(spec) for spec in PROC_SPECS}

        self.stack = QtWidgets.QStackedWidget(); self.setCentralWidget(self.stack)
        self.home = HomeScreen(); self.instructions = InstructionsScreen(DEFAULT_PDF_PATHS); self.experiment = ExperimentScreen()
        self.stack.addWidget(self.home); self.stack.addWidget(self.instructions); self.stack.addWidget(self.experiment)

        self.home.instructions_clicked.connect(lambda: self.stack.setCurrentWidget(self.instructions))
        self.instructions.back_clicked.connect(lambda: self.stack.setCurrentWidget(self.home))
        self.home.start_clicked.connect(self._start_clicked)
        self.experiment.back_to_home.connect(self._back_from_experiment)

        self._io_threads: list[threading.Thread] = []
        self._io_stop = threading.Event()

        # ELG UDP listener (gaze arrow/telemetry)
        self._udp_thread = None
        self._udp_stop = threading.Event()

        # ELG ZMQ subscriber (processed video)
        self._zmq_ctx = None
        self._elg_sub = None
        self._elg_sub_thread = None
        self._elg_sub_stop = threading.Event()

    # ---------------- Navigation ---------------- #
    def _start_clicked(self):
        # Start all processes (includes elg_run.py and ablink.py if present)
        for p in self.procs.values():
            p.start()
        # Start stdout readers
        self._start_readers()
        # Start ELG listeners
        self._start_udp_listener()
        self._start_elg_subscriber()
        # Go to experiment screen
        self.stack.setCurrentWidget(self.experiment)

    def _back_from_experiment(self):
        self.experiment.stop_video()
        self.stack.setCurrentWidget(self.home)

    # ---------------- Process IO ---------------- #
    def closeEvent(self, e: QtGui.QCloseEvent):
        self._shutdown()
        super().closeEvent(e)

    def _shutdown(self):
        self._io_stop.set()
        for t in self._io_threads:
            t.join(timeout=1)
        self._stop_udp_listener()
        self._stop_elg_subscriber()
        for p in self.procs.values():
            p.stop()

    def _start_readers(self):
        if self._io_threads: return
        self._io_stop.clear()
        for name, mproc in self.procs.items():
            t = threading.Thread(target=self._reader_thread, args=(name, mproc), daemon=True)
            t.start(); self._io_threads.append(t)

    def _reader_thread(self, name: str, mproc: ManagedProc):
        popen = mproc.popen
        if not popen or not popen.stdout:
            return
        for line in iter(popen.stdout.readline, ''):
            if self._io_stop.is_set(): break
            self._handle_process_output(name, line.rstrip())
        try: popen.stdout.close()
        except Exception: pass

    def _handle_process_output(self, name: str, line: str):
        print(f"[{name}] {line}")

        if name == "drone_gaze_blink_control":
            # Telemetry
            alt, bat = self._parse_telemetry(line)
            if alt is not None or bat is not None:
                QtCore.QMetaObject.invokeMethod(
                    self.experiment, "update_telemetry", Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(float, alt if alt is not None else float('nan')),
                    QtCore.Q_ARG(float, bat if bat is not None else float('nan')),
                )
            # Recording toggle (double blink)
            if self._is_double_blink(line):
                new_state = not self.experiment.recording
                QtCore.QMetaObject.invokeMethod(self.experiment, "set_recording",
                    Qt.ConnectionType.QueuedConnection, QtCore.Q_ARG(bool, new_state))
            # Blink status text (Right/Left/Double)
            blink_text = self._blink_text_from_line(line)
            if blink_text:
                QtCore.QMetaObject.invokeMethod(self.experiment, "update_blink_status",
                    Qt.ConnectionType.QueuedConnection, QtCore.Q_ARG(str, blink_text))

        elif name == "ablink":
            # Blink status from ablink.py (uses same regex rules)
            blink_text = self._blink_text_from_line(line)
            if blink_text:
                QtCore.QMetaObject.invokeMethod(
                    self.experiment, "update_blink_status",
                    Qt.ConnectionType.QueuedConnection, QtCore.Q_ARG(str, blink_text)
                )
            # Optional: toggle recording on "double" patterns from ablink
            if self._is_double_blink(line):
                new_state = not self.experiment.recording
                QtCore.QMetaObject.invokeMethod(
                    self.experiment, "set_recording",
                    Qt.ConnectionType.QueuedConnection, QtCore.Q_ARG(bool, new_state)
                )

    @staticmethod
    def _parse_telemetry(line: str):
        alt = bat = None
        alt_match = re.search(r"(?i)(alt(?:itude)?|height)\s*[=:]?\s*([0-9]+(?:\.[0-9]+)?)", line)
        bat_match = re.search(r"(?i)(bat(?:tery)?)\s*[=:]?\s*([0-9]+(?:\.[0-9]+)?)", line)
        if alt_match:
            try: alt = float(alt_match.group(2))
            except: pass
        if bat_match:
            try: bat = float(bat_match.group(2))
            except: pass
        return alt, bat

    @staticmethod
    def _is_double_blink(line: str) -> bool:
        phrases = [r"double\s*blink", r"recording\s*toggle", r"\[REC\]\s*toggle", r"\[REC\]\s*toggle\s*→\s*(start|stop)"]
        return any(re.search(p, line, re.IGNORECASE) for p in phrases)

    @staticmethod
    def _blink_text_from_line(line: str) -> str | None:
        if re.search(r"consecutive\s+right", line, re.I):   return "Right x2"
        if re.search(r"consecutive\s+left",  line, re.I):   return "Left x2"
        if re.search(r"double\s*blink|DOUBLE", line, re.I): return "Double"
        if re.search(r"right\s*blink|RIGHT", line, re.I):   return "Right"
        if re.search(r"left\s*blink|LEFT",  line, re.I):    return "Left"
        return None

    # ---------------- ELG UDP listener (gaze) ---------------- #
    def _start_udp_listener(self):
        if self._udp_thread and self._udp_thread.is_alive():
            return
        self._udp_stop.clear()
        def _loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((ELG_UDP_HOST, ELG_UDP_PORT))
            sock.settimeout(0.2)
            try:
                while not self._udp_stop.is_set():
                    try:
                        data, _ = sock.recvfrom(4096)
                        msg = json.loads(data.decode("utf-8"))
                        gx = float(msg.get("gx", 0.0))
                        gy = float(msg.get("gy", 0.0))
                        ok = bool(msg.get("ok", False))
                        QtCore.QMetaObject.invokeMethod(
                            self.experiment, "update_gaze", Qt.ConnectionType.QueuedConnection,
                            QtCore.Q_ARG(float, gx), QtCore.Q_ARG(float, gy), QtCore.Q_ARG(bool, ok)
                        )
                    except socket.timeout:
                        continue
                    except Exception:
                        continue
            finally:
                try: sock.close()
                except Exception: pass
        self._udp_thread = threading.Thread(target=_loop, name="ELG_UDP", daemon=True)
        self._udp_thread.start()

    def _stop_udp_listener(self):
        self._udp_stop.set()
        if self._udp_thread:
            self._udp_thread.join(timeout=1.0)
        self._udp_thread = None

    # ---------------- ELG ZMQ subscriber (processed video) ---------------- #
    def _start_elg_subscriber(self):
        if self._elg_sub_thread and self._elg_sub_thread.is_alive():
            return
        try:
            self._zmq_ctx = zmq.Context.instance()
            self._elg_sub = self._zmq_ctx.socket(zmq.SUB)
            self._elg_sub.connect(ELG_ZMQ_ENDPOINT)     # default tcp://127.0.0.1:5556
            self._elg_sub.setsockopt(zmq.SUBSCRIBE, b"")
            self._elg_sub_stop.clear()
        except Exception:
            return

        def _loop():
            poller = zmq.Poller()
            poller.register(self._elg_sub, zmq.POLLIN)
            while not self._elg_sub_stop.is_set():
                try:
                    socks = dict(poller.poll(200))
                    if self._elg_sub in socks and socks[self._elg_sub] == zmq.POLLIN:
                        data = self._elg_sub.recv(zmq.NOBLOCK)
                        # hand off JPEG bytes to the GUI thread
                        self.experiment.elg_jpeg_received.emit(data)
                except Exception:
                    pass

        self._elg_sub_thread = threading.Thread(target=_loop, name="ELG_SUB", daemon=True)
        self._elg_sub_thread.start()

    def _stop_elg_subscriber(self):
        self._elg_sub_stop.set()
        t = self._elg_sub_thread
        if t:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        try:
            if self._elg_sub:
                self._elg_sub.close(0)
        except Exception:
            pass
        self._elg_sub_thread = None

# ------------------------------- Entrypoint ------------------------------- #
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
