"""
DeepMosaicsPlus — Dataset & Training tool

One window for the whole training workflow:

  Workflow tab   The four steps as cards, each with its own settings and a live status
                 line (what footage/clips/models exist right now). Tick the steps you
                 want and press "Run selected steps": they run one after another, e.g.
                 generate clips and then train, with nothing to retype in between.
                   1. Real pairs      tools/bucket_real_pairs.py
                   2. Synthetic pool  make_datasets/make_video_dataset.py
                   3. Generated clips tools/make_generated_dataset.py
                   4. Train           tools/train_all_buckets.py
  Folders & models   where your footage, outputs and model files are
  Pair check         compares censored/ and uncensored/ files before you build from them
  Buckets            the size manifest (which checkpoint serves which mosaic size)

The log on the right shows the exact command of every step, so nothing is hidden.
Run with: python tools/dataset_prep_ui.pyw
"""
import os
import re
import sys
import json
import time
import shutil
import subprocess

try:
    from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, QUrl
    from PyQt6.QtGui import QFont, QColor, QTextCursor, QDesktopServices
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
        QTableWidget, QTableWidgetItem, QTextEdit, QPlainTextEdit, QTabWidget,
        QGroupBox, QHeaderView, QMessageBox, QAbstractItemView, QFrame, QScrollArea,
        QSplitter, QProgressBar, QSizePolicy,
    )
except ImportError:
    print("This tool needs PyQt6. Install it with: pip install PyQt6")
    sys.exit(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.dataset_prep_config.json')

DEFAULTS = {
    'censored_dir': os.path.join(ROOT, 'datasets', 'raw', 'censored'),
    'uncensored_dir': os.path.join(ROOT, 'datasets', 'raw', 'uncensored'),
    'synthetic_source_dir': os.path.join(ROOT, 'datasets', 'raw', 'synthetic_source'),
    'real_pairs_dir': os.path.join(ROOT, 'datasets', 'real_pairs'),
    'synthetic_pool_dir': os.path.join(ROOT, 'datasets', 'synthetic_pool'),
    'model_manifest': os.path.join(ROOT, 'pretrained_models', 'mosaic', 'size_manifest.json'),
    'mosaic_position_model_path': '',
    'roi_model_path': os.path.join(ROOT, 'pretrained_models', 'mosaic', 'add_face.pth'),
    'fps': 24,
    'reuse_uncensored': True,
    'gpu_id': '0',
    'batchsize': 8,
    'n_epoch': 200,
    'model_size': 512,
    'built_outsize': 0,
    'save_freq': 1000,
    'scene_accept_ratio': 1.0,
    'gen_kind': 'generated',
    'gen_clips': 200,
    'gen_style': 'mixed',
    'gen_skin_pct': 60,
    'gen_detail_pct': 100,
    'gen_texture_dir': os.path.join(ROOT, 'datasets', 'raw', 'textures'),
    'train_buckets': '',
    'train_mode': 'single',          # 'single' = original DeepMosaics training (one model); 'buckets' = one model per size
    'redo_finished': False,
    'steps_enabled': [False, False, True, True],
    'extra_dataset_args': '',
    'extra_train_args': '',
}

MODEL_SIZES = (256, 384, 512)
# Measured memory for one training step (generator + discriminator + VGG loss), GB per sample,
# by model size. Measured on CPU, so treat as a guide: GPU memory differs somewhat.
GB_PER_SAMPLE = {256: 0.9, 384: 2.0, 512: 3.6}   # 512: measured on an RTX 2080 Ti, batch 7 ran out of 22 GB

def loadsize_for(size):
    """Frames are resized to this before the random crop: the original 256 -> 286 ratio."""
    return int(round(size * 286 / 256))

def max_batch(vram_gb, size):
    return max(1, int((vram_gb * 0.85 - 1.0) / GB_PER_SAMPLE[size]))

def vram_needed(batch, size):
    return 1.0 + batch * GB_PER_SAMPLE[size]

VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')

# Same palette as the main DeepMosaicsPlus window, so the two tools look like one program.
C = {
    "bg": "#0a0c0e", "surface": "#111417", "surface2": "#161b21", "border": "#1e2530", "border2": "#252d38",
    "accent": "#2d6fd4", "accent_hi": "#3d82ea", "accent_dim": "#1a3a6a",
    "text": "#c8d0db", "text_dim": "#6b7a8d", "text_bright": "#e8edf4",
    "green": "#4caf76", "green_dim": "#1a3a26", "amber": "#c8943a", "red": "#c85050", "red_dim": "#3a1a1a",
}

QSS = f"""
QMainWindow, QWidget {{ background: {C['bg']}; color: {C['text']};
    font-family: "Inter", "Segoe UI", "SF Pro Text", sans-serif; font-size: 13px; }}
QToolTip {{ background: {C['surface2']}; color: {C['text_bright']}; border: 1px solid {C['border2']}; padding: 5px; }}
QLabel {{ background: transparent; }}
QLabel#h1 {{ font-size: 17px; font-weight: 600; color: {C['text_bright']}; }}
QLabel#dim {{ color: {C['text_dim']}; font-size: 12px; }}
QLabel#cardTitle {{ font-size: 14px; font-weight: 600; color: {C['text_bright']}; }}
QLabel#sectionTitle {{ font-size: 11px; font-weight: 700; color: {C['text_dim']}; letter-spacing: 1px; }}
QFrame#card {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 8px; }}
QFrame#card[state="running"] {{ border: 1px solid {C['accent']}; }}
QFrame#card[state="done"] {{ border: 1px solid {C['green']}; }}
QFrame#card[state="failed"] {{ border: 1px solid {C['red']}; }}
QFrame#card QLabel, QFrame#card QCheckBox {{ background: transparent; }}
QFrame#bar {{ background: {C['surface']}; border-top: 1px solid {C['border']}; }}
QFrame#bar QLabel {{ background: transparent; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {C['surface2']}; border: 1px solid {C['border2']};
    border-radius: 5px; padding: 5px 8px; color: {C['text_bright']}; min-height: 18px;
    selection-background-color: {C['accent']}; }}
QFrame#card QLineEdit, QFrame#card QSpinBox, QFrame#card QDoubleSpinBox, QFrame#card QComboBox {{ background: {C['surface2']}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {C['accent']}; }}
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{ color: {C['text_dim']}; }}
QComboBox QAbstractItemView {{ background: {C['surface2']}; color: {C['text_bright']}; border: 1px solid {C['border2']};
    selection-background-color: {C['accent_dim']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QPushButton {{ background: {C['surface2']}; border: 1px solid {C['border2']}; border-radius: 5px;
    padding: 6px 14px; color: {C['text']}; }}
QPushButton:hover {{ border: 1px solid {C['accent']}; color: {C['text_bright']}; }}
QPushButton:disabled {{ color: {C['text_dim']}; border: 1px solid {C['border']}; }}
QPushButton#primary {{ background: {C['accent']}; border: 1px solid {C['accent']}; color: white;
    font-weight: 600; font-size: 14px; padding: 9px 22px; }}
QPushButton#primary:hover {{ background: {C['accent_hi']}; }}
QPushButton#primary:disabled {{ background: {C['accent_dim']}; border: 1px solid {C['accent_dim']}; color: {C['text_dim']}; }}
QPushButton#danger {{ background: {C['red_dim']}; border: 1px solid {C['red']}; color: {C['text_bright']}; font-weight: 600; }}
QPushButton#danger:disabled {{ background: {C['surface2']}; border: 1px solid {C['border']}; color: {C['text_dim']}; }}
QPushButton#link {{ background: transparent; border: none; color: {C['accent_hi']}; padding: 2px 4px; }}
QPushButton#link:hover {{ color: {C['text_bright']}; }}
QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {C['border2']}; border-radius: 4px; background: {C['surface2']}; }}
QCheckBox::indicator:checked {{ background: {C['accent']}; border: 3px solid {C['surface2']}; outline: 1px solid {C['accent']}; }}
QCheckBox#stepCheck {{ font-size: 14px; font-weight: 600; color: {C['text_bright']}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {C['text_dim']}; padding: 9px 16px; border: none;
    border-bottom: 2px solid transparent; font-weight: 600; }}
QTabBar::tab:selected {{ color: {C['text_bright']}; border-bottom: 2px solid {C['accent']}; }}
QTabBar::tab:hover {{ color: {C['text']}; }}
QPlainTextEdit, QTextEdit {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 6px;
    color: {C['text']}; padding: 6px; }}
QTableWidget {{ background: {C['surface']}; border: 1px solid {C['border']}; border-radius: 6px;
    gridline-color: {C['border']}; }}
QHeaderView::section {{ background: {C['surface2']}; color: {C['text_dim']}; border: none;
    border-bottom: 1px solid {C['border']}; padding: 6px; font-weight: 600; }}
QTableWidget::item:selected {{ background: {C['accent_dim']}; }}
QProgressBar {{ background: {C['surface2']}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; }}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 3px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {C['border2']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QSplitter::handle {{ background: {C['border']}; width: 1px; }}
"""


def load_config():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"Could not save config: {e}")


def resolve_python_exe():
    """Prefer python.exe over pythonw.exe for launching subprocesses. This
    GUI is a .pyw file, which Windows associates with pythonw.exe by
    default -- meaning sys.executable here may report pythonw.exe, and that
    choice would otherwise propagate to every tool this GUI launches.
    pythonw.exe has no console/real stdio handles, which can make output
    capture through a subprocess chain behave inconsistently. python.exe is
    installed alongside pythonw.exe on every standard Windows install, so
    substituting it costs nothing and removes that whole class of doubt.
    """
    exe = sys.executable
    if exe.lower().endswith('pythonw.exe'):
        candidate = exe[:-len('pythonw.exe')] + 'python.exe'
        if os.path.exists(candidate):
            return candidate
    return exe


def is_video(path):
    return os.path.isfile(path) and path.lower().endswith(VIDEO_EXTS)


def ffprobe_info(path):
    """Return (fps, duration) via ffprobe directly, or (None, None) on
    failure. Deliberately does NOT import util.ffmpeg here — that module
    pulls in torch (for hwaccel detection), which this lightweight prep
    tool has no other reason to require.
    """
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
               '-show_format', '-show_streams', '-i', path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 stdin=subprocess.DEVNULL, timeout=15)
        if result.returncode != 0:
            return None, None
        infos = json.loads(result.stdout.decode('utf-8', errors='replace'))
        stream = infos['streams'][0]
        fps_str = stream.get('avg_frame_rate') or stream.get('r_frame_rate') or '0/1'
        num, _, den = fps_str.partition('/')
        fps = float(num) / float(den or 1) if float(den or 1) else 0
        duration = float(infos['format']['duration'])
        if not fps:
            return None, duration
        return fps, duration
    except Exception:
        return None, None


def gpu_memory_gb():
    """Total memory of the first NVIDIA GPU in GB, or None (no NVIDIA GPU / nvidia-smi)."""
    try:
        kw = {'creationflags': 0x08000000} if os.name == 'nt' else {}      # no console window flash
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=4, **kw)
        return float(out.stdout.strip().splitlines()[0]) / 1024.0 if out.returncode == 0 else None
    except Exception:
        return None


# ------------------------------------------------------------------- what's on disk

def count_files(folder, exts, recursive=False):
    if not folder or not os.path.isdir(folder):
        return None
    n = 0
    try:
        if recursive:
            for _, _, files in os.walk(folder):
                n += sum(1 for f in files if f.lower().endswith(exts))
        else:
            n = sum(1 for f in os.listdir(folder) if f.lower().endswith(exts))
    except OSError:
        return None
    return n


def subfolders(folder):
    try:
        return [n for n in os.listdir(folder) if os.path.isdir(os.path.join(folder, n))]
    except OSError:
        return []


def pool_summary(folder):
    """(total clips, generated, region-crop) in a synthetic pool folder."""
    names = subfolders(folder)
    return len(names), sum(n.startswith('gen_') for n in names), sum(n.startswith('crop_') for n in names)


def real_pairs_summary(folder):
    """{bucket label: clip count} from bucket_<N> folders."""
    out = {}
    for n in subfolders(folder):
        if n.startswith('bucket_'):
            out[n[len('bucket_'):]] = len(subfolders(os.path.join(folder, n)))
    return out


def _bucket_finished(savename):
    """Same test step 4 uses to skip finished buckets (completed.json, else the training log)."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('_tab', os.path.join(ROOT, 'tools', 'train_all_buckets.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.bucket_finished(savename)
    except Exception as e:
        return False, f"could not check ({e})"


SINGLE_MODEL_DST = os.path.join(ROOT, 'pretrained_models', 'clean_single.pth')


def _savename(bucket):
    return 'clean_single' if bucket == 'single' else f'clean_{bucket}pct'


def _label(bucket):
    return 'Single model' if bucket == 'single' else f'{bucket}%'


def plan_model_install(manifest_path, single=False):
    """For every bucket in the manifest: the newest checkpoint training produced
    (checkpoints/clean_<bucket>pct/<N>_G.pth + .json) and where the manifest wants it.
    single: the one model of single-model training (checkpoints/clean_single) instead,
    installed as pretrained_models/clean_single.pth."""
    if single:
        manifest, base = {'single': SINGLE_MODEL_DST}, ROOT
        items = list(manifest.items())
    else:
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)
        base = os.path.dirname(os.path.abspath(manifest_path))
        items = sorted(manifest.items(), key=lambda kv: float(kv[0]))
    plan = []
    for bucket, rel in items:
        dst = rel if os.path.isabs(rel) else os.path.join(base, rel)
        folder = os.path.join(ROOT, 'checkpoints', _savename(bucket))
        item = {'bucket': bucket, 'dst': dst, 'src': None, 'iteration': None,
                'finished': False, 'note': '', 'has_json': False, 'replaces': os.path.exists(dst)}
        iters = []
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                m = re.fullmatch(r'(\d+)_G\.pth', name)
                if m:
                    iters.append(int(m.group(1)))
        if not iters:
            item['note'] = 'no checkpoint yet'
        else:
            it = max(iters)
            item['src'] = os.path.join(folder, f'{it}_G.pth')
            item['iteration'] = it
            item['has_json'] = os.path.exists(os.path.join(folder, f'{it}_G.json'))
            item['finished'], detail = _bucket_finished(_savename(bucket))
            item['note'] = 'finished' if item['finished'] else ('not finished' + (f': {detail}' if detail else ''))
        plan.append(item)
    return plan


def install_model(item):
    """Copy one bucket's checkpoint (and its .json) to the manifest path. An existing model there
    is kept as <name>.previous.pth / .json, replacing an older backup."""
    dst = item['dst']
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src_json = os.path.splitext(item['src'])[0] + '.json'
    dst_json = os.path.splitext(dst)[0] + '.json'
    stem = os.path.splitext(dst)[0]
    for old, backup in ((dst, stem + '.previous.pth'), (dst_json, stem + '.previous.json')):
        if os.path.exists(old):
            os.replace(old, backup)
    shutil.copy2(item['src'], dst + '.part')
    os.replace(dst + '.part', dst)
    if os.path.exists(src_json):
        shutil.copy2(src_json, dst_json)


def manifest_summary(path):
    """(buckets, how many have their checkpoint file) or None if unreadable."""
    try:
        with open(path) as f:
            manifest = json.load(f)
    except Exception:
        return None
    base = os.path.dirname(os.path.abspath(path))
    have = sum(os.path.exists(p if os.path.isabs(p) else os.path.join(base, p)) for p in manifest.values())
    return len(manifest), have


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


# ------------------------------------------------------------------- widgets

def dim(text):
    lab = QLabel(text)
    lab.setObjectName("dim")
    lab.setWordWrap(True)
    return lab


def field(label, widget, tip=None):
    """Small 'label above control' block for a card's settings row."""
    box = QVBoxLayout()
    box.setSpacing(3)
    lab = QLabel(label)
    lab.setObjectName("dim")
    box.addWidget(lab)
    box.addWidget(widget)
    if tip:
        lab.setToolTip(tip)
        widget.setToolTip(tip)
    return box


class StepCard(QFrame):
    """One workflow step: tick box + title, a live status line, its settings, and a
    button to run just this step."""

    def __init__(self, number, title, what, run_one):
        super().__init__()
        self.setObjectName("card")
        self.number = number
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 14)
        outer.setSpacing(8)
        head = QHBoxLayout()
        self.check = QCheckBox(f"{number}.  {title}")
        self.check.setObjectName("stepCheck")
        self.check.setToolTip("Include this step when you press 'Run selected steps'")
        head.addWidget(self.check)
        head.addStretch()
        self.state_label = QLabel("")
        self.state_label.setObjectName("dim")
        head.addWidget(self.state_label)
        self.run_btn = QPushButton("Run only this")
        self.run_btn.clicked.connect(lambda: run_one(number - 1))
        head.addWidget(self.run_btn)
        outer.addLayout(head)
        outer.addWidget(dim(what))
        self.status = QLabel("")
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self.body = QHBoxLayout()
        self.body.setSpacing(14)
        outer.addLayout(self.body)

    def set_status(self, text, level='info'):
        color = {'info': C['text_dim'], 'ok': C['green'], 'warn': C['amber'], 'bad': C['red']}[level]
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {color}; font-size: 12px;")

    def set_state(self, state, text=''):
        """'' | running | done | failed -- coloured border + a word on the right."""
        self.setProperty("state", state)
        self.style().unpolish(self); self.style().polish(self)
        color = {'running': C['accent_hi'], 'done': C['green'], 'failed': C['red']}.get(state, C['text_dim'])
        self.state_label.setText(text)
        self.state_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600;")


class DatasetPrepWindow(QMainWindow):
    STEP_NAMES = ["Real pairs", "Synthetic pool", "Generated clips", "Train"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepMosaicsPlus 1.3.1 — Dataset & Training")
        self.resize(1360, 860)
        self.cfg = load_config()
        self._process = None
        self._queue = []            # steps still to run: (index, title, cmd, cwd)
        self._current = None
        self._run_total = 0
        self._run_started = 0.0
        self._step_started = 0.0
        self._stopping = False
        self._partial = ''
        self._cr = False
        self._live = False
        self._gpu_gb = gpu_memory_gb()
        self._build_ui()
        self._load_into_ui()
        self._refresh_status()
        self._clock = QTimer(self)
        self._clock.timeout.connect(self._tick)
        self._clock.start(1000)

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split, stretch=1)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_workflow_tab(), "Workflow")
        self.tabs.addTab(self._build_paths_tab(), "Folders && models")
        self.tabs.addTab(self._build_scan_tab(), "Pair check")
        self.tabs.addTab(self._build_manifest_tab(), "Buckets")
        self.tabs.currentChanged.connect(lambda _: self._refresh_status())
        split.addWidget(self.tabs)

        # ---- right side: progress + log, always visible
        side = QWidget()
        side_l = QVBoxLayout(side)
        side_l.setContentsMargins(14, 12, 14, 12)
        side_l.setSpacing(8)
        top = QHBoxLayout()
        self.run_title = QLabel("Idle")
        self.run_title.setObjectName("cardTitle")
        top.addWidget(self.run_title, stretch=1)
        self.elapsed_label = QLabel("")
        self.elapsed_label.setObjectName("dim")
        top.addWidget(self.elapsed_label)
        side_l.addLayout(top)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        side_l.addWidget(self.progress)
        self.run_detail = dim("Nothing is running. Tick the steps you want on the Workflow tab and press Run.")
        side_l.addWidget(self.run_detail)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas" if sys.platform == 'win32' else "Monospace", 9))
        self.log.setMaximumBlockCount(5000)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        side_l.addWidget(self.log, stretch=1)
        btns = QHBoxLayout()
        clear = QPushButton("Clear log")
        clear.clicked.connect(self.log.clear)
        copy = QPushButton("Copy log")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.log.toPlainText()))
        btns.addWidget(clear); btns.addWidget(copy); btns.addStretch()
        side_l.addLayout(btns)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([800, 560])

    # ---- Workflow -------------------------------------------------------
    def _build_workflow_tab(self):
        page = QWidget()
        page_l = QVBoxLayout(page)
        page_l.setContentsMargins(0, 0, 0, 0)
        page_l.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setContentsMargins(18, 14, 18, 14)
        col.setSpacing(12)
        scroll.setWidget(inner)
        page_l.addWidget(scroll, stretch=1)

        # -- settings shared by every step
        common = QFrame(); common.setObjectName("card")
        cl = QVBoxLayout(common); cl.setContentsMargins(16, 12, 16, 14); cl.setSpacing(8)
        t = QLabel("USED BY EVERY STEP"); t.setObjectName("sectionTitle"); cl.addWidget(t)
        row = QHBoxLayout(); row.setSpacing(14)
        self.model_size_combo = QComboBox()
        for sz in MODEL_SIZES:
            self.model_size_combo.addItem(f"{sz} px", sz)
        row.addLayout(field("Model size", self.model_size_combo,
                            "Size the network trains on and later runs at. Clips are stored a little larger "
                            "(572 px for 512) and randomly cropped.\nChanging it means rebuilding the clips "
                            "(steps 1-3): clips stored smaller would only be enlarged."))
        self.fps_spin = QSpinBox(); self.fps_spin.setRange(1, 120)
        row.addLayout(field("Frame rate (fps)", self.fps_spin, "Frames per second extracted from your footage in steps 1 and 2."))
        self.gpu_edit = QLineEdit(); self.gpu_edit.setFixedWidth(70)
        row.addLayout(field("GPU id", self.gpu_edit, "Which GPU to train on (0 = first). -1 = CPU, far too slow for real training."))
        row.addStretch()
        cl.addLayout(row)
        self.size_hint = dim("")
        cl.addWidget(self.size_hint)
        col.addWidget(common)

        self.cards = []

        # -- 1 real pairs
        c1 = StepCard(1, "Real pairs", "Cuts your censored / uncensored pairs to the mosaic area and sorts them into "
                      "buckets by mosaic size.", self._run_single)
        self.reuse_check = QCheckBox("Also use the uncensored clips as clean footage for step 2")
        c1.body.addWidget(self.reuse_check); c1.body.addStretch()
        self.cards.append(c1); col.addWidget(c1)

        # -- 2 synthetic pool
        c2 = StepCard(2, "Synthetic pool from clean footage", "Finds areas to censor in your clean footage with the ROI "
                      "model and stores them as clips. Training adds the mosaic itself.", self._run_single)
        self.scene_accept_spin = QDoubleSpinBox()
        self.scene_accept_spin.setRange(0.1, 1.0); self.scene_accept_spin.setSingleStep(0.1); self.scene_accept_spin.setDecimals(2)
        c2.body.addLayout(field("Scene acceptance", self.scene_accept_spin,
                                "Share of the sampled frames in a scene that must pass ROI detection for the scene to be kept.\n"
                                "1.0 = all of them (strict: one weak frame discards the scene).\n"
                                "Try 0.5-0.6 if you get few clips from plenty of footage."))
        self.extra_dataset_edit = QLineEdit(); self.extra_dataset_edit.setPlaceholderText("optional")
        c2.body.addLayout(field("Extra arguments", self.extra_dataset_edit, "Passed to make_datasets/make_video_dataset.py as typed."), stretch=1)
        self.cards.append(c2); col.addWidget(c2)

        # -- 3 generated clips
        c3 = StepCard(3, "Generated clips", "Adds clips to the synthetic pool without the ROI model: procedural animation, "
                      "or random regions of your clean footage (works for anime). Footage is much better: procedural "
                      "clips are smooth, so a model trained mostly on them learns to smooth blocks, not to restore detail.", self._run_single)
        self.gen_kind_combo = QComboBox()
        self.gen_kind_combo.addItem("Procedural animation (no footage)", "generated")
        self.gen_kind_combo.addItem("Random regions of clean footage", "crops")
        self.gen_kind_combo.currentIndexChanged.connect(self._on_gen_kind)
        c3.body.addLayout(field("Source", self.gen_kind_combo,
                                "Procedural: shapes, gradients and patterns drawn by the program. No real texture or anatomy,\n"
                                "so use it alongside real footage.\n"
                                "Random regions: square areas of the videos/images in your clean footage folder, mosaic placed\n"
                                "at random. No detection, so it works on content the ROI model misses."))
        self.gen_clips_spin = QSpinBox(); self.gen_clips_spin.setRange(1, 100000)
        c3.body.addLayout(field("Clips to add", self.gen_clips_spin, "Each run ADDS this many clips (gen_00000... / crop_00000...). 100 frames each."))
        self.gen_style_combo = QComboBox()
        for label, value in (("Mixed", "mixed"), ("Anime", "anime"), ("Real life", "real")):
            self.gen_style_combo.addItem(label, value)
        c3.body.addLayout(field("Look", self.gen_style_combo,
                                "Anime: flat colours, cel-shaded gradients, dark outlines, sharp.\n"
                                "Real life: smooth shading, soft edges, surface texture, blur and noise.\n"
                                "Mixed: half of the clips each."))
        self.gen_skin_spin = QSpinBox(); self.gen_skin_spin.setRange(0, 100); self.gen_skin_spin.setSuffix(" %")
        c3.body.addLayout(field("Beige / pink tones", self.gen_skin_spin,
                                "Share of colours taken from beige, pink and tan shades.\n0 % = any colour, 100 % = only those shades."))
        self.gen_detail_spin = QSpinBox(); self.gen_detail_spin.setRange(0, 100); self.gen_detail_spin.setSuffix(" %")
        c3.body.addLayout(field("Detail", self.gen_detail_spin,
                                "Fine detail for the model to restore: multi-scale texture, line art and hair-like strokes,\n"
                                "and video-like compression. 0 % = the old smooth look, which mostly teaches the model\n"
                                "to smooth blocks over."))
        tex_box = QWidget(); tex_row = QHBoxLayout(tex_box); tex_row.setContentsMargins(0, 0, 0, 0); tex_row.setSpacing(6)
        self.gen_texture_edit = QLineEdit(); self.gen_texture_edit.setPlaceholderText("optional: folder of images")
        tex_btn = QPushButton("…"); tex_btn.setFixedWidth(32)
        tex_btn.clicked.connect(self._browse_textures)
        self.gen_texture_edit.editingFinished.connect(self._refresh_status)
        tex_row.addWidget(self.gen_texture_edit, 1); tex_row.addWidget(tex_btn)
        c3.body.addLayout(field("Texture images", tex_box,
                                "Patches of these images fill backgrounds and shapes: photos or texture packs of fabric,\n"
                                "skin-like surfaces, hair, wood, foliage, stone... Any size and format; subfolders included.\n"
                                "Default: datasets/raw/textures. While the folder is empty, only drawn detail is used."), stretch=1)
        c3.body.addStretch()
        self.cards.append(c3); col.addWidget(c3)

        # -- 4 train
        c4 = StepCard(4, "Train", "Single model: the original DeepMosaics training, one model from the whole synthetic pool "
                      "and all real pairs (checkpoints/clean_single/). Size buckets (experimental): one model per bucket "
                      "(checkpoints/clean_<N>pct/).", self._run_single)
        self.train_mode_combo = QComboBox()
        self.train_mode_combo.addItem("Single model (original)", 'single')
        self.train_mode_combo.addItem("Size buckets (experimental)", 'buckets')
        self.train_mode_combo.currentIndexChanged.connect(self._on_train_mode)
        c4.body.addLayout(field("Training mode", self.train_mode_combo,
                                "Single model: one model for every mosaic size, with the original wide random mosaic sizes.\n"
                                "Use it in the main program as a normal model (Install trained models puts it at\n"
                                "pretrained_models/clean_single.pth).\n\n"
                                "Size buckets: one model per mosaic size in the size manifest; the main program's\n"
                                "'Auto-select model by mosaic size' picks one per video. Experimental: it is not yet\n"
                                "known to beat a single model."))
        self.batchsize_spin = QSpinBox(); self.batchsize_spin.setRange(1, 256)
        c4.body.addLayout(field("Batch size", self.batchsize_spin, "Samples per training step. Limited by GPU memory: see the estimate below."))
        self.epoch_spin = QSpinBox(); self.epoch_spin.setRange(1, 100000)
        c4.body.addLayout(field("Epochs", self.epoch_spin, "How many times training goes through every clip. One epoch is already 82 iterations\n"
                                "per clip, so large pools need few epochs: see the length estimate below."))
        self.save_freq_spin = QSpinBox(); self.save_freq_spin.setRange(1, 1000000); self.save_freq_spin.setSingleStep(100)
        c4.body.addLayout(field("Save every (iterations)", self.save_freq_spin,
                                "A checkpoint is written every N iterations. Use 500-1000 for small datasets so you get\n"
                                "intermediate checkpoints. A final checkpoint is always saved."))
        self.buckets_edit = QLineEdit(); self.buckets_edit.setPlaceholderText("all"); self.buckets_edit.setFixedWidth(90)
        c4.body.addLayout(field("Buckets", self.buckets_edit, "Which buckets to train, comma separated (e.g. 4,5,6). Empty = all buckets in the manifest."))
        self.redo_check = QCheckBox("Train finished buckets again")
        self.redo_check.setToolTip("Normally buckets that already finished are skipped, so a run that was interrupted\ncontinues with the first unfinished bucket. Tick this to retrain all of them.")
        c4.body.addWidget(self.redo_check)
        self.extra_train_edit = QLineEdit(); self.extra_train_edit.setPlaceholderText("optional")
        c4.body.addLayout(field("Extra arguments", self.extra_train_edit, "Passed to train/clean/train.py as typed (e.g. --lr 0.0001 --continue_train)."), stretch=1)
        self.cards.append(c4); col.addWidget(c4)
        self.mem_hint = QLabel("")
        self.mem_hint.setWordWrap(True)
        c4.layout().addWidget(self.mem_hint)
        self.len_hint = QLabel("")
        self.len_hint.setWordWrap(True)
        c4.layout().addWidget(self.len_hint)
        open_row = QHBoxLayout()
        inst = QPushButton("Install trained models…")
        inst.setToolTip("Copy each bucket's newest checkpoint (and its .json) to the file the size manifest\n"
                        "expects in pretrained_models/mosaic, so the main program's\n"
                        "'Auto-select model by mosaic size' uses them.")
        inst.clicked.connect(self._install_models)
        open_row.addWidget(inst)
        for text, target in (("Open checkpoints folder", lambda: os.path.join(ROOT, 'checkpoints')),
                             ("Open synthetic pool folder", lambda: self._path_fields['synthetic_pool_dir'].text())):
            b = QPushButton(text); b.setObjectName("link")
            b.clicked.connect(lambda _, t=target: self._open_folder(t()))
            open_row.addWidget(b)
        open_row.addStretch()
        c4.layout().addLayout(open_row)

        col.addStretch()

        for w in (self.model_size_combo, ):
            w.currentIndexChanged.connect(self._update_hints)
        self.batchsize_spin.valueChanged.connect(self._update_hints)
        self.epoch_spin.valueChanged.connect(self._update_hints)
        self.save_freq_spin.valueChanged.connect(self._update_hints)
        for card in self.cards:
            card.check.toggled.connect(self._update_run_summary)

        # -- fixed bottom bar: the one button
        bar = QFrame(); bar.setObjectName("bar")
        bl = QHBoxLayout(bar); bl.setContentsMargins(18, 12, 18, 12); bl.setSpacing(12)
        self.run_summary = QLabel("")
        self.run_summary.setObjectName("dim")
        bl.addWidget(self.run_summary, stretch=1)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_process)
        bl.addWidget(self.stop_btn)
        self.run_all_btn = QPushButton("Run selected steps")
        self.run_all_btn.setObjectName("primary")
        self.run_all_btn.clicked.connect(self._run_selected)
        bl.addWidget(self.run_all_btn)
        page_l.addWidget(bar)
        return page

    # ---- Folders & models ----------------------------------------------
    def _build_paths_tab(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        w = QWidget(); scroll.setWidget(w)
        col = QVBoxLayout(w); col.setContentsMargins(18, 14, 18, 14); col.setSpacing(12)
        self._path_fields, self._path_marks = {}, {}
        groups = [
            ("YOUR FOOTAGE", [
                ("Censored clips", 'censored_dir', False, "Step 1. Same file names as the uncensored versions."),
                ("Uncensored clips", 'uncensored_dir', False, "Step 1. The clean versions of the same clips."),
                ("Clean footage", 'synthetic_source_dir', False, "Steps 2 and 3. Any footage without mosaic (videos; images too for step 3)."),
            ]),
            ("WHERE CLIPS ARE WRITTEN", [
                ("Real pairs", 'real_pairs_dir', False, "Output of step 1 (bucket_<N> folders)."),
                ("Synthetic pool", 'synthetic_pool_dir', False, "Output of steps 2 and 3; shared by every bucket in training."),
            ]),
            ("MODEL FILES", [
                ("Size manifest", 'model_manifest', True, "size_manifest.json: which buckets exist and which checkpoint serves each."),
                ("Mosaic position model", 'mosaic_position_model_path', True, "Needed by step 1 to find the mosaic (mosaic_position.pth)."),
                ("ROI model", 'roi_model_path', True, "Needed by step 2 to choose areas to censor (add_face.pth)."),
            ]),
        ]
        for title, rows in groups:
            card = QFrame(); card.setObjectName("card")
            grid = QGridLayout(card); grid.setContentsMargins(16, 12, 16, 14); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(6)
            t = QLabel(title); t.setObjectName("sectionTitle"); grid.addWidget(t, 0, 0, 1, 5)
            for i, (label, key, is_file, hint) in enumerate(rows):
                r = 1 + i * 2
                mark = QLabel("●"); mark.setFixedWidth(14)
                edit = QLineEdit()
                edit.textChanged.connect(lambda _, k=key: self._update_path_mark(k))
                browse = QPushButton("Browse…")
                browse.clicked.connect(lambda _, k=key, f=is_file: self._browse(k, is_file=f))
                opener = QPushButton("Open"); opener.setObjectName("link")
                opener.clicked.connect(lambda _, k=key, f=is_file: self._open_folder(
                    os.path.dirname(self._path_fields[k].text()) if f else self._path_fields[k].text()))
                grid.addWidget(mark, r, 0); grid.addWidget(QLabel(label), r, 1); grid.addWidget(edit, r, 2)
                grid.addWidget(browse, r, 3); grid.addWidget(opener, r, 4)
                grid.addWidget(dim(hint), r + 1, 2, 1, 3)
                self._path_fields[key], self._path_marks[key] = edit, mark
            grid.setColumnStretch(2, 1)
            grid.setColumnMinimumWidth(1, 150)
            col.addWidget(card)
        col.addWidget(dim("Green dot = found, amber = not found yet. Settings are saved automatically when you run a step or close the window."))
        col.addStretch()
        return scroll

    # ---- Pair check ----------------------------------------------------
    def _build_scan_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w); layout.setContentsMargins(18, 14, 18, 14); layout.setSpacing(10)
        btn_row = QHBoxLayout()
        scan_btn = QPushButton("Check censored / uncensored pairs")
        scan_btn.clicked.connect(self._scan_pairs)
        btn_row.addWidget(scan_btn)
        btn_row.addWidget(dim("Matches files by name and compares frame rate and length, to catch pairs that aren't the same cut."), stretch=1)
        layout.addLayout(btn_row)

        self.pairs_table = QTableWidget(0, 6)
        self.pairs_table.setHorizontalHeaderLabels(
            ["Name", "Censored fps", "Uncensored fps", "Censored length (s)", "Uncensored length (s)", "Status"])
        self.pairs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pairs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pairs_table.verticalHeader().setVisible(False)
        layout.addWidget(self.pairs_table, stretch=1)

        self.warnings_box = QTextEdit()
        self.warnings_box.setReadOnly(True)
        self.warnings_box.setMaximumHeight(150)
        self.warnings_box.setPlaceholderText("Results of the check appear here.")
        layout.addWidget(self.warnings_box)
        return w

    # ---- Buckets -------------------------------------------------------
    def _build_manifest_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w); layout.setContentsMargins(18, 14, 18, 14); layout.setSpacing(10)
        layout.addWidget(dim("Each bucket is a mosaic block size (% of the square the model sees: the crop around the mosaic area) with the model file that handles it. "
                             "Training makes one model per bucket; copy the checkpoint you like to the path shown here (with its .json)."))
        btn_row = QHBoxLayout()
        for text, slot in (("Install trained models…", self._install_models),
                           ("Reload", self._load_manifest), ("Add bucket", self._add_bucket_row),
                           ("Delete selected", self._delete_bucket_row), ("Save manifest", self._save_manifest)):
            b = QPushButton(text); b.clicked.connect(slot); btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.manifest_table = QTableWidget(0, 4)
        self.manifest_table.setHorizontalHeaderLabels(["Bucket %", "Model file", "Model", "Real-pair clips"])
        self.manifest_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.manifest_table.verticalHeader().setVisible(False)
        layout.addWidget(self.manifest_table, stretch=1)
        return w

    # ------------------------------------------------------------ config --
    def _update_hints(self, *_):
        size = self.model_size_combo.currentData() or 512
        load = loadsize_for(size)
        built = self.cfg.get('built_outsize', 0)
        text = f"Clips are stored at {load} px and trained on as {size} px crops."
        if built and built != load:
            text += f"  Your clips were last built at {built} px: rebuild steps 1-3 before training at this size."
        self.size_hint.setText(text)
        self.size_hint.setStyleSheet(f"color: {C['amber'] if built and built != load else C['text_dim']}; font-size: 12px;")

        batch = self.batchsize_spin.value()
        need = vram_needed(batch, size)
        text = f"Estimated GPU memory: about {need:.0f} GB for batch {batch} at {size} px ({GB_PER_SAMPLE[size]:.1f} GB per sample)."
        color = C['text_dim']
        if self._gpu_gb:
            best = max_batch(self._gpu_gb, size)
            if batch > best:
                text += f"  Your GPU has {self._gpu_gb:.0f} GB: batch {best} is about the most that fits."
                color = C['amber']
            else:
                text += f"  Fits your {self._gpu_gb:.0f} GB GPU (up to about batch {best})."
                color = C['green']
        else:
            text += "  Largest batch by card: " + ", ".join(f"{g} GB → {max_batch(g, size)}" for g in (8, 12, 16, 24)) + "."
        self.mem_hint.setText(text)
        self.mem_hint.setStyleSheet(f"color: {color}; font-size: 12px;")

        # Length of the run. Each clip gives 82 consecutive iterations per epoch (100 frames,
        # 5-frame windows, stride 3), and 80% of the clips are used for training.
        clips = pool_summary(self._path_fields['synthetic_pool_dir'].text().strip())[0] if hasattr(self, '_path_fields') else 0
        epochs = self.epoch_spin.value()
        if clips:
            per_epoch = int(clips * 0.8) // batch * 82
            total = per_epoch * epochs
            unit = 'bucket' if self.train_mode_combo.currentData() == 'buckets' else 'model'
            text = (f"Length: about {total:,} iterations per {unit} ({plural(clips, 'pool clip')} → {per_epoch:,} per epoch × "
                    f"{plural(epochs, 'epoch')}), a checkpoint every {self.save_freq_spin.value():,}. "
                    f"At 1 second per iteration that is {total / 3600:,.0f} hours per {unit}.")
            long_run = total > 300000
            if long_run:
                want = max(1, round(100000 / max(1, per_epoch)))
                text += f"  That is a very long run: {plural(want, 'epoch')} would give about {per_epoch * want:,} iterations."
            self.len_hint.setText(text)
            self.len_hint.setStyleSheet(f"color: {C['amber'] if long_run else C['text_dim']}; font-size: 12px;")
        else:
            self.len_hint.setText("")

    def _on_gen_kind(self, *_):
        procedural = self.gen_kind_combo.currentData() == 'generated'
        self.gen_style_combo.setEnabled(procedural)
        self.gen_skin_spin.setEnabled(procedural)
        self.gen_detail_spin.setEnabled(procedural)
        self.gen_texture_edit.setEnabled(procedural)
        self._refresh_status()

    def _browse_textures(self):
        path = QFileDialog.getExistingDirectory(self, "Folder of texture images", self.gen_texture_edit.text() or ROOT)
        if path:
            self.gen_texture_edit.setText(path)

    def _update_run_summary(self, *_):
        chosen = [c.number for c in self.cards if c.check.isChecked()]
        if chosen:
            self.run_summary.setText("Will run, in order:  " + "  →  ".join(f"{n}. {self.STEP_NAMES[n-1]}" for n in chosen))
        else:
            self.run_summary.setText("Tick at least one step to run.")
        self.run_all_btn.setEnabled(bool(chosen) and not self._busy())

    def _update_path_mark(self, key):
        path = self._path_fields[key].text().strip()
        ok = bool(path) and os.path.exists(path)
        self._path_marks[key].setStyleSheet(f"color: {C['green'] if ok else C['amber']};")
        self._path_marks[key].setToolTip("Found" if ok else "Not found")

    def _load_into_ui(self):
        for key, edit in self._path_fields.items():
            edit.setText(self.cfg.get(key, ''))
            self._update_path_mark(key)
        self.fps_spin.setValue(self.cfg.get('fps', 24))
        self.reuse_check.setChecked(self.cfg.get('reuse_uncensored', True))
        self.gpu_edit.setText(str(self.cfg.get('gpu_id', '0')))
        self.batchsize_spin.setValue(self.cfg.get('batchsize', 8))
        self.epoch_spin.setValue(self.cfg.get('n_epoch', 200))
        self.model_size_combo.setCurrentIndex(max(0, self.model_size_combo.findData(self.cfg.get('model_size', 512))))
        self.save_freq_spin.setValue(self.cfg.get('save_freq', 1000))
        self.scene_accept_spin.setValue(self.cfg.get('scene_accept_ratio', 1.0))
        self.gen_kind_combo.setCurrentIndex(max(0, self.gen_kind_combo.findData(self.cfg.get('gen_kind', 'generated'))))
        self.gen_clips_spin.setValue(int(self.cfg.get('gen_clips', 200)))
        self.gen_style_combo.setCurrentIndex(max(0, self.gen_style_combo.findData(self.cfg.get('gen_style', 'mixed'))))
        self.gen_skin_spin.setValue(int(self.cfg.get('gen_skin_pct', 60)))
        self.gen_detail_spin.setValue(int(self.cfg.get('gen_detail_pct', 100)))
        self.gen_texture_edit.setText(self.cfg.get('gen_texture_dir') or DEFAULTS['gen_texture_dir'])
        self.buckets_edit.setText(self.cfg.get('train_buckets', ''))
        i = self.train_mode_combo.findData(self.cfg.get('train_mode', 'single'))
        self.train_mode_combo.setCurrentIndex(max(0, i))
        self._on_train_mode()
        self.redo_check.setChecked(bool(self.cfg.get('redo_finished', False)))
        self.extra_dataset_edit.setText(self.cfg.get('extra_dataset_args', ''))
        self.extra_train_edit.setText(self.cfg.get('extra_train_args', ''))
        enabled = list(self.cfg.get('steps_enabled', DEFAULTS['steps_enabled'])) + [False] * 4
        for card, on in zip(self.cards, enabled):
            card.check.setChecked(bool(on))
        self._on_gen_kind()
        self._update_hints()
        self._update_run_summary()
        self._load_manifest()

    def _save_from_ui(self, quiet=True):
        for key, edit in self._path_fields.items():
            self.cfg[key] = edit.text().strip()
        self.cfg['fps'] = self.fps_spin.value()
        self.cfg['reuse_uncensored'] = self.reuse_check.isChecked()
        self.cfg['gpu_id'] = self.gpu_edit.text().strip() or '0'
        self.cfg['batchsize'] = self.batchsize_spin.value()
        self.cfg['n_epoch'] = self.epoch_spin.value()
        self.cfg['model_size'] = self.model_size_combo.currentData() or 512
        self.cfg['save_freq'] = self.save_freq_spin.value()
        self.cfg['scene_accept_ratio'] = self.scene_accept_spin.value()
        self.cfg['gen_kind'] = self.gen_kind_combo.currentData() or 'generated'
        self.cfg['gen_clips'] = self.gen_clips_spin.value()
        self.cfg['gen_style'] = self.gen_style_combo.currentData() or 'mixed'
        self.cfg['gen_skin_pct'] = self.gen_skin_spin.value()
        self.cfg['gen_detail_pct'] = self.gen_detail_spin.value()
        self.cfg['gen_texture_dir'] = self.gen_texture_edit.text().strip()
        self.cfg['train_buckets'] = self.buckets_edit.text().strip()
        self.cfg['train_mode'] = self.train_mode_combo.currentData()
        self.cfg['redo_finished'] = self.redo_check.isChecked()
        self.cfg['steps_enabled'] = [c.check.isChecked() for c in self.cards]
        self.cfg['extra_dataset_args'] = self.extra_dataset_edit.text().strip()
        self.cfg['extra_train_args'] = self.extra_train_edit.text().strip()
        save_config(self.cfg)
        if not quiet:
            self._log("Settings saved.")

    def _browse(self, key, is_file=False):
        current = self._path_fields[key].text() or ROOT
        if is_file:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", current)
        else:
            path = QFileDialog.getExistingDirectory(self, "Select folder", current)
        if path:
            self._path_fields[key].setText(path)
            self._refresh_status()

    def _open_folder(self, path):
        if path and os.path.isdir(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            self._log(f"Folder does not exist yet: {path}")

    # ------------------------------------------------------------ status --
    def _refresh_status(self):
        """Fill each card's status line from what is on disk right now."""
        p = {k: e.text().strip() for k, e in self._path_fields.items()}
        # 1 real pairs
        cen, unc = count_files(p['censored_dir'], VIDEO_EXTS), count_files(p['uncensored_dir'], VIDEO_EXTS)
        buckets = real_pairs_summary(p['real_pairs_dir'])
        built = (f"Built so far: {plural(sum(buckets.values()), 'clip')} in {plural(len(buckets), 'bucket')}."
                 if buckets else "Nothing built yet.")
        if cen is None or unc is None:
            self.cards[0].set_status("Censored or uncensored folder not found (Folders & models tab).  " + built, 'warn')
        elif not p['mosaic_position_model_path'] or not os.path.exists(p['mosaic_position_model_path']):
            self.cards[0].set_status(f"{plural(cen, 'censored video')}, {unc} uncensored.  Mosaic position model not set (Folders & models tab).  " + built, 'warn')
        else:
            self.cards[0].set_status(f"{plural(cen, 'censored video')}, {unc} uncensored.  " + built, 'ok' if cen and unc else 'info')
        # 2 + 3 pool
        total, gen, crop = pool_summary(p['synthetic_pool_dir'])
        src_v = count_files(p['synthetic_source_dir'], VIDEO_EXTS, recursive=True)
        src_i = count_files(p['synthetic_source_dir'], IMAGE_EXTS, recursive=True)
        pool_txt = f"Pool: {plural(total, 'clip')} ({total - gen - crop} from detection, {gen} procedural, {crop} region)."
        if not os.path.exists(p['roi_model_path']):
            self.cards[1].set_status("ROI model not found (Folders & models tab).  " + pool_txt, 'warn')
        elif not src_v:
            extra = " (step 1 can copy the uncensored clips there)" if self.reuse_check.isChecked() else ""
            self.cards[1].set_status(f"No videos in the clean footage folder yet{extra}.  " + pool_txt, 'warn' if not extra else 'info')
        else:
            self.cards[1].set_status(f"{plural(src_v, 'clean video')} to scan.  " + pool_txt, 'ok')
        if self.gen_kind_combo.currentData() == 'crops':
            if not (src_v or src_i):
                self.cards[2].set_status("No videos or images in the clean footage folder.  " + pool_txt, 'warn')
            else:
                self.cards[2].set_status(f"Source: {plural(src_v or 0, 'video')}, {plural(src_i or 0, 'image')} of clean footage.  " + pool_txt, 'ok')
        else:
            tex = self.gen_texture_edit.text().strip()
            n_tex = count_files(tex, IMAGE_EXTS, recursive=True) if tex and os.path.isdir(tex) else 0
            tex_txt = f"Textures: {plural(n_tex, 'image')}." if n_tex else "No texture images (drawn detail only)."
            self.cards[2].set_status(f"Needs no footage or model. {tex_txt}  " + pool_txt, 'ok')
        # 4 train
        man = manifest_summary(p['model_manifest'])
        if self.train_mode_combo.currentData() == 'single':
            if not total and not buckets:
                self.cards[3].set_status("Single model: no training clips yet. Run a step above first "
                                         "(or tick it to run before training).", 'warn')
            else:
                self.cards[3].set_status(f"Single model from {plural(total, 'pool clip')} + "
                                         f"{plural(sum(buckets.values()), 'real-pair clip')} (all buckets).", 'ok')
        elif man is None:
            self.cards[3].set_status("Size manifest not found or unreadable (Folders & models tab).", 'bad')
        elif not total and not buckets:
            self.cards[3].set_status(f"{plural(man[0], 'bucket')} in the manifest, but no training clips yet: run a step above first "
                                     f"(or tick it to run before training).", 'warn')
        else:
            self.cards[3].set_status(f"{plural(man[0], 'bucket')} to train; {man[1]} already have a model file.  "
                                     f"Data: {plural(total, 'pool clip')} + {plural(sum(buckets.values()), 'real-pair clip')}.", 'ok')
        self._update_hints()

    # --------------------------------------------------------------- log --
    def _log(self, text):
        self.log.appendPlainText(text)

    def _show_line(self, line, final):
        """Show the current output line. A line that hasn't ended yet (a progress bar
        that is still updating) is shown straight away and replaced in place by its
        next update, like in a terminal -- instead of staying hidden until it ends,
        which made a finished bar look stuck while the next one was already running."""
        if self._live:
            cur = self.log.textCursor()
            cur.movePosition(QTextCursor.MoveOperation.End)
            cur.select(QTextCursor.SelectionType.BlockUnderCursor)
            cur.removeSelectedText()
        self.log.appendPlainText(line)
        self._live = not final
        self._parse_progress(line)

    def _emit_line(self, line, replace=False):
        self._show_line(line, True)

    def _feed(self, text):
        for part in re.split(r'(\r\n|\n|\r)', text):
            if part in ('\n', '\r\n'):
                self._show_line(self._partial, True)
                self._partial, self._cr = '', False
            elif part == '\r':
                self._cr = True                     # what follows overwrites this line
            elif part:
                if self._cr:
                    self._partial, self._cr = '', False
                self._partial += part
                self._show_line(self._partial, False)

    def _parse_progress(self, line):
        m = re.search(r'=== Bucket ([\d.]+)% ===', line)
        if m:
            self._sub = f"bucket {m.group(1)}%"
            self.progress.setRange(0, 0)
            self._last_pct = None
            return
        if line.startswith(('Starting up to', 'Waiting for the first batch', 'Downloading:')):
            self.progress.setRange(0, 0)            # busy, no percentage yet
            self._last_pct = None
            return
        pct = None
        m = re.search(r'(\d{1,3})%\|', line)                 # tqdm bar
        if m:
            pct = int(m.group(1))
        else:
            m = re.search(r'(?<![\d.:/])(\d+)\s*/\s*(\d+)(?![\d.:/])', line)     # "45/200" (not dates, times or paths)
            if m and int(m.group(2)) > 1 and int(m.group(1)) <= int(m.group(2)):
                pct = int(100 * int(m.group(1)) / int(m.group(2)))
        if pct is not None:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, pct)))
            self._last_pct = pct

    # ------------------------------------------------------------- scan --
    def _scan_pairs(self):
        self._save_from_ui()
        censored_dir = self.cfg['censored_dir']
        uncensored_dir = self.cfg['uncensored_dir']
        self.pairs_table.setRowCount(0)
        warnings = []

        if not os.path.isdir(censored_dir):
            warnings.append(f"Censored folder not found: {censored_dir}")
        if not os.path.isdir(uncensored_dir):
            warnings.append(f"Uncensored folder not found: {uncensored_dir}")
        if warnings:
            self.warnings_box.setPlainText("\n".join(warnings))
            return

        censored_files = {f for f in os.listdir(censored_dir) if is_video(os.path.join(censored_dir, f))}
        uncensored_files = {f for f in os.listdir(uncensored_dir) if is_video(os.path.join(uncensored_dir, f))}
        common = sorted(censored_files & uncensored_files)
        only_c = sorted(censored_files - uncensored_files)
        only_u = sorted(uncensored_files - censored_files)

        n_fps_mismatch = 0
        n_dur_mismatch = 0
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for name in common:
                c_fps, c_dur = ffprobe_info(os.path.join(censored_dir, name))
                u_fps, u_dur = ffprobe_info(os.path.join(uncensored_dir, name))
                status_parts = []
                if c_fps is None or u_fps is None:
                    status_parts.append("Could not read (ffprobe failed)")
                else:
                    if abs(c_fps - u_fps) > 0.5:
                        status_parts.append("Frame rates differ")
                        n_fps_mismatch += 1
                    if c_dur is not None and u_dur is not None and abs(c_dur - u_dur) > 1.0:
                        status_parts.append(f"Lengths differ ({c_dur:.1f}s vs {u_dur:.1f}s)")
                        n_dur_mismatch += 1
                status = "; ".join(status_parts) if status_parts else "OK"
                row = self.pairs_table.rowCount()
                self.pairs_table.insertRow(row)
                values = [name,
                          f"{c_fps:.2f}" if c_fps else "?", f"{u_fps:.2f}" if u_fps else "?",
                          f"{c_dur:.1f}" if c_dur else "?", f"{u_dur:.1f}" if u_dur else "?", status]
                for col, val in enumerate(values):
                    item = QTableWidgetItem(val)
                    if col == 5:
                        item.setForeground(QColor(C['green'] if status == "OK" else C['red']))
                    self.pairs_table.setItem(row, col, item)
                QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()

        for names, where in ((only_c, "uncensored"), (only_u, "censored")):
            for name in names:
                row = self.pairs_table.rowCount()
                self.pairs_table.insertRow(row)
                self.pairs_table.setItem(row, 0, QTableWidgetItem(name))
                for col in range(1, 5):
                    self.pairs_table.setItem(row, col, QTableWidgetItem("-"))
                item_status = QTableWidgetItem(f"No file with this name in {where}/")
                item_status.setForeground(QColor(C['amber']))
                self.pairs_table.setItem(row, 5, item_status)

        warnings = [f"Matched pairs: {len(common)}"]
        if only_c:
            warnings.append(f"{len(only_c)} file(s) only in censored/ (no match): {only_c}")
        if only_u:
            warnings.append(f"{len(only_u)} file(s) only in uncensored/ (no match): {only_u}")
        if n_fps_mismatch:
            warnings.append(f"{n_fps_mismatch} pair(s) have different frame rates in the two files. This can mean the pair "
                             f"isn't the same content, or was re-encoded inconsistently. Worth spot-checking.")
        if n_dur_mismatch:
            warnings.append(f"{n_dur_mismatch} pair(s) differ in length by more than 1 second: a stronger sign of "
                             f"misalignment (extra intro/outro, different cut). Check these before training on them.")
        if not os.path.exists(self.cfg['model_manifest']):
            warnings.append(f"Size manifest not found at {self.cfg['model_manifest']}: needed before step 1 can run.")
        if not os.path.exists(self.cfg['mosaic_position_model_path']):
            warnings.append("Mosaic position model not found: needed before step 1 can run.")
        self.warnings_box.setPlainText("\n".join(warnings))
        self._log(f"Pair check: {len(common)} matched, {len(only_c)+len(only_u)} without a partner.")
        self._refresh_status()

    # ---------------------------------------------------------- manifest --
    def _load_manifest(self):
        self.manifest_table.setRowCount(0)
        path = self._path_fields['model_manifest'].text() if hasattr(self, '_path_fields') else self.cfg['model_manifest']
        if not path or not os.path.exists(path):
            return
        try:
            with open(path) as f:
                manifest = json.load(f)
        except Exception as e:
            self._log(f"Could not read manifest: {e}")
            return
        base_dir = os.path.dirname(os.path.abspath(path))
        real = real_pairs_summary(self._path_fields['real_pairs_dir'].text())
        for bucket, rel_path in sorted(manifest.items(), key=lambda kv: float(kv[0])):
            row = self.manifest_table.rowCount()
            self.manifest_table.insertRow(row)
            self.manifest_table.setItem(row, 0, QTableWidgetItem(str(bucket)))
            self.manifest_table.setItem(row, 1, QTableWidgetItem(str(rel_path)))
            abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(base_dir, rel_path)
            found = os.path.exists(abs_path)
            text = "not trained yet"
            if found:
                text = "ready"
                try:
                    with open(os.path.splitext(abs_path)[0] + '.json') as f:
                        text = f"ready ({int(json.load(f)['finesize'])} px)"
                except Exception:
                    text = "ready (size not recorded)"
            exists_item = QTableWidgetItem(text)
            exists_item.setForeground(QColor(C['green'] if found else C['amber']))
            exists_item.setFlags(exists_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.manifest_table.setItem(row, 2, exists_item)
            label = f"{float(bucket):g}"
            n_item = QTableWidgetItem(str(real.get(label, real.get(str(bucket), 0))))
            n_item.setFlags(n_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.manifest_table.setItem(row, 3, n_item)

    def _add_bucket_row(self):
        row = self.manifest_table.rowCount()
        self.manifest_table.insertRow(row)
        self.manifest_table.setItem(row, 0, QTableWidgetItem("0"))
        self.manifest_table.setItem(row, 1, QTableWidgetItem("clean_Xpct.pth"))
        self.manifest_table.setItem(row, 2, QTableWidgetItem(""))
        self.manifest_table.setItem(row, 3, QTableWidgetItem(""))

    def _on_train_mode(self, *_):
        buckets = self.train_mode_combo.currentData() == 'buckets'
        self.buckets_edit.setEnabled(buckets)
        self.redo_check.setText("Train finished buckets again" if buckets else "Train the finished model again")
        if hasattr(self, 'cards') and len(self.cards) > 3 and hasattr(self, '_path_fields'):
            self._refresh_status()

    def _install_models(self):
        if self._busy():
            QMessageBox.information(self, "Busy", "Wait for the running step to finish (a bucket may still be training).")
            return
        path = self._path_fields['model_manifest'].text()
        single = self.train_mode_combo.currentData() == 'single'
        try:
            plan = plan_model_install(path, single=single)
        except Exception as e:
            QMessageBox.warning(self, "No manifest", f"Could not read the size manifest:\n{path}\n\n{e}")
            return
        ready = [p for p in plan if p['src'] and p['finished']]
        unfinished = [p for p in plan if p['src'] and not p['finished']]
        lines = []
        for p in plan:
            name = os.path.basename(p['dst'])
            if not p['src']:
                lines.append(f"{_label(p['bucket'])}: skipped, {p['note']}")
                continue
            state = "" if p['finished'] else f"  [{p['note']}]"
            extra = " (replaces the current one, kept as .previous)" if p['replaces'] else ""
            nojson = "  [no .json: it would run at 256 px]" if not p['has_json'] else ""
            lines.append(f"{_label(p['bucket'])}: {os.path.basename(p['src'])} -> {name}{extra}{state}{nojson}")
        if not ready and not unfinished:
            QMessageBox.information(self, "Nothing to install", "No trained checkpoint yet.\n\n" + "\n".join(lines))
            return
        box = QMessageBox(self)
        box.setWindowTitle("Install trained models")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(f"Copy {len(ready)} finished model(s) to {os.path.dirname(os.path.abspath(plan[0]['dst']))}?")
        box.setInformativeText("\n".join(lines))
        b_fin = box.addButton(f"Install finished ({len(ready)})", QMessageBox.ButtonRole.AcceptRole) if ready else None
        b_all = box.addButton(f"Install all ({len(ready) + len(unfinished)})", QMessageBox.ButtonRole.AcceptRole) if unfinished else None
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        chosen = box.clickedButton()
        if chosen is None or chosen not in (b_fin, b_all):
            return
        todo = ready + (unfinished if chosen is b_all else [])
        self._log(f"\n=== Installing {len(todo)} model(s) ===")
        failed = 0
        for p in todo:
            try:
                install_model(p)
                self._log(f"  {p['bucket']}%: {os.path.basename(p['src'])} -> {p['dst']}"
                          + ("" if p['finished'] else "  (not finished)"))
            except Exception as e:
                failed += 1
                self._log(f"  {p['bucket']}%: FAILED ({e})")
        self._log("Done." + (f" {failed} failed." if failed else "") +
                  " In the main program, tick 'Auto-select model by mosaic size' to use them.")
        self._load_manifest()

    def _delete_bucket_row(self):
        rows = sorted({idx.row() for idx in self.manifest_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.manifest_table.removeRow(r)

    def _save_manifest(self):
        path = self._path_fields['model_manifest'].text()
        if not path:
            QMessageBox.warning(self, "No path", "Set the size manifest path on the Folders & models tab first.")
            return
        manifest = {}
        for row in range(self.manifest_table.rowCount()):
            bucket_item = self.manifest_table.item(row, 0)
            path_item = self.manifest_table.item(row, 1)
            if bucket_item and path_item and bucket_item.text().strip():
                manifest[bucket_item.text().strip()] = path_item.text().strip()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2)
        self._log(f"Manifest saved to {path}")
        self._load_manifest()
        self._refresh_status()

    # ------------------------------------------------------ step commands --
    def _build_step(self, index):
        """(command, working dir) for a step, or raise ValueError with what is missing."""
        cfg = self.cfg
        py = resolve_python_exe()
        outsize = str(loadsize_for(cfg['model_size']))
        if index == 0:
            if not os.path.isdir(cfg['censored_dir']) or not os.path.isdir(cfg['uncensored_dir']):
                raise ValueError("the censored or uncensored folder does not exist")
            if not cfg['mosaic_position_model_path'] or not os.path.exists(cfg['mosaic_position_model_path']):
                raise ValueError("the mosaic position model is not set or not found")
            if not os.path.exists(cfg['model_manifest']):
                raise ValueError("the size manifest was not found")
            return [py, os.path.join(ROOT, 'tools', 'bucket_real_pairs.py'),
                    '--censored_dir', cfg['censored_dir'],
                    '--uncensored_dir', cfg['uncensored_dir'],
                    '--output_dir', cfg['real_pairs_dir'],
                    '--mosaic_position_model_path', cfg['mosaic_position_model_path'],
                    '--model_manifest', cfg['model_manifest'],
                    '--fps', str(cfg['fps']),
                    '--reuse_uncensored_for_synthetic', 'yes' if cfg['reuse_uncensored'] else 'no',
                    '--synthetic_source_dir', cfg['synthetic_source_dir'],
                    '--outsize', outsize], ROOT
        if index == 1:
            if not os.path.exists(cfg['roi_model_path']):
                raise ValueError(f"the ROI model was not found at {cfg['roi_model_path']}")
            cmd = [py, os.path.join(ROOT, 'make_datasets', 'make_video_dataset.py'),
                   '--datadir', cfg['synthetic_source_dir'],
                   '--savedir', cfg['synthetic_pool_dir'],
                   '--model_path', cfg['roi_model_path'],
                   '--fps', str(cfg['fps']),
                   '--scene_accept_ratio', str(cfg['scene_accept_ratio']),
                   '--outsize', outsize]
            return cmd + cfg['extra_dataset_args'].split(), os.path.join(ROOT, 'make_datasets')
        if index == 2:
            cmd = [py, os.path.join(ROOT, 'tools', 'make_generated_dataset.py'),
                   '--mode', cfg['gen_kind'],
                   '--savedir', cfg['synthetic_pool_dir'],
                   '--clips', str(cfg['gen_clips']),
                   '--outsize', outsize]
            if cfg['gen_kind'] == 'crops':
                if not os.path.isdir(cfg['synthetic_source_dir']):
                    raise ValueError("the clean footage folder does not exist")
                cmd += ['--source', cfg['synthetic_source_dir']]
            else:
                cmd += ['--style', cfg['gen_style'], '--skin_tone', str(cfg['gen_skin_pct'] / 100.0),
                        '--detail', str(cfg.get('gen_detail_pct', 100) / 100.0)]
                tex = cfg.get('gen_texture_dir', '')
                if tex:
                    if not os.path.isdir(tex):
                        raise ValueError("the texture images folder does not exist")
                    if count_files(tex, IMAGE_EXTS, recursive=True):     # an empty folder is simply not used
                        cmd += ['--textures', tex]
            return cmd, ROOT
        single = cfg.get('train_mode', 'single') == 'single'
        if not single and not os.path.exists(cfg['model_manifest']):
            raise ValueError("the size manifest was not found (needed for size buckets)")
        cmd = [py, os.path.join(ROOT, 'tools', 'train_all_buckets.py'),
               '--synthetic_dataset', cfg['synthetic_pool_dir'],
               '--real_pairs_dataset', cfg['real_pairs_dir'],
               '--model_manifest', cfg['model_manifest'],
               '--work_dir', os.path.join(ROOT, 'datasets', '_merged')]
        if single:
            cmd += ['--single']
        elif cfg['train_buckets']:
            cmd += ['--buckets', cfg['train_buckets'].replace(' ', '')]
        if cfg.get('redo_finished'):
            cmd += ['--redo_finished']
        cmd += ['--', '--gpu_id', cfg['gpu_id'], '--batchsize', str(cfg['batchsize']), '--n_epoch', str(cfg['n_epoch']),
                '--save_freq', str(cfg['save_freq']),
                '--finesize', str(cfg['model_size']), '--loadsize', outsize]
        return cmd + cfg['extra_train_args'].split(), ROOT

    # -------------------------------------------------------------- run --
    def _busy(self):
        return self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning

    def _run_single(self, index):
        self._start([index])

    def _run_selected(self):
        self._start([i for i, c in enumerate(self.cards) if c.check.isChecked()])

    def _start(self, indices):
        if self._busy():
            QMessageBox.information(self, "Busy", "A step is already running. Stop it first, or wait for it to finish.")
            return
        if not indices:
            return
        self._save_from_ui()
        cfg = self.cfg
        # check EVERYTHING before starting, so a long run can't fail at step 4 over a missing file
        plan, problems = [], []
        for i in indices:
            try:
                cmd, cwd = self._build_step(i)
                plan.append((i, cmd, cwd))
            except ValueError as e:
                problems.append(f"{i + 1}. {self.STEP_NAMES[i]}: {e}")
        if 3 in indices and not any(i in indices for i in (0, 1, 2)):
            total = pool_summary(cfg['synthetic_pool_dir'])[0]
            if not total and not real_pairs_summary(cfg['real_pairs_dir']):
                problems.append("4. Train: there are no training clips yet. Tick a step that creates some (1-3) as well.")
        if problems:
            QMessageBox.warning(self, "Can't start yet", "Fix these first:\n\n" + "\n".join(problems) +
                                "\n\nPaths are on the Folders & models tab.")
            return
        load = loadsize_for(cfg['model_size'])
        built = cfg.get('built_outsize', 0)
        if 3 in indices and built and built != load and not (0 in indices or 1 in indices):
            ans = QMessageBox.question(self, "Clips were built at a different size",
                f"Your clips were last built at {built} px, but training at {cfg['model_size']} px uses {load} px clips.\n\n"
                f"They would be resized {load/built:.1f}x"
                f"{' (enlarged: no extra detail, more memory)' if load > built else ' (shrunk: detail is thrown away)'}.\n\n"
                f"Train anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                return
        if 0 in indices or 1 in indices:
            self.cfg['built_outsize'] = load
            save_config(self.cfg)
        for card in self.cards:
            card.set_state('')
        for i, _, _ in plan:
            self.cards[i].set_state('', 'queued')
        self._queue = plan
        self._run_total = len(plan)
        self._run_started = time.time()
        self._stopping = False
        self._set_running(True)
        self._next_step()

    def _set_running(self, running):
        for card in self.cards:
            card.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self._update_run_summary()
        if running:
            self.run_all_btn.setEnabled(False)

    def _next_step(self):
        if not self._queue:
            took = self._fmt(time.time() - self._run_started)
            self.run_title.setText("Finished")
            self.run_detail.setText(f"All {plural(self._run_total, 'step')} finished in {took}.")
            self.progress.setRange(0, 100); self.progress.setValue(100)
            self._log(f"\n=== All done in {took} ===\n")
            self._current = None
            self._set_running(False)
            self._refresh_status(); self._load_manifest()
            QApplication.alert(self)
            return
        index, cmd, cwd = self._queue.pop(0)
        self._current = index
        self._sub = ''
        self._last_pct = None
        self._step_started = time.time()
        self.cards[index].set_state('running', 'running…')
        self.progress.setRange(0, 0)       # busy until the step reports progress
        self._log(f"\n=== Step {index + 1}: {self.STEP_NAMES[index]} ===\n$ {' '.join(cmd)}\n")
        self._partial, self._cr, self._live = '', False, False
        self._process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")        # progress appears as it happens, not in bursts
        env.insert("PYTHONIOENCODING", "utf-8")
        self._process.setProcessEnvironment(env)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.setWorkingDirectory(cwd)
        self._process.readyReadStandardOutput.connect(self._on_process_output)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        self._process.start(cmd[0], cmd[1:])
        self._tick()

    def _on_process_output(self):
        self._feed(self._process.readAllStandardOutput().data().decode('utf-8', errors='replace'))

    def _on_error(self, err):
        if err == QProcess.ProcessError.FailedToStart:
            self._log("Could not start the process (is Python available?).")
            self._on_finished(-1, None)

    def _on_finished(self, code, _status):
        if self._current is None:
            return
        index, self._current = self._current, None
        try:      # anything still in the pipe when the process ended (its last words are the important ones)
            rest = self._process.readAllStandardOutput().data().decode('utf-8', errors='replace')
            if rest:
                self._feed(rest)
        except Exception:
            pass
        if self._live:
            self._show_line(self._partial, True)
        self._partial, self._cr, self._live = '', False, False
        took = self._fmt(time.time() - self._step_started)
        if code == 0 and not self._stopping:
            self.cards[index].set_state('done', f"done in {took}")
            self._refresh_status()
            self._next_step()
            return
        skipped = [i for i, _, _ in self._queue]
        self._queue = []
        for i in skipped:
            self.cards[i].set_state('', 'not run')
        if self._stopping:
            self.cards[index].set_state('failed', 'stopped')
            self.run_title.setText("Stopped")
            self.run_detail.setText(f"Stopped during step {index + 1} ({self.STEP_NAMES[index]}).")
        else:
            self.cards[index].set_state('failed', f"failed (exit code {code})")
            self.run_title.setText("Failed")
            self.run_detail.setText(f"Step {index + 1} ({self.STEP_NAMES[index]}) failed: the reason is at the end of the log."
                                    + (f" {plural(len(skipped), 'later step')} not run." if skipped else ""))
            self._log(f"\n=== Step {index + 1} failed (exit code {code}) ===")
            if index == 3:
                self._log("The complete training output is also saved in checkpoints/clean_<bucket>pct/train_log.txt\n")
            QApplication.alert(self)
        self.progress.setRange(0, 100); self.progress.setValue(0)
        self._set_running(False)
        self._refresh_status()

    def _stop_process(self):
        if not self._busy():
            return
        self._stopping = True
        pid = int(self._process.processId())
        # The training step starts train.py as its own child process; killing only the
        # parent would leave it running (and holding the GPU) on Windows.
        if os.name == 'nt' and pid:
            try:
                subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True, timeout=15,
                               creationflags=0x08000000)
            except Exception:
                pass
        self._process.kill()
        self._log("\nStopped by you.")

    @staticmethod
    def _fmt(seconds):
        seconds = int(seconds)
        h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
        return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")

    def _tick(self):
        if not self._busy() or self._current is None:
            self.elapsed_label.setText("")
            return
        done = self._run_total - len(self._queue)
        name = self.STEP_NAMES[self._current]
        self.run_title.setText(f"Step {done} of {self._run_total}: {name}" if self._run_total > 1 else name)
        bits = [b for b in (getattr(self, '_sub', ''), f"{self._last_pct}%" if getattr(self, '_last_pct', None) is not None else '') if b]
        waiting = [self.STEP_NAMES[i] for i, _, _ in self._queue]
        self.run_detail.setText((" · ".join(bits) or "working…") + (f"      Next: {', '.join(waiting)}" if waiting else ""))
        self.elapsed_label.setText(f"{self._fmt(time.time() - self._step_started)} this step · "
                                   f"{self._fmt(time.time() - self._run_started)} total")

    def closeEvent(self, event):
        if self._busy():
            ans = QMessageBox.question(self, "A step is still running",
                                       "Closing will stop the running step. Close anyway?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)
            if ans != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._stop_process()
        self._save_from_ui()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")      # same look on every Windows version; the stylesheet builds on it
    app.setStyleSheet(QSS)
    win = DatasetPrepWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
