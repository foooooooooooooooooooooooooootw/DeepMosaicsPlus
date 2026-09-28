"""
DeepMosaicsPlus — Dataset Prep & Training UI

A small front-end over the dataset-building workflow discussed in chat:
  - Lets you point at your censored/, uncensored/, and synthetic_source/
    folders (see datasets/raw/README.md) instead of remembering CLI paths.
  - Scans your censored/uncensored pairs, shows which filenames matched,
    which didn't, and flags fps/duration mismatches that suggest a pair
    might not actually be aligned.
  - Lets you view/edit your size_manifest.json buckets and see which
    checkpoint files actually exist yet.
  - Runs tools/bucket_real_pairs.py, make_datasets/make_video_dataset.py,
    and tools/train_all_buckets.py as subprocesses with live log output,
    so you don't have to remember or retype the full command lines.

This does not replace understanding what those tools do — it's meant to
reduce how much of the workflow you have to hold in your head at once,
not to hide it. Run with: python tools/dataset_prep_ui.pyw
"""
import os
import sys
import json
import glob

try:
    from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment
    from PyQt6.QtGui import QFont, QColor
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox,
        QTableWidget, QTableWidgetItem, QTextEdit, QPlainTextEdit, QTabWidget,
        QGroupBox, QHeaderView, QMessageBox, QAbstractItemView,
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
    'save_freq': 1000,
    'scene_accept_ratio': 1.0,
    'extra_dataset_args': '',
    'extra_train_args': '',
}

VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm')


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
    import subprocess
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
        fps = eval(fps_str) if fps_str != '0/0' else 0
        duration = float(infos['format']['duration'])
        if not fps:
            return None, duration
        return fps, duration
    except Exception:
        return None, None


def path_row(label_text, key, browse_mode='dir'):
    """Returns (label, lineedit, browse_button) — caller adds to layout."""
    label = QLabel(label_text)
    edit = QLineEdit()
    btn = QPushButton("Browse…")
    return label, edit, btn


class DatasetPrepWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepMosaicsPlus 1.2.0 — Dataset Prep & Training")
        self.resize(1040, 780)
        self.cfg = load_config()
        self._process = None
        self._build_ui()
        self._load_into_ui()

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        tabs = QTabWidget()
        root.addWidget(tabs, stretch=1)

        tabs.addTab(self._build_paths_tab(), "1. Paths && Settings")
        tabs.addTab(self._build_scan_tab(), "2. Scan Pairs")
        tabs.addTab(self._build_manifest_tab(), "3. Buckets")
        tabs.addTab(self._build_run_tab(), "4. Run")

        # Log panel, shared across all actions, always visible at the bottom
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas" if sys.platform == 'win32' else "Monospace", 9))
        self.log.setMaximumBlockCount(5000)
        log_layout.addWidget(self.log)
        root.addWidget(log_group, stretch=1)

    def _build_paths_tab(self):
        w = QWidget()
        grid = QGridLayout(w)
        self._path_fields = {}
        rows = [
            ("Censored clips folder:", 'censored_dir'),
            ("Uncensored clips folder:", 'uncensored_dir'),
            ("Synthetic source (clean) folder:", 'synthetic_source_dir'),
            ("Real pairs output folder:", 'real_pairs_dir'),
            ("Synthetic pool output folder:", 'synthetic_pool_dir'),
            ("Model manifest (size_manifest.json):", 'model_manifest'),
            ("Mosaic position model (.pth):", 'mosaic_position_model_path'),
            ("ROI/face model for synthetic pool (add_face.pth):", 'roi_model_path'),
        ]
        for i, (label_text, key) in enumerate(rows):
            label = QLabel(label_text)
            edit = QLineEdit()
            is_file = key in ('model_manifest', 'mosaic_position_model_path', 'roi_model_path')
            btn = QPushButton("Browse…")
            btn.clicked.connect(lambda _, k=key, f=is_file: self._browse(k, is_file=f))
            grid.addWidget(label, i, 0)
            grid.addWidget(edit, i, 1)
            grid.addWidget(btn, i, 2)
            self._path_fields[key] = edit

        r = len(rows)
        grid.addWidget(QLabel("Target extraction FPS (used for all steps):"), r, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        grid.addWidget(self.fps_spin, r, 1)
        r += 1

        self.reuse_check = QCheckBox("Reuse uncensored clips as synthetic source too (recommended)")
        grid.addWidget(self.reuse_check, r, 0, 1, 3)
        r += 1

        grid.addWidget(QLabel("GPU id (-1 for CPU):"), r, 0)
        self.gpu_edit = QLineEdit()
        grid.addWidget(self.gpu_edit, r, 1)
        r += 1

        grid.addWidget(QLabel("Batch size:"), r, 0)
        self.batchsize_spin = QSpinBox()
        self.batchsize_spin.setRange(1, 256)
        grid.addWidget(self.batchsize_spin, r, 1)
        r += 1

        grid.addWidget(QLabel("Epochs:"), r, 0)
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 100000)
        grid.addWidget(self.epoch_spin, r, 1)
        r += 1

        grid.addWidget(QLabel("Save checkpoint every N iterations (--save_freq):"), r, 0)
        self.save_freq_spin = QSpinBox()
        self.save_freq_spin.setRange(1, 1000000)
        self.save_freq_spin.setSingleStep(100)
        grid.addWidget(self.save_freq_spin, r, 1)
        r += 1
        save_freq_hint = QLabel("Lower this for small datasets (e.g. 500-1000) so you get intermediate "
                                 "checkpoints during training, not just whatever the final iteration produces. "
                                 "A guaranteed final save always happens regardless of this value.")
        save_freq_hint.setWordWrap(True)
        grid.addWidget(save_freq_hint, r, 0, 1, 3)
        r += 1

        grid.addWidget(QLabel("Scene acceptance ratio (synthetic pool, --scene_accept_ratio):"), r, 0)
        self.scene_accept_spin = QDoubleSpinBox()
        self.scene_accept_spin.setRange(0.1, 1.0)
        self.scene_accept_spin.setSingleStep(0.1)
        self.scene_accept_spin.setDecimals(2)
        grid.addWidget(self.scene_accept_spin, r, 1)
        r += 1
        scene_accept_hint = QLabel("Fraction of sampled points within a candidate scene that must pass ROI "
                                    "detection for that scene to be kept. 1.0 = ALL must pass (strict, the "
                                    "original default -- one weak frame throws out the whole scene). Try 0.5-0.6 "
                                    "if buckets are coming out with very little data despite plenty of source "
                                    "footage.")
        scene_accept_hint.setWordWrap(True)
        grid.addWidget(scene_accept_hint, r, 0, 1, 3)
        r += 1

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_from_ui)
        grid.addWidget(save_btn, r, 0, 1, 3)

        grid.setRowStretch(r + 1, 1)
        return w

    def _build_scan_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        btn_row = QHBoxLayout()
        scan_btn = QPushButton("Scan censored/ + uncensored/ for pairs")
        scan_btn.clicked.connect(self._scan_pairs)
        btn_row.addWidget(scan_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.pairs_table = QTableWidget(0, 6)
        self.pairs_table.setHorizontalHeaderLabels(
            ["Name", "Censored FPS", "Uncensored FPS", "Censored Dur (s)", "Uncensored Dur (s)", "Status"])
        self.pairs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pairs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.pairs_table, stretch=1)

        self.warnings_box = QTextEdit()
        self.warnings_box.setReadOnly(True)
        self.warnings_box.setMaximumHeight(160)
        layout.addWidget(QLabel("Warnings:"))
        layout.addWidget(self.warnings_box)

        return w

    def _build_manifest_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        btn_row = QHBoxLayout()
        reload_btn = QPushButton("Reload manifest")
        reload_btn.clicked.connect(self._load_manifest)
        add_btn = QPushButton("Add bucket row")
        add_btn.clicked.connect(self._add_bucket_row)
        del_btn = QPushButton("Delete selected row")
        del_btn.clicked.connect(self._delete_bucket_row)
        save_btn = QPushButton("Save manifest")
        save_btn.clicked.connect(self._save_manifest)
        for b in (reload_btn, add_btn, del_btn, save_btn):
            btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.manifest_table = QTableWidget(0, 3)
        self.manifest_table.setHorizontalHeaderLabels(["Bucket %", "Checkpoint path", "Exists?"])
        self.manifest_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.manifest_table, stretch=1)

        return w

    def _build_run_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        for text, handler, hint in [
            ("1) Bucket real pairs (detect size + sort into buckets)", self._run_bucket_pairs,
             "Runs tools/bucket_real_pairs.py"),
            ("2) Build synthetic pool from synthetic_source/", self._run_make_synthetic,
             "Runs make_datasets/make_video_dataset.py"),
            ("3) Train all buckets", self._run_train_all,
             "Runs tools/train_all_buckets.py"),
        ]:
            box = QGroupBox(text)
            box_layout = QVBoxLayout(box)
            box_layout.addWidget(QLabel(hint))
            run_btn = QPushButton("Run")
            run_btn.clicked.connect(handler)
            box_layout.addWidget(run_btn)
            layout.addWidget(box)

        extra_group = QGroupBox("Extra arguments (advanced, optional)")
        extra_layout = QGridLayout(extra_group)
        extra_layout.addWidget(QLabel("Extra args for make_video_dataset.py:"), 0, 0)
        self.extra_dataset_edit = QLineEdit()
        extra_layout.addWidget(self.extra_dataset_edit, 0, 1)
        extra_layout.addWidget(QLabel("Extra args for train_all_buckets.py (after --):"), 1, 0)
        self.extra_train_edit = QLineEdit()
        extra_layout.addWidget(self.extra_train_edit, 1, 1)
        layout.addWidget(extra_group)

        stop_btn = QPushButton("Stop running process")
        stop_btn.clicked.connect(self._stop_process)
        layout.addWidget(stop_btn)

        layout.addStretch()
        return w

    # ------------------------------------------------------------ config --
    def _load_into_ui(self):
        for key, edit in self._path_fields.items():
            edit.setText(self.cfg.get(key, ''))
        self.fps_spin.setValue(self.cfg.get('fps', 24))
        self.reuse_check.setChecked(self.cfg.get('reuse_uncensored', True))
        self.gpu_edit.setText(str(self.cfg.get('gpu_id', '0')))
        self.batchsize_spin.setValue(self.cfg.get('batchsize', 8))
        self.epoch_spin.setValue(self.cfg.get('n_epoch', 200))
        self.save_freq_spin.setValue(self.cfg.get('save_freq', 1000))
        self.scene_accept_spin.setValue(self.cfg.get('scene_accept_ratio', 1.0))
        self.extra_dataset_edit.setText(self.cfg.get('extra_dataset_args', ''))
        self.extra_train_edit.setText(self.cfg.get('extra_train_args', ''))
        self._load_manifest()

    def _save_from_ui(self):
        for key, edit in self._path_fields.items():
            self.cfg[key] = edit.text().strip()
        self.cfg['fps'] = self.fps_spin.value()
        self.cfg['reuse_uncensored'] = self.reuse_check.isChecked()
        self.cfg['gpu_id'] = self.gpu_edit.text().strip()
        self.cfg['batchsize'] = self.batchsize_spin.value()
        self.cfg['n_epoch'] = self.epoch_spin.value()
        self.cfg['save_freq'] = self.save_freq_spin.value()
        self.cfg['scene_accept_ratio'] = self.scene_accept_spin.value()
        self.cfg['extra_dataset_args'] = self.extra_dataset_edit.text().strip()
        self.cfg['extra_train_args'] = self.extra_train_edit.text().strip()
        save_config(self.cfg)
        self._log("Settings saved.")

    def _browse(self, key, is_file=False):
        current = self._path_fields[key].text() or ROOT
        if is_file:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", current)
        else:
            path = QFileDialog.getExistingDirectory(self, "Select folder", current)
        if path:
            self._path_fields[key].setText(path)

    # --------------------------------------------------------------- log --
    def _log(self, text):
        self.log.appendPlainText(text)

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

        target_fps = self.fps_spin.value()
        n_fps_mismatch = 0
        n_dur_mismatch = 0

        for name in common:
            c_path = os.path.join(censored_dir, name)
            u_path = os.path.join(uncensored_dir, name)
            c_fps, c_dur = ffprobe_info(c_path)
            u_fps, u_dur = ffprobe_info(u_path)

            status_parts = []
            if c_fps is None or u_fps is None:
                status_parts.append("COULD NOT READ (ffprobe failed)")
            else:
                if abs(c_fps - u_fps) > 0.5:
                    status_parts.append("FPS MISMATCH between pair")
                    n_fps_mismatch += 1
                if c_dur is not None and u_dur is not None and abs(c_dur - u_dur) > 1.0:
                    status_parts.append(f"DURATION MISMATCH ({c_dur:.1f}s vs {u_dur:.1f}s)")
                    n_dur_mismatch += 1
            status = "; ".join(status_parts) if status_parts else "OK"

            row = self.pairs_table.rowCount()
            self.pairs_table.insertRow(row)
            values = [
                name,
                f"{c_fps:.2f}" if c_fps else "?",
                f"{u_fps:.2f}" if u_fps else "?",
                f"{c_dur:.1f}" if c_dur else "?",
                f"{u_dur:.1f}" if u_dur else "?",
                status,
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                if col == 5 and status != "OK":
                    item.setForeground(QColor("#e05555"))
                self.pairs_table.setItem(row, col, item)

        for name in only_c:
            row = self.pairs_table.rowCount()
            self.pairs_table.insertRow(row)
            item_name = QTableWidgetItem(name)
            item_status = QTableWidgetItem("ORPHAN — no match in uncensored/")
            item_status.setForeground(QColor("#e0a555"))
            self.pairs_table.setItem(row, 0, item_name)
            for col in range(1, 5):
                self.pairs_table.setItem(row, col, QTableWidgetItem("-"))
            self.pairs_table.setItem(row, 5, item_status)

        for name in only_u:
            row = self.pairs_table.rowCount()
            self.pairs_table.insertRow(row)
            item_name = QTableWidgetItem(name)
            item_status = QTableWidgetItem("ORPHAN — no match in censored/")
            item_status.setForeground(QColor("#e0a555"))
            self.pairs_table.setItem(row, 0, item_name)
            for col in range(1, 5):
                self.pairs_table.setItem(row, col, QTableWidgetItem("-"))
            self.pairs_table.setItem(row, 5, item_status)

        # Aggregate warnings
        warnings = []
        warnings.append(f"Matched pairs: {len(common)}")
        if only_c:
            warnings.append(f"{len(only_c)} file(s) only in censored/ (no match): {only_c}")
        if only_u:
            warnings.append(f"{len(only_u)} file(s) only in uncensored/ (no match): {only_u}")
        if n_fps_mismatch:
            warnings.append(f"{n_fps_mismatch} pair(s) have mismatched native fps between the two "
                             f"files — this can be a sign the pair isn't actually the same content, "
                             f"or was re-encoded inconsistently. Worth spot-checking.")
        if n_dur_mismatch:
            warnings.append(f"{n_dur_mismatch} pair(s) have duration differing by more than 1 second "
                             f"— a stronger sign of misalignment (extra intro/outro, different cut, "
                             f"etc.). Recommend checking these before including them in training.")
        synth_dir = self.cfg['synthetic_source_dir']
        if os.path.isdir(synth_dir) and not any(is_video(os.path.join(synth_dir, f)) for f in os.listdir(synth_dir)):
            warnings.append(f"synthetic_source/ is empty — that's fine if you're training real-pairs-"
                             f"only, otherwise add clean footage there or enable 'reuse uncensored' above.")
        if not os.path.exists(self.cfg['model_manifest']):
            warnings.append(f"model_manifest not found at {self.cfg['model_manifest']} — needed before "
                             f"bucket detection can run.")
        if not os.path.exists(self.cfg['mosaic_position_model_path']):
            warnings.append(f"mosaic_position_model_path not found — needed before bucket detection can run.")
        if not os.path.exists(self.cfg['roi_model_path']):
            warnings.append(f"roi_model_path (add_face.pth) not found at {self.cfg['roi_model_path']} — "
                             f"needed before Step 2 (Build synthetic pool) can run.")

        self.warnings_box.setPlainText("\n".join(warnings) if warnings else "No issues found.")
        self._log(f"Scan complete: {len(common)} matched, {len(only_c)+len(only_u)} orphan(s).")

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
        for bucket, rel_path in sorted(manifest.items(), key=lambda kv: float(kv[0])):
            row = self.manifest_table.rowCount()
            self.manifest_table.insertRow(row)
            self.manifest_table.setItem(row, 0, QTableWidgetItem(str(bucket)))
            self.manifest_table.setItem(row, 1, QTableWidgetItem(str(rel_path)))
            abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(base_dir, rel_path)
            exists_item = QTableWidgetItem("yes" if os.path.exists(abs_path) else "NOT FOUND")
            if not os.path.exists(abs_path):
                exists_item.setForeground(QColor("#e0a555"))
            self.manifest_table.setItem(row, 2, exists_item)

    def _add_bucket_row(self):
        row = self.manifest_table.rowCount()
        self.manifest_table.insertRow(row)
        self.manifest_table.setItem(row, 0, QTableWidgetItem("0"))
        self.manifest_table.setItem(row, 1, QTableWidgetItem("clean_Xpct.pth"))
        self.manifest_table.setItem(row, 2, QTableWidgetItem(""))

    def _delete_bucket_row(self):
        rows = sorted({idx.row() for idx in self.manifest_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.manifest_table.removeRow(r)

    def _save_manifest(self):
        path = self._path_fields['model_manifest'].text()
        if not path:
            QMessageBox.warning(self, "No path", "Set a model_manifest path first.")
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

    # -------------------------------------------------------------- run --
    def _run_command(self, cmd, cwd=None):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Busy", "A process is already running. Stop it first.")
            return
        self._log(f"\n$ {' '.join(cmd)}\n")
        self._process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        self._process.setProcessEnvironment(env)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        if cwd:
            self._process.setWorkingDirectory(cwd)
        self._process.readyReadStandardOutput.connect(self._on_process_output)
        self._process.finished.connect(lambda code, status: self._log(f"\n[process exited with code {code}]\n"))
        self._process.start(cmd[0], cmd[1:])

    def _on_process_output(self):
        data = self._process.readAllStandardOutput().data().decode('utf-8', errors='replace')
        self.log.appendPlainText(data.rstrip('\n'))

    def _stop_process(self):
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
            self._log("Process stopped.")
        else:
            self._log("No process running.")

    def _run_bucket_pairs(self):
        self._save_from_ui()
        cfg = self.cfg
        if not os.path.exists(cfg['mosaic_position_model_path']):
            QMessageBox.warning(self, "Missing model", "Set a valid mosaic_position_model_path first.")
            return
        cmd = [resolve_python_exe(), os.path.join(ROOT, 'tools', 'bucket_real_pairs.py'),
               '--censored_dir', cfg['censored_dir'],
               '--uncensored_dir', cfg['uncensored_dir'],
               '--output_dir', cfg['real_pairs_dir'],
               '--mosaic_position_model_path', cfg['mosaic_position_model_path'],
               '--model_manifest', cfg['model_manifest'],
               '--fps', str(cfg['fps']),
               '--reuse_uncensored_for_synthetic', 'yes' if cfg['reuse_uncensored'] else 'no',
               '--synthetic_source_dir', cfg['synthetic_source_dir']]
        self._run_command(cmd, cwd=ROOT)

    def _run_make_synthetic(self):
        self._save_from_ui()
        cfg = self.cfg
        if not os.path.exists(cfg['roi_model_path']):
            QMessageBox.warning(self, "Missing model",
                f"ROI/face model not found at {cfg['roi_model_path']}. "
                f"Set the correct path on the Paths tab first.")
            return
        cmd = [resolve_python_exe(), os.path.join(ROOT, 'make_datasets', 'make_video_dataset.py'),
               '--datadir', cfg['synthetic_source_dir'],
               '--savedir', cfg['synthetic_pool_dir'],
               '--model_path', cfg['roi_model_path'],
               '--fps', str(cfg['fps']),
               '--scene_accept_ratio', str(cfg['scene_accept_ratio'])]
        if self.extra_dataset_edit.text().strip():
            cmd += self.extra_dataset_edit.text().strip().split()
        self._run_command(cmd, cwd=os.path.join(ROOT, 'make_datasets'))

    def _run_train_all(self):
        self._save_from_ui()
        cfg = self.cfg
        cmd = [resolve_python_exe(), os.path.join(ROOT, 'tools', 'train_all_buckets.py'),
               '--synthetic_dataset', cfg['synthetic_pool_dir'],
               '--real_pairs_dataset', cfg['real_pairs_dir'],
               '--model_manifest', cfg['model_manifest'],
               '--work_dir', os.path.join(ROOT, 'datasets', '_merged'),
               '--']
        cmd += ['--gpu_id', cfg['gpu_id'], '--batchsize', str(cfg['batchsize']), '--n_epoch', str(cfg['n_epoch']),
                '--save_freq', str(cfg['save_freq'])]
        if self.extra_train_edit.text().strip():
            cmd += self.extra_train_edit.text().strip().split()
        self._run_command(cmd, cwd=ROOT)


def main():
    app = QApplication(sys.argv)
    win = DatasetPrepWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
