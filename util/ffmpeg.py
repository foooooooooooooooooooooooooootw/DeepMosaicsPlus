import os, json, contextlib, tempfile, platform
import subprocess
from multiprocessing import Pool, Manager
from pathlib import Path
import threading
from tqdm import tqdm
import time
import re

hwaccel = None

# ── ffmpeg/ffprobe availability + hardware-decode detection (LAZY) ───────────
# These used to run at import time, which meant every run -- including
# single-image runs that never touch ffmpeg -- spawned three subprocesses
# (ffmpeg -version, ffprobe -version, ffmpeg -hwaccels) and printed a
# "[ffmpeg] Hardware decode" line. They now run once, on first real use:
# the first frame extraction/encode, or an explicit check_ffmpeg_available().
FFMPEG_AVAILABLE = False
FFMPEG_VERSION = None
FFPROBE_AVAILABLE = False
_probed = False

def _directml_available():
    import importlib.util
    return importlib.util.find_spec('torch_directml') is not None

def _probe_binary_version(binary):
    """Return (available, version_string_or_None) for a command-line tool."""
    try:
        r = subprocess.run([binary, '-version'], capture_output=True, text=True, timeout=5)
        first_line = (r.stdout or r.stderr or '').splitlines()[0] if (r.stdout or r.stderr) else ''
        return True, first_line.strip()
    except FileNotFoundError:
        return False, None
    except Exception:
        # e.g. permissions error, corrupt binary — treat as "found but broken"
        return True, None

def _probe_ffmpeg_hwaccels():
    """Return set of hwaccel names ffmpeg was built with."""
    if not FFMPEG_AVAILABLE:
        return set()
    try:
        r = subprocess.run(['ffmpeg', '-hwaccels', '-hide_banner'],
                           capture_output=True, text=True, timeout=5)
        lines = (r.stdout + r.stderr).splitlines()
        accels = set()
        capture = False
        for line in lines:
            if 'Hardware acceleration methods:' in line:
                capture = True
                continue
            if capture and line.strip():
                accels.add(line.strip())
        return accels
    except Exception:
        return set()

def _ensure_probed():
    """Run the ffmpeg/ffprobe/hwaccel probes once, on first need."""
    global _probed, FFMPEG_AVAILABLE, FFMPEG_VERSION, FFPROBE_AVAILABLE, hwaccel
    if _probed:
        return
    _probed = True
    # Worker processes: the main process already probed and passed its result
    # down (environment variables are inherited by spawned workers). Reuse it
    # silently instead of re-running three ffmpeg probes and printing the
    # "Hardware decode" line once per worker.
    if os.environ.get('DMP_FFMPEG_PROBED') == '1':
        FFMPEG_AVAILABLE = FFPROBE_AVAILABLE = True
        hw = os.environ.get('DMP_HWACCEL', 'none')
        hwaccel = None if hw.lower() == 'none' else hw
        return
    FFMPEG_AVAILABLE, FFMPEG_VERSION = _probe_binary_version('ffmpeg')
    FFPROBE_AVAILABLE, _ = _probe_binary_version('ffprobe')
    if not FFMPEG_AVAILABLE:
        hwaccel = None
        return   # check_ffmpeg_available() reports the missing binaries clearly
    supported = _probe_ffmpeg_hwaccels()
    try:
        import torch
        _sys = platform.system()
        if torch.cuda.is_available():
            # nvdec is the correct ffmpeg decoder name for NVIDIA GPU decode
            hwaccel = 'nvdec' if 'nvdec' in supported else ('cuda' if 'cuda' in supported else None)
        elif _directml_available() or _sys == 'Windows':
            # d3d11va works on Windows for both AMD and Intel
            hwaccel = 'd3d11va' if 'd3d11va' in supported else ('dxva2' if 'dxva2' in supported else None)
        elif _sys == 'Linux':
            # vaapi covers AMD and Intel on Linux
            hwaccel = 'vaapi' if 'vaapi' in supported else None
    except Exception:
        hwaccel = None
    # DMP_HWACCEL=none forces software decode. Set automatically by
    # _disable_hwaccel() so worker processes (which re-probe independently,
    # especially under Windows' spawn) inherit the decision instead of each
    # rediscovering the failure; can also be set by hand to force software.
    if os.environ.get('DMP_HWACCEL', '').lower() == 'none':
        hwaccel = None
    else:
        print(f"[ffmpeg] Hardware decode: {hwaccel}" if hwaccel else "[ffmpeg] Hardware decode: software fallback")
    # hand the result to any worker processes started from here on
    if FFPROBE_AVAILABLE:
        os.environ['DMP_FFMPEG_PROBED'] = '1'
        os.environ['DMP_HWACCEL'] = hwaccel or 'none'

def _disable_hwaccel():
    """A hardware-decoded ffmpeg call failed: switch this process to software
    decoding for the rest of the run. hwaccel is chosen from what ffmpeg was
    BUILT with, which doesn't guarantee a working device (no GPU, VM, broken
    or outdated driver) -- in that case ffmpeg exits immediately with e.g.
    'No device available for decoder'. Retrying in software is always safe."""
    global hwaccel
    if hwaccel is not None:
        print(f"[ffmpeg] Hardware decode ({hwaccel}) failed on this system -- falling back to software decode")
        hwaccel = None
    os.environ['DMP_HWACCEL'] = 'none'

def _validate_hwaccel(videopath):
    """Test-decode a fraction of a second of the real input with the chosen
    hardware decoder, once, in the main process -- before worker processes
    start -- so a broken decoder is detected (and reported) once rather than
    once per worker."""
    _ensure_probed()
    if hwaccel is None:
        return
    r = subprocess.run(['ffmpeg', '-v', 'error', '-hwaccel', hwaccel, '-t', '0.1',
                        '-i', videopath, '-f', 'null', '-'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       stdin=subprocess.DEVNULL, timeout=60)
    if r.returncode != 0:
        _disable_hwaccel()

def check_ffmpeg_available(raise_on_missing=True):
    """Explicit preflight check. Call this early (before model loading) when
    the run involves video, so missing ffmpeg/ffprobe fails fast with a clear
    message instead of a confusing WinError/OSError deep inside a later
    subprocess call.
    """
    _ensure_probed()
    missing = [name for name, ok in (('ffmpeg', FFMPEG_AVAILABLE), ('ffprobe', FFPROBE_AVAILABLE)) if not ok]
    if not missing:
        return True
    msg = (
        f"[ffmpeg] ERROR: {', '.join(missing)} not found on PATH.\n"
        "DeepMosaicsPlus needs ffmpeg (which includes ffprobe) to read/write video and audio.\n"
        "  1. Download a build from https://ffmpeg.org/download.html\n"
        "     (Windows users: the 'essentials' or 'full' build from gyan.dev is fine)\n"
        "  2. Extract it and add its 'bin' folder (containing ffmpeg.exe / ffprobe.exe) to your PATH\n"
        "  3. Close and reopen your terminal/command prompt so the updated PATH takes effect\n"
        "  4. Verify with: ffmpeg -version\n"
    )
    if raise_on_missing:
        raise FileNotFoundError(msg)
    print(msg)
    return False

def ffmpeg_status():
    """(ffmpeg_available, ffmpeg_version, ffprobe_available) — probes on first call."""
    _ensure_probed()
    return FFMPEG_AVAILABLE, FFMPEG_VERSION, FFPROBE_AVAILABLE

# ── Safe path: hardlink trick ────────────────────────────────────────────────
def _is_safe_path(p):
    """Plain ASCII and no glob brackets -- what ffmpeg/OpenCV handle reliably."""
    try:
        p.encode('ascii')
    except UnicodeEncodeError:
        return False
    return '[' not in p and ']' not in p

@contextlib.contextmanager
def safe_input_path(path, temp_dir=None):
    """Yield a plain-ASCII path pointing to *path* (brackets / non-ASCII names
    are otherwise mishandled by ffmpeg and OpenCV on Windows).

    Tries, in order, and never the root of a drive (writing there needs admin
    rights -- the old fallback, which failed with "Access is denied"):
      1. hard link next to the program's temp folder, if on the same drive
      2. hard link in the source file's own folder (same drive by definition),
         if that folder's own path is safe
      3. copy next to the temp folder (works across drives; slower for big files)
      4. copy in the system temp folder
    Falls back to the original path if every attempt fails.
    """
    import shutil
    path = str(path)
    if _is_safe_path(path):
        yield path
        return

    def same_drive(a, b):
        return os.path.splitdrive(a)[0].lower() == os.path.splitdrive(b)[0].lower()

    src_dir = os.path.dirname(os.path.abspath(path))
    # Parent of temp_dir: ours and writable. Not temp_dir itself -- video_init's
    # file_init() wipes and recreates it right after this link is made. Made
    # absolute BEFORE comparing drives: temp_dir is usually relative ('./tmp/..'),
    # so the old comparison never matched on Windows and always fell through
    # to the drive root.
    work_dir = os.path.dirname(os.path.abspath(temp_dir)) if temp_dir else tempfile.gettempdir()
    sys_tmp = tempfile.gettempdir()
    attempts = []
    if same_drive(work_dir, src_dir):
        attempts.append((work_dir, 'link'))
    attempts.append((src_dir, 'link'))
    attempts.append((work_dir, 'copy'))
    if os.path.abspath(sys_tmp) != os.path.abspath(work_dir):
        attempts.append((sys_tmp, 'copy'))

    name = f"_dmp_input_{os.getpid()}{os.path.splitext(path)[1]}"
    link, made, errors = path, None, []
    for folder, method in attempts:
        candidate = os.path.join(folder, name)
        if not _is_safe_path(candidate):
            continue                     # e.g. a folder or username with non-ASCII characters
        try:
            os.makedirs(folder, exist_ok=True)
            if os.path.lexists(candidate):
                os.remove(candidate)     # stale file from a crashed run
            if method == 'link':
                os.link(path, candidate)
            else:
                shutil.copy2(path, candidate)
            link, made = candidate, method
            print(f"[safe_input_path] {'hard link' if method == 'link' else 'copy'}: {candidate}")
            break
        except OSError as e:
            errors.append(f"{method} in {folder}: {e}")
    if made is None:
        print("[safe_input_path] could not create a safe-named link or copy; using the original path")
        for err in errors:
            print(f"    {err}")

    # Yield exactly ONCE, outside the try above: a yield inside it turned any
    # error from the code using this path into "generator didn't stop after
    # throw()", hiding the real error.
    try:
        yield link
    finally:
        if made:
            try:
                os.remove(link)
            except OSError:
                pass


def safe_output_filename(path):
    """Return path with brackets and non-ASCII stripped from the filename only.
    The directory part is kept as-is (it already exists and is accessible).
    ffmpeg glob-expands output paths too, so we sanitise them here.
    """
    import unicodedata
    dirpart  = os.path.dirname(path)
    basename = os.path.basename(path)
    # Replace brackets (glob chars) and strip non-ASCII from the stem
    stem, ext = os.path.splitext(basename)
    # Keep alphanumerics, spaces, hyphens, underscores, dots
    safe_stem = ''
    for ch in stem:
        try:
            ch.encode('ascii')
            if ch in r'[]':
                safe_stem += '_'
            else:
                safe_stem += ch
        except UnicodeEncodeError:
            # Transliterate if possible, else drop
            nfkd = unicodedata.normalize('NFKD', ch)
            ascii_ch = nfkd.encode('ascii', 'ignore').decode('ascii')
            safe_stem += ascii_ch if ascii_ch.strip() else '_'
    # Collapse multiple underscores
    import re as _re
    safe_stem = _re.sub(r'_+', '_', safe_stem).strip('_')
    return os.path.join(dirpart, safe_stem + ext)

# ── Core helpers ─────────────────────────────────────────────────────────────
def run(args, mode=0):
    if mode == 0:
        subprocess.run(args, check=False, stdin=subprocess.DEVNULL)
    elif mode == 1:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL)
        return result.stdout.decode('utf-8', errors='replace')
    elif mode == 2:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL)
        return result.stdout.splitlines(keepends=True)

def video2image(videopath, imagepath, fps=0, start_time='00:00:00', last_time='00:00:00', qv=1):
    _ensure_probed()
    if not _video2image_once(videopath, imagepath, fps, start_time, last_time, qv, use_hw=True) \
            and hwaccel is not None:
        _disable_hwaccel()
        _video2image_once(videopath, imagepath, fps, start_time, last_time, qv, use_hw=False)

def _video2image_once(videopath, imagepath, fps, start_time, last_time, qv, use_hw):
    args = ['ffmpeg', '-y']
    if use_hw and hwaccel is not None:
        args += ['-hwaccel', hwaccel]
    if last_time != '00:00:00':
        args += ['-ss', start_time, '-t', last_time]
    args += ['-i', videopath]
    if fps != 0:
        args += ['-r', str(fps)]
    args += ['-f', 'image2', '-q:v', str(qv), imagepath]
    return subprocess.run(args, stdin=subprocess.DEVNULL).returncode == 0

def video_to_gif(videopath, gifpath):
    """Convert a processed video back into a looping GIF.

    GIFs hold 256 colours. The source GIF fits that already, but the model
    repaints the mosaic region with smooth new tones (tens of thousands of
    colours), so every pixel has to be re-quantized. Settings chosen by
    measurement on a photographic GIF with a simulated cleaned region:
      * palettegen stats_mode=full -- palette from ALL pixels. stats_mode=diff
        (used before) built it mostly from the changing pixels, i.e. the
        cleaned region, starving the untouched background: 5x more error
        there (2.63 vs 0.53 on 0-255).
      * dither=bayer (ordered) -- a fixed pattern that stays put between
        frames. Error-diffusion (sierra2_4a, used before) scatters a
        different speckle each frame, which reads as crawling noise over
        static areas (flicker 0.21 -> 0.00). Also ~half the file size.
      * diff_mode=rectangle -- only the changed area of each frame is
        re-dithered, so static regions stay identical frame to frame.
    Frame timing comes from the video's timestamps, preserving the
    original GIF's frame rate.
    """
    cmd = ['ffmpeg', '-y', '-i', videopath,
           '-filter_complex',
           '[0:v]split[a][b];[a]palettegen=stats_mode=full[p];'
           '[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle',
           '-loop', '0', gifpath]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def video2voice(videopath, voicepath, start_time='00:00:00', last_time='00:00:00'):
    # Probe native audio codec to decide whether to stream-copy
    probe_cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-select_streams', 'a:0', '-i', videopath
    ]
    probe = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           stdin=subprocess.DEVNULL)
    probe_out = probe.stdout.decode('utf-8', errors='replace') if probe.stdout else ''
    native_codec = None
    streams = None
    try:
        streams = json.loads(probe_out).get('streams', [])
        if streams:
            native_codec = streams[0].get('codec_name', '')
    except Exception:
        pass
    if streams == []:
        # No audio track: ffmpeg would fail here (with check=True, crashing
        # the run). Skip it; image2video() already omits audio when the voice
        # file doesn't exist. Remove any stale file from a previous run so it
        # can't get muxed into this video by mistake.
        if os.path.exists(voicepath):
            os.remove(voicepath)
        print("[ffmpeg] Source has no audio track -- output will be silent")
        return

    target_ext = os.path.splitext(voicepath)[1].lower()
    mp3_compat = target_ext == '.mp3' and native_codec == 'mp3'
    aac_compat = target_ext in ('.aac', '.m4a') and native_codec == 'aac'

    # Large I/O buffers keep the HDD streaming continuously instead of
    # stalling between ffmpeg's internal read/write bursts.
    # 64 MB read buffer, 32 MB output buffer — safe upper bound for HDDs.
    IO_BUF = '67108864'   # 64 MiB  (ffmpeg -readrate_initial_burst / -bufsize)
    OUT_BUF = '33554432'  # 32 MiB

    cmd = ['ffmpeg', '-y',
           '-thread_queue_size', '4096',
           '-readrate_initial_burst', '0',  # read as fast as possible (no rate cap)
           '-i', videopath]
    if last_time != '00:00:00':
        cmd += ['-ss', start_time, '-t', last_time]
    cmd += ['-vn']
    if mp3_compat or aac_compat:
        cmd += ['-acodec', 'copy', '-bufsize', OUT_BUF, voicepath]
    else:
        cmd += ['-acodec', 'libmp3lame', '-b:a', '320k', '-bufsize', OUT_BUF, voicepath]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)

def get_duration(video_path):
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
           '-show_format', '-i', video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL)
    stdout_text = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
    info = json.loads(stdout_text)
    return float(info['format']['duration'])

def to_seconds(timestr):
    h, m, s = map(float, timestr.split(":"))
    return int(h * 3600 + m * 60 + s)

def run_ffmpeg_segment(args):
    """Runs one extraction segment, retrying in software if hardware decode fails."""
    _ensure_probed()
    try:
        return _run_ffmpeg_segment_once(args, use_hw=True)
    except RuntimeError:
        if hwaccel is None:
            raise
        _disable_hwaccel()
        return _run_ffmpeg_segment_once(args, use_hw=False)

def _run_ffmpeg_segment_once(args, use_hw):
    videopath, output_template, fps, start_time, duration, part_num, ext, start_frame, expected_frames, progress, qv = args

    cmd = ['ffmpeg', '-y']
    if use_hw and hwaccel is not None:
        cmd += ['-hwaccel', hwaccel]
    cmd += ['-ss', str(start_time), '-t', str(duration), '-i', videopath]
    if fps != 0:
        cmd += ['-r', str(fps)]
    cmd += ['-f', 'image2', '-q:v', str(qv), '-start_number', str(start_frame), output_template]

    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stdin=subprocess.DEVNULL)
    frame_count = 0
    for raw in process.stderr:
        line = raw.decode('utf-8', errors='replace')
        if 'frame=' in line:
            m = re.search(r'frame=\s*(\d+)', line)
            if m:
                f = int(m.group(1))
                delta = f - frame_count
                if delta > 0:
                    progress.value += delta
                    frame_count = f
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg segment {part_num} failed")

def video2image_parallel(videopath, imagepath, fps=0, start_time='00:00:00', last_time='00:00:00', segments=None, qv=1):
    _validate_hwaccel(videopath)
    folder = os.path.dirname(imagepath)
    ext = os.path.basename(imagepath).split('.')[-1]
    output_template = os.path.join(folder, f"output_%06d.{ext}")

    start_sec = to_seconds(start_time)
    total_dur = get_duration(videopath)
    # Bug fix: use float duration, not int-truncated seconds, for accurate frame count
    dur = (float(last_time.replace(':', ' ').split()[0]) * 3600
           + float(last_time.replace(':', ' ').split()[1]) * 60
           + float(last_time.replace(':', ' ').split()[2])
           ) if last_time != '00:00:00' else total_dur - start_sec

    if segments is None:
        segments = max(1, min(os.cpu_count() or 4, 16))

    total_frames = int(round(fps * dur)) if fps != 0 else 0
    # Splitting only pays off for longer videos; short clips (and GIFs) run as
    # one segment. Also required when fps is unknown (no frame grid to split on).
    if total_frames <= 0:
        segments = 1
    else:
        segments = max(1, min(segments, total_frames // 60))

    manager = Manager()
    # Bug fix: use a Lock so read-modify-write on progress is atomic
    progress = manager.Value('i', 0)
    progress_lock = manager.Lock()
    done_flag = manager.Value('b', False)  # signals watcher to stop

    # Segment layout. This used to cut the video into equal TIME slices but
    # number each slice's files as if it held exactly total_frames//segments
    # frames. Slices actually hold that +/- a frame, so numbering ranges
    # overlapped: later segments overwrote earlier frames and the end of the
    # video was never reached (e.g. a 20-frame clip came out as 16 frames;
    # long videos got a duplicated/dropped frame at each boundary).
    #   * Boundaries are placed half a frame BEFORE each segment's first frame,
    #     so no frame sits on a boundary and gets taken twice or not at all.
    #   * Segment 0 always starts at frame 1, so it writes straight into the
    #     final folder with final names -- this is what lets the GUI preview
    #     frames while extraction is still running. Every later segment writes
    #     to its own subfolder with its own numbering and is appended in order
    #     afterwards, so segments can never overwrite each other.
    bounds = [round(k * total_frames / segments) for k in range(segments + 1)]
    seg_dirs = []
    args_list = []
    for i in range(segments):
        if i == 0:
            seg_template = output_template
        else:
            seg_dir = os.path.join(folder, f"_seg{i:02d}")
            os.makedirs(seg_dir, exist_ok=True)
            seg_dirs.append(seg_dir)
            seg_template = os.path.join(seg_dir, f"%06d.{ext}")
        seg_start = start_sec + (max(0.0, (bounds[i] - 0.5) / fps) if i > 0 else 0.0)
        if i < segments - 1:
            seg_dur = start_sec + (bounds[i + 1] - 0.5) / fps - seg_start
        else:
            seg_dur = (start_sec + dur - seg_start) if last_time != '00:00:00' else None
        expected = bounds[i + 1] - bounds[i]
        args_list.append((videopath, seg_template, fps, seg_start, seg_dur,
                          i, ext, 1, expected, progress, progress_lock, qv))

    with tqdm(total=total_frames if total_frames else 1, desc="Extracting frames", unit="frame") as pbar:
        def progress_watcher():
            last = 0
            while not done_flag.value:
                current = progress.value
                if current > last:
                    pbar.update(current - last)
                    last = current
                time.sleep(0.1)
            # Drain any final progress after workers finish
            current = progress.value
            if current > last:
                pbar.update(current - last)
            # Clamp bar to 100%
            if pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)

        watcher = threading.Thread(target=progress_watcher, daemon=True)
        watcher.start()
        try:
            with Pool(segments) as pool:
                pool.map(run_ffmpeg_segment_with_progress, args_list)
        finally:
            # Bug fix: always signal the watcher so join() never hangs,
            # even if pool.map raised an exception
            done_flag.value = True
        watcher.join(timeout=2.0)  # bounded join — never hangs forever

    # Append segments 1..N after segment 0's frames (already in place)
    prefix = os.path.basename(output_template).split('%')[0]
    n = len([f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith('.' + ext)])
    for seg_dir in seg_dirs:
        for name in sorted(os.listdir(seg_dir)):
            n += 1
            os.replace(os.path.join(seg_dir, name), output_template % n)
        os.rmdir(seg_dir)
    if total_frames and abs(n - total_frames) > 1:
        print(f"[ffmpeg] Note: extracted {n} frames (expected ~{total_frames} from duration x fps)")

def run_ffmpeg_with_progress(cmd, total_duration_sec, progress_callback=None):
    cmd = cmd + ['-progress', 'pipe:1', '-nostats']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               stdin=subprocess.DEVNULL, bufsize=1)

    def reader():
        for raw in process.stdout:
            line = raw.decode('utf-8', errors='replace').strip()
            if line.startswith('out_time_ms='):
                try:
                    out_time_ms = int(line.split('=')[1])
                    pct = min(1.0, out_time_ms / (total_duration_sec * 1_000_000))
                    if progress_callback:
                        progress_callback(pct)
                except ValueError:
                    pass
            elif line == 'progress=end':
                if progress_callback:
                    progress_callback(1.0)

    t = threading.Thread(target=reader)
    t.start()
    process.wait()
    t.join()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {process.returncode}")

def run_ffmpeg_segment_with_progress(args):
    """Runs one extraction segment, retrying in software if hardware decode fails."""
    _ensure_probed()
    try:
        return _run_ffmpeg_segment_with_progress_once(args, use_hw=True)
    except RuntimeError:
        if hwaccel is None:
            raise
        _disable_hwaccel()
        return _run_ffmpeg_segment_with_progress_once(args, use_hw=False)

def _run_ffmpeg_segment_with_progress_once(args, use_hw):
    (videopath, output_template, fps,
     start_time, duration, part_num, ext,
     start_frame, expected_frames, progress, progress_lock, qv) = args

    cmd = ['ffmpeg', '-y']
    if use_hw and hwaccel is not None:
        cmd += ['-hwaccel', hwaccel]
    cmd += ['-ss', str(start_time)]
    if duration is not None:          # None = run to the end of the range
        cmd += ['-t', str(duration)]
    cmd += ['-i', videopath]
    if fps != 0:
        cmd += ['-r', str(fps)]
    # -progress pipe:1 emits structured key=value lines to stdout regardless of
    # whether we have a TTY. Without this, ffmpeg suppresses frame= stats when
    # stderr is a pipe. -nostats suppresses the human-readable stderr overlay.
    cmd += ['-progress', 'pipe:1', '-nostats']
    cmd += ['-f', 'image2', '-q:v', str(qv), '-start_number', str(start_frame), output_template]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               stdin=subprocess.DEVNULL)
    frame_count = 0
    for raw in process.stdout:
        line = raw.decode('utf-8', errors='replace').strip()
        if line.startswith('frame='):
            try:
                f = int(line.split('=', 1)[1])
                delta = f - frame_count
                if delta > 0:
                    with progress_lock:
                        progress.value += delta
                    frame_count = f
            except ValueError:
                pass
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg segment {part_num} failed")

# Encoder settings per user-facing codec name.
# 'crf_flag' is the argument ffmpeg expects for that encoder's quality knob
# (libaom-av1 and svt-av1 both use -crf too, but some builds expect -qp; -crf
# is correct and preferred for both as of recent ffmpeg/aom releases).
# Encoder settings per user-facing codec name.
# 'crf_flag' is the argument ffmpeg expects for that encoder's quality knob
# (libaom-av1 and svt-av1 both use -crf too, but some builds expect -qp; -crf
# is correct and preferred for both as of recent ffmpeg/aom releases).
# 'default_crf'/'crf_range' differ per encoder because their CRF scales aren't
# comparable — an AV1/VP9 CRF of 18 is far higher quality (and much bigger)
# than an x264 CRF of 18. 'min_ffmpeg' is the oldest FFmpeg release known to
# ship that encoder (see docs/options_introduction.md for sources).
VIDEO_CODECS = {
    'h264':       {'vcodec': 'libx264',    'crf_flag': '-crf', 'extra': ['-preset', 'fast'], 'pix_fmt': 'yuv420p',
                   'default_crf': 18, 'crf_range': (0, 51),  'min_ffmpeg': '2.1'},
    'h265':       {'vcodec': 'libx265',    'crf_flag': '-crf', 'extra': ['-preset', 'fast'], 'pix_fmt': 'yuv420p',
                   'default_crf': 22, 'crf_range': (0, 51),  'min_ffmpeg': '2.1'},
    'hevc':       {'vcodec': 'libx265',    'crf_flag': '-crf', 'extra': ['-preset', 'fast'], 'pix_fmt': 'yuv420p',
                   'default_crf': 22, 'crf_range': (0, 51),  'min_ffmpeg': '2.1'},
    'av1':        {'vcodec': 'libsvtav1',  'crf_flag': '-crf', 'extra': ['-preset', '8'],    'pix_fmt': 'yuv420p',
                   'default_crf': 30, 'crf_range': (0, 63),  'min_ffmpeg': '4.4'},
    'vp9':        {'vcodec': 'libvpx-vp9', 'crf_flag': '-crf', 'extra': ['-b:v', '0'],       'pix_fmt': 'yuv420p',
                   'default_crf': 31, 'crf_range': (0, 63),  'min_ffmpeg': '2.4'},
    # Hardware encoders (require compatible GPU/driver; quality knob differs from CRF)
    'h264_nvenc': {'vcodec': 'h264_nvenc', 'crf_flag': '-cq', 'extra': ['-preset', 'p5'],    'pix_fmt': 'yuv420p',
                   'default_crf': 19, 'crf_range': (0, 51),  'min_ffmpeg': '3.1'},
    'hevc_nvenc': {'vcodec': 'hevc_nvenc', 'crf_flag': '-cq', 'extra': ['-preset', 'p5'],    'pix_fmt': 'yuv420p',
                   'default_crf': 19, 'crf_range': (0, 51),  'min_ffmpeg': '3.1'},
    # Internal: mathematically lossless RGB intermediate, used automatically
    # for GIF inputs (the .mp4 is only read back by ffmpeg to build the GIF,
    # then deleted). Regular yuv420p H.264 halves colour resolution and adds
    # compression noise, which smears the hard flat-colour edges GIFs are
    # made of; libx264rgb at CRF 0 keeps every pixel exactly.
    'rgb_lossless': {'vcodec': 'libx264rgb', 'crf_flag': '-crf', 'extra': ['-preset', 'ultrafast'], 'pix_fmt': None,
                   'default_crf': 0, 'crf_range': (0, 0),  'min_ffmpeg': '2.1'},
}

_CODEC_ALIASES = {'x264': 'h264', 'x265': 'h265', 'h.264': 'h264', 'h.265': 'hevc', 'libaom-av1': 'av1'}

def list_video_codecs():
    """Return the list of vcodec names accepted by image2video()."""
    return sorted(VIDEO_CODECS.keys())

def _resolve_codec(name):
    name = _CODEC_ALIASES.get(name.lower(), name.lower())
    if name not in VIDEO_CODECS:
        raise ValueError(
            f"Unknown vcodec '{name}'. Supported: {', '.join(list_video_codecs())}"
        )
    return VIDEO_CODECS[name]

def get_default_crf(vcodec):
    """Return the sensible default CRF/CQ for a given codec name."""
    return _resolve_codec(vcodec)['default_crf']

def get_crf_range(vcodec):
    """Return the (min, max) CRF/CQ range accepted by a given codec's encoder."""
    return _resolve_codec(vcodec)['crf_range']

# Paths image2video() actually wrote. The output name can differ from the
# input's name (safe_output_filename strips brackets and non-ASCII), so
# anything post-processing the result (GIF conversion) must use these rather
# than rebuilding the name from the input -- that silently found nothing.
VIDEO_OUTPUTS = []

def image2video(fps, imagepath, voicepath, videopath, crf=None, vcodec='h264'):
    """Encode a numbered image sequence (+ optional audio) to a video file.

    crf:    quality value passed straight to the selected encoder's quality
            flag (CRF for the software x264/x265/av1/vp9 encoders, CQ for the
            nvenc hardware encoders). Lower = higher quality/bitrate. Scales
            differ per encoder (x264/x265/nvenc: 0-51, av1/vp9: 0-63) — if
            omitted (None), the codec's own sane default is used rather than
            silently reusing an x264-tuned value.
    vcodec: one of list_video_codecs(), e.g. 'h264', 'hevc'/'h265', 'av1', 'vp9'.
    """
    codec = _resolve_codec(vcodec)
    if crf is None:
        crf = codec['default_crf']

    # Single-pass encode: images + audio simultaneously, stream-copy audio
    cmd = ['ffmpeg', '-y', '-r', str(fps), '-i', imagepath]
    if os.path.exists(voicepath):
        cmd += ['-i', voicepath]
    cmd += ['-vcodec', codec['vcodec'], codec['crf_flag'], str(crf)] + codec['extra']
    if codec['pix_fmt']:
        # Explicit pixel format: without it the encoder guessed from its input --
        # 4:4:4 for PNG (RGB) frames, which many players/browsers/phones can't
        # play, and full-range yuvj420p for JPEG frames. yuv420p plays anywhere.
        cmd += ['-pix_fmt', codec['pix_fmt']]
        if codec['pix_fmt'].endswith('420p'):
            # 4:2:0 needs even width/height (GIFs are often odd, e.g. 245x139):
            # pad one pixel instead of failing. No-op for even sizes.
            cmd += ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2']
    if os.path.exists(voicepath):
        cmd += ['-acodec', 'copy', '-shortest']
    # Sanitise output path — ffmpeg glob-expands output paths too
    safe_out = safe_output_filename(videopath)
    if safe_out != videopath:
        print(f"[image2video] output sanitised: {os.path.basename(safe_out)}")
    cmd += [safe_out]
    print(f"[image2video] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    VIDEO_OUTPUTS.append(safe_out)

def get_pixel_aspect_ratio(videopath):
    """Return the video's Sample Aspect Ratio (SAR / pixel aspect ratio) as a
    float (pixel_width_scale / pixel_height_scale). 1.0 means square pixels
    (the normal case for a full-raster 1080p export). Anything else means
    the stored pixel grid is stretched relative to how it's meant to be
    displayed — which would also stretch anything authored square within
    that grid, including mosaic blocks. Returns 1.0 if SAR is unset/unknown
    rather than raising, since "unknown" and "1:1" both mean "no correction
    needed" for our purposes.
    """
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
               '-show_entries', 'stream=sample_aspect_ratio,display_aspect_ratio',
               '-select_streams', 'v:0', '-i', videopath]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, timeout=10)
        infos = json.loads(result.stdout.decode('utf-8', errors='replace'))
        sar = infos['streams'][0].get('sample_aspect_ratio', '1:1')
        if not sar or sar in ('1:1', 'N/A', '0:1'):
            return 1.0
        num, den = sar.split(':')
        return float(num) / float(den)
    except Exception:
        return 1.0

def get_video_infos(videopath):
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
           '-show_format', '-show_streams', '-i', videopath]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL)
    stdout_text = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
    stderr_text = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {stderr_text}")
    if not stdout_text.strip():
        raise RuntimeError(f"ffprobe returned empty output for {videopath}")
    infos = json.loads(stdout_text)
    try:
        fps = eval(infos['streams'][0]['avg_frame_rate'])
        endtime = float(infos['format']['duration'])
        width = int(infos['streams'][0]['width'])
        height = int(infos['streams'][0]['height'])
    except Exception as e:
        try:
            fps = eval(infos['streams'][1]['r_frame_rate'])
            endtime = float(infos['format']['duration'])
            width = int(infos['streams'][1]['width'])
            height = int(infos['streams'][1]['height'])
        except Exception as e2:
            raise RuntimeError(f"Could not extract video info: {e}, {e2}")
    return fps, endtime, height, width

def cut_video(in_path, start_time, last_time, out_path, vcodec='h265'):
    if vcodec == 'copy':
        cmd = ['ffmpeg', '-ss', start_time, '-t', last_time, '-i', in_path,
               '-vcodec', 'copy', '-acodec', 'copy', out_path]
    elif vcodec == 'h264':
        cmd = ['ffmpeg', '-ss', start_time, '-t', last_time, '-i', in_path,
               '-vcodec', 'libx264', '-b:v', '12M', out_path]
    elif vcodec == 'h265':
        cmd = ['ffmpeg', '-ss', start_time, '-t', last_time, '-i', in_path,
               '-vcodec', 'libx265', '-b:v', '12M', out_path]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)

def continuous_screenshot(videopath, savedir, fps):
    videoname = os.path.splitext(os.path.basename(videopath))[0]
    cmd = ['ffmpeg', '-i', videopath, '-vf', f'fps={fps}', '-q:v', '1',
           os.path.join(savedir, f'{videoname}_%06d.jpg')]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)