import os, json, contextlib, tempfile, platform
import subprocess
from multiprocessing import Pool, Manager
from pathlib import Path
import threading
from tqdm import tqdm
import time
import re
import torch

hwaccel = None

def _probe_ffmpeg_hwaccels():
    """Return set of hwaccel names ffmpeg was built with."""
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

_supported_hwaccels = _probe_ffmpeg_hwaccels()

try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False

try:
    _sys = platform.system()
    if torch.cuda.is_available():
        # nvdec is the correct ffmpeg decoder name for NVIDIA GPU decode
        hwaccel = 'nvdec' if 'nvdec' in _supported_hwaccels else ('cuda' if 'cuda' in _supported_hwaccels else None)
    elif DIRECTML_AVAILABLE or _sys == 'Windows':
        # d3d11va works on Windows for both AMD and Intel
        hwaccel = 'd3d11va' if 'd3d11va' in _supported_hwaccels else ('dxva2' if 'dxva2' in _supported_hwaccels else None)
    elif _sys == 'Linux':
        # vaapi covers AMD and Intel on Linux
        hwaccel = 'vaapi' if 'vaapi' in _supported_hwaccels else None
except Exception:
    hwaccel = None

if hwaccel:
    print(f"[ffmpeg] Hardware decode: {hwaccel}")
else:
    print("[ffmpeg] Hardware decode: software fallback")

# ── Safe path: hardlink trick ────────────────────────────────────────────────
@contextlib.contextmanager
def safe_input_path(path, temp_dir=None):
    """Yield a plain-ASCII path pointing to *path*.

    Creates a hardlink (or copy for cross-drive) so ffmpeg never sees
    brackets or non-ASCII characters. Prints what it does so failures are visible.
    """
    import shutil
    path = str(path)
    needs_link = False
    try:
        path.encode('ascii')
        if '[' in path or ']' in path:
            needs_link = True
    except UnicodeEncodeError:
        needs_link = True

    if not needs_link:
        yield path
        return

    # Prefer temp dir on the SAME drive as source to allow hardlinks
    src_drive = os.path.splitdrive(path)[0]  # e.g. 'S:'
    if temp_dir and os.path.splitdrive(temp_dir)[0].lower() == src_drive.lower():
        base = temp_dir
    elif src_drive:
        # Use root of source drive as temp — guaranteed same drive
        base = src_drive + '\\'
    else:
        base = temp_dir or tempfile.gettempdir()

    ext  = os.path.splitext(path)[1]
    link = os.path.join(base, f'_dmp_input{ext}')

    # Remove stale link/file from a previous run
    for attempt in range(3):
        try:
            if os.path.exists(link) or os.path.islink(link):
                os.remove(link)
            break
        except OSError:
            import time as _time
            _time.sleep(0.2)
    else:
        # Can't remove stale link — use a unique name instead
        import uuid
        link = os.path.join(base, f'_dmp_input_{uuid.uuid4().hex[:6]}{ext}')

    created = False
    copied  = False
    try:
        # Try hardlink first (instant, zero-copy, same drive required)
        try:
            os.link(path, link)
            created = True
            print(f"[safe_input_path] hardlink: {link}")
        except OSError as e:
            # Cross-drive or permissions — fall back to copy
            print(f"[safe_input_path] hardlink failed ({e}), copying...")
            shutil.copy2(path, link)
            created = True
            copied  = True
            print(f"[safe_input_path] copied to: {link}")
        yield link
    except Exception as e:
        print(f"[safe_input_path] all attempts failed ({e}), using original path")
        yield path
    finally:
        if created:
            try:
                os.remove(link)
                if copied:
                    print(f"[safe_input_path] removed copy: {link}")
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
    args = ['ffmpeg', '-y']
    if hwaccel is not None:
        args += ['-hwaccel', hwaccel]
    if last_time != '00:00:00':
        args += ['-ss', start_time, '-t', last_time]
    args += ['-i', videopath]
    if fps != 0:
        args += ['-r', str(fps)]
    args += ['-f', 'image2', '-q:v', str(qv), imagepath]
    run(args)

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
    try:
        streams = json.loads(probe_out).get('streams', [])
        if streams:
            native_codec = streams[0].get('codec_name', '')
    except Exception:
        pass

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
    videopath, output_template, fps, start_time, duration, part_num, ext, start_frame, expected_frames, progress, qv = args

    cmd = ['ffmpeg', '-y']
    if hwaccel is not None:
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

    # Bug fix: use float dur so frame count is accurate
    total_frames = int(fps * dur) if fps != 0 else 0
    frames_per_segment = max(1, total_frames // segments) if total_frames else 1

    manager = Manager()
    # Bug fix: use a Lock so read-modify-write on progress is atomic
    progress = manager.Value('i', 0)
    progress_lock = manager.Lock()
    done_flag = manager.Value('b', False)  # signals watcher to stop

    args_list = []
    for i in range(segments):
        seg_start = start_sec + i * (dur / segments)
        seg_dur = (dur / segments) if i < segments - 1 else (dur - (dur / segments) * i)
        start_frame = i * frames_per_segment + 1
        expected = frames_per_segment if i < segments - 1 else max(0, total_frames - frames_per_segment * (segments - 1))
        args_list.append((videopath, output_template, fps, seg_start, seg_dur, i, ext,
                          start_frame, expected, progress, progress_lock, qv))

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
    (videopath, output_template, fps,
     start_time, duration, part_num, ext,
     start_frame, expected_frames, progress, progress_lock, qv) = args

    cmd = ['ffmpeg', '-y']
    if hwaccel is not None:
        cmd += ['-hwaccel', hwaccel]
    cmd += ['-ss', str(start_time), '-t', str(duration), '-i', videopath]
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

def image2video(fps, imagepath, voicepath, videopath, crf=18):
    # Single-pass encode: images + audio simultaneously, stream-copy audio
    cmd = ['ffmpeg', '-y', '-r', str(fps), '-i', imagepath]
    if os.path.exists(voicepath):
        cmd += ['-i', voicepath]
    cmd += ['-vcodec', 'libx264', '-crf', str(crf), '-preset', 'fast']
    if os.path.exists(voicepath):
        cmd += ['-acodec', 'copy', '-shortest']
    # Sanitise output path — ffmpeg glob-expands output paths too
    safe_out = safe_output_filename(videopath)
    if safe_out != videopath:
        print(f"[image2video] output sanitised: {os.path.basename(safe_out)}")
    cmd += [safe_out]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)

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