import os,json
import subprocess
from multiprocessing import Pool, Manager
from pathlib import Path
import threading
from tqdm import tqdm
import time
import re
import torch

hwaccel = None

try:
    import torch_directml
    DIRECTML_AVAILABLE = True
    print("DirectML available for acceleration")
except ImportError:
    DIRECTML_AVAILABLE = False

try:
    if torch.cuda.is_available():
        hwaccel = 'cuda'
    elif DIRECTML_AVAILABLE:
        hwaccel = 'd3d11va'
except ImportError:
    hwaccel = None

def args2cmd(args):
    quoted_args = []
    for arg in args:
        if ' ' in arg or '[' in arg or ']' in arg:
            quoted_args.append(f'"{arg}"')
        else:
            quoted_args.append(arg)
    return ' '.join(quoted_args)

def run(args,mode = 0):

    if mode == 0:
        cmd = args2cmd(args)
        os.system(cmd)

    elif mode == 1:
        cmd = args2cmd(args)
        stream = os.popen(cmd)._stream
        sout = stream.buffer.read().decode(encoding='utf-8')
        return sout

    elif mode == 2:
        cmd = args2cmd(args)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sout = p.stdout.readlines()
        return sout

def video2image(videopath, imagepath, fps=0, start_time='00:00:00', last_time='00:00:00'):
    args = ['ffmpeg']
    if hwaccel != None:
        args += ['-hwaccel', hwaccel]
    if last_time != '00:00:00':
        args += ['-ss', start_time]
        args += ['-t', last_time]
    args += ['-i', videopath]
    if fps != 0:
        args += ['-r', str(fps)]
    args += ['-f', 'image2','-q:v','-0', imagepath]
    run(args)

def video2voice(videopath, voicepath, start_time='00:00:00', last_time='00:00:00'):
    import subprocess
    
    cmd = ['ffmpeg', '-y', '-i', videopath]
    
    if last_time != '00:00:00':
        cmd += ['-ss', start_time, '-t', last_time]
    
    cmd += ['-vn', '-acodec', 'libmp3lame', '-b:a', '320k', voicepath]
    
    subprocess.run(cmd, check=True)

def get_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-i", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    duration = float(info['format']['duration'])
    return duration

def to_seconds(timestr):
    h, m, s = map(float, timestr.split(":"))
    return int(h * 3600 + m * 60 + s)

def run_ffmpeg_segment(args):
    import subprocess
    import re

    videopath, output_template, fps, start_time, duration, part_num, ext, start_frame, expected_frames, progress = args

    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'd3d11va',
        '-ss', str(start_time),
        '-t', str(duration),
        '-i', videopath
    ]
    if fps != 0:
        cmd += ['-r', str(fps)]
    cmd += [
        '-f', 'image2',
        '-q:v', '1',
        '-start_number', str(start_frame),
        output_template
    ]

    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    frame_count = 0
    for line in process.stderr:
        if "frame=" in line:
            match = re.search(r"frame=\s*(\d+)", line)
            if match:
                f = int(match.group(1))
                delta = f - frame_count
                if delta > 0:
                    progress.value += delta
                    frame_count = f

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg segment {part_num} failed")

def video2image_parallel(videopath, imagepath, fps=0, start_time='00:00:00', last_time='00:00:00', segments=None):
    folder = os.path.dirname(imagepath)
    pattern = os.path.basename(imagepath)
    ext = pattern.split('.')[-1]
    output_template = os.path.join(folder, f"output_%06d.{ext}")

    start_sec = to_seconds(start_time)
    total_dur = get_duration(videopath)
    if last_time != '00:00:00':
        dur = to_seconds(last_time)
    else:
        dur = total_dur - start_sec

    if segments is None:
        segments = max(1, min(os.cpu_count() or 4, 16))

    total_frames = int(fps * dur)
    frames_per_segment = total_frames // segments

    manager = Manager()
    progress = manager.Value('i', 0)
    segments_done = manager.Value('i', 0)

    args_list = []
    for i in range(segments):
        seg_start = start_sec + i * (dur / segments)
        seg_dur = (dur / segments) if i < segments - 1 else (dur - (dur / segments) * i)
        start_frame = i * frames_per_segment + 1
        expected_frames = frames_per_segment if i < segments - 1 else total_frames - frames_per_segment * (segments - 1)
        args_list.append((videopath, output_template, fps, seg_start, seg_dur,i, ext, start_frame, expected_frames, progress, segments_done))

    # Live global progress bar
    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        def progress_watcher():
            last = 0
            while True:
                current = progress.value
                pbar.update(current - last)
                last = current
                if segments_done.value >= segments:
                    break
                time.sleep(0.1)
            pbar.update(total_frames - pbar.n)

        import threading
        watcher = threading.Thread(target=progress_watcher, daemon=True)
        watcher.start()

        with Pool(segments) as pool:
            pool.map(run_ffmpeg_segment_with_progress, args_list)

        watcher.join()



def run_ffmpeg_with_progress(cmd, total_duration_sec, progress_callback=None):
    """
    Runs ffmpeg command with -progress pipe:1 and calls progress_callback(percent) if provided.
    """
    cmd = cmd + ['-progress', 'pipe:1', '-nostats']  # ensure progress info only
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    
    def reader():
        for line in process.stdout:
            line = line.strip()
            if line.startswith('out_time_ms='):
                out_time_ms = int(line.split('=')[1])
                progress = min(1.0, out_time_ms / (total_duration_sec * 1_000_000))
                if progress_callback:
                    progress_callback(progress)
            elif line == 'progress=end':
                if progress_callback:
                    progress_callback(1.0)
    
    thread = threading.Thread(target=reader)
    thread.start()
    process.wait()
    thread.join()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {process.returncode}")
    

def run_ffmpeg_segment_with_progress(args):
    (
        videopath, output_template, fps,
        start_time, duration, part_num, ext,
        start_frame, expected_frames, progress, segments_done
    ) = args

    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'd3d11va',
        '-ss', str(start_time),
        '-t', str(duration),
        '-i', videopath,
    ]
    if fps != 0:
        cmd += ['-r', str(fps)]
    cmd += [
        '-f', 'image2',
        '-q:v', '1',
        '-start_number', str(start_frame),
        output_template
    ]

    # Launch ffmpeg and parse its stderr
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)

    frame_count = 0
    for line in process.stderr:
        if "frame=" in line:
            match = re.search(r"frame=\s*(\d+)", line)
            if match:
                f = int(match.group(1))
                delta = f - frame_count
                if delta > 0:
                    # Just increase shared progress counter
                    progress.value += delta
                    frame_count = f

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg segment {part_num} failed")
    segments_done.value += 1

def image2video(fps, imagepath, voicepath, videopath):
    import subprocess
    
    video_tmp = os.path.join(os.path.split(voicepath)[0], 'video_tmp.mp4')
    
    # create video from images
    cmd1 = [
        'ffmpeg', '-y',
        '-r', str(fps),
        '-i', imagepath,
        '-vcodec', 'libx264',
        video_tmp
    ]
    subprocess.run(cmd1, check=True)
    
    # merge with audio
    if os.path.exists(voicepath):
        cmd2 = [
            'ffmpeg', '-y',
            '-i', video_tmp,
            '-i', voicepath,
            '-vcodec', 'copy',
            '-acodec', 'aac',
            videopath
        ]
        subprocess.run(cmd2, check=True)
    else:
        cmd2 = [
            'ffmpeg', '-y',
            '-i', video_tmp,
            videopath
        ]
        subprocess.run(cmd2, check=True)

def get_video_infos(videopath):
    import subprocess
    import json
    
    # Use subprocess with list (proper quoting)
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        '-i', videopath
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    
    if not result.stdout.strip():
        raise RuntimeError(f"ffprobe returned empty output for {videopath}")
    
    infos = json.loads(result.stdout)
    
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
    import subprocess
    
    if vcodec == 'copy':
        cmd = [
            'ffmpeg',
            '-ss', start_time,
            '-t', last_time,
            '-i', in_path,
            '-vcodec', 'copy',
            '-acodec', 'copy',
            out_path
        ]
    elif vcodec == 'h264':    
        cmd = [
            'ffmpeg',
            '-ss', start_time,
            '-t', last_time,
            '-i', in_path,
            '-vcodec', 'libx264',
            '-b:v', '12M',
            out_path
        ]
    elif vcodec == 'h265':
        cmd = [
            'ffmpeg',
            '-ss', start_time,
            '-t', last_time,
            '-i', in_path,
            '-vcodec', 'libx265',
            '-b:v', '12M',
            out_path
        ]
    
    subprocess.run(cmd, check=True)

def continuous_screenshot(videopath, savedir, fps):
    '''
    videopath: input video path
    savedir:   images will save here
    fps:       save how many images per second
    '''
    import subprocess
    videoname = os.path.splitext(os.path.basename(videopath))[0]
    
    cmd = [
        'ffmpeg',
        '-i', videopath,
        '-vf', f'fps={fps}',
        '-q:v', '1',
        os.path.join(savedir, f'{videoname}_%06d.jpg')
    ]
    subprocess.run(cmd, check=True)