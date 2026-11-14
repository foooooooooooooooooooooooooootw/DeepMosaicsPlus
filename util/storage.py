"""
Keep the program's caches and temporary files next to the project instead of on C:.

By default several libraries write into the user's profile on the system drive:
  PyTorch      downloaded weights (the 548 MB VGG19 used by training)   %USERPROFILE%\\.cache\\torch
  ROCm MIOpen  compiled GPU kernels and tuning results                  %USERPROFILE%\\.miopen, ...\\.cache\\miopen
  Python       temporary files (tempfile module, multiprocessing)        %LOCALAPPDATA%\\Temp

redirect_caches() points all of them at one folder BESIDE the project folder,
e.g. project S:\\dmpai\\DMP2  ->  cache S:\\dmpai\\DMP2_cache\\{torch,miopen,temp}.
Set the environment variable DMP_CACHE_DIR to use a different folder.

It only sets environment variables (inherited by every process the program starts)
and tempfile's default folder, so it is safe to call more than once, and a variable
the user has already set themselves (e.g. their own TORCH_HOME) is left alone.
Nothing here imports torch.
"""
import os
import shutil
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_done = False


def cache_dir():
    custom = os.environ.get('DMP_CACHE_DIR')
    if custom:
        return os.path.abspath(custom)
    return os.path.join(os.path.dirname(ROOT), os.path.basename(ROOT) + '_cache')


def _adopt_existing_vgg(torch_home):
    """Training downloads VGG19 (548 MB) once. If an earlier run already put it in the
    default cache, copy it over instead of downloading it again."""
    name = 'vgg19-dcbb9e9d.pth'
    dst = os.path.join(torch_home, 'hub', 'checkpoints', name)
    if os.path.exists(dst):
        return
    old_home = os.path.join(os.path.expanduser('~'), '.cache', 'torch')
    src = os.path.join(old_home, 'hub', 'checkpoints', name)
    if os.path.exists(src):
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst + '.part')
            os.replace(dst + '.part', dst)
            print(f"[cache] Copied {name} to {os.path.dirname(dst)} (you can delete the copy in {os.path.dirname(src)})",
                  flush=True)
        except OSError:
            pass


def redirect_caches():
    global _done
    if _done:
        return
    _done = True
    if os.environ.get('DMP_KEEP_DEFAULT_CACHES') == '1':
        return
    base = cache_dir()
    dirs = {name: os.path.join(base, name) for name in ('torch', 'miopen', 'temp')}
    try:
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
    except OSError:
        return                                   # e.g. read-only parent folder: keep the defaults

    if 'TORCH_HOME' not in os.environ:
        os.environ['TORCH_HOME'] = dirs['torch']
        _adopt_existing_vgg(dirs['torch'])
    # MIOpen (AMD ROCm): kernel binaries + per-GPU tuning database
    os.environ.setdefault('MIOPEN_USER_DB_PATH', dirs['miopen'])
    os.environ.setdefault('MIOPEN_CUSTOM_CACHE_DIR', dirs['miopen'])
    # temporary files of this program and everything it starts
    if not os.environ.get('DMP_TEMP_REDIRECTED'):
        for var in ('TMP', 'TEMP', 'TMPDIR'):
            os.environ[var] = dirs['temp']
        os.environ['DMP_TEMP_REDIRECTED'] = '1'
    tempfile.tempdir = os.environ.get('TMP', dirs['temp'])
