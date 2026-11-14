"""
Small helpers that make a failed run explain itself. No torch import: usable from the
launcher scripts and the GUI.

  system_status()   one line: free RAM, free commit (RAM + page file), free disk space
  explain_exit()    what a process exit code means, in words
  low_resources()   list of warnings when memory or disk is nearly exhausted
"""
import os
import shutil
import sys


def _windows_memory():
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(st)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    gb = 1024.0 ** 3
    return {'ram_total': st.ullTotalPhys / gb, 'ram_free': st.ullAvailPhys / gb,
            'commit_total': st.ullTotalPageFile / gb, 'commit_free': st.ullAvailPageFile / gb}


def _linux_memory():
    info = {}
    with open('/proc/meminfo') as f:
        for line in f:
            k, _, v = line.partition(':')
            info[k] = float(v.split()[0]) / 1024.0 / 1024.0
    return {'ram_total': info['MemTotal'], 'ram_free': info.get('MemAvailable', info['MemFree']),
            'commit_total': info['MemTotal'] + info.get('SwapTotal', 0),
            'commit_free': info.get('MemAvailable', info['MemFree']) + info.get('SwapFree', 0)}


def memory():
    try:
        return _windows_memory() if os.name == 'nt' else _linux_memory()
    except Exception:
        return None


def disks(project_dir=None):
    """{label: free GB} for the system drive and the project's drive (if different)."""
    out = {}
    paths = [('system drive', os.environ.get('SystemDrive', 'C:') + '\\' if os.name == 'nt' else '/')]
    if project_dir:
        paths.append(('project drive', project_dir))
    seen = set()
    for label, p in paths:
        try:
            drive = os.path.splitdrive(os.path.abspath(p))[0].upper() or '/'
            if drive in seen:
                continue
            seen.add(drive)
            out[f"{label} {drive}"] = shutil.disk_usage(p).free / 1024.0 ** 3
        except Exception:
            pass
    return out


def system_status(project_dir=None):
    parts = []
    m = memory()
    if m:
        parts.append(f"RAM {m['ram_free']:.1f} of {m['ram_total']:.0f} GB free")
        parts.append(f"RAM+page file {m['commit_free']:.1f} of {m['commit_total']:.0f} GB free")
    for label, free in disks(project_dir).items():
        parts.append(f"{label} {free:.1f} GB free")
    return "; ".join(parts) if parts else "unavailable"


def low_resources(project_dir=None):
    warn = []
    m = memory()
    if m and m['commit_free'] < 4:
        warn.append(f"Only {m['commit_free']:.1f} GB of memory (RAM + page file) is left. Processes fail to start or die "
                    f"without an error message when it runs out. Close other programs, lower --load_thread / batch size, "
                    f"or let Windows manage a larger page file.")
    for label, free in disks(project_dir).items():
        if free < 3:
            warn.append(f"The {label} has only {free:.1f} GB free. Windows cannot grow the page file on a full system "
                        f"drive, which makes programs crash without an error message.")
    return warn


_WINDOWS_CODES = {
    0xC0000005: "access violation: a crash inside native code (GPU driver, ROCm/CUDA library or PyTorch)",
    0xC000001D: "illegal instruction: a library was built for a CPU feature this processor lacks",
    0xC0000409: "a native library aborted (stack buffer overrun / fast-fail); usually the GPU runtime",
    0xC00000FD: "stack overflow inside native code",
    0xC000013A: "interrupted (Ctrl+C or the window was closed)",
    0xC0000142: "a DLL failed to initialise: typically Windows ran out of memory (page file) while starting the process",
    0xC0000017: "not enough memory to start (page file exhausted)",
    0xC000012D: "the system ran out of commit memory (page file full)",
    0xC0000135: "a required DLL was not found (broken PyTorch / GPU runtime install)",
}


def explain_exit(code):
    """Readable meaning of a process exit code (Windows NTSTATUS values included)."""
    if code is None:
        return "unknown"
    unsigned = code & 0xFFFFFFFF
    if unsigned in _WINDOWS_CODES:
        return f"0x{unsigned:08X}: {_WINDOWS_CODES[unsigned]}"
    if code < 0:
        try:
            import signal
            return f"killed by signal {signal.Signals(-code).name}"
        except Exception:
            return f"killed by signal {-code}"
    if code == 1:
        return "1: the program ended with an error (a Python error prints a 'Traceback' above; if there is none, see below)"
    if code == 2:
        return "2: a command-line argument was not accepted (see the 'error:' line above)"
    if code == 3:
        return "3: a native library called abort()"
    if unsigned >= 0xC0000000:
        return f"0x{unsigned:08X}: Windows terminated the process (a crash in native code)"
    return str(code)


if __name__ == '__main__':
    print(system_status(os.getcwd()))
    for w in low_resources(os.getcwd()):
        print("WARNING:", w)
    if len(sys.argv) > 1:
        print(explain_exit(int(sys.argv[1], 0)))
