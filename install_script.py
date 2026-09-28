"""
DeepMosaicsPlus dependency installer.

Detects your GPU, installs the matching PyTorch build, VERIFIES it actually
works on the GPU, installs everything in requirements.txt, and creates
launchers that start the GUI with this same Python (so a venv keeps working
when you double-click).

Run it with the Python you want DeepMosaicsPlus to use -- e.g. from an
activated venv:   python install_script.py
Re-running it is safe, and also repairs an environment that has the wrong
PyTorch build (e.g. a CPU-only build on an NVIDIA PC).

Options:  --dry-run   show what would be done without installing anything
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DRY_RUN = "--dry-run" in sys.argv
PY = sys.version_info[:2]

# Newest first. The installer tries each CUDA build the driver supports and
# keeps the first one that works on the GPU; builds that don't exist for this
# PyTorch/Python just fail to install and are skipped, so the list can include
# versions from several PyTorch releases.
CUDA_BUILDS = ["cu132", "cu131", "cu130", "cu129", "cu128", "cu126", "cu124", "cu121", "cu118"]
TORCH_INDEX = "https://download.pytorch.org/whl/"
DIRECTML_PYTHON = ((3, 8), (3, 12))     # torch-directml only publishes wheels for these

# ------------------------------------------------------------

def cls():
    if not DRY_RUN:
        os.system("cls" if os.name == "nt" else "clear")

def title():
    cls()
    print("=" * 55)
    print("         DeepMosaicsPlus Dependency Installer")
    print("=" * 55)
    print()

def pause():
    if not DRY_RUN:
        input("\nPress Enter to exit...")

def pip(*args):
    """Run pip with THIS Python. Returns True on success (never raises)."""
    cmd = [sys.executable, "-m", "pip"] + list(args)
    print(">", " ".join(cmd))
    if DRY_RUN:
        return True
    return subprocess.call(cmd) == 0

def python_check(code):
    """Run a snippet in THIS Python; returns (ok, output)."""
    if DRY_RUN:
        return True, "(dry run)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()

# ------------------------------------------------------------

def driver_cuda_version():
    """Highest CUDA version the installed NVIDIA driver supports, from
    nvidia-smi's header ('CUDA Version: 12.4'), or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output([exe], text=True, timeout=20)
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None

def video_controllers():
    if os.name != "nt":
        return ""
    try:
        return subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            text=True, timeout=30).lower()
    except Exception:
        return ""

def detect_backend():
    print("Detecting graphics hardware...")
    names = video_controllers()
    if driver_cuda_version() or "nvidia" in names:
        print("  NVIDIA GPU detected -> CUDA\n")
        return "cuda"
    if os.name == "nt" and any(k in names for k in ("amd", "radeon", "intel")):
        print("  AMD/Intel GPU detected -> DirectML\n")
        return "directml"
    print("  No supported GPU detected -> CPU\n")
    return "cpu"

def choose_backend():
    detected = detect_backend()
    if DRY_RUN:
        return detected
    ans = input(f"Install for [{detected}]? Press Enter to accept, or type cuda / directml / cpu: ").strip().lower()
    return ans if ans in ("cuda", "directml", "cpu") else detected

# ------------------------------------------------------------

def remove_existing_torch():
    # Needed so re-running actually changes the build: pip otherwise treats an
    # existing CPU-only torch as "already satisfied" and skips the CUDA one.
    print("\nRemoving any existing PyTorch so the right build can be installed...")
    pip("uninstall", "-y", "torch", "torchvision", "torch-directml")

def install_cuda():
    drv = driver_cuda_version()
    if drv:
        print(f"\nNVIDIA driver supports CUDA up to {drv[0]}.{drv[1]}")
        builds = [b for b in CUDA_BUILDS if (int(b[2:-1]), int(b[-1])) <= drv]
    else:
        print("\nWARNING: nvidia-smi not found -- is the NVIDIA driver installed?")
        print("Trying CUDA builds anyway; if none work, CPU will be installed.")
        builds = list(CUDA_BUILDS)
    for b in builds:
        print(f"\nTrying PyTorch with {b} ...")
        if not pip("install", "torch", "torchvision", "--index-url", TORCH_INDEX + b):
            print(f"  {b}: not available for this Python/PyTorch, trying the next one")
            continue
        ok, out = python_check(
            "import torch; assert torch.cuda.is_available(), 'CUDA not available'; "
            "x = torch.ones(8, device='cuda') * 2; torch.cuda.synchronize(); "
            "print(torch.__version__, '|', torch.cuda.get_device_name(0))")
        if ok:
            print(f"  OK: {out}")
            return "cuda"
        print(f"  {b} installed but does not work on this GPU ({out.splitlines()[-1] if out else 'unknown error'})")
        pip("uninstall", "-y", "torch", "torchvision")
    print("\nNo CUDA build worked on this system. Installing the CPU build instead.")
    return install_cpu()

def install_directml():
    lo, hi = DIRECTML_PYTHON
    if not (lo <= PY <= hi):
        print(f"\nDirectML needs Python {lo[0]}.{lo[1]}-{hi[0]}.{hi[1]}, but this is Python {PY[0]}.{PY[1]}.")
        print("torch-directml has no build for this Python version. To use your AMD/Intel GPU,")
        print(f"install Python {hi[0]}.{hi[1]} (python.org) and run this installer with it.")
        if DRY_RUN or input("Install the CPU build instead? [Y/n]: ").strip().lower() in ("", "y", "yes"):
            return install_cpu()
        sys.exit(1)
    print("\nInstalling torch-directml (it pulls the torch/torchvision versions it requires)...")
    if pip("install", "torch-directml"):
        ok, out = python_check(
            "import torch, torch_directml; d = torch_directml.device(); "
            "x = (torch.ones(8) * 2).to(d); print(torch.__version__, '|', torch_directml.device_name(0))")
        if ok:
            print(f"  OK: {out}")
            return "directml"
        print(f"  torch-directml installed but failed a test ({out.splitlines()[-1] if out else 'unknown error'})")
    print("\nDirectML could not be set up. Installing the CPU build instead.")
    pip("uninstall", "-y", "torch", "torchvision", "torch-directml")
    return install_cpu()

def install_cpu():
    print("\nInstalling PyTorch (CPU)...")
    if not pip("install", "torch", "torchvision", "--index-url", TORCH_INDEX + "cpu"):
        sys.exit("PyTorch installation failed -- see the pip output above.")
    return "cpu"

def install_requirements():
    print("\nInstalling other packages (requirements.txt)...\n")
    if not pip("install", "-r", os.path.join(HERE, "requirements.txt")):
        sys.exit("Installing requirements.txt failed -- see the pip output above.")

# ------------------------------------------------------------

def check_ffmpeg():
    print("\nChecking FFmpeg...")
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        print("  FFmpeg found.")
    else:
        print()
        print("WARNING")
        print("-------------------------------------")
        print("FFmpeg was not found in PATH. It is needed for video and GIF")
        print("input (images work without it). Download it from")
        print("https://ffmpeg.org/download.html and add its 'bin' folder to PATH.")
        print()

def pick_gui(choice):
    if choice == "1":
        for name in ("deepmosaicui_stable_old.pyw", "deepmosaicui.pyw"):
            if os.path.isfile(os.path.join(HERE, name)):
                return name
    return "deepmosaicui_modern_NEW.pyw"

def create_launchers(gui):
    """Launchers that start the GUI with THIS Python -- so a venv is used even
    when double-clicking. (Double-clicking a .pyw uses the system Python.)"""
    if os.name != "nt":
        print(f"\nStart the program with:  {sys.executable} {gui}")
        return
    exe_dir = os.path.dirname(sys.executable)
    pyw = os.path.join(exe_dir, "pythonw.exe")
    runner = pyw if os.path.isfile(pyw) else sys.executable
    launchers = {"Launch DeepMosaicsPlus.bat": gui}
    if os.path.isfile(os.path.join(HERE, "tools", "dataset_prep_ui.pyw")):
        launchers["Launch Dataset Tool.bat"] = os.path.join("tools", "dataset_prep_ui.pyw")
    print("\nCreating launchers...")
    for bat, target in launchers.items():
        path = os.path.join(HERE, bat)
        print(f"  {bat} -> {target}")
        if DRY_RUN:
            continue
        with open(path, "w") as f:
            f.write('@echo off\n')
            f.write('cd /d "%~dp0"\n')
            f.write(f'start "" "{runner}" "{target}"\n')

# ------------------------------------------------------------

def main():
    title()
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"Environment: {'virtual environment' if in_venv else 'system Python'}")
    if DRY_RUN:
        print("DRY RUN: nothing will be installed.")
    print()

    print("Select interface\n")
    print("1) Stable UI (CustomTkinter)")
    print("2) Modern UI (PyQt6)")
    print()
    choice = "2" if DRY_RUN else None
    while choice not in ("1", "2"):
        choice = input("Selection: ").strip()
    print()

    backend = choose_backend()
    remove_existing_torch()
    installed = {"cuda": install_cuda, "directml": install_directml, "cpu": install_cpu}[backend]()
    install_requirements()
    check_ffmpeg()
    gui = pick_gui(choice)
    create_launchers(gui)

    print()
    print("=" * 55)
    print(f"Installation complete  (PyTorch backend: {installed.upper()})")
    print("=" * 55)
    if os.name == "nt":
        print("\nStart the program with 'Launch DeepMosaicsPlus.bat'.")
        print("It uses this Python; double-clicking the .pyw files directly may")
        print("use a different Python (e.g. outside your venv).")
    pause()

# ------------------------------------------------------------

if __name__ == "__main__":
    main()
