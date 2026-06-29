import os
import shutil
import subprocess
import sys

# ------------------------------------------------------------

def cls():
    os.system("cls" if os.name == "nt" else "clear")

def title():
    cls()
    print("=" * 55)
    print("              Dependency Installer")
    print("=" * 55)
    print()

def pause():
    input("\nPress Enter to exit...")

# ------------------------------------------------------------

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)

# ------------------------------------------------------------

def detect_gpu():

    print("Detecting graphics hardware...")

    try:
        output = subprocess.check_output(
            [
                "powershell",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
            ],
            text=True
        ).lower()

        if "nvidia" in output:
            print("✓ NVIDIA GPU detected\n")
            return "cuda"

        if "amd" in output or "radeon" in output:
            print("✓ AMD GPU detected\n")
            return "directml"

        if "intel" in output:
            print("✓ Intel GPU detected\n")
            return "directml"

    except Exception:
        pass

    print("No compatible GPU detected.")
    print("Installing CPU version.\n")

    return "cpu"

# ------------------------------------------------------------

def install_pytorch(backend):

    print("Installing PyTorch...")

    if backend == "cuda":

        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision"
        ])

    elif backend == "directml":

        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torch-directml"
        ])

    else:

        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/cpu"
        ])

# ------------------------------------------------------------

def install_common():

    print("\nInstalling common packages...\n")

    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "opencv-python",
    ])

# ------------------------------------------------------------

def install_ui(choice):

    print("\nInstalling UI...\n")

    if choice == "1":

        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "customtkinter"
        ])

        return "stable"

    else:

        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "PyQt6"
        ])

        return "modern"

# ------------------------------------------------------------

def check_ffmpeg():

    print("\nChecking FFmpeg...")

    if shutil.which("ffmpeg"):

        print("✓ FFmpeg found.")

    else:

        print()
        print("WARNING")
        print("-------------------------------------")
        print("FFmpeg was not found in PATH.")
        print()
        print("Please install FFmpeg before")
        print("using this application.")
        print()

# ------------------------------------------------------------

def create_launcher(ui):

    print("\nCreating launcher...")

    filename = "Launch Dependency_Installer.bat"

    target = "stable_ui.pyw" if ui == "stable" else "modern_ui.pyw"

    with open(filename, "w") as f:

        f.write(f'@echo off\n')
        f.write(f'"{sys.executable}" "{target}"\n')

    print("✓ Launcher created.")

# ------------------------------------------------------------

def main():

    title()

    print("Python:", sys.version.split()[0])
    print()

    print("Select interface\n")
    print("1) Stable UI (CustomTkinter)")
    print("2) Modern UI (PyQt6)")
    print()

    while True:

        choice = input("Selection: ").strip()

        if choice in ("1", "2"):
            break

    print()

    backend = detect_gpu()

    install_pytorch(backend)

    install_common()

    ui = install_ui(choice)

    check_ffmpeg()

    create_launcher(ui)

    print()
    print("=" * 55)
    print("Installation Complete!")
    print("=" * 55)

    print("\nYou can now launch the program using by double-clicking the pyw files.")


    pause()

# ------------------------------------------------------------

if __name__ == "__main__":
    main()