"""
Diagnose why DeepMosaicsPlus can't use an NVIDIA GPU (or hits CUDA errors).

Run it with the SAME Python that runs the program (e.g. from your activated venv):
    python tools/check_gpu.py

It checks the things that actually go wrong, in order:
  * the Python / PyTorch / torchvision / numpy / OpenCV versions and whether they work together
  * whether the PyTorch build can use CUDA at all (CPU-only builds are common on Windows)
  * your NVIDIA driver vs. what that PyTorch build needs
  * whether the build has GPU code for your card's architecture (RTX 50-series "sm_120"
    needs PyTorch 2.7+ built for CUDA 12.8+; CUDA 13 builds dropped GTX 10 / V100 and older)
  * a real self-test: padding, convolution, upsampling and matrix multiply on the GPU

then prints a verdict with the next step. Please send the whole output if you ask for help.
It is standalone (it only needs torch) so it still works when the rest of the app can't start.
"""
import os
import re
import shutil
import subprocess
import sys

# Minimum NVIDIA driver (approximate, Windows/Linux) for each CUDA version a PyTorch build can use.
DRIVER_MIN = [((13, 0), 580), ((12, 9), 575), ((12, 8), 570), ((12, 6), 560),
              ((12, 4), 550), ((12, 1), 530), ((11, 8), 520)]


def driver_min_for(cuda):
    for ver, drv in DRIVER_MIN:
        if cuda >= ver:
            return drv
    return None


def parse_ver(text):
    m = re.match(r"\s*(\d+)\.(\d+)", text or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


# --------------------------------------------------------------------------- gathering

def gather_nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    info = {"gpus": [], "cuda_max": None}
    for query in ("name,driver_version,compute_cap", "name,driver_version"):   # old drivers lack compute_cap
        try:
            out = subprocess.run([exe, f"--query-gpu={query}", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30)
        except Exception:
            continue
        if out.returncode == 0 and out.stdout.strip():
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                info["gpus"].append({"name": parts[0], "driver": parts[1] if len(parts) > 1 else "",
                                     "cc": parse_ver(parts[2]) if len(parts) > 2 else None})
            break
    try:
        head = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", head)
        if m:
            info["cuda_max"] = (int(m.group(1)), int(m.group(2)))
    except Exception:
        pass
    return info


def gather_python_packages():
    pk = {"python": sys.version.split()[0], "executable": sys.executable,
          "venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix)}
    for mod in ("numpy", "cv2"):
        try:
            m = __import__(mod)
            pk[mod] = m.__version__
        except Exception as e:
            pk[mod] = None
            pk[mod + "_error"] = f"{type(e).__name__}: {e}"
    return pk


def gather_torch():
    t = {"import_error": None}
    try:
        import torch
    except Exception as e:
        t["import_error"] = f"{type(e).__name__}: {e}"
        return t
    t.update(version=torch.__version__, cuda_build=torch.version.cuda, available=False,
             arch_list=[], device=None, cc=None, ops={}, tv_version=None, tv_error=None)
    try:
        import torchvision
        t["tv_version"] = torchvision.__version__
        import torch as _t
        torchvision.ops.nms(_t.tensor([[0., 0., 1., 1.]]), _t.tensor([1.]), 0.5)   # fails if torchvision doesn't match torch
    except Exception as e:
        t["tv_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
    try:
        t["arch_list"] = list(torch.cuda.get_arch_list())
        t["available"] = bool(torch.cuda.is_available())
    except Exception as e:
        t["cuda_error"] = f"{type(e).__name__}: {e}"
    if t["available"]:
        try:
            t["device"] = torch.cuda.get_device_name(0)
            t["cc"] = tuple(torch.cuda.get_device_capability(0))
        except Exception:
            pass
        import torch.nn.functional as F
        steps = {
            "elementwise math": lambda: (torch.ones(8, device="cuda") * 2).sum(),
            "matrix multiply": lambda: torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"),
            "reflection padding": lambda: F.pad(torch.randn(1, 3, 16, 16, device="cuda"), (1, 1, 1, 1), mode="reflect"),
            "convolution (cuDNN)": lambda: F.conv2d(torch.randn(1, 3, 32, 32, device="cuda"), torch.randn(8, 3, 3, 3, device="cuda")),
            "bilinear upsample": lambda: F.interpolate(torch.randn(1, 4, 16, 16, device="cuda"), scale_factor=2, mode="bilinear", align_corners=False),
        }
        for name, fn in steps.items():
            try:
                fn()
                torch.cuda.synchronize()
                t["ops"][name] = None
            except Exception as e:
                t["ops"][name] = (str(e).strip().splitlines() or [type(e).__name__])[0][:200]
    return t


# --------------------------------------------------------------------------- diagnosing (pure, testable)

def arch_supported(cc, arch_list):
    """Whether a build with these kernels can run on a GPU of compute capability cc.
    Exact SASS (sm_XY) or same-major lower-minor SASS works; PTX (compute_XY) can be JIT-compiled
    by the driver but only for architectures the build's CUDA toolkit knew about."""
    if cc is None or not arch_list:
        return None
    sass = [parse_ver(a.split("_")[1][:-1] + "." + a.split("_")[1][-1]) for a in arch_list if a.startswith("sm_")]
    return any(s and s[0] == cc[0] and s[1] <= cc[1] for s in sass)


def diagnose(pk, smi, t):
    """Return (problems, notes). Each problem is (headline, next_step)."""
    problems, notes = [], []
    if t.get("import_error"):
        return [(f"PyTorch can't be imported: {t['import_error']}",
                 "Reinstall it: python install_script.py (run it in your venv).")], notes
    for mod, pkg in (("numpy", "numpy"), ("cv2", "opencv-python<5")):
        if pk.get(mod) is None:
            problems.append((f"{mod} fails to import ({pk.get(mod + '_error')}).",
                             f"pip install -U \"{pkg}\" numpy    (numpy 2.x needs a recent OpenCV 4.x)"))
    if t.get("tv_error"):
        problems.append((f"torchvision {t.get('tv_version')} doesn't work with torch {t['version']} ({t['tv_error']}).",
                         "torch and torchvision must be installed TOGETHER from the same index. Downgrading only torch leaves a mismatched "
                         "torchvision behind -- the most common cause of 'it still doesn't work' after changing versions. "
                         "Fix: python install_script.py (it reinstalls both)."))
    cuda_build = parse_ver(t.get("cuda_build") or "")
    gpus = (smi or {}).get("gpus", [])
    if cuda_build is None:
        if gpus:
            problems.append((f"This PyTorch ({t['version']}) is a CPU-only build, but an NVIDIA GPU ({gpus[0]['name']}) is installed.",
                             "On Windows a plain 'pip install torch' gives the CPU build. Run: python install_script.py"))
        else:
            notes.append("CPU-only PyTorch and no NVIDIA GPU found: the program will run on the CPU.")
        return problems, notes
    if smi is None:
        problems.append(("nvidia-smi was not found, so no NVIDIA driver is installed (or it isn't on PATH).",
                         "Install/repair the NVIDIA driver from nvidia.com, then reopen the terminal."))
        return problems, notes
    gpu = gpus[0] if gpus else {}
    cc = t.get("cc") or gpu.get("cc")
    if not t.get("available"):
        need = driver_min_for(cuda_build)
        drv = parse_ver(gpu.get("driver", ""))
        if need and drv and drv[0] < need:
            problems.append((f"The NVIDIA driver ({gpu.get('driver')}) is too old for this PyTorch's CUDA {t['cuda_build']} (needs about {need}+).",
                             "Update the NVIDIA driver, or install a PyTorch built for an older CUDA (python install_script.py picks one that fits)."))
        else:
            problems.append(("PyTorch has CUDA support but can't use the GPU.",
                             "Update the NVIDIA driver and reinstall PyTorch (python install_script.py). If you use a laptop with hybrid graphics, "
                             "make sure the program runs on the NVIDIA GPU."))
        return problems, notes
    if cc is not None and arch_supported(cc, t.get("arch_list")) is False:
        blackwell = cc[0] >= 12
        problems.append((f"This PyTorch build ({t['version']}, CUDA {t['cuda_build']}) has no GPU code for your {t.get('device') or gpu.get('name')} "
                         f"(compute capability {cc[0]}.{cc[1]} = sm_{cc[0]}{cc[1]}); it was built for: {', '.join(t['arch_list'])}.",
                         "RTX 50-series (Blackwell) needs PyTorch 2.7+ built for CUDA 12.8 or newer (or a CUDA 13 build with driver 580+). "
                         "Do a clean install in a NEW venv: python install_script.py." if blackwell else
                         "Install a PyTorch build that includes your GPU's architecture (python install_script.py). CUDA 13 builds dropped "
                         "GPUs older than the GTX 16 / RTX 20 series; use a CUDA 12.x build for those."))
    failed = {k: v for k, v in t.get("ops", {}).items() if v}
    if failed:
        first = next(iter(failed.items()))
        if not problems:
            problems.append((f"A real GPU operation failed ({first[0]}): {first[1]}",
                             "Reinstall PyTorch (python install_script.py in a fresh venv) and update the NVIDIA driver."))
        notes.append("Failed steps: " + "; ".join(f"{k}: {v}" for k, v in failed.items()))
    return problems, notes


# --------------------------------------------------------------------------- output

def main():
    pk, smi = gather_python_packages(), gather_nvidia_smi()
    t = gather_torch()
    print("=== Environment ===")
    print(f"Python {pk['python']}  ({'virtual environment' if pk['venv'] else 'system Python'})")
    print(f"  {pk['executable']}")
    print(f"numpy {pk.get('numpy')}  |  OpenCV {pk.get('cv2')}")
    print("\n=== PyTorch ===")
    if t.get("import_error"):
        print(f"torch failed to import: {t['import_error']}")
    else:
        print(f"torch {t['version']}  |  built for CUDA {t['cuda_build']}  |  torchvision {t.get('tv_version')}"
              f"{'  (BROKEN: ' + t['tv_error'] + ')' if t.get('tv_error') else ''}")
        print(f"GPU kernels in this build: {', '.join(t['arch_list']) or 'none (CPU-only build)'}")
        print(f"CUDA usable: {t['available']}" + (f"  |  {t['device']}  (compute capability {t['cc'][0]}.{t['cc'][1]})" if t.get('device') else ""))
    print("\n=== NVIDIA driver ===")
    if smi is None:
        print("nvidia-smi not found")
    else:
        for g in smi["gpus"]:
            print(f"{g['name']}  |  driver {g['driver']}" + (f"  |  compute capability {g['cc'][0]}.{g['cc'][1]}" if g['cc'] else ""))
        if smi["cuda_max"]:
            print(f"driver supports CUDA up to {smi['cuda_max'][0]}.{smi['cuda_max'][1]}")
    if t.get("ops"):
        print("\n=== GPU self-test ===")
        for name, err in t["ops"].items():
            print(f"  {'ok  ' if not err else 'FAIL'} {name}" + (f"  --  {err}" if err else ""))
    problems, notes = diagnose(pk, smi, t)
    print("\n=== VERDICT ===")
    if not problems:
        print("No problem found." + ("  " + notes[0] if notes else "  PyTorch can use your GPU."))
    for i, (head, step) in enumerate(problems, 1):
        print(f"{i}. {head}\n   -> {step}")
    for n in notes:
        if problems:
            print(f"   note: {n}")
    print("\nPlease send this whole output if you ask for help.")


if __name__ == "__main__":
    main()
