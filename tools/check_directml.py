"""
Check whether DirectML computes the video clean model correctly.

Runs the SAME input through the same model on the CPU (reference) and on
DirectML, then compares:
  1. Device transfer  -- are tensors copied to DirectML and back unchanged?
                         (checks contiguous and non-contiguous tensors)
  2. Conversion       -- does the 2-D form of the video model match the
                         original on the CPU? (isolates DirectML from the
                         Conv3d -> Conv2d conversion)
  3. Layer by layer   -- CPU vs DirectML after every layer, reporting the
                         first operation whose output goes wrong
  4. Final output     -- difference in 0-255 levels; optionally a side-by-side
                         image (CPU | DirectML | difference x8) with --save_image

Usage:
  python tools/check_directml.py --model_path path/to/clean_video_model.pth
  python tools/check_directml.py --model_path ... --media_path some.gif   (use real frames)
  python tools/check_directml.py --model_path ... --device 1               (second GPU)
  python tools/check_directml.py --model_path ... --save_image check.png   (visual comparison)
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch

import torch.nn.functional as F
from models.BVDNet import define_G, make_directml_compatible, _Conv3dAs2d, _BilinearUp2x

N, T, SIZE = 2, 5, 256
WARN, FAIL = 1e-3, 1e-2          # relative error thresholds per layer


def load_frames(media_path):
    """Up to 5 frames from a video/GIF/image, resized to the model's 256x256."""
    frames = []
    if media_path:
        cap = cv2.VideoCapture(media_path)
        while len(frames) < T:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            img = cv2.imread(media_path)
            if img is None:
                sys.exit(f"Could not read any frames from {media_path}")
            frames = [img]
    else:  # deterministic synthetic content: gradients + texture
        yy, xx = np.mgrid[0:SIZE, 0:SIZE] / SIZE
        rng = np.random.default_rng(0)
        for t in range(T):
            img = np.stack([xx, yy, (xx + yy + t * 0.05) % 1], axis=2) * 200
            img += rng.normal(0, 12, img.shape)
            frames.append(np.clip(img, 0, 255).astype(np.uint8))
    while len(frames) < T:
        frames.append(frames[-1])
    frames = [cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA)[:, :, ::-1] for f in frames]
    arr = np.stack(frames).astype(np.float32) / 255.0 * 2 - 1        # T,H,W,C in [-1, 1]
    stream = torch.from_numpy(arr.transpose(3, 0, 1, 2)[None].copy())  # 1,C,T,H,W
    previous = stream[:, :, N].clone()
    return stream, previous


def load_model(path, convert):
    net = define_G(N, 4, gpu_id='-1')
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)   # plain weights; no pickle code
    except TypeError:                                                     # torch too old for weights_only
        state = torch.load(path, map_location='cpu')
    net.load_state_dict(state)
    net.eval()
    return make_directml_compatible(net) if convert else net


def capture_layers(net):
    """Record each layer's output (moved to CPU) in execution order."""
    records, hooks = [], []
    for name, m in net.named_modules():
        if isinstance(m, _Conv3dAs2d) or (len(list(m.children())) == 0 and not isinstance(m, torch.nn.ParameterList)):
            def hook(mod, inp, out, name=name):
                t = torch.stack(out, 2) if isinstance(out, list) else out
                records.append((name, type(mod).__name__, t.detach().float().cpu()))
            hooks.append(m.register_forward_hook(hook))
    return records, hooks


def compare(net_cpu, net_dev, stream, previous):
    """Run both models on the same input; return (cpu_out, dev_out, per-layer rows).
    The models are recorded one at a time: BVDNet takes its LeakyReLU from a
    default argument, so every instance shares that one module object, and
    hooks on both models at once would fire during either model's pass."""
    rec_c, h_c = capture_layers(net_cpu)
    with torch.no_grad():
        out_c = net_cpu(stream, previous)
    for h in h_c:
        h.remove()
    rec_d, h_d = capture_layers(net_dev)
    with torch.no_grad():
        out_d = net_dev(stream, previous).cpu()
    for h in h_d:
        h.remove()
    rows = []
    for (name, kind, a), (_, _, b) in zip(rec_c, rec_d):
        rel = ((a - b).abs().max() / (a.abs().max() + 1e-8)).item()
        rows.append((name, kind, rel))
    return out_c, out_d, rows


def transfer_test(dev):
    x = torch.randn(1, 3, T, 64, 64)
    views = {
        'contiguous tensor': x[:, :, 1].contiguous(),
        'non-contiguous slice (frame cut from the 5-frame stack)': x[:, :, 1],
        'non-contiguous transpose (HWC->CHW image)': torch.randn(64, 64, 3).permute(2, 0, 1),
    }
    results = {}
    for label, t in views.items():
        back = t.to(dev).cpu()
        results[label] = torch.equal(back, t)
    return results


def op_checks(dev):
    """Single operations, CPU vs device. Shows which building blocks the
    device computes correctly -- including PyTorch's bilinear upsample, which
    the DirectML conversion replaces with matrix multiplications."""
    torch.manual_seed(0)
    x = torch.randn(1, 64, 32, 32)
    w = torch.randn(32, 64, 3, 3)
    up = _BilinearUp2x()
    ops = {
        'bilinear upsample (PyTorch built-in)': lambda t: F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False),
        'bilinear upsample (matrix form, used by the conversion)': lambda t: up(t),
        'conv2d 3x3': lambda t: F.conv2d(t, w.to(t.device)),
        'reflection pad': lambda t: F.pad(t, (1, 1, 1, 1), mode='reflect'),
        'matrix multiply': lambda t: torch.matmul(t, t.transpose(-1, -2)),
    }
    results = {}
    with torch.no_grad():
        for label, op in ops.items():
            a = op(x)
            b = op(x.to(dev)).cpu()
            results[label] = ((a - b).abs().max() / (a.abs().max() + 1e-8)).item()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model_path', required=True, help='the video clean model (.pth) you use')
    ap.add_argument('--media_path', default='', help='optional video/GIF/image to take real frames from')
    ap.add_argument('--device', type=int, default=0, help='DirectML device index (like --gpu_id)')
    ap.add_argument('--save_image', default='', help='optional: save a CPU | device | difference image here')
    ap.add_argument('--test_device', default='', help=argparse.SUPPRESS)   # self-test: e.g. "cpu"
    args = ap.parse_args()

    if args.test_device:
        dev, dev_name = torch.device(args.test_device), f"{args.test_device} (self-test)"
    else:
        try:
            import torch_directml
        except ImportError:
            sys.exit("torch-directml is not installed in this Python environment.")
        count = torch_directml.device_count()
        if args.device >= count:
            sys.exit(f"--device {args.device} not found ({count} DirectML device(s)).")
        dev, dev_name = torch_directml.device(args.device), torch_directml.device_name(args.device)
    print(f"Device under test: {dev_name}\n")

    stream, previous = load_frames(args.media_path)

    print("1. Device transfer (to device and back must be exact)")
    tr = transfer_test(dev)
    for label, ok in tr.items():
        print(f"   {'OK  ' if ok else 'FAIL'} {label}")
    print()

    print("2. Individual operations, CPU vs device (relative error)")
    ops = op_checks(dev)
    for label, rel in ops.items():
        print(f"   {'FAIL' if rel > FAIL else 'OK  '} {rel:9.2e}  {label}")
    print()

    ref = load_model(args.model_path, convert=False)
    net_cpu = load_model(args.model_path, convert=True)
    net_dev = load_model(args.model_path, convert=True).to(dev)

    with torch.no_grad():
        conv_err = (ref(stream, previous) - net_cpu(stream, previous)).abs().max().item()
    print(f"3. Conversion (original vs DirectML form, both on CPU): max difference {conv_err * 127.5:.3f} levels (0-255)")
    print(f"   {'OK' if conv_err * 127.5 < 1 else 'UNEXPECTED -- the conversion itself differs'}\n")

    t0 = time.time(); out_c, out_d, rows = compare(net_cpu, net_dev, stream, previous); t1 = time.time()
    print("4. Layer by layer, CPU vs device (relative error per layer)")
    first_bad, n_warn, shown = None, 0, 0
    for name, kind, rel in rows:
        if kind == 'LeakyReLU':
            name = '(activation, shared by all layers)'
        flag = 'FAIL' if rel > FAIL else ('warn' if rel > WARN else None)
        if flag == 'FAIL' and first_bad is None:
            first_bad = (name, kind, rel)
        if flag == 'warn':
            n_warn += 1
        if flag and shown < 12:              # only flagged layers; errors propagate, so cap the list
            print(f"   {flag} {rel:9.2e}  {kind:16s} {name}")
            shown += 1
    print(f"   {len(rows)} layer outputs compared: "
          f"{sum(r[2] > FAIL for r in rows)} failing (>{FAIL:g}), {n_warn} small differences (>{WARN:g})")
    print()

    diff = (out_c - out_d).abs() * 127.5
    print(f"5. Final output: max difference {diff.max().item():.2f} levels, mean {diff.mean().item():.3f} levels (0-255)")
    if args.save_image:
        to_img = lambda t: ((t[0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)[:, :, ::-1]
        amp = (np.abs(to_img(out_c).astype(int) - to_img(out_d).astype(int)) * 8).clip(0, 255).astype(np.uint8)
        os.makedirs(os.path.dirname(args.save_image) or '.', exist_ok=True)
        cv2.imwrite(args.save_image, np.hstack([to_img(out_c), to_img(out_d), amp]))
        print(f"   side-by-side saved: {args.save_image}   (CPU | device | difference x8)")
    print()

    print("VERDICT")
    builtin_bad = ops['bilinear upsample (PyTorch built-in)'] > FAIL
    if not all(tr.values()):
        bad = [k for k, ok in tr.items() if not ok]
        print(f"   DirectML corrupts tensor copies: {', '.join(bad)}.")
        print("   This garbles model inputs and would show up as noise.")
    elif first_bad:
        print(f"   DirectML computes '{first_bad[1]}' incorrectly (first failing layer: {first_bad[0]},")
        print(f"   relative error {first_bad[2]:.1e}). Errors after it are knock-on effects.")
    elif diff.max().item() > 2:
        print("   No single layer fails, but small differences add up to a visible change in the output.")
    elif builtin_bad:
        print("   DirectML's built-in bilinear upsample is broken on this system, but the converted")
        print("   model (which uses the matrix form instead) matches the CPU. Cleaning on DirectML")
        print("   should now look the same as on CPU.")
    else:
        print("   DirectML matches the CPU. The noise is not caused by DirectML computing the model")
        print("   differently -- compare a CPU run of the same file (GPU ID -1) to confirm.")
    print("\nPlease send this whole output back.")


if __name__ == '__main__':
    main()
