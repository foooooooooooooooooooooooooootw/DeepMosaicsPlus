"""
Train one clean model per size bucket in your manifest, combining:
  - a shared pool of synthetic origin_image/+mask/ video folders (same data
    reused for every bucket — see docs/training_with_your_own_dataset.md /
    make_datasets/make_video_dataset.py for how to build this), with
    train/clean/train.py's --dataset_mosaic_pct constraining synthesis to
    each bucket's size band at train time, and
  - that bucket's real censored/uncensored pairs, if any (from
    tools/bucket_real_pairs.py's output).

For each bucket, this builds one merged dataset view (via symlinks/junctions
so the — possibly large — synthetic frame data isn't duplicated on disk),
then runs train/clean/train.py against it with --dataset_mosaic_pct set,
looping through every bucket in your manifest sequentially.

This has NOT been run end-to-end (no GPU/torch in the environment this was
built in) — recommend a quick dry run on one bucket with a tiny synthetic
pool and --n_epoch 1 before committing to a full multi-bucket run.

Usage:
    python tools/train_all_buckets.py \
        --synthetic_dataset datasets/synthetic_pool \
        --real_pairs_dataset datasets/real_pairs \
        --model_manifest pretrained_models/mosaic/size_manifest.json \
        --work_dir datasets/_merged \
        -- --batchsize 16 --n_epoch 200 --gpu_id 0

Anything after a bare `--` is forwarded as-is to train/clean/train.py
(finesize, batchsize, n_epoch, gpu_id, lr, etc.) — this script only injects
--dataset, --dataset_mosaic_pct, and --savename itself.
"""
import os
import sys
import shutil
import platform
import subprocess
import argparse
import json

# Windows consoles often default to a non-UTF-8 code page (cp1252/cp437),
# which mangles Unicode characters like em-dashes into '?' or '�'. Force
# UTF-8 stdout so any character in this script's output prints correctly
# regardless of the system's console code page.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def link_or_copy(src, dst):
    """Create dst -> src as a symlink (Unix/macOS/Windows dev mode), falling
    back to an NTFS directory junction (Windows, no special privileges),
    falling back to a full copy (always works, costs disk + time) as a last
    resort. Skips silently if dst already exists (idempotent re-runs).
    """
    if os.path.exists(dst):
        return
    src = os.path.abspath(src)
    try:
        os.symlink(src, dst, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if platform.system() == 'Windows':
        try:
            subprocess.run(['cmd', '/c', 'mklink', '/J', dst, src],
                            check=True, capture_output=True)
            return
        except Exception:
            pass
    print(f"  (falling back to copying {os.path.basename(src)} — symlink/junction unavailable; "
          f"this uses extra disk space and time)")
    shutil.copytree(src, dst)


def build_merged_dataset(work_dir, bucket_label, synthetic_dataset, real_pairs_dataset):
    merged_dir = os.path.join(work_dir, f'bucket_{bucket_label}_merged')
    os.makedirs(merged_dir, exist_ok=True)

    n_synth = 0
    if synthetic_dataset and os.path.isdir(synthetic_dataset):
        for name in os.listdir(synthetic_dataset):
            src = os.path.join(synthetic_dataset, name)
            if os.path.isdir(src):
                link_or_copy(src, os.path.join(merged_dir, name))
                n_synth += 1

    n_real = 0
    if real_pairs_dataset:
        bucket_dir = os.path.join(real_pairs_dataset, f'bucket_{bucket_label}')
        if os.path.isdir(bucket_dir):
            for name in os.listdir(bucket_dir):
                src = os.path.join(bucket_dir, name)
                if os.path.isdir(src):
                    # prefix to avoid name collisions with synthetic folders
                    link_or_copy(src, os.path.join(merged_dir, f'realpair_{name}'))
                    n_real += 1

    return merged_dir, n_synth, n_real


def resolve_python_exe():
    """Prefer python.exe over pythonw.exe for launching subprocesses.
    If this script itself was launched via pythonw.exe (e.g. because it was
    started from a .pyw GUI, which Windows associates with pythonw.exe by
    default), sys.executable reports pythonw.exe -- and that choice
    propagates to every subprocess launched with sys.executable, including
    train.py here. pythonw.exe has no console and no real stdio handles,
    which can make output capture through a multi-level subprocess chain
    (GUI -> this script -> train.py) behave inconsistently. python.exe is
    installed alongside pythonw.exe on every standard Windows install, so
    substituting it costs nothing and removes that whole class of doubt.
    """
    exe = sys.executable
    if exe.lower().endswith('pythonw.exe'):
        candidate = exe[:-len('pythonw.exe')] + 'python.exe'
        if os.path.exists(candidate):
            return candidate
    return exe


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--synthetic_dataset', default=None,
        help='shared pool of origin_image/+mask/ folders (the OUTPUT of make_datasets/'
             'make_video_dataset.py, not your raw source videos), reused for every bucket. '
             'Optional — omit this, or point it at an empty/nonexistent folder, to train purely '
             'from real-pair data (see --real_pairs_dataset) with no synthetic mosaics at all.')
    parser.add_argument('--real_pairs_dataset', default=None,
        help='output of tools/bucket_real_pairs.py (contains bucket_<N>/ subfolders)')
    parser.add_argument('--model_manifest', default=os.path.join('pretrained_models','mosaic','size_manifest.json'))
    parser.add_argument('--work_dir', default=os.path.join('datasets', '_merged'))
    parser.add_argument('--train_script', default=os.path.join('train', 'clean', 'train.py'))
    parser.add_argument('--buckets', default=None,
        help='comma-separated subset of buckets to run, e.g. "1,3" — default: all buckets in the manifest')
    parser.add_argument('--dry_run', action='store_true', help='build merged folders and print commands, but do not actually launch training')
    args, train_args = parser.parse_known_args()
    if train_args and train_args[0] == '--':
        train_args = train_args[1:]

    if not os.path.exists(args.model_manifest):
        print(f"--model_manifest not found at {args.model_manifest}")
        sys.exit(1)
    with open(args.model_manifest) as f:
        manifest = json.load(f)
    bucket_labels = list(manifest.keys())
    if args.buckets:
        wanted = set(args.buckets.split(','))
        bucket_labels = [b for b in bucket_labels if b in wanted]

    if not args.synthetic_dataset and not args.real_pairs_dataset:
        print("Provide at least one of --synthetic_dataset or --real_pairs_dataset.")
        sys.exit(1)

    print(f"Buckets to train: {bucket_labels}")
    os.makedirs(args.work_dir, exist_ok=True)

    for bucket in bucket_labels:
        print(f"\n=== Bucket {bucket}% ===")
        merged_dir, n_synth, n_real = build_merged_dataset(
            args.work_dir, bucket, args.synthetic_dataset, args.real_pairs_dataset)
        print(f"  merged dataset: {n_synth} synthetic folder(s) + {n_real} real-pair folder(s) -> {merged_dir}")

        if n_synth == 0 and n_real == 0:
            print(f"  no data for this bucket at all, skipping.")
            continue

        savename = f'clean_{bucket}pct'
        cmd = [resolve_python_exe(), args.train_script,
               '--dataset', merged_dir,
               '--savename', savename]
        # Only pass --dataset_mosaic_pct if there's synthetic data to constrain —
        # harmless either way, but skip the noise if this bucket is real-pairs-only.
        if n_synth > 0:
            cmd += ['--dataset_mosaic_pct', bucket]
        cmd += train_args

        print(f"  command: {' '.join(cmd)}")
        if args.dry_run:
            continue
        # train.py's own path setup is now anchored to its file location (not
        # CWD), so this doesn't strictly need cwd= to work correctly anymore —
        # but setting it explicitly (to the project root, matching how the
        # rest of this project's relative defaults, e.g. model paths, assume
        # CWD=root) costs nothing and removes any doubt.
        #
        # Explicit PIPE + line-by-line streaming here (rather than
        # subprocess.run(cmd, cwd=ROOT) with no redirection) guarantees
        # train.py's output flows through THIS script's own stdout — which is
        # what the GUI actually captures — regardless of how this script
        # itself was launched (python.exe vs pythonw.exe, with or without a
        # real console). Relying on inherited stdio handles through a
        # multi-level subprocess chain is exactly the kind of thing that can
        # work in one launch context and silently fail in another.
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
        process.wait()
        if process.returncode != 0:
            print(f"  ** training for bucket {bucket}% exited with code {process.returncode} - "
                  f"stopping here rather than continuing to the next bucket. **")
            sys.exit(process.returncode)

    print("\nDone. Trained checkpoints are under checkpoints/<savename>/ as usual for train/clean/train.py — "
          "pick the checkpoint you want per bucket and copy/rename it into your --model_manifest's expected "
          "path (e.g. pretrained_models/mosaic/clean_1pct.pth).")


if __name__ == '__main__':
    main()
