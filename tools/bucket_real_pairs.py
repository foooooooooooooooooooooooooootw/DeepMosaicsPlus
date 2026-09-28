"""
Auto-bucket real censored/uncensored video pairs by detected mosaic size.

Given two folders — one of censored clips, one of matching uncensored clips,
each pair sharing the SAME filename across the two folders — this:
  1. Extracts frames from both videos into origin_image/ (uncensored) and
     mosaic_image/ (censored) — the exact layout util/dataloader.py's
     real-pair training path expects (see util.dataloader.VideoLoader).
  2. Runs the same block-size detector used by tools/test_mosaic_size.py on
     the censored frames to estimate the mosaic's size as % of frame short
     side.
  3. Copies the resulting pair folder into <output_dir>/bucket_<N>/<name>/,
     where N is the nearest bucket in your --model_manifest.

IMPORTANT — alignment: this assumes each pair's two videos are already the
same content, frame-for-frame, just with/without censoring (same cuts, same
timing). It does NOT verify that for you. Misaligned pairs are worse than
no real-pair data at all (see the earlier discussion) — spot check a few
extracted frame pairs before trusting a bucket's real-pair data in training.

Folder layout — see datasets/raw/README.md for the full picture:
    datasets/raw/censored/clip1.mp4
    datasets/raw/uncensored/clip1.mp4      <- same filename, different folder
    datasets/raw/censored/scene04.mp4
    datasets/raw/uncensored/scene04.mp4

Usage:
    python tools/bucket_real_pairs.py \
        --censored_dir datasets/raw/censored \
        --uncensored_dir datasets/raw/uncensored \
        --output_dir datasets/real_pairs \
        --mosaic_position_model_path mosaic_position.pth \
        --model_manifest pretrained_models/mosaic/size_manifest.json
"""
import os
import sys

# Windows consoles often default to a non-UTF-8 code page (cp1252/cp437),
# which mangles Unicode characters like em-dashes into '?' or '�'. Force
# UTF-8 stdout so any character in this script's output prints correctly
# regardless of the system's console code page.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from cores import Options
from models import loadmodel, runmodel
from util import mosaic as mosaic_util
from util import ffmpeg as ffmpeg_util
from util import util as util_


def find_pairs(censored_dir, uncensored_dir):
    censored_files = {f for f in os.listdir(censored_dir) if util_.is_video(os.path.join(censored_dir, f))}
    uncensored_files = {f for f in os.listdir(uncensored_dir) if util_.is_video(os.path.join(uncensored_dir, f))}
    common = sorted(censored_files & uncensored_files)
    pairs = []
    for f in common:
        name, _ext = os.path.splitext(f)
        pairs.append((name, os.path.join(censored_dir, f), os.path.join(uncensored_dir, f)))

    only_censored = censored_files - uncensored_files
    only_uncensored = uncensored_files - censored_files
    if only_censored:
        print(f"Note: {len(only_censored)} file(s) in censored/ have no matching filename in "
              f"uncensored/, skipped: {sorted(only_censored)}")
    if only_uncensored:
        print(f"Note: {len(only_uncensored)} file(s) in uncensored/ have no matching filename in "
              f"censored/, skipped: {sorted(only_uncensored)}")
    return pairs


def extract_frames(videopath, out_dir, fps):
    util_.makedirs(out_dir)
    ffmpeg_util.video2image(videopath, os.path.join(out_dir, '%05d.jpg'), fps=fps)


def detect_pct(opt, netM, censored_video, n_samples, min_pct, max_pct):
    cap = cv2.VideoCapture(censored_video)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    idxs = sorted(set(int(i * total / n_samples) for i in range(n_samples)))
    pairs = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        mask, x, y, size = runmodel.get_mosaic_position(frame, netM, opt)
        if mask is None or size <= getattr(opt, 'min_mosaic_size', 100):
            continue
        pairs.append((frame, mask))
    cap.release()
    if not pairs:
        return None
    result = mosaic_util.estimate_mosaic_block_pct_multi(
        pairs, min_confidence=getattr(opt, 'auto_model_min_confidence', 0.15),
        min_pct=min_pct, max_pct=max_pct)
    return result


def maybe_reuse_uncensored(pairs, mode, synthetic_source_dir):
    """Optionally copy each pair's uncensored clip into a shared synthetic-
    source folder, so it can ALSO be used as raw material for synthetic
    mosaic generation at other bucket sizes (make_video_dataset.py + train's
    --dataset_mosaic_pct) — not just the one real mosaic size this specific
    pair happens to have. Worthwhile if your real pairs for a bucket come
    from a small number of distinct sources: this multiplies effective
    content diversity per bucket without needing brand-new source footage.
    Skips files already copied (safe to re-run).
    """
    if mode == 'no':
        return
    if mode == 'ask':
        print(f"\nYou have {len(pairs)} uncensored clip(s). These can ALSO be reused as source "
              f"footage for synthetic mosaic generation at any bucket size (not just whatever "
              f"real size these particular pairs have) — useful if a bucket's real pairs come "
              f"from only a few distinct videos, since it adds content diversity for free.")
        answer = input(f"Copy them into {synthetic_source_dir} for that purpose? [y/N]: ").strip().lower()
        if answer not in ('y', 'yes'):
            print("Skipping — real pairs will only be used for their own detected bucket.")
            return
    os.makedirs(synthetic_source_dir, exist_ok=True)
    copied = 0
    for _name, _censored_path, uncensored_path in pairs:
        dst = os.path.join(synthetic_source_dir, os.path.basename(uncensored_path))
        if os.path.exists(dst):
            continue
        shutil.copy2(uncensored_path, dst)
        copied += 1
    print(f"Copied {copied} new uncensored clip(s) into {synthetic_source_dir}.")
    print(f"Note: make_datasets/make_video_dataset.py reprocesses everything in its --datadir on "
          f"every run and does not skip already-done videos — if you've already built a synthetic "
          f"pool from an earlier version of this folder, either point --startcnt past your existing "
          f"output count, or move already-processed source clips elsewhere before re-running it, "
          f"to avoid overwriting previous output.")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--censored_dir', required=True, help='folder of real censored clips')
    parser.add_argument('--uncensored_dir', required=True, help='folder of matching uncensored clips (same filenames)')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--fps', type=int, default=15,
        help='extraction frame rate for both videos in each pair. Match this to whatever --fps '
             'you use for make_datasets/make_video_dataset.py when building your synthetic pool, '
             'if combining both into the same bucket for training (train/clean/train.py\'s '
             'temporal model uses a frame-count stride, so mixed extraction rates across your '
             'combined dataset mean inconsistent real-world motion per training sample).')
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--reuse_uncensored_for_synthetic', choices=['ask', 'yes', 'no'], default='ask',
        help='Also copy each pair\'s uncensored clip into --synthetic_source_dir, so it can be '
             'reused as raw material for synthetic mosaic generation at OTHER bucket sizes (via '
             'make_datasets/make_video_dataset.py + --dataset_mosaic_pct) — not just the one real '
             'size this pair happens to have. Worth doing if your real pairs for a bucket come '
             'from few distinct sources, since it multiplies content diversity per bucket without '
             'new source footage. \'ask\' (default) prompts once interactively; \'yes\'/\'no\' skip '
             'the prompt, useful for scripting.')
    parser.add_argument('--synthetic_source_dir', default=os.path.join('datasets', 'raw', 'synthetic_source'))
    known, remaining = parser.parse_known_args()

    # Rebuild sys.argv with just the args Options() understands, plus the
    # --model_path shim (same reasoning as tools/test_mosaic_size.py: this
    # tool never runs the clean/generator model, only position detection).
    sys.argv = [sys.argv[0]] + remaining
    if '--model_path' not in sys.argv and '--mosaic_position_model_path' in sys.argv:
        pos_idx = sys.argv.index('--mosaic_position_model_path')
        sys.argv += ['--model_path', sys.argv[pos_idx + 1], '--mode', 'clean', '--netG', 'unet_128']

    opt = Options().getparse(test_flag=True)

    if not os.path.isdir(known.censored_dir):
        print(f"--censored_dir not found: {known.censored_dir}")
        sys.exit(1)
    if not os.path.isdir(known.uncensored_dir):
        print(f"--uncensored_dir not found: {known.uncensored_dir}")
        sys.exit(1)

    print("Loading mosaic-position model...")
    netM = loadmodel.bisenet(opt, 'mosaic')

    manifest_path = opt.model_manifest
    if not os.path.exists(manifest_path):
        print(f"--model_manifest not found at {manifest_path}. Create it first (see "
              f"pretrained_models/mosaic/README.md) so this tool knows your bucket boundaries.")
        sys.exit(1)
    manifest = loadmodel.load_size_manifest(manifest_path)
    min_pct, max_pct = min(manifest.keys()), max(manifest.keys())
    print(f"Bucket range from manifest: {min_pct:g}%-{max_pct:g}%")

    pairs = find_pairs(known.censored_dir, known.uncensored_dir)
    if not pairs:
        print("No matching filenames found between --censored_dir and --uncensored_dir.")
        sys.exit(1)
    print(f"Found {len(pairs)} pair(s).")

    maybe_reuse_uncensored(pairs, known.reuse_uncensored_for_synthetic, known.synthetic_source_dir)

    results = []
    for name, censored_path, uncensored_path in pairs:
        print(f"\n--- {name} ---")
        result = detect_pct(opt, netM, censored_path, known.n_samples, min_pct, max_pct)
        if result is None:
            print(f"  Could not confidently detect mosaic size, skipping. "
                  f"(Check the clip actually contains a detectable mosaic.)")
            continue
        _, bucket = loadmodel.select_model_by_pct(result['pct'], manifest_path)
        note = ""
        if result['n_nonsquare'] > 0:
            note = f"  ({result['n_nonsquare']}/{result['n_samples']} samples non-square)"
        print(f"  detected pct={result['pct']:.2f}%  confidence={result['confidence']:.2f}{note}  -> bucket {bucket:g}%")

        bucket_dir = os.path.join(known.output_dir, f'bucket_{bucket:g}', name)
        origin_dir = os.path.join(bucket_dir, 'origin_image')
        mosaic_dir = os.path.join(bucket_dir, 'mosaic_image')
        print(f"  extracting frames -> {bucket_dir}")
        extract_frames(uncensored_path, origin_dir, known.fps)
        extract_frames(censored_path, mosaic_dir, known.fps)

        n_orig = len(os.listdir(origin_dir))
        n_mosaic = len(os.listdir(mosaic_dir))
        if n_orig != n_mosaic:
            print(f"  ** WARNING: frame count mismatch (origin={n_orig}, mosaic={n_mosaic}) — "
                  f"the two source videos may differ in length/framerate. This pair's alignment "
                  f"is suspect; recommend checking manually before training on it. **")

        results.append((name, result['pct'], bucket))

    print("\n=== Summary ===")
    for name, pct, bucket in results:
        print(f"  {name}: {pct:.2f}% -> bucket {bucket:g}%")
    print(f"\nDone. {len(results)}/{len(pairs)} pairs bucketed under {known.output_dir}/bucket_<N>/")
    print("Recommended next step: spot-check a few frame pairs in each bucket folder for alignment "
          "before training (see this script's docstring).")


if __name__ == '__main__':
    main()
