# Raw training data — where to put things

This folder is where you drop your source material before running any of
the dataset-building tools. Nothing in here is consumed directly by
training — these are inputs to `tools/bucket_real_pairs.py` and
`make_datasets/make_video_dataset.py`, which turn this into the actual
training datasets under `datasets/real_pairs/` and `datasets/synthetic_pool/`.

## censored/  +  uncensored/   (real paired clips)

For each real-world source where you have BOTH a censored and an
uncensored version, put them in these two folders **under the exact same
filename**:

```
datasets/raw/censored/clip1.mp4        datasets/raw/uncensored/clip1.mp4
datasets/raw/censored/scene04.mp4      datasets/raw/uncensored/scene04.mp4
```

The filename is how they get matched up — `clip1.mp4` in `censored/` is
paired with `clip1.mp4` in `uncensored/`, nothing else. A file with no
matching name in the other folder is skipped (and reported) rather than
silently ignored.

**Alignment matters more than anything else here.** Both files need to be
the same content, frame-for-frame — same cuts, same timing, no extra
intro/outro on one but not the other. Misaligned pairs actively hurt
training, worse than not including them. Spot-check a few frames before
trusting a pair.

Once your files are here, run:
```
python tools/bucket_real_pairs.py \
    --censored_dir datasets/raw/censored \
    --uncensored_dir datasets/raw/uncensored \
    --output_dir datasets/real_pairs \
    --mosaic_position_model_path mosaic_position.pth \
    --model_manifest pretrained_models/mosaic/size_manifest.json
```
This auto-detects each pair's mosaic size and sorts it into
`datasets/real_pairs/bucket_<N>/`.

**Note:** it'll also ask whether to copy each pair's uncensored clip into
`synthetic_source/` below — worth saying yes to if your real pairs for a
given bucket come from only a handful of distinct source videos, since it
reuses that same footage as raw material for synthetic mosaics at *other*
bucket sizes too, adding content diversity per bucket without new source
footage. Pass `--reuse_uncensored_for_synthetic no` to skip this and keep
the two data sources fully separate, or `yes` to skip the interactive
prompt (e.g. for scripting).

## synthetic_source/   (clean footage for synthetic mosaic generation)

Uncensored source videos with **no real mosaic in them at all** — this is
the raw material `make_datasets/make_video_dataset.py` uses to build the
shared synthetic pool (crops + masks), which `train/clean/train.py` then
synthesizes mosaics onto on-the-fly at whatever size each bucket needs via
`--dataset_mosaic_pct`. One shared extraction serves all 6 buckets — you
don't need to duplicate this footage per bucket.

Just drop video files in here, no naming convention needed:
```
datasets/raw/synthetic_source/anything.mp4
```

Then run (see docs/training_with_your_own_dataset.md for the full option
list):
```
cd make_datasets
python make_video_dataset.py --datadir ../datasets/raw/synthetic_source \
    --savedir ../datasets/synthetic_pool --fps 24
```

**This folder can be left empty if you only want to train on real pairs** —
see the note in `tools/train_all_buckets.py`'s docstring; leaving
`--synthetic_dataset` unset (or pointing at an empty/nonexistent folder)
trains purely from whatever's in `datasets/real_pairs/` for each bucket.

## textures/   (images for generated clips, optional)

Close-up photos or texture packs that step 3 ("Generated clips", procedural
mode) cuts patches from to fill backgrounds and shapes: fabric, hair and fur,
skin-like surfaces, wood, stone, sand, foliage, paper, anime screenshots
without mosaic... Any size and format (jpg, png, webp, tiff); subfolders are
searched too. High-resolution, sharp close-ups work best; avoid blurry or
flat images and ones with text or watermarks. WebP at quality 90+ is fine.

Each run loads a random 80 of them, each kept at up to 2048 px, so a large
collection costs no extra memory: every run just picks a different set.

The dataset tool uses this folder by default and skips it while it is empty.
From the command line:
```
python tools/make_generated_dataset.py --clips 200 --textures datasets/raw/textures
```
