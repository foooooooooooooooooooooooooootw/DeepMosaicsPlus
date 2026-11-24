# Size-based model manifest (`--auto_model`)

`size_manifest.json` maps a detected mosaic block size to a specific
pretrained clean-model checkpoint.

- Keys are bucket labels in percent of **the square the clean model sees**:
  the crop around the mosaic area, resized to the model's input size (see
  `util.mosaic.estimate_crop_pct_multi`). `"3"` = blocks ~3% of that square,
  e.g. 15 px blocks in a 500 px crop. (Measuring against the whole frame, as
  earlier versions did, picks models trained on much smaller blocks whenever
  the mosaic area is small.)
- The `[auto_model]` log line prints this value for each video, so you can see
  which range your videos need and choose buckets to match.
- Values are paths to `.pth` checkpoints, **relative to this file's folder**
  (or absolute paths, if you'd rather keep checkpoints elsewhere).
- Detected sizes outside your min/max buckets are clamped to the nearest
  bucket rather than erroring — e.g. a 0.4% detection with buckets 1-6 will
  use the "1" bucket's model.

The filenames above (`clean_1pct.pth` ... `clean_6pct.pth`) are placeholders
— drop in your actual trained checkpoints with matching names, or edit the
JSON to point at whatever filenames/paths you're actually using.

Point `--model_manifest` at a different file if you want multiple manifests
(e.g. one per model architecture/`--netG` setting).
