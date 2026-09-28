# Size-based model manifest (`--auto_model`)

`size_manifest.json` maps a detected mosaic block size (as a % of the frame's
shorter side — see `util.mosaic.estimate_mosaic_block_pct`) to a specific
pretrained clean-model checkpoint.

- Keys are bucket labels in percent (e.g. `"3"` = a mosaic block that's ~3%
  of the frame's shorter side).
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
