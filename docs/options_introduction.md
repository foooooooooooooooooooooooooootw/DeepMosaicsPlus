## Introduction to options
If you need more effects,  use '--option your-parameters' to enter what you need.

### Base

|    Option    |        Description         |                 Default                 |
| :----------: | :------------------------: | :-------------------------------------: |
|  --gpu_id   |   if -1, do not use gpu    |                    0                    |
| --media_path | your videos or images path |            ./imgs/ruoruo.jpg            |
| --start_time | start position of video, default is the beginning of video | '00:00:00' |
| --last_time | limit the duration of the video, default is the entire video | '00:00:00' |
|    --mode    |    program running mode(auto/clean/add/style)    |                 'auto'                  |
| --model_path |   pretrained model path    | ./pretrained_models/mosaic/add_face.pth |
| --result_dir |  output media will be saved here|                 ./result          |
| --temp_dir | Temporary files will go here | ./tmp |
|    --fps    |    read and output fps, if 0-> origin    |                 0                  |
| --no_preview | if specified,do not preview images when processing video. eg.(when run it on server) | Flase |

### AddMosaic

|    Option    |        Description         |                 Default                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --mosaic_mod | type of mosaic -> squa_avg/ squa_random/ squa_avg_circle_edge/ rect_avg/random |                    squa_avg                    |
| --mosaic_size | mosaic size,if 0 -> auto size |            0            |
|    --mask_extend    |    extend mosaic area    |         10  |
| --mask_threshold | threshold of recognize mosaic position 0~255 | 64 |

### CleanMosaic

|    Option    |        Description         |                 Default                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --traditional | if specified, use traditional image processing methods to clean mosaic |                                        |
| --tr_blur | ksize of blur when using traditional method, it will affect final quality |            10            |
|    --tr_down    |    downsample when using traditional method,it will affect final quality    |         10  |
| --medfilt_num | medfilt window of mosaic movement in the video | 11 |

### VideoEncode

Output videos are re-encoded from the processed frame sequence with ffmpeg. `--encode_vcodec`
selects the encoder; `--encode_crf` sets its quality knob (CRF, or CQ for the nvenc encoders).

|      Option      |                              Description                              |  Default  |
| :--------------: | :--------------------------------------------------------------------: | :-------: |
| --encode_vcodec  | video codec -> h264 / hevc(h265) / av1 / vp9 / h264_nvenc / hevc_nvenc |   h264    |
|  --encode_crf    | CRF/CQ quality. If unset, uses the selected codec's own default below.|  (auto)   |

**Important: CRF is not comparable across codecs.** Each encoder has its own scale, so the
same number means a different quality/size trade-off on each one. If you don't pass
`--encode_crf`, DeepMosaics uses these per-codec defaults automatically:

|  Codec  |  Encoder     | CRF/CQ range | Default | Minimum ffmpeg version |
| :-----: | :----------: | :----------: | :-----: | :---------------------: |
| h264    | libx264      |   0 – 51     |   18    | 2.1 (Nov 2013)           |
| hevc/h265 | libx265    |   0 – 51     |   22    | 2.1 (Nov 2013)           |
| av1     | libsvtav1    |   0 – 63     |   30    | 4.4 (Apr 2021)           |
| vp9     | libvpx-vp9   |   0 – 63     |   31    | 2.4 (Oct 2014)           |
| h264_nvenc | h264_nvenc (NVIDIA GPU) | 0 – 51 | 19 | 3.1 (Jun 2016), plus an NVENC-capable driver/GPU |
| hevc_nvenc | hevc_nvenc (NVIDIA GPU) | 0 – 51 | 19 | 3.1 (Jun 2016), plus an NVENC-capable driver/GPU |

Notes:
- `libsvtav1` is the fastest practical AV1 encoder, but 4.4+ is only the *minimum*; recent ffmpeg
  (5.0+) ships more mature SVT-AV1 integration and is recommended for AV1 encoding.
- AV1 encoding is significantly slower than h264/hevc even on the SVT-AV1 encoder — expect longer
  processing times.
- The nvenc encoders require an NVIDIA GPU with hardware encode support and an ffmpeg build
  compiled with `--enable-nvenc`; check `ffmpeg -encoders | grep nvenc` to confirm availability.
- Run `ffmpeg -version` to check your installed version, and `ffmpeg -encoders | grep -E "libsvtav1|libx265|libvpx-vp9|nvenc"` to confirm your build actually includes the encoder you want.

### Style Transfer

|    Option    |        Description         |                 Default                 |
| :----------: | :------------------------: | :-------------------------------------: |
| --output_size | size of output media, if 0 -> origin |512|