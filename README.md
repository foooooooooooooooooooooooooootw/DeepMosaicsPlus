<div align="center">
  <img src="./imgs/logo.png" width="250"><br><br>
</div>

# DeepMosaicsPlus

**English**<br>
You can use it to remove mosaics in images and videos.<br>

### Examples

![image](./imgs/hand.gif)

- Compared with [DeepCreamPy](https://github.com/deeppomf/DeepCreamPy)

|                mosaic image                |            DeepCreamPy             |                   ours                    |
| :----------------------------------------: | :--------------------------------: | :---------------------------------------: |
| ![image](./imgs/example/face_a_mosaic.jpg) | ![image](./imgs/example/a_dcp.png) | ![image](./imgs/example/face_a_clean.jpg) |
| ![image](./imgs/example/face_b_mosaic.jpg) | ![image](./imgs/example/b_dcp.png) | ![image](./imgs/example/face_b_clean.jpg) |

### TODO

- Fix bugs relating to non-english letters and spaces in filenames
- Increase performance further by parallelizing/async I/O, saturating GPU with batches etc. 

## Run DeepMosaicsPlus

You can run DeepMosaicsPlus from source. Executable might come in the future.<br>

### Arguments

#### Base arguments
| Argument              | Type  | Default                                     | Description                                                                                        |
| --------------------- | ----- | ------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `--debug`             | flag  | `False`                                     | Enable debug mode.                                                                                 |
| `--gpu_id`            | `str` | `'0'`                                       | GPU ID to use; set to `-1` for CPU.                                                                |
| `--media_path`        | `str` | `'./imgs/ruoruo.jpg'`                       | Path to input media (image or video).                                                              |
| `-ss`, `--start_time` | `str` | `'00:00:00'`                                | Start time in the video (HH\:MM\:SS).                                                              |
| `-t`, `--last_time`   | `str` | `'00:00:00'`                                | Duration to process; `00:00:00` means full video.                                                  |
| `--mode`              | `str` | `'auto'`                                    | For now it's "clean" only, or at least its the only thing I optimized                              |
| `--model_path`        | `str` | `'./clean_youknow_video.pth'`               | Path to the pretrained model.                                                                      |
| `--result_dir`        | `str` | `'./result'`                                | Directory where results are saved.                                                                 |
| `--temp_dir`          | `str` | `'./tmp'`                                   | Directory for temporary files.                                                                     |
| `--tempimage_type`    | `str` | `'jpg'`                                     | Image format for temp files: `jpg` or `png`.                                                       |
| `--netG`              | `str` | `'auto'`                                    | Network to use for `clean/style`: `auto`, `unet_128`, `unet_256`, `resnet_9blocks`, `HD`, `video`. |
| `--fps`               | `int` | `0`                                         | Output FPS; `0` keeps original FPS.                                                                |
| `--no_preview`        | flag  | `False`                                     | Disable preview window (useful for servers).                                                       |
| `--output_size`       | `int` | `0`                                         | Output size; `0` preserves original resolution.                                                    |
| `--mask_threshold`    | `int` | `48`                                        | Mosaic detection sensitivity (0–255); lower = more sensitive.                                      |

#### Clean Arguments
| Argument                       | Type  | Default  | Description                                            |
| ------------------------------ | ----- | -------- | ------------------------------------------------------ |
| `--mosaic_position_model_path` | `str` | `'auto'` | Path or name of model for detecting mosaic positions.  |
| `--traditional`                | flag  | `False`  | Use traditional (non-AI) method for mosaic cleaning.   |
| `--tr_blur`                    | `int` | `10`     | Blur kernel size for traditional cleaning.             |
| `--tr_down`                    | `int` | `10`     | Downsampling factor for traditional cleaning.          |
| `--no_feather`                 | flag  | `False`  | Disable edge feathering and color correction (faster). |
| `--all_mosaic_area`            | flag  | `True`   | Find all mosaic regions instead of just the largest.   |
| `--medfilt_num`                | `int` | `5`      | Median filter window for smoothing mosaic detection.   |
| `--ex_mult`                    | `str` | `'auto'` | Area expansion multiplier for mosaics.                 |


### Pre-built binary package

For GUI just double click deepmosaicplusui.py or execute via commandline - run the command under "[Example](####Example)"

### Run From Source

#### Prerequisites

- Windows
- Python 3.6+
- [ffmpeg 3.4.6](http://ffmpeg.org/)
- [Pytorch 1.0+](https://pytorch.org/)
- torch_directml
- Any GPU. <br>

#### Modification Note:
<br>
 I changed it to use Directml because the CUDA version I made worked on my 2060 but not my 1070ti and I just gave up. 
 <br> <br>
 With directml it can be used on AMD GPUs as well. With my 7800XT at 75% clock speed processing for step 2 has sped up by 6x (1h -> 10 mins). It has been hardcoded to use directml (your gpu) and will only fallback to cpu if something fails. I tweaked ffmpeg args so it uses DirectX 11 for hardware acceleration. There is no need to use the gpu-id argument because if torch_directml is installed on your system it will autodetect any available directml devices. 
 <br> <br>
 The nature of directml means this is only available for windows. 
<br> <br>

#### Dependencies

This code depends on opencv-python, torchvision available via pip install.

#### Clone this repo

```bash
git clone https://github.com/foooooooooooooooooooooooooootw/DeepMosaicsPlus.git
cd DeepMosaicsPlus
```

#### Get Pre-Trained Models

You can download pre_trained models and put them into './pretrained_models'.<br>
[[Google Drive]](https://drive.google.com/open?id=1LTERcN33McoiztYEwBxMuRjjgxh4DEPs) [[百度云,提取码1x0a]](https://pan.baidu.com/s/10rN3U3zd5TmfGpO_PEShqQ)<br>
[[Introduction to pre-trained models]](./docs/pre-trained_models_introduction.md)<br>

In order to add/remove mosaic, there must be a model file `mosaic_position.pth` at `./pretrained_models/mosaic/mosaic_position.pth`

#### Example
```
python deepmosaic.py --media_path "weenus.mkv" --model_path "clean_youknow_video.pth"
```
#### More Parameters

If you want to test other images or videos, please refer to this file.<br>
[[options_introduction.md]](./docs/options_introduction.md) <br>

## Training With Your Own Dataset

If you want to train with your own dataset, please refer to [training_with_your_own_dataset.md](./docs/training_with_your_own_dataset.md)

## Acknowledgements

This code borrows heavily from [[pytorch-CycleGAN-and-pix2pix]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) [[Pytorch-UNet]](https://github.com/milesial/Pytorch-UNet) [[pix2pixHD]](https://github.com/NVIDIA/pix2pixHD) [[BiSeNet]](https://github.com/ooooverflow/BiSeNet) [[DFDNet]](https://github.com/csxmli2016/DFDNet) [[GFRNet_pytorch_new]](https://github.com/sonack/GFRNet_pytorch_new).

#### This codebase was also forked from [[DeepMosaics]](https://github.com/HypoX64/DeepMosaics) and optimized since I noticed it didn't use much of my CPU and GPU :) I plan on optimizing it further but this is a start.
