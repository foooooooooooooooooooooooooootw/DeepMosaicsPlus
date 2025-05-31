<div align="center">
  <img src="./imgs/logo.png" width="250"><br><br>
</div>

# DeepMosaicsPlus

**English | [中文](./README_CN.md)**<br>
You can use it to automatically remove the mosaics in images and videos, or add mosaics to them.<br>

### Examples

![image](./imgs/hand.gif)

|                origin                |             auto add mosaic              |             auto clean mosaic              |
| :----------------------------------: | :--------------------------------------: | :----------------------------------------: |
|  ![image](./imgs/example/lena.jpg)   |  ![image](./imgs/example/lena_add.jpg)   |  ![image](./imgs/example/lena_clean.jpg)   |
| ![image](./imgs/example/youknow.png) | ![image](./imgs/example/youknow_add.png) | ![image](./imgs/example/youknow_clean.png) |

- Compared with [DeepCreamPy](https://github.com/deeppomf/DeepCreamPy)

|                mosaic image                |            DeepCreamPy             |                   ours                    |
| :----------------------------------------: | :--------------------------------: | :---------------------------------------: |
| ![image](./imgs/example/face_a_mosaic.jpg) | ![image](./imgs/example/a_dcp.png) | ![image](./imgs/example/face_a_clean.jpg) |
| ![image](./imgs/example/face_b_mosaic.jpg) | ![image](./imgs/example/b_dcp.png) | ![image](./imgs/example/face_b_clean.jpg) |

- Style Transfer

|              origin              |               to Van Gogh                |                   to winter                    |
| :------------------------------: | :--------------------------------------: | :--------------------------------------------: |
| ![image](./imgs/example/SZU.jpg) | ![image](./imgs/example/SZU_vangogh.jpg) | ![image](./imgs/example/SZU_summer2winter.jpg) |

An interesting example:[Ricardo Milos to cat](https://www.bilibili.com/video/BV1Q7411W7n6)

### TODO

- Fix bugs relating to non-english letters and spaces in filenames
- Increase performance further by parallelizing/async I/O, saturating GPU with batches etc. 
- Create GUI

## Run DeepMosaicsPlus

You can run DeepMosaicsPlus from source. Executable might come in the future.<br>

### Try it on web

You can simply try to remove the mosaic on the **face** at this [website](http://118.89.27.46:5000/).<br>

### Pre-built binary package

No GUI yet, execution is commandline only - run the command under "[Example](####Example)"

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
python deepmosaic.py --media_path "path/to/video.mp4" --model_path "/path/to/model.pth"
```
#### More Parameters

If you want to test other images or videos, please refer to this file.<br>
[[options_introduction.md]](./docs/options_introduction.md) <br>

## Training With Your Own Dataset

If you want to train with your own dataset, please refer to [training_with_your_own_dataset.md](./docs/training_with_your_own_dataset.md)

## Acknowledgements

This code borrows heavily from [[pytorch-CycleGAN-and-pix2pix]](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) [[Pytorch-UNet]](https://github.com/milesial/Pytorch-UNet) [[pix2pixHD]](https://github.com/NVIDIA/pix2pixHD) [[BiSeNet]](https://github.com/ooooverflow/BiSeNet) [[DFDNet]](https://github.com/csxmli2016/DFDNet) [[GFRNet_pytorch_new]](https://github.com/sonack/GFRNet_pytorch_new).

#### This codebase was also forked from [[DeepMosaics]](https://github.com/HypoX64/DeepMosaics) and optimized since I noticed it didn't use much of my CPU and GPU :) I plan on optimizing it further but this is a start.
