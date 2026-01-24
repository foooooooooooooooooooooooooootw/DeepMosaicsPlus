<div align="center">
  <img src="./imgs/logo_new.png" width="400"><br>
  
  # DeepMosaicsPlus
  
  **AI-powered mosaic removal for images and videos**
  
  [![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-ee4c2c.svg)](https://pytorch.org/)
  [![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](https://www.microsoft.com/windows)

</div>

---

## ✨ What's New in Plus

This is an optimized fork of the original DeepMosaics project with significant performance improvements:

- 🚀 **6x faster processing** - Optimized GPU utilization (1h → 10 mins on AMD 7800XT)
- 🎮 **AMD GPU support** - DirectML integration for AMD graphics cards
- ⚡ **Hardware acceleration** - DirectX 11 for AMD, CUDA for NVIDIA
- 🔧 **Auto-detection** - Automatically detects and uses available GPU
- 🐛 **Bug fixes planned** - Auto dependency instllation, async I/O, batch processing

> **Note**: The original project was revolutionary for the time - a proof of concept of sorts - and as such does not use CPU/GPU resources efficiently. This fork addresses that with substantial performance gains. Most programs also only use CUDA which locks all AMD GPU users out. Only the mosaic removal part was optimized, adding mosaics was not touched by me.

---

## 🎯 Features

- **Image & Video Processing** - Remove mosaics from both static images and video files
- **Multiple Network Architectures** - UNet, ResNet, HD models for different use cases
- **Flexible Detection** - Adjustable sensitivity for mosaic detection
- **Traditional Fallback** - Non-AI method available for edge cases
- **GUI & CLI** - Choose between graphical interface or command-line operation
- **Customizable Output** - Control resolution, FPS, and processing parameters

---

## 🖼️ Results

![Demo](./imgs/hand.gif)

### Comparison with DeepCreamPy

|                Mosaic Image                |            DeepCreamPy             |                DeepMosaicsPlus                |
| :----------------------------------------: | :--------------------------------: | :-------------------------------------------: |
| ![image](./imgs/example/face_a_mosaic.jpg) | ![image](./imgs/example/a_dcp.png) |     ![image](./imgs/example/face_a_clean.jpg) |
| ![image](./imgs/example/face_b_mosaic.jpg) | ![image](./imgs/example/b_dcp.png) |     ![image](./imgs/example/face_b_clean.jpg) |

---

## 🚀 Quick Start

### Option 1: GUI (Easiest)

Double-click `deepmosaicplusui.py` or run:
```bash
python deepmosaicplusui.py
```

### Option 2: Command Line

```bash
python deepmosaic.py --media_path "weenus.mkv" --model_path "clean_youknow_video.pth"
```

---

## 📋 Requirements

### System Requirements
- **OS**: Windows (DirectML requirement)
- **Python**: 3.6 or higher
- **GPU**: Any AMD or NVIDIA GPU (CPU fallback available)

### Dependencies

#### Core Libraries
- [PyTorch 1.0+](https://pytorch.org/)
- [FFmpeg 3.4.6](http://ffmpeg.org/)
- opencv-python
- torchvision

#### GPU Support
- **AMD GPUs**: torch_directml (auto-detects DirectML devices)
- **NVIDIA GPUs**: CUDA toolkit

#### GUI (Optional)
- customtkinter

---

## ⚙️ Installation

### Option A: Download Release (Easiest)

Download the latest pre-packaged release with models included:

**[📦 Download DeepMosaicsPlus.zip](https://github.com/foooooooooooooooooooooooooootw/DeepMosaicsPlus/releases/latest/download/DeepMosaicsPlus.zip)**

Extract the zip file and you're ready to go! Skip to the [Usage](#-usage) section.

### Option B: Clone from Source

### 1. Clone the Repository
```bash
git clone https://github.com/foooooooooooooooooooooooooootw/DeepMosaicsPlus.git
cd DeepMosaicsPlus
```

### 2. Install Dependencies
```bash
pip install torch torchvision opencv-python customtkinter
```

**For AMD GPUs:**
```bash
pip install torch-directml
```

**For NVIDIA GPUs:** Install CUDA toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)

### 3. Download Pre-trained Models

Download models and place them in `./pretrained_models/`

- [[Google Drive]](https://drive.google.com/open?id=1LTERcN33McoiztYEwBxMuRjjgxh4DEPs) 
- [[百度云, 提取码 1x0a]](https://pan.baidu.com/s/10rN3U3zd5TmfGpO_PEShqQ)
- [[Model Information]](./docs/pre-trained_models_introduction.md)

**Required files**: Place `mosaic_position.pth` and `clean_youknow_video.pth` in the same directory as `deepmosaic.py`

---

## 🎮 Usage

### Basic Command
```bash
python deepmosaic.py --media_path "input.mp4" --model_path "clean_youknow_video.pth"
```

### Process Specific Time Range
```bash
python deepmosaic.py --media_path "video.mp4" \
                      --start_time "00:01:30" \
                      --last_time "00:05:00" \
                      --model_path "clean_youknow_video.pth"
```

### Adjust Detection Sensitivity
```bash
python deepmosaic.py --media_path "image.jpg" \
                      --mask_threshold 30 \
                      --model_path "clean_youknow_video.pth"
```
*Lower threshold = more sensitive detection*

---

## 📖 Command-Line Arguments

### Essential Arguments

| Argument         | Type  | Default                   | Description                                    |
| ---------------- | ----- | ------------------------- | ---------------------------------------------- |
| `--media_path`   | `str` | `'./imgs/ruoruo.jpg'`     | Path to input image or video                   |
| `--model_path`   | `str` | `'./clean_youknow_video.pth'` | Path to the pretrained model               |
| `--result_dir`   | `str` | `'./result'`              | Output directory                               |
| `--gpu_id`       | `str` | `'0'`                     | GPU ID (auto-detected with DirectML)           |

### Video Processing

| Argument              | Type  | Default       | Description                              |
| --------------------- | ----- | ------------- | ---------------------------------------- |
| `-ss`, `--start_time` | `str` | `'00:00:00'`  | Start time (HH:MM:SS)                    |
| `-t`, `--last_time`   | `str` | `'00:00:00'`  | Duration (`00:00:00` = entire video)     |
| `--fps`               | `int` | `0`           | Output FPS (`0` = keep original)         |

### Detection & Processing

| Argument                       | Type   | Default  | Description                                         |
| ------------------------------ | ------ | -------- | --------------------------------------------------- |
| `--mask_threshold`             | `int`  | `48`     | Mosaic detection sensitivity (0-255, lower = more)  |
| `--netG`                       | `str`  | `'auto'` | Network: `unet_128`, `unet_256`, `resnet_9blocks`, `HD`, `video` |
| `--mosaic_position_model_path` | `str`  | `'auto'` | Model for detecting mosaic positions                |
| `--all_mosaic_area`            | `flag` | `True`   | Find all mosaic regions (not just largest)         |
| `--no_feather`                 | `flag` | `False`  | Disable edge feathering (faster, lower quality)     |

### Advanced Options

| Argument           | Type   | Default | Description                                  |
| ------------------ | ------ | ------- | -------------------------------------------- |
| `--traditional`    | `flag` | `False` | Use non-AI traditional method                |
| `--no_preview`     | `flag` | `False` | Disable preview window (for servers)         |
| `--output_size`    | `int`  | `0`     | Output resolution (`0` = original)           |
| `--temp_dir`       | `str`  | `'./tmp'` | Temporary files directory                  |
| `--debug`          | `flag` | `False` | Enable debug mode                            |

For complete documentation, see [[options_introduction.md]](./docs/options_introduction.md)

---

## 🔧 Technical Details

### GPU Acceleration

**DirectML (AMD GPUs)**
- Automatically detects DirectML-capable devices
- Hardcoded to use GPU with CPU fallback
- Uses DirectX 11 for hardware video acceleration
- No need for `--gpu_id` argument

**CUDA (NVIDIA GPUs)**
- Uses CUDA for hardware acceleration
- Specify GPU with `--gpu_id` (e.g., `--gpu_id 0` by default)

### Performance Optimization

This fork improves on the original by:
- Better GPU utilization (original barely used GPU resources)
- Optimized FFmpeg arguments for hardware acceleration
- DirectML integration for broader GPU support

---

## 🏗️ Training Your Own Models

Want to train on custom datasets? Check out the [training guide](./docs/training_with_your_own_dataset.md).

---

## 📝 Roadmap

- [ ] Auto detect dependencies and install if missing
- [ ] Add batch processing for multiple files
- [ ] GPU workload saturation improvements

---

## 🙏 Acknowledgements

This project builds upon excellent work from:

- [DeepMosaics](https://github.com/HypoX64/DeepMosaics) - Original codebase
- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
- [BiSeNet](https://github.com/ooooverflow/BiSeNet)
- [DFDNet](https://github.com/csxmli2016/DFDNet)
- [GFRNet_pytorch_new](https://github.com/sonack/GFRNet_pytorch_new)

**Special thanks** to the original DeepMosaics project. This fork exists to optimize performance and add AMD GPU support. 🚀

---

## 💬 Support

Found a bug or have a feature request? [Open an issue](https://github.com/foooooooooooooooooooooooooootw/DeepMosaicsPlus/issues)

---

<div align="center">
  
**If this project helped you, consider giving it a ⭐!**

</div>