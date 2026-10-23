# Audio & Speech Enhancement
<p align="center">
</p>


<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue" />
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" />
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows-lightgrey" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-11.8%20%7C%2012.1%20%7C%2012.4-76b900?logo=nvidia&logoColor=white" />
  <img alt="Status" src="https://img.shields.io/badge/status-stable-brightgreen" />
  <img alt="License" src="https://img.shields.io/badge/License-GPLv3-blue" />
</p>

Desktop application for enhancing audio quality using AI models. Load a file, pick a model, press process — the app upsamples, denoises or restores speech and saves the result in the format you choose.

---

<img width="1303" height="874" alt="bettervoice" src="https://github.com/user-attachments/assets/85dc8bb4-09f8-4d07-a02f-1d3aead4cb27" />

---

Audio before:

https://soundcloud.com/mubumbutu/test

Audio after (RE-USE):

https://soundcloud.com/mubumbutu/testnew

---

## Changelog

<details>
<summary><b>Version history</b> (click to expand)</summary>

<br>

### [1.0.0] - 2026-10-23
**Added**
- Initial public release version 1.0.0.

</details>

---

## What it does

Process and restore audio files using a range of AI-powered enhancement models. Import individual files or entire folders, including common audio formats as well as video files with automatic audio extraction. Optional vocal isolation via Demucs can separate speech or vocals before enhancement, while advanced stem-based workflows allow additional mixing and mastering adjustments. Multiple files can be processed in batch with real-time progress estimation, and built-in waveform players make it easy to compare original and enhanced versions side by side before exporting the final result in WAV, FLAC, MP3 or OGG format.

---

## Models

| Backend                     | Task                                             |
| --------------------------- | ------------------------------------------------ |
| **RE-USE** (NVIDIA)         | Speech enhancement + bandwidth extension         |
| **AudioSR**                 | Audio super-resolution                           |
| **UniverSR**                | Universal audio super-resolution (flow matching) |
| **DeepFilterNet** (v2 / v3) | Real-time noise suppression                      |
| **FlashSR**                 | Fast super-resolution (ONNX / PyTorch)           |
| **LavaSR**                  | Super-resolution with optional denoising         |
| **NovaSR**                  | Super-resolution (fp16 / fp32)                   |
| **ClearerVoice** (Alibaba)  | Super-resolution + speech enhancement            |
| **FlowHigh** (Resemble AI)  | Flow-based super-resolution                      |
| **VoiceFixer**              | End-to-end speech restoration                    |

---

## Requirements

- Python 3.12
- ffmpeg in `PATH` — required for non-WAV inputs and video files
- NVIDIA GPU with CUDA
- Windows

---

## Installation

Run `install.bat` and pick which models you want. Each model installs into its own isolated virtual environment so their dependencies don't conflict.

```
[1]  RE-USE Speech Enhancement
[2]  AudioSR Super-Resolution
[3]  DeepFilterNet Noise Suppression
[4]  FlashSR Super-Resolution
[5]  UniverSR Super-Resolution
[6]  NovaSR Super-Resolution
[7]  LavaSR Super-Resolution
[8]  FlowHigh Super-Resolution
[9]  ClearerVoice Studio
[10] VoiceFixer Speech Restoration
```

After each install the script asks whether you want to install another model.

---

## Running

Run `start.bat`. It lists all installed environments and launches the app in the one you select.

On first use, download the model weights using the **Download Model** button inside the app.

---

## License

[GNU General Public License v3.0](LICENSE)
