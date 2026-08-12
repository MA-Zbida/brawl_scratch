# Environment Setup

Windows, Python 3.12, CUDA 12.x. Verified against an RTX 3050 Ti Laptop (4 GB VRAM,
driver 592.82).

## Already done on this machine

- **Python 3.12.10** at `C:\Python312` (NuGet portable distribution — full layout,
  so `venv`, `pip`, `ssl` and `sqlite3` all work).
- **Virtualenv** at `C:\venvs\brawl312`, empty except pip/setuptools/wheel.

> The `py` launcher still advertises a 3.12 at
> `arc-prize-2026-arc-agi-3\tools\Python312\` — that path no longer exists, so
> `py -3.12` will fail. Use the explicit paths below.

### Why the venv is not in the project folder

The project lives under `C:\Users\mazbi\OneDrive\...`. A torch + TensorRT venv is
roughly 8 GB across tens of thousands of files. OneDrive's Files-On-Demand
dehydrates infrequently-touched files, and a dehydrated `.dll` or `.pyd` surfaces
as `DLL load failed while importing ...` at run time — slow to diagnose, trivial
to avoid. A venv also cannot be relocated after creation, so this is worth getting
right the first time.

## Activate

PowerShell:

```powershell
C:\venvs\brawl312\Scripts\Activate.ps1
```

If that is blocked by execution policy (`... cannot be loaded because running
scripts is disabled`), allow it for the current shell only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
C:\venvs\brawl312\Scripts\Activate.ps1
```

cmd.exe:

```bat
C:\venvs\brawl312\Scripts\activate.bat
```

Or skip activation entirely and call the interpreter directly — this is what the
VS Code interpreter picker should point at, and it is the most reliable option for
scripts and scheduled runs:

```powershell
C:\venvs\brawl312\Scripts\python.exe -m train.train_curriculum --phase movement_fluency
```

Confirm you are in the right environment:

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
# C:\venvs\brawl312\Scripts\python.exe
# 3.12.10 ...
```

## Install

**Order matters.** torch must come from the PyTorch CUDA index in its own command.

```powershell
# 1. torch, CUDA 12.6
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2. everything else, from PyPI
python -m pip install -r requirements-llc.txt
```

Installing both at once lets the resolver satisfy `torch` from PyPI, which gives
you the **CPU-only** build. Nothing errors: every import works, `torch.cuda.is_available()`
is `False`, and the detector quietly runs on CPU at a few frames per second.

### If pip hangs during resolution

pip's HTTP cache on this machine is around **48 GB**, on a disk with ~47 GB free.
A cache that size makes dependency resolution crawl, and low free space makes
large wheel unpacking fail late. Clear it:

```powershell
python -m pip cache purge
```

Then retry. Add `-v` to watch actual download progress rather than guessing:

```powershell
python -m pip install -v torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Verify

```powershell
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import tensorrt; print('tensorrt', tensorrt.__version__)"
python -c "import ultralytics, cv2, dxcam, pydirectinput, keyboard, gymnasium, stable_baselines3; print('full stack OK')"
python -m pytest tests -q
```

Expected: `cuda 12.6`, `available True`, `NVIDIA GeForce RTX 3050 Ti Laptop GPU`,
and 108 passing tests.

Then the project's own preflight:

```powershell
python tools/llc_preflight.py --device cuda
```

## Building the TensorRT engine

TensorRT engines are **not portable** across GPU architecture, driver version, or
TensorRT version. An engine exported on Kaggle will not load here.

Export `.pt` from wherever you trained, then build the engine **on this machine**:

```powershell
python -c "from ultralytics import YOLO; YOLO('feature_extractor/yolo/best.pt').export(format='engine', half=True, imgsz=(384,640), device=0)"
```

`half=True` matters on 4 GB of VRAM. Keep `imgsz` consistent with the inference
size in [`feature_extractor/yolo/extract.py`](../feature_extractor/yolo/extract.py);
training and inference resolution must agree.

[`Extract`](../feature_extractor/yolo/extract.py) reads class names from the model's
own metadata, so `['character', 'indicator_self', 'weapon']` from `data.yaml` is
carried through automatically — there is no hardcoded list to keep in sync.

## Notes for this GPU

- **4 GB VRAM.** Keep PPO on `--device cpu` (the default). The policy is a small
  MLP where GPU launch overhead exceeds the compute, and it leaves the whole GPU
  for the detector.
- Do not run periodic evaluation with `--eval-include-previous` on later phases:
  it constructs one full environment per evaluated phase, each with its own
  capture instance and TensorRT engine.
