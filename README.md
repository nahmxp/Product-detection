# Product Detection — Quickstart

This repository contains scripts and helpers for training and running YOLO-based object detection/segmentation models (Ultralytics). The repo intentionally excludes large data and model files from version control.

## Quick setup (recommended)
1. Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source ./venv/bin/activate
```

2. Upgrade pip and install runtime dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
# Install torch separately to match your CUDA (example for CUDA 12.8):
# pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Files & what they do
- `app.py` — Example training script that uses `ultralytics.YOLO` to train a segmentation model (calls `model.train`). Adjust `data`, `model` and hyperparameters inside the file.
- `train.py` — Another training wrapper/config used for running experiments with the Ultralytics trainer. Use this for standard training runs in the repo.
- `tf_test.py` — Run TFLite inference on a single image. Prints raw detections and filtered detections (by `--min-conf`) and saves annotated outputs. Use `--show-all` to save an image with all raw detections.
- `camera.py` — Streamlit realtime camera UI that loads a YOLO model and runs live inference. Start it with `streamlit run camera.py`.
- `camera.py` (Streamlit UI) and `app.py` (training) are separate: use `app.py` for training, `camera.py` for demo/inference.
- `tf_cnv.py`, `tf_tes_confi.py`, `tf_test.py` — utility scripts for converting/testing TensorFlow/TFLite outputs (see top of each file for usage).
- `oggy.py`, `oggy_v1.py`, `oggy`* — experimental or dataset-specific utilities (project-specific). Inspect each file header to see exact behavior.

## Common commands
- Run Streamlit camera UI (after loading model path in UI or editing `camera.py` default):

```bash
streamlit run camera.py
```

- Run TFLite test (default uses file paths in `tf_test.py`):

```bash
python tf_test.py --image "./WhatsApp Image 2025-11-12 at 13.43.02.jpeg" --min-conf 0.75 --show-all
```

- Train a model (example — review `app.py` or `train.py` for settings):

```bash
python app.py
# or
python train.py
```

## Notes & tips
- Large directories such as `dataset/`, `Model/`, and `runs/` are excluded from git by `.gitignore`.
- Install `torch` with the correct CUDA build for your GPU; the `requirements.txt` does not pin a specific `torch` wheel to avoid mismatches.
- If using VS Code: select the interpreter from `./venv/bin/python` (Command Palette → `Python: Select Interpreter`).
- If you plan to push to a Git repo later, run `git init`, add files, and make your initial commit. The `.gitignore` is already in place to avoid committing large artifacts.

## Repro & troubleshooting
- If conversion to TFLite changed confidences significantly, compare outputs between the PyTorch `.pt` model and TFLite using `tf_test.py` (use `--show-all` to see all raw detections).
- If GPU is available but not used, verify `torch` and CUDA versions and that the CUDA drivers are installed.

If you want, I can:
- Create a `scripts/` wrapper to run the most common commands.
- Initialize a new git repo, make the first commit, and create a remote (your choice).
