# Collision-Avoidance-Framework

A CARLA 0.9.15 data toolkit for transformer-based, multimodal collision prediction (targeting ~3 seconds before impact). It automates the data side so you can focus on modeling:

- Drive and record an ego Tesla Model 3 in dense traffic, logging RGB, LiDAR, IMU, and collision events into rolling shelve runs.
- Extract and label frames, marking collision windows and tracking which modalities are missing per frame.
- Preprocess modalities into normalized numpy arrays and build scenario-aware train/val/test windows.
- Tokenize each window into per-frame 384-d embeddings (image, LiDAR, IMU, availability, CLS) ready for a dynamic transformer.

The research target is a dynamic transformer with per-layer gating and a lightweight binary head that outputs the probability of a collision within ~3 seconds. This repository implements data collection, extraction, preprocessing, windowing, and tokenization. The transformer and prediction head are described in [Concept.pdf](Concept.pdf) and are not yet implemented here.

## Requirements

- OS: Linux
- CARLA: 0.9.15 with additional maps installed; set `CARLA_PATH` to the server binary (for example `~/Carla/CarlaUE4.sh`).
- Python: 3.10.x. `uv` is recommended for reproducible environments; `pip` works with `pyproject.toml`.
- GPU: NVIDIA GPU with CUDA 11.8+ recommended for tokenization. 8 GB VRAM is sufficient for defaults; CPU runs are possible but slower.
- System: ~30-50 GB free disk for raw runs plus derived artifacts; 16 GB RAM recommended for preprocessing and tokenization jobs.
- Python deps (installed via `uv run` or `pip install -e .`): carla, torch, torchvision, numpy, pillow, pygame, psutil, python-dotenv, tqdm, and the rest of the scientific stack in `pyproject.toml`.

## Installation

1. Install CARLA 0.9.15 and additional maps:

   ```bash
   mkdir -p ~/Carla && cd ~/Carla
   aria2c -x 16 -s 16 -k 1M https://tiny.carla.org/carla-0-9-15-linux
   tar -xzf CARLA_0.9.15.tar.gz
   cd ~/Carla/Import
   aria2c -x 16 -s 16 -k 1M https://tiny.carla.org/additional-maps-0-9-15-linux
   cd .. && ./ImportAssets.sh
   ```

2. Set the server path:

   ```bash
   export CARLA_PATH=$HOME/Carla/CarlaUE4.sh
   ```

3. Install `uv` (recommended) to auto-manage Python dependencies:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Note: `uv` honors `requires-python ==3.10.*` in `pyproject.toml` and will provision Python 3.10 automatically when running `uv run`.

4. Clone and configure the project:

   ```bash
   git clone https://github.com/MaherMuhtadi/Collision-Avoidance-Framework.git
   cd Collision-Avoidance-Framework
   echo "CARLA_PATH=$HOME/Carla/CarlaUE4.sh" > .env
   ```

5. Ensure `lidar_settings.json` exists (auto-written on first run). If you need defaults, use:

   ```json
   {
     "CHANNELS": "32",
     "ROT_FREQ": "10.0",
     "PPS": "56000",
     "FOV_UP": "10.0",
     "FOV_DOWN": "-30.0",
     "MAX_RANGE": "50.0"
   }
   ```

6. First run (auto-installs dependencies with uv):

   ```bash
   uv run src/main.py
   ```

   Prefer a manual install? Run `uv pip install -e .` (or `pip install -e .`) then `python src/main.py`.

7. (Optional) Reset working folders:

   ```bash
   python src/clearLogs.py
   ```

   This recreates SensorData, ExtractedData, PreprocessedData, Dataset, Tokens, and Results.

## Data pipeline

Each step assumes the previous one completed successfully.

1. **Collect sensor data** — starts CARLA server, writes rolling shelve files under `SensorData/run_<N>`, and captures LiDAR settings.

   ```bash
   python src/main.py
   ```

   - Runs up to 30 minutes or until collision/ESC.
   - Spawns NPC traffic and an ego Tesla Model 3; use arrow keys/space for manual control (or lane changes if Traffic Manager autopilot is enabled in code).

2. **Extract raw frames** — converts shelve runs to per-frame artifacts and labels the collision window.

   ```bash
   python src/extractData.py
   ```

   Outputs `ExtractedData/<run>/Image/*.png`, `ExtractedData/<run>/Lidar/*.npy`, `extracted_data.json`, and run summaries.

3. **Preprocess modalities** — normalizes RGB to ImageNet stats, rasterizes LiDAR to range and mask grids using `lidar_settings.json`, normalizes IMU, and pads missing modalities.

   ```bash
   python src/preprocessData.py
   ```

   Outputs `PreprocessedData/<run>/preprocessed_data.json` plus per-frame numpy arrays.

4. **Create frame windows and splits** — builds scenario-aware windows and train/val/test splits.

   ```bash
   python src/createFrameWindows.py
   ```

   Outputs `Dataset/train.jsonl`, `Dataset/val.jsonl`, `Dataset/test.jsonl`, and `Dataset/stats.json`.

5. **Tokenize windows** — embeds each window into fixed tokens (image, LiDAR, IMU, availability, CLS) with carry-forward imputation for missing modalities.

   ```bash
   python src/embeddingLayer.py
   ```

   Outputs `Tokens/<split>/*.pt`, `Tokens/<split>/manifest.jsonl`, and `Tokens/summary.json`.
