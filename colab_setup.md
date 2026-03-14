# Google Colab Setup for SD 2.1 BBScore Experiments

## Cell 1: Mount Google Drive & set paths

```python
from google.colab import drive
drive.mount('/content/drive')

import os

# Persistent storage on Drive
DRIVE_ROOT = "/content/drive/MyDrive/bbscore"
os.makedirs(DRIVE_ROOT, exist_ok=True)

# Data dir on Drive (survives session restarts)
DATA_DIR = f"{DRIVE_ROOT}/data"
os.makedirs(DATA_DIR, exist_ok=True)

# HuggingFace cache on Drive (avoid re-downloading SD weights ~5GB)
HF_CACHE = f"{DRIVE_ROOT}/hf_cache"
os.makedirs(HF_CACHE, exist_ok=True)

os.environ["SCIKIT_LEARN_DATA"] = DATA_DIR
os.environ["HF_HOME"] = HF_CACHE

print(f"Data dir:    {DATA_DIR}")
print(f"HF cache:    {HF_CACHE}")
```

## Cell 2: Clone repo (or pull if already cloned)

```python
REPO_DIR = "/content/bbscore_public"

if os.path.exists(REPO_DIR):
    !cd {REPO_DIR} && git pull
else:
    !git clone https://github.com/lgonzalez876/bbscore_public.git {REPO_DIR}

%cd {REPO_DIR}
```

## Cell 3: Install dependencies

```python
!pip install -q -r requirements.txt
```

## Cell 4: Verify GPU & imports

```python
import torch
print(f"PyTorch:  {torch.__version__}")
print(f"CUDA:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"VRAM:     {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

import diffusers
print(f"Diffusers: {diffusers.__version__}")
```

## Cell 5: Run Tier 1 validation (environment check, ~30s)

```python
!python validate.py --tier 1
```

## Cell 6: Run Tier 2 validation (model loading + inference, ~2-3 min)

```python
!python validate.py --tier 2
```

## Cell 7: SD-specific validation — load model and test hook extraction

```python
import sys
sys.path.insert(0, "/content/bbscore_public")

import torch
from models import get_model_instance
from extractor_wrapper import FeatureExtractor

# Load SD 2.1 at timestep 200
model_wrapper = get_model_instance("sd21_t200")
model = model_wrapper.model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create extractor with a single U-Net layer
extractor = FeatureExtractor(
    model=model,
    layer_name="unet.up_blocks.3",
    device=device,
    postprocess_fn=model_wrapper.postprocess_fn,
    batch_size=2,
    static=True,
)

# Test with random input: list of 2 HxWx3 uint8 numpy arrays
import numpy as np
dummy = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(2)]
preprocessed = model_wrapper.preprocess_fn(dummy)

# Quick forward pass
with torch.inference_mode():
    extractor.get_activations(preprocessed.to(device))

# Check captured features
for layer, feats in extractor.features.items():
    for i, f in enumerate(feats):
        print(f"  {layer}[{i}]: shape={f.shape}, dtype={f.dtype}, device={f.device}")
        print(f"    requires_grad={f.requires_grad}")

# Check postprocess output shape
feat = feats[0]
pooled = model_wrapper.postprocess_fn(feat)
print(f"\nRaw shape:    {feat.shape}")
print(f"Pooled shape: {pooled.shape}  (should be (B, C), NOT (B, C*H*W))")

# Memory check
if torch.cuda.is_available():
    print(f"\nGPU mem used:      {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU mem reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Cleanup
del extractor, model, model_wrapper, dummy, preprocessed
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nSD validation PASSED")
```

## Cell 8: Run Tier 3 validation (end-to-end with NSD data download, ~10-30 min first run)

```python
# This downloads NSD data to Google Drive (persistent).
# Subsequent sessions skip the download.
!python validate.py --tier 3
```

## Cell 9: Quick single-benchmark test with SD (smoke test before full sweep)

```python
!python run.py \
    --model sd21_t200 \
    --layer unet.up_blocks.3 \
    --benchmark NSDV1Shared \
    --metric ridge \
    --batch-size 2 \
    --debug
```

## Cell 10: Full SD benchmark sweep (resume-safe)

```python
import os

RESULTS_DIR = os.path.join(os.environ["SCIKIT_LEARN_DATA"], "results")
BATCH_SIZE = 2  # Use 1 for T4 (16GB), 2-4 for A100 (40GB)

TIMESTEPS = [50, 200, 500, 999]
SD_LAYERS = [
    "vae.encoder.down_blocks.0", "vae.encoder.down_blocks.1", "vae.encoder.down_blocks.2",
    "unet.down_blocks.0", "unet.down_blocks.1", "unet.down_blocks.2", "unet.down_blocks.3",
    "unet.mid_block",
    "unet.up_blocks.0", "unet.up_blocks.1", "unet.up_blocks.2", "unet.up_blocks.3",
]
NSD_BENCHMARKS = [
    "NSDV1Shared", "NSDV1dShared", "NSDV1vShared",
    "NSDV2Shared", "NSDV2dShared", "NSDV2vShared",
    "NSDV3Shared", "NSDV3dShared", "NSDV3vShared",
    "NSDV4Shared",
    "NSDLateralShared", "NSDVentralShared", "NSDParietalShared",
    "NSDMidLateralShared", "NSDMidVentralShared", "NSDMidParietalShared",
    "NSDHighLateralShared", "NSDHighVentralShared", "NSDHighParietalShared",
]

total = len(TIMESTEPS) * len(SD_LAYERS) * len(NSD_BENCHMARKS)
done, skipped = 0, 0

for t in TIMESTEPS:
    model_id = f"sd21_t{t}"
    for layer in SD_LAYERS:
        layer_fn = layer.replace(".", "_")
        for bench in NSD_BENCHMARKS:
            result_file = os.path.join(RESULTS_DIR, f"{model_id}_{layer}_{bench}.pkl")
            if os.path.exists(result_file):
                skipped += 1
                continue
            done += 1
            print(f"\n[{done + skipped}/{total}] t={t} | {layer} | {bench}")
            !python run.py \
                --model {model_id} \
                --layer {layer} \
                --benchmark {bench} \
                --metric ridge \
                --batch-size {BATCH_SIZE}

print(f"\nDone. Ran {done}, skipped {skipped} (already computed), total {total}.")
```

## Cell 11: Copy results to Drive (run before session ends)

```python
import shutil, glob

results_src = os.path.join(os.environ["SCIKIT_LEARN_DATA"], "results")
results_dst = os.path.join(DRIVE_ROOT, "results")

if os.path.exists(results_src):
    shutil.copytree(results_src, results_dst, dirs_exist_ok=True)
    n_files = len(glob.glob(f"{results_dst}/**/*.pkl", recursive=True))
    print(f"Copied {n_files} result files to Drive: {results_dst}")
else:
    print("No results directory found yet.")
```

---

## Notes
- **Session breaks**: Data and HF model weights are on Drive, so re-mounting + `pip install` is the only setup needed on restart. Cell 2 does `git pull` if repo already exists.
- **T4 (16GB) vs A100 (40GB)**: Use `--batch-size 1` on T4, `--batch-size 2-4` on A100.
- **Disk**: NSD data is ~30GB. Ensure Google Drive has space. HF cache for SD 2.1 is ~5GB.
- **Total storage needed**: ~40GB on Drive (data + model cache + results).
