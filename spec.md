# Stable Diffusion 2.1 Brain Encoding — BBScore Implementation Spec

## Overview

Add Stable Diffusion 2.1 as a new model in the BBScore framework. Extract features from the VAE encoder, U-Net encoder blocks, U-Net decoder blocks (pre-skip, post-skip), and run ridge regression against all 19 NSD fMRI ROIs. Sweep across 4 diffusion timesteps.

**Repo**: `bbscore_public/` (local clone of `github.com/neuroailab/bbscore_public`)

---

## 1. Files to Create

```
models/
  stable_diffusion/
    __init__.py          # MODEL_REGISTRY entries
    sd_wrapper.py        # StableDiffusion class (BBScore model interface)
    sd_unet_wrapper.py   # nn.Module wrapper that encapsulates VAE+noise+U-Net forward
scripts/
  run_sd_benchmarks.py   # Orchestration script: loops over layers × timesteps × ROIs
```

## 2. Files to Modify

```
models/__init__.py       # Add "stable_diffusion" to MODEL_MODULES list
requirements.txt         # Add: diffusers, transformers, accelerate
```

---

## 3. Architecture & Key Engineering Challenge

### The Problem

BBScore's `FeatureExtractor.get_activations()` does:

```python
def get_activations(self, inputs):
    for l in self.layer_names:
        self.features[l] = []
    inputs = inputs.to(self.device)
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        return self.model(inputs)
```

It calls `self.model(inputs)` where `inputs` are preprocessed image tensors (B, 3, H, W). Forward hooks on named submodules capture intermediate activations.

But Stable Diffusion's U-Net doesn't take raw images. The actual inference path is:
1. VAE encoder: image → latent `z₀` (B, 4, 64, 64)
2. Add noise at timestep `t`: `z_t = √ᾱ_t · z₀ + √(1-ᾱ_t) · ε`
3. U-Net forward: `unet(z_t, t, encoder_hidden_states=null_text_emb)` → noise prediction

### The Solution

Create `SDUNetWrapper(nn.Module)` that wraps the entire pipeline. Its `forward(images)` method internally:
1. Runs the VAE encoder to get `z₀`
2. Adds noise at a configurable `self.timestep`
3. Runs the U-Net with null prompt conditioning
4. Returns the U-Net output (noise prediction)

This wrapper exposes the VAE and U-Net as named submodules, so `FeatureExtractor` can attach hooks via standard `model.named_modules()` paths.

---

## 4. Model Implementation Details

### 4.1 `sd_unet_wrapper.py` — The Core nn.Module

```python
class SDUNetWrapper(nn.Module):
    """
    Wraps SD 2.1 VAE + U-Net into a single nn.Module whose forward()
    accepts raw image tensors and runs the full encode → noise → denoise pipeline.

    Exposes submodules via standard naming so FeatureExtractor hooks work:
      - "vae.encoder.*"       → VAE encoder layers
      - "unet.down_blocks.*"  → U-Net encoder (downsampling) blocks
      - "unet.mid_block.*"    → U-Net bottleneck
      - "unet.up_blocks.*"    → U-Net decoder (upsampling) blocks
    """
    def __init__(self, timestep=200):
        super().__init__()
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16,
            safety_checker=None,
        )

        # Expose as named submodules (critical for hook registration)
        self.vae = pipe.vae
        self.unet = pipe.unet

        # Precompute null text embedding (77 tokens × 1024 dim for SD 2.1)
        # SD 2.1 uses OpenCLIP-ViT/H with 1024-dim embeddings
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder

        with torch.no_grad():
            null_tokens = tokenizer(
                [""], padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids
            self.register_buffer(
                "null_text_emb",
                text_encoder(null_tokens.to(text_encoder.device))[0]
            )

        # Free text encoder memory — not needed after computing null embedding
        del pipe.tokenizer, pipe.text_encoder
        del tokenizer, text_encoder

        # Noise scheduler for alpha values
        self.noise_scheduler = pipe.scheduler

        # Configurable timestep
        self.timestep = timestep

        # Freeze everything
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.eval()

    def forward(self, images):
        """
        Args:
            images: (B, 3, 512, 512) float tensor, normalized to [-1, 1]
        Returns:
            noise_pred: U-Net output (for the standard forward pass to complete;
                        actual features are captured by hooks)
        """
        # 1. VAE encode → latent z_0
        with torch.no_grad():
            latent_dist = self.vae.encode(images).latent_dist
            z_0 = latent_dist.mean  # deterministic, no sampling noise
            z_0 = z_0 * self.vae.config.scaling_factor  # 0.18215

        # 2. Add noise at timestep t
        noise = torch.randn_like(z_0)
        # Use a fixed seed per-batch for reproducibility across runs
        timesteps = torch.full(
            (z_0.shape[0],), self.timestep,
            device=z_0.device, dtype=torch.long
        )
        z_t = self.noise_scheduler.add_noise(z_0, noise, timesteps)

        # 3. U-Net forward with null conditioning
        null_emb = self.null_text_emb.expand(z_0.shape[0], -1, -1)
        noise_pred = self.unet(
            z_t, timesteps, encoder_hidden_states=null_emb
        ).sample

        return noise_pred
```

**Critical implementation notes:**

- Use `latent_dist.mean` (not `.sample()`) to avoid stochastic sampling noise polluting features. This gives deterministic latents for the same input image.
- Set a **fixed random seed** before generating noise `ε` so features are reproducible across runs. Specifically: call `torch.manual_seed(42)` before `torch.randn_like(z_0)` inside `forward()`. Or better: precompute noise per-image if feasible, but for simplicity a fixed seed per forward call is acceptable since BBScore processes images in deterministic dataloader order with `shuffle=False`.
- The `null_text_emb` is a buffer (moves with `.to(device)` automatically).
- VAE scaling factor is 0.18215 for SD 2.1 (access via `self.vae.config.scaling_factor`).

### 4.2 `sd_wrapper.py` — BBScore Model Interface

```python
class StableDiffusion:
    """BBScore model wrapper for Stable Diffusion 2.1."""

    def __init__(self):
        self.static = True  # image (not video) model
        self.model_mappings = {
            "SD21-T50":   50,
            "SD21-T200":  200,
            "SD21-T500":  500,
            "SD21-T999":  999,
        }

    def preprocess_fn(self, input_data, fps=None):
        """
        Preprocess to (B, 3, 512, 512), normalized to [-1, 1].
        SD 2.1 was trained on 512×512 images.
        """
        # Handle PIL, numpy, list inputs → produce (B, 3, 512, 512) tensor
        # Use torchvision v2 transforms:
        #   Resize(512, bicubic) → CenterCrop(512) → ToImage → ToDtype(float32) → Normalize(mean=0.5, std=0.5)
        # This maps [0,255] → [0,1] → [-1,1]
        ...

    def get_model(self, identifier):
        """Load SDUNetWrapper with the appropriate timestep."""
        timestep = self.model_mappings[identifier]
        self.model = SDUNetWrapper(timestep=timestep)
        return self.model

    def postprocess_fn(self, features_np):
        """Flatten spatial dims: (B, C, H, W) → (B, C*H*W)."""
        if features_np.ndim == 4:
            B = features_np.shape[0]
            return features_np.reshape(B, -1)
        elif features_np.ndim == 3:
            B = features_np.shape[0]
            return features_np.reshape(B, -1)
        return features_np
```

### 4.3 `__init__.py` — Registration

```python
from models import MODEL_REGISTRY
from .sd_wrapper import StableDiffusion

MODEL_REGISTRY["sd21_t50"]  = {"class": StableDiffusion, "model_id_mapping": "SD21-T50"}
MODEL_REGISTRY["sd21_t200"] = {"class": StableDiffusion, "model_id_mapping": "SD21-T200"}
MODEL_REGISTRY["sd21_t500"] = {"class": StableDiffusion, "model_id_mapping": "SD21-T500"}
MODEL_REGISTRY["sd21_t999"] = {"class": StableDiffusion, "model_id_mapping": "SD21-T999"}
```

---

## 5. Layer Names for Hook Registration

The layer names below are the actual `named_modules()` paths from the HuggingFace `diffusers` implementation of SD 2.1. Claude Code should verify these by running:

```python
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet")
for name, _ in unet.named_modules():
    print(name)
```

### 5.1 Expected Layer Structure

**VAE encoder** (operates on pixel-space, 512² → 64²):
- `vae.encoder.down_blocks.0` — 512 → 256
- `vae.encoder.down_blocks.1` — 256 → 128
- `vae.encoder.down_blocks.2` — 128 → 64
- `vae.encoder.mid_block` — 64×64 bottleneck

**U-Net encoder (down_blocks)** — processes noisy latent:
- `unet.down_blocks.0` — 64² → 32², channels: 320
- `unet.down_blocks.1` — 32² → 16², channels: 640
- `unet.down_blocks.2` — 16² → 8², channels: 1280
- `unet.down_blocks.3` — 8² (no downsampling), channels: 1280

**U-Net bottleneck:**
- `unet.mid_block` — 8×8, channels: 1280

**U-Net decoder (up_blocks)** — upsamples + skip connections:
- `unet.up_blocks.0` — 8², channels: 1280
- `unet.up_blocks.1` — 8² → 16², channels: 1280
- `unet.up_blocks.2` — 16² → 32², channels: 640
- `unet.up_blocks.3` — 32² → 64², channels: 320

### 5.2 Encoder vs Decoder vs Combined (Post-Skip) Features

This is the core experimental manipulation. The U-Net decoder's `up_blocks` receive skip connections from corresponding encoder `down_blocks`. Inside each `up_block`, the architecture is:

```
encoder_features (from skip) ──concat──► ResNet block → Attention → (optional upsample)
decoder_features (from below) ─┘
```

The skip concatenation happens at the **input** to each ResNet block within the up_block. To capture pre-skip (decoder-only) vs post-skip (combined), we need hooks at:

1. **Encoder-only**: Hook on `unet.down_blocks.{i}` output
2. **Decoder pre-skip (top-down only)**: This is trickier. The concatenation happens inside the up_block's resnets. To get the decoder signal before skip concat, hook on the **upsample** output of the previous up_block (or mid_block for the first decoder level). Specifically:
   - For `up_blocks.0`: the input comes from `mid_block` output → that IS the pre-skip decoder signal for this level
   - For `up_blocks.1`: hook on `unet.up_blocks.0` output (which is the pre-skip signal going into up_blocks.1 before its skip concat)
   - And so on...
3. **Post-skip (combined)**: Hook on `unet.up_blocks.{i}` output (after skip concat + processing)

**Recommended layer name scheme for the experiment:**

| Condition | Layer Name | What It Captures |
|-----------|-----------|-----------------|
| Encoder-only @ 32² | `unet.down_blocks.0` | Bottom-up features at 32² |
| Encoder-only @ 16² | `unet.down_blocks.1` | Bottom-up features at 16² |
| Encoder-only @ 8² | `unet.down_blocks.2` | Bottom-up features at 8² |
| Bottleneck @ 8² | `unet.mid_block` | Deepest representation |
| Combined @ 8² | `unet.up_blocks.0` | After skip from down_blocks.3 |
| Combined @ 16² | `unet.up_blocks.1` | After skip from down_blocks.2 |
| Combined @ 32² | `unet.up_blocks.2` | After skip from down_blocks.1 |
| Combined @ 64² | `unet.up_blocks.3` | After skip from down_blocks.0 |
| VAE encoder @ 256² | `vae.encoder.down_blocks.0` | Pixel-space early features |
| VAE encoder @ 128² | `vae.encoder.down_blocks.1` | Pixel-space mid features |
| VAE encoder @ 64² | `vae.encoder.down_blocks.2` | Pixel-space late features |

**IMPORTANT**: Claude Code must verify these layer names empirically by instantiating the model and printing `named_modules()`. The diffusers library may nest things differently (e.g., `up_blocks.0.resnets.0` vs `up_blocks.0`). Get the exact paths and ensure hooks fire.

### 5.3 Isolating Pre-Skip Decoder Features

Getting the decoder signal *before* skip concatenation requires hooking into the internals of each `up_block`. In diffusers, each `CrossAttnUpBlock2D` has this structure:

```
up_blocks.{i}.resnets.{j}    — ResNet blocks (receive concatenated input)
up_blocks.{i}.attentions.{j} — cross-attention blocks
up_blocks.{i}.upsamplers.0   — upsampler (if present)
```

The skip connection is concatenated to the hidden states *before* being passed to `resnets.0`. This means the **output** of `up_blocks.{i}` already includes skip information. To get a clean pre-skip signal, you would need to hook the hidden_states *before* the first resnet in each up_block.

**Pragmatic approach**: For the initial implementation, focus on:
- **Encoder features**: `unet.down_blocks.{0,1,2,3}`
- **Combined (post-skip) features**: `unet.up_blocks.{0,1,2,3}`
- **Bottleneck**: `unet.mid_block`
- **VAE features**: `vae.encoder.down_blocks.{0,1,2}`

The pre-skip isolation can be added as a follow-up by registering hooks on `unet.up_blocks.{i}.resnets.0` and intercepting the input (not output) of that module. The hook signature `hook_fn(module, input, output)` provides `input` which would be the concatenated tensor — but you'd need to split off the skip portion. This is non-trivial and should be Phase 2.

---

## 6. Running Benchmarks

### 6.1 `scripts/run_sd_benchmarks.py`

This script loops over the experimental grid and invokes BBScore's standard pipeline:

```python
"""
Run SD 2.1 brain encoding benchmarks across:
  - 4 timesteps: {50, 200, 500, 999}
  - 11+ layers: VAE encoder (3) + U-Net encoder (4) + bottleneck (1) + U-Net decoder (4)
  - 19 NSD ROIs

Usage:
    python scripts/run_sd_benchmarks.py --timestep 200 --layer unet.up_blocks.3
    python scripts/run_sd_benchmarks.py --all  # full sweep
"""

import argparse
from benchmarks import BENCHMARK_REGISTRY
from models import MODEL_REGISTRY

# All 19 NSD benchmarks
NSD_BENCHMARKS = [
    "NSDV1Shared", "NSDV1dShared", "NSDV1vShared",
    "NSDV2Shared", "NSDV2dShared", "NSDV2vShared",
    "NSDV3Shared", "NSDV3dShared", "NSDV3vShared",
    "NSDV4Shared",
    "NSDLateralShared", "NSDVentralShared", "NSDParietalShared",
    "NSDMidLateralShared", "NSDMidVentralShared", "NSDMidParietalShared",
    "NSDHighLateralShared", "NSDHighVentralShared", "NSDHighParietalShared",
]

# Layers to extract (verify these names empirically!)
SD_LAYERS = [
    # VAE encoder
    "vae.encoder.down_blocks.0",
    "vae.encoder.down_blocks.1",
    "vae.encoder.down_blocks.2",
    # U-Net encoder
    "unet.down_blocks.0",
    "unet.down_blocks.1",
    "unet.down_blocks.2",
    "unet.down_blocks.3",
    # Bottleneck
    "unet.mid_block",
    # U-Net decoder (combined / post-skip)
    "unet.up_blocks.0",
    "unet.up_blocks.1",
    "unet.up_blocks.2",
    "unet.up_blocks.3",
]

TIMESTEPS = [50, 200, 500, 999]

def run_single(model_id, layer, benchmark_id, batch_size=2, debug=False):
    """Run a single (model, layer, benchmark) combo."""
    benchmark_class = BENCHMARK_REGISTRY[benchmark_id]
    pipeline = benchmark_class(
        model_identifier=model_id,
        layer_name=layer,
        batch_size=batch_size,
        debug=debug,
    )
    pipeline.add_metric("ridge")
    results = pipeline.run()
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep", type=int, choices=TIMESTEPS)
    parser.add_argument("--layer", type=str)
    parser.add_argument("--benchmark", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.all:
        for t in TIMESTEPS:
            model_id = f"sd21_t{t}"
            for layer in SD_LAYERS:
                for bench in NSD_BENCHMARKS:
                    print(f"\n{'='*60}")
                    print(f"t={t} | layer={layer} | ROI={bench}")
                    print(f"{'='*60}")
                    try:
                        results = run_single(model_id, layer, bench, args.batch_size, args.debug)
                        print(f"  → {results}")
                    except Exception as e:
                        print(f"  → FAILED: {e}")
    else:
        model_id = f"sd21_t{args.timestep or 200}"
        layer = args.layer or "unet.up_blocks.3"
        bench = args.benchmark or "NSDV1Shared"
        results = run_single(model_id, layer, bench, args.batch_size, args.debug)
        print(results)
```

### 6.2 Batch Size & Memory

SD 2.1 U-Net at fp16 uses ~3.5 GB VRAM. VAE uses ~200 MB. With 512×512 images:
- **batch_size=2** is safe for a 24 GB GPU (A10/L4/3090)
- **batch_size=4** may work on 40 GB+ (A100)
- Use `--batch-size 1` if OOM

The features themselves can be large (e.g., `up_blocks.3` outputs 320×64×64 = 1.3M floats per image). Use BBScore's built-in random projection (`--random-projection sparse`) if ridge regression runs OOM on the feature matrix.

### 6.3 CLI Examples

```bash
# Smoke test: single layer, single ROI, single timestep
python run.py --model sd21_t200 --layer unet.up_blocks.3 --benchmark NSDV1Shared --metric ridge --debug

# Full sweep for one timestep
python scripts/run_sd_benchmarks.py --timestep 200 --all

# Compare encoder vs decoder at one resolution
python run.py --model sd21_t200 --layer unet.down_blocks.1 unet.up_blocks.2 --benchmark NSDV4Shared --metric ridge --aggregation-mode none
```

---

## 7. Preprocessing Spec

SD 2.1 expects 512×512 images normalized to [-1, 1]. NSD stimuli are 425×425 COCO crops.

```python
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode

transform = T.Compose([
    T.ToImage(),
    T.Resize(512, interpolation=InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(512),
    T.ToDtype(torch.float32, scale=True),          # [0,255] → [0,1]
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1,1]
])
```

Handle the same input types as other BBScore models (PIL Image, numpy array, file path, list of PIL Images). Follow the pattern in `models/dinov2/dinov2.py` for input normalization.

---

## 8. Noise Reproducibility

**Critical**: Features must be deterministic for the same image at the same timestep across runs. Two sources of randomness:

1. **VAE encoding**: Use `latent_dist.mean` (not `.sample()`) → deterministic
2. **Noise ε for z_t**: Must use a fixed seed. Set `torch.manual_seed(42)` before each `torch.randn_like(z_0)` call in `forward()`.

This ensures that given the same input image and timestep, the noisy latent `z_t` and therefore all downstream features are identical.

---

## 9. Dependencies

Add to `requirements.txt`:
```
diffusers>=0.25.0
transformers>=4.36.0
accelerate>=0.25.0
```

The model weights (~5 GB) will be auto-downloaded by HuggingFace on first use to `~/.cache/huggingface/hub/`. On RunPod, set `HF_HOME=/workspace/hf_cache` to avoid filling root disk.

---

## 10. Validation Checklist

Before running full experiments, verify:

1. **Model loads**: `python -c "from models import get_model_instance; m = get_model_instance('sd21_t200')"`
2. **Layer names resolve**: Instantiate `SDUNetWrapper`, call `dict(model.named_modules())`, and confirm every layer in `SD_LAYERS` exists.
3. **Hooks fire**: Run a single forward pass with hooks on `unet.up_blocks.3`, verify `extractor.features` is non-empty.
4. **Features are deterministic**: Run the same image twice at the same timestep, assert features are identical (bitwise for fp16).
5. **Smoke test**: `python run.py --model sd21_t200 --layer unet.up_blocks.3 --benchmark NSDV1Shared --metric ridge --debug`
6. **Feature shapes are reasonable**: Print shapes at each hook point. Expected:
   - `unet.down_blocks.0`: (B, 320, 32, 32)
   - `unet.down_blocks.1`: (B, 640, 16, 16)
   - `unet.down_blocks.2`: (B, 1280, 8, 8)
   - `unet.mid_block`: (B, 1280, 8, 8)
   - `unet.up_blocks.3`: (B, 320, 64, 64)

---

## 11. Edge Cases & Gotchas

1. **FeatureExtractor checks `model.named_modules()`** at init (line 87-91 of `extractor_wrapper.py`). If a layer name doesn't match, it raises `ValueError`. The wrapper must expose VAE and U-Net as proper submodules (not just attributes stored in `self.__dict__`) — using `self.vae = ...` and `self.unet = ...` inside `nn.Module.__init__` handles this correctly.

2. **`_process_sequence_features` expects tensors or tuples-of-tensors** (line 223-253). SD U-Net blocks may return `UNet2DConditionOutput` (a dataclass). The hook captures the output — if it's not a raw tensor, the hook should extract `.sample` from it. Handle this in the hook function:

```python
def hook_fn_factory(layer_id):
    def hook_fn(module, input, output):
        if hasattr(output, 'sample'):
            self.features[layer_id].append(output.sample)
        elif isinstance(output, tuple):
            self.features[layer_id].append(output[0])
        else:
            self.features[layer_id].append(output)
    return hook_fn
```

**This is a modification to `extractor_wrapper.py` line 114-117.** The current hook blindly appends `output`, which will break for diffusers model outputs. Either modify the extractor OR override hooking in the SDUNetWrapper.

**Recommended approach**: Override by patching in `StableDiffusion.get_model()` — or simpler, modify the global `hook_fn_factory` in `extractor_wrapper.py` to handle non-tensor outputs. The latter is cleaner since it also future-proofs for other HuggingFace models:

```python
# In extractor_wrapper.py, replace hook_fn_factory:
def hook_fn_factory(layer_id):
    def hook_fn(module, input, output):
        # Handle HuggingFace model outputs (BaseModelOutput, etc.)
        if hasattr(output, 'sample'):
            output = output.sample
        elif hasattr(output, 'last_hidden_state'):
            output = output.last_hidden_state
        elif isinstance(output, tuple):
            output = output[0]
        self.features[layer_id].append(output)
    return hook_fn
```

3. **VAE encoder `down_blocks` output tuples**: The VAE's `DownEncoderBlock2D` returns a tensor directly. But verify empirically — if it returns a tuple, the hook must handle it.

4. **U-Net `down_blocks` return `(hidden_states, res_hidden_states_tuple)`**: The encoder down_blocks return a tuple where index 0 is the output hidden states and index 1 is a tuple of residual hidden states (for skip connections). The hook must extract index 0.

5. **postprocess_fn flattening**: SD features are 4D (B, C, H, W). The postprocess must flatten to (B, C*H*W) for ridge regression. But high-res layers like `up_blocks.3` produce 320×64×64 = 1,310,720 features per image, which may be too large for ridge. Options:
   - Use random projection (`sparse`, target_dim=4096)
   - Spatially average-pool before flattening
   - Use BBScore's built-in `downsample_factor`

   **Recommendation**: Implement spatial average pooling in `postprocess_fn` to reduce spatial dims to a fixed size (e.g., 8×8) before flattening. This is what many brain encoding papers do. Make it configurable.

6. **Module type for hooks**: `FeatureExtractor` (line 122-128) checks if the target is a `ModuleList` and hooks the last element. Some diffusers blocks may be `ModuleList`. Verify behavior.

---

## 12. Implementation Order

1. **Create `models/stable_diffusion/sd_unet_wrapper.py`** — the nn.Module wrapper
2. **Create `models/stable_diffusion/sd_wrapper.py`** — the BBScore interface class
3. **Create `models/stable_diffusion/__init__.py`** — registry entries
4. **Add `"stable_diffusion"` to `models/__init__.py` MODEL_MODULES**
5. **Modify `extractor_wrapper.py` hook_fn_factory** to handle non-tensor outputs
6. **Verify layer names** by instantiating and printing named_modules
7. **Smoke test** with `--debug` flag on a single ROI
8. **Create `scripts/run_sd_benchmarks.py`** — the sweep orchestrator
9. **Run full sweep** across timesteps × layers × ROIs
10. **Collect results** from pickle files in `$SCIKIT_LEARN_DATA/results/`

---

## 13. Results Collection

BBScore saves results as pickle files at:
```
$SCIKIT_LEARN_DATA/results/{model_id}_{layer_name}_{BenchmarkName}.pkl
```

Each contains `{"metrics": {"ridge": score, ...}, "ceiling": float}`.

The analysis script (separate from this spec) will:
1. Load all pkl files matching `sd21_t*`
2. Build a 3D tensor: `(layer, timestep, ROI)` → ridge score
3. Generate heatmaps
4. Compare encoder vs decoder vs combined at matched resolutions
5. Compare against ResNet/DINOv2 baselines (already in BBScore results)