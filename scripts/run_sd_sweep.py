"""
Optimized SD 2.1 sweep with feature caching.

Features are extracted once per (timestep, layer) and cached to disk.
Brain data is loaded once per ROI and reused across all (timestep, layer) combos.

Usage:
    python scripts/run_sd_sweep.py                    # 64-combo subset
    python scripts/run_sd_sweep.py --full              # all 836 combos
    python scripts/run_sd_sweep.py --sanity-check      # 2 combos to verify
"""

import argparse
import datetime
import gc
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.datasets import get_data_home

from extractor_wrapper import FeatureExtractor
from metrics import METRICS
from models import get_model_class_and_id
from data.NSDShared import NSDStimulusSet
from data.utils import custom_collate

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

TIMESTEPS = [50, 200, 500, 999]

ALL_LAYERS = [
    "vae.encoder.down_blocks.0",
    "vae.encoder.down_blocks.1",
    "vae.encoder.down_blocks.2",
    "unet.down_blocks.0",
    "unet.down_blocks.1",
    "unet.down_blocks.2",
    "unet.down_blocks.3",
    "unet.mid_block",
    "unet.up_blocks.0",
    "unet.up_blocks.1",
    "unet.up_blocks.2",
    "unet.up_blocks.3",
]

SUBSET_LAYERS = [
    "unet.down_blocks.1",
    "unet.mid_block",
    "unet.up_blocks.1",
    "unet.up_blocks.3",
]

ALL_BENCHMARKS = [
    "NSDV1Shared", "NSDV1dShared", "NSDV1vShared",
    "NSDV2Shared", "NSDV2dShared", "NSDV2vShared",
    "NSDV3Shared", "NSDV3dShared", "NSDV3vShared",
    "NSDV4Shared",
    "NSDLateralShared", "NSDVentralShared", "NSDParietalShared",
    "NSDMidLateralShared", "NSDMidVentralShared", "NSDMidParietalShared",
    "NSDHighLateralShared", "NSDHighVentralShared", "NSDHighParietalShared",
]

SUBSET_BENCHMARKS = [
    "NSDV1Shared",
    "NSDV4Shared",
    "NSDVentralShared",
    "NSDLateralShared",
]

# Benchmark name -> assembly class (lazy import to avoid loading all at once)
_ASSEMBLY_MAP = None


def _get_assembly_map():
    global _ASSEMBLY_MAP
    if _ASSEMBLY_MAP is None:
        from data.NSDShared import (
            NSDAssemblyV1, NSDAssemblyV1d, NSDAssemblyV1v,
            NSDAssemblyV2, NSDAssemblyV2d, NSDAssemblyV2v,
            NSDAssemblyV3, NSDAssemblyV3d, NSDAssemblyV3v,
            NSDAssemblyV4,
            NSDAssemblyLateral, NSDAssemblyVentral, NSDAssemblyParietal,
            NSDAssemblyMidLateral, NSDAssemblyMidVentral, NSDAssemblyMidParietal,
            NSDAssemblyHighLateral, NSDAssemblyHighVentral, NSDAssemblyHighParietal,
        )
        _ASSEMBLY_MAP = {
            "NSDV1Shared": NSDAssemblyV1,
            "NSDV1dShared": NSDAssemblyV1d,
            "NSDV1vShared": NSDAssemblyV1v,
            "NSDV2Shared": NSDAssemblyV2,
            "NSDV2dShared": NSDAssemblyV2d,
            "NSDV2vShared": NSDAssemblyV2v,
            "NSDV3Shared": NSDAssemblyV3,
            "NSDV3dShared": NSDAssemblyV3d,
            "NSDV3vShared": NSDAssemblyV3v,
            "NSDV4Shared": NSDAssemblyV4,
            "NSDLateralShared": NSDAssemblyLateral,
            "NSDVentralShared": NSDAssemblyVentral,
            "NSDParietalShared": NSDAssemblyParietal,
            "NSDMidLateralShared": NSDAssemblyMidLateral,
            "NSDMidVentralShared": NSDAssemblyMidVentral,
            "NSDMidParietalShared": NSDAssemblyMidParietal,
            "NSDHighLateralShared": NSDAssemblyHighLateral,
            "NSDHighVentralShared": NSDAssemblyHighVentral,
            "NSDHighParietalShared": NSDAssemblyHighParietal,
        }
    return _ASSEMBLY_MAP


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def extract_or_load_features(model_id, layer, cache_dir, stimulus, batch_size=1):
    """Extract features for (model_id, layer) or load from cache.

    Cache key: {cache_dir}/{model_id}_{layer}.npz
    Returns: np.ndarray of shape (N, D)
    """
    safe_layer = layer.replace(".", "_")
    cache_file = os.path.join(cache_dir, f"{model_id}_{safe_layer}.npz")

    if os.path.exists(cache_file):
        print(f"  [CACHE HIT] Loading features from {cache_file}")
        data = np.load(cache_file)
        return data["features"]

    print(f"  [CACHE MISS] Extracting features for {model_id} / {layer}")
    t0 = time.time()

    # Instantiate model
    model_class, model_id_mapping = get_model_class_and_id(model_id)
    model_instance = model_class()
    model = model_instance.get_model(model_id_mapping)

    # Create extractor
    extractor = FeatureExtractor(
        model, [layer],
        postprocess_fn=model_instance.postprocess_fn,
        batch_size=batch_size,
        num_workers=0,
        static=model_instance.static,
        aggregation_mode="none",
    )

    # Extract
    features_dict, _ = extractor.extract_features(stimulus)

    # Get the array for this layer
    features = features_dict[layer]
    print(f"  Features shape: {features.shape}, dtype: {features.dtype}")

    # Save cache (convert to float16 to save space)
    features_f16 = features.astype(np.float16)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_file, features=features_f16)
    print(f"  Saved cache: {cache_file} ({os.path.getsize(cache_file) / 1024:.0f} KB)")

    # Cleanup model
    del extractor, model, model_instance
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Extraction took {elapsed:.1f}s")

    return features_f16


def load_brain_data(benchmark_id):
    """Load brain data for a given benchmark/ROI.

    Returns: (target_data, ceiling)
    """
    assembly_map = _get_assembly_map()
    assembly_class = assembly_map[benchmark_id]
    assembly = assembly_class()
    target_data, ceiling = assembly.get_assembly()
    print(f"  Brain data: shape={target_data.shape}, dtype={target_data.dtype}, "
          f"size={target_data.nbytes / 2**20:.1f} MB")
    return target_data, ceiling


def run_ridge(features, target, ceiling):
    """Run ridge regression metric.

    Returns: results dict with metric outputs + timestamp.
    """
    metric_class = METRICS["ridge"]
    metric = metric_class(ceiling=ceiling)
    results = metric.compute(features, target)
    results["timestamp"] = datetime.datetime.utcnow().isoformat()
    return results


def result_path(model_id, layer, benchmark_id, results_dir):
    """Build result file path (same format as BBS.py)."""
    return os.path.join(
        results_dir,
        f"{model_id}_{layer}_{benchmark_id}.pkl"
    )


def result_exists(model_id, layer, benchmark_id, results_dir):
    """Check if a result file already exists for this combo."""
    return os.path.exists(result_path(model_id, layer, benchmark_id, results_dir))


def save_result(results, ceiling, model_id, layer, benchmark_id, results_dir):
    """Save results as pkl, appending to existing file if present."""
    fpath = result_path(model_id, layer, benchmark_id, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    if os.path.exists(fpath):
        try:
            with open(fpath, "rb") as f:
                prev = pickle.load(f)
            prev_metrics = prev.get("metrics", [])
            if isinstance(prev_metrics, dict):
                prev_metrics = [prev_metrics]
            prev_metrics.append(results)
            merged = {"metrics": prev_metrics, "ceiling": ceiling}
        except Exception:
            merged = {"metrics": results, "ceiling": ceiling}
    else:
        merged = {"metrics": results, "ceiling": ceiling}

    with open(fpath, "wb") as f:
        pickle.dump(merged, f)
    print(f"  Saved: {fpath}")
    return fpath


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(timesteps, layers, benchmarks, batch_size=1, sanity_check=False):
    """Run the optimized sweep.

    Loop: ROI -> timestep -> layer (features cached across ROIs).
    """
    data_home = get_data_home()
    cache_dir = os.path.join(data_home, "sd_feature_cache")
    results_base = os.environ.get("RESULTS_PATH", data_home)
    results_dir = os.path.join(results_base, "results")
    os.makedirs(results_dir, exist_ok=True)

    total_combos = len(benchmarks) * len(timesteps) * len(layers)
    print(f"\n{'=' * 70}")
    print(f"SD 2.1 Sweep: {len(timesteps)} timesteps x {len(layers)} layers x "
          f"{len(benchmarks)} ROIs = {total_combos} combos")
    print(f"Cache dir: {cache_dir}")
    print(f"Results dir: {results_dir}")
    print(f"{'=' * 70}\n")

    # Prepare stimulus dataset once (uses SD preprocess)
    model_class, _ = get_model_class_and_id(f"sd21_t{timesteps[0]}")
    model_instance = model_class()
    stimulus = NSDStimulusSet(preprocess=model_instance.preprocess_fn)
    del model_instance
    gc.collect()

    combo_idx = 0
    sweep_start = time.time()
    first_combo_done = False

    for bench_idx, benchmark_id in enumerate(benchmarks):
        print(f"\n{'=' * 70}")
        print(f"[ROI {bench_idx + 1}/{len(benchmarks)}] Loading brain data: {benchmark_id}")
        print(f"{'=' * 70}")
        t0 = time.time()
        target_data, ceiling = load_brain_data(benchmark_id)
        brain_load_time = time.time() - t0
        print(f"  Brain data loaded in {brain_load_time:.1f}s")

        for ts in timesteps:
            model_id = f"sd21_t{ts}"

            for layer in layers:
                combo_idx += 1
                tag = f"[{combo_idx}/{total_combos}]"

                # Resume support: skip existing results
                if result_exists(model_id, layer, benchmark_id, results_dir):
                    print(f"\n{tag} SKIP (exists): {model_id} / {layer} / {benchmark_id}")
                    continue

                print(f"\n{tag} {model_id} / {layer} / {benchmark_id}")
                combo_start = time.time()

                # 1. Features (cached across ROIs)
                features = extract_or_load_features(
                    model_id, layer, cache_dir, stimulus, batch_size
                )

                # 2. Ridge regression
                print(f"  Running ridge CV...")
                t_ridge = time.time()
                results = run_ridge(features, target_data, ceiling)
                ridge_time = time.time() - t_ridge

                # 3. Save
                fpath = save_result(
                    results, ceiling, model_id, layer, benchmark_id, results_dir
                )

                combo_time = time.time() - combo_start
                elapsed_total = time.time() - sweep_start

                # Print summary
                ridge_results = results.get("ridge", results)
                if isinstance(ridge_results, dict):
                    pearson = ridge_results.get("final_pearson", "N/A")
                    r2 = ridge_results.get("final_r2", "N/A")
                else:
                    pearson, r2 = "N/A", "N/A"

                print(f"  Result: pearson={pearson}, r2={r2}")
                print(f"  Ridge: {ridge_time:.1f}s | Combo: {combo_time:.1f}s | "
                      f"Elapsed: {format_time(elapsed_total)}")

                # ETA
                combos_done = combo_idx  # includes skipped
                combos_remaining = total_combos - combos_done
                if combos_done > 0:
                    avg_per_combo = elapsed_total / combos_done
                    eta = avg_per_combo * combos_remaining
                    print(f"  ETA: ~{format_time(eta)} remaining")

                # Sanity check on first combo
                if not first_combo_done:
                    first_combo_done = True
                    print(f"\n  --- First combo sanity check ---")
                    print(f"  Feature shape: {features.shape}")
                    print(f"  Target shape: {target_data.shape}")
                    print(f"  Ceiling shape: {ceiling.shape}")

                    # Verify pkl roundtrip
                    with open(fpath, "rb") as f:
                        loaded = pickle.load(f)
                    print(f"  PKL roundtrip OK: keys={list(loaded.keys())}")
                    if isinstance(loaded["metrics"], dict):
                        print(f"  Metrics keys: {list(loaded['metrics'].keys())}")

                    if sanity_check and combo_idx >= 2:
                        print(f"\n  Sanity check complete after {combo_idx} combos.")
                        return

        # Free brain data between ROIs
        del target_data, ceiling
        gc.collect()
        print(f"\n  Freed brain data for {benchmark_id}")

    total_time = time.time() - sweep_start
    print(f"\n{'=' * 70}")
    print(f"Sweep complete: {total_combos} combos in {format_time(total_time)}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimized SD 2.1 sweep with feature caching.")
    parser.add_argument("--full", action="store_true",
                        help="Full sweep: all 836 combos")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run just 2 combos to verify everything works")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for feature extraction (default: 1)")
    args = parser.parse_args()

    if args.full:
        run_sweep(TIMESTEPS, ALL_LAYERS, ALL_BENCHMARKS,
                  batch_size=args.batch_size)
    elif args.sanity_check:
        # 2 combos: t200/unet.up_blocks.3 x V1 + V4
        run_sweep(
            timesteps=[200],
            layers=["unet.up_blocks.3"],
            benchmarks=["NSDV1Shared", "NSDV4Shared"],
            batch_size=args.batch_size,
            sanity_check=True,
        )
    else:
        # Default: 64-combo subset
        run_sweep(TIMESTEPS, SUBSET_LAYERS, SUBSET_BENCHMARKS,
                  batch_size=args.batch_size)


if __name__ == "__main__":
    main()
