"""
Run SD 2.1 brain encoding benchmarks across:
  - 4 timesteps: {50, 200, 500, 999}
  - 11 layers: VAE encoder (3) + U-Net encoder (4) + bottleneck (1) + U-Net decoder (4)
  - 19 NSD ROIs

Usage:
    python scripts/run_sd_benchmarks.py --timestep 200 --layer unet.up_blocks.3
    python scripts/run_sd_benchmarks.py --all
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks import BENCHMARK_REGISTRY

NSD_BENCHMARKS = [
    "NSDV1Shared", "NSDV1dShared", "NSDV1vShared",
    "NSDV2Shared", "NSDV2dShared", "NSDV2vShared",
    "NSDV3Shared", "NSDV3dShared", "NSDV3vShared",
    "NSDV4Shared",
    "NSDLateralShared", "NSDVentralShared", "NSDParietalShared",
    "NSDMidLateralShared", "NSDMidVentralShared", "NSDMidParietalShared",
    "NSDHighLateralShared", "NSDHighVentralShared", "NSDHighParietalShared",
]

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
    # U-Net decoder (post-skip)
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
        model_id,
        layer,
        batch_size=batch_size,
        debug=debug,
    )
    pipeline.add_metric("ridge")
    results = pipeline.run()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run SD 2.1 brain encoding benchmarks.")
    parser.add_argument("--timestep", type=int, choices=TIMESTEPS)
    parser.add_argument("--layer", type=str, choices=SD_LAYERS)
    parser.add_argument("--benchmark", type=str, choices=NSD_BENCHMARKS)
    parser.add_argument("--all", action="store_true",
                        help="Full sweep: all timesteps x layers x ROIs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.all:
        for t in TIMESTEPS:
            model_id = f"sd21_t{t}"
            for layer in SD_LAYERS:
                for bench in NSD_BENCHMARKS:
                    print(f"\n{'=' * 60}")
                    print(f"t={t} | layer={layer} | ROI={bench}")
                    print(f"{'=' * 60}")
                    try:
                        results = run_single(
                            model_id, layer, bench,
                            args.batch_size, args.debug,
                        )
                        print(f"  -> {results}")
                    except Exception as e:
                        print(f"  -> FAILED: {e}")
    else:
        model_id = f"sd21_t{args.timestep or 200}"
        layer = args.layer or "unet.up_blocks.3"
        bench = args.benchmark or "NSDV1Shared"
        print(f"Running: model={model_id} | layer={layer} | ROI={bench}")
        results = run_single(
            model_id, layer, bench,
            args.batch_size, args.debug,
        )
        print(results)


if __name__ == "__main__":
    main()
