# Copyright (c) 2024-2026 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
# ruff: noqa: T201
"""Unified benchmark CLI."""

from __future__ import annotations

import argparse
import warnings

from utils.config import (
    BATCH_FRAMEWORKS,
    MODEL_NAMES,
    MODEL_TO_DIR,
    MODEL_TO_IMGSIZES,
    ULTRALYTICS_MODELS,
)


def cmd_models(args: argparse.Namespace) -> None:
    """Run single-image model benchmarks."""
    from utils.runners import benchmark_trtutils_models, benchmark_ultralytics_models

    models = MODEL_NAMES if args.model == "all" else [args.model]

    for model in models:
        if model not in MODEL_TO_DIR:
            warnings.warn(f"Unknown model: {model}")
            continue

        image_sizes = MODEL_TO_IMGSIZES.get(model, [])
        if args.imgsz is not None:
            image_sizes = [args.imgsz] if args.imgsz in image_sizes else []
        if not image_sizes:
            warnings.warn(f"No image sizes configured for {model}")
            continue

        if args.trtutils:
            try:
                benchmark_trtutils_models(
                    args.device, model, image_sizes,
                    args.warmup, args.iterations, overwrite=args.overwrite,
                )
            except Exception as e:
                warnings.warn(f"Failed {model} with trtutils: {e}")

        if args.ultralytics:
            if model not in ULTRALYTICS_MODELS:
                warnings.warn(f"{model} is not a supported ultralytics model")
            else:
                try:
                    benchmark_ultralytics_models(
                        args.device, model, image_sizes,
                        args.warmup, args.iterations, overwrite=args.overwrite,
                    )
                except Exception as e:
                    warnings.warn(f"Failed {model} with ultralytics: {e}")


def cmd_batch(args: argparse.Namespace) -> None:
    """Run batch throughput benchmarks."""
    from utils.data import get_data
    from utils.runners import benchmark_trtutils_batch, benchmark_ultralytics_batch

    if args.nvtx:
        from trtutils import enable_nvtx
        enable_nvtx()

    if args.model not in MODEL_TO_DIR:
        err_msg = f"Unknown model: {args.model}. Supported: {list(MODEL_TO_DIR.keys())}"
        raise ValueError(err_msg)

    print(f"Batch benchmark: {args.model} @ {args.imgsz}x{args.imgsz}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}\n")

    if args.trtutils:
        benchmark_trtutils_batch(
            args.device, args.model, args.imgsz, args.batch_sizes,
            args.warmup, args.iterations, overwrite=args.overwrite,
        )

    if args.ultralytics:
        if args.model not in ULTRALYTICS_MODELS:
            warnings.warn(f"{args.model} is not a supported ultralytics model")
        else:
            benchmark_ultralytics_batch(
                args.device, args.model, args.imgsz, args.batch_sizes,
                args.warmup, args.iterations, overwrite=args.overwrite,
            )

    # Print summary
    data = get_data(args.device, "batch", BATCH_FRAMEWORKS)
    print("\n" + "=" * 70)
    print(f"Summary: {args.model} @ {args.imgsz}")
    print(
        f"{'Framework':<22} {'Batch':<8} {'Mean(ms)':<10} "
        f"{'CI95(ms)':<10} {'Throughput(img/s)':<18}",
    )
    print("=" * 70)
    for fw in BATCH_FRAMEWORKS:
        model_data = data.get(fw, {}).get(args.model, {})
        for bs in args.batch_sizes:
            entry = model_data.get(str(bs))
            if entry:
                print(
                    f"{fw:<22} {bs:<8} {entry['mean']:<10.2f} "
                    f"{entry.get('ci95', 0.0):<10.2f} "
                    f"{entry['throughput']:<18.1f}",
                )
    print("=" * 70)


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run optimization grid benchmarks."""
    from utils.runners import benchmark_optimizations

    benchmark_optimizations(
        args.device, args.model, args.imgsz,
        args.warmup, args.iterations,
    )


def cmd_sahi(args: argparse.Namespace) -> None:
    """Run SAHI comparison benchmarks."""
    from utils.runners import benchmark_sahi

    if args.model not in ULTRALYTICS_MODELS:
        warnings.warn(f"SAHI only supports ultralytics models, got: {args.model}")
        return

    benchmark_sahi(
        args.device, args.model,
        args.warmup, args.iterations, overwrite=args.overwrite,
    )


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate benchmark plots."""
    from utils.config import DATA_DIR

    skip_devices: set[str] = set()
    if args.skip_devices:
        skip_devices = {
            d.strip() for d in args.skip_devices.split(",") if d.strip()
        }

    if args.batch:
        from plotting.batch import plot_batch

        batch_dir = DATA_DIR / "batch"
        if not batch_dir.exists():
            print("Warning: data/batch/ not found, nothing to plot.")
            return
        devices = [
            f.stem for f in batch_dir.iterdir()
            if f.is_file() and f.suffix == ".json"
        ]
        if args.device:
            devices = [d for d in devices if d == args.device]
        for device in devices:
            if device not in skip_devices:
                try:
                    plot_batch(device, overwrite=args.overwrite)
                except Exception as e:
                    print(f"Warning: Failed batch plots for {device}: {e}")

    elif args.optimizations:
        from plotting.optimizations import plot_optimizations

        opt_dir = DATA_DIR / "optimizations"
        if not opt_dir.exists():
            print("Warning: data/optimizations/ not found, nothing to plot.")
            return
        devices = [
            f.stem for f in opt_dir.iterdir()
            if f.is_file() and f.suffix == ".json"
        ]
        if args.device:
            devices = [d for d in devices if d == args.device]
        for device in devices:
            if device not in skip_devices:
                try:
                    plot_optimizations(device, overwrite=args.overwrite)
                except Exception as e:
                    print(f"Warning: Failed optimization plot for {device}: {e}")

    else:
        from plotting.models import (
            load_all_device_data,
            load_model_info,
            plot_device,
            plot_pareto,
        )

        all_data = load_all_device_data()
        model_info = load_model_info()

        for name, data in all_data:
            if name in skip_devices or (args.device and name != args.device):
                continue
            try:
                if args.latency:
                    plot_device(name, data, overwrite=args.overwrite)
                if args.pareto:
                    plot_pareto(
                        name, data, model_info, args.framework,
                        overwrite=args.overwrite,
                    )
            except Exception as e:
                print(f"Warning: Failed plots for {name}: {e}")


def cmd_table(args: argparse.Namespace) -> None:
    """Generate RST tables."""
    from plotting.table import generate_tables

    skip: set[str] = set()
    if args.skip_devices:
        skip = {d.strip() for d in args.skip_devices.split(",") if d.strip()}
    generate_tables(skip_devices=skip)


def cmd_bootstrap(args: argparse.Namespace) -> None:
    """Download models upfront."""
    from utils.runners import bootstrap_models

    models = MODEL_NAMES if args.model == "all" else [args.model]
    bootstrap_models(models)


def main() -> None:
    """Run the benchmark CLI."""
    parser = argparse.ArgumentParser("Benchmark trtutils.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- models ---
    p = subparsers.add_parser("models", help="Single-image model benchmarks")
    p.add_argument("--device", required=True)
    p.add_argument("--model", default="yolov10n", help="Model name or 'all'")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--trtutils", action="store_true")
    p.add_argument("--ultralytics", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_models)

    # --- batch ---
    p = subparsers.add_parser("batch", help="Batch throughput benchmarks")
    p.add_argument("--device", required=True)
    p.add_argument("--model", default="yolov10n")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--trtutils", action="store_true")
    p.add_argument("--ultralytics", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--nvtx", action="store_true")
    p.set_defaults(func=cmd_batch)

    # --- optimize ---
    p = subparsers.add_parser("optimize", help="Optimization grid benchmarks")
    p.add_argument("--device", required=True)
    p.add_argument("--model", default="yolov10n")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iterations", type=int, default=200)
    p.set_defaults(func=cmd_optimize)

    # --- sahi ---
    p = subparsers.add_parser("sahi", help="SAHI comparison benchmarks")
    p.add_argument("--device", required=True)
    p.add_argument("--model", default="yolov10n")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_sahi)

    # --- plot ---
    p = subparsers.add_parser("plot", help="Generate plots")
    p.add_argument("--device", default=None)
    p.add_argument("--latency", action="store_true")
    p.add_argument("--pareto", action="store_true")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--optimizations", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-devices", type=str, default="")
    p.add_argument("--framework", default="trtutils(trt)")
    p.set_defaults(func=cmd_plot)

    # --- table ---
    p = subparsers.add_parser("table", help="Generate RST tables")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-devices", type=str, default="")
    p.set_defaults(func=cmd_table)

    # --- bootstrap ---
    p = subparsers.add_parser("bootstrap", help="Download models upfront")
    p.add_argument("--model", default="all", help="Model name or 'all'")
    p.set_defaults(func=cmd_bootstrap)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
