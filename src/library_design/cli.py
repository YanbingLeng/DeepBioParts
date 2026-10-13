"""Command-line orchestration for iterative DeepBioParts library design."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

from .constraints import NrpSequenceConstraints
from .domain import DesignConfig
from .model_adapters import FeedbackAwareDiffusionGenerator, CheckpointActivityPredictor
from .reporting import write_design_outputs
from .workflow import design_library


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_LENGTHS = {"promoter": 40, "rbs": 15, "terminator": 50}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Iteratively coordinate a trained DDPM and activity "
            "predictor to design a continuous-coverage library without repeats "
            "longer than Lmax within or between sequences."
        )
    )
    parser.add_argument(
        "--biopart",
        choices=("promoter", "rbs", "terminator"),
        required=True,
    )
    parser.add_argument("--generator-dir", type=Path, default=None)
    parser.add_argument("--predictor-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-library-size", type=int, default=100)
    parser.add_argument("--lmax", type=int, default=10)
    parser.add_argument("--generation-batch", type=int, default=1_000)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--max-swaps-per-round", type=int, default=10)
    parser.add_argument("--min-coverage-gain", type=float, default=0.0)
    parser.add_argument("--max-stagnant-rounds", type=int, default=3)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--oversample-factor", type=float, default=2.0)
    parser.add_argument("--conflict-probe-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _resolve_paths(args: argparse.Namespace) -> None:
    from src.config import load_config

    if args.generator_dir is None:
        checkpoint_root = PROJECT_ROOT / "diffusion_checkpoints"
        canonical_dir = checkpoint_root / f"{args.biopart}_diffusion"
        candidates = sorted(
            (
                path
                for path in checkpoint_root.glob(f"{args.biopart}*")
                if (path / "checkpoints" / "best.pth").is_file()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        args.generator_dir = candidates[0] if candidates else canonical_dir
    if args.predictor_dir is None:
        config = load_config(args.biopart)
        args.predictor_dir = PROJECT_ROOT / config["default_predictor"]
    if args.output_dir is None:
        args.output_dir = (
            PROJECT_ROOT / "results" / "iterative_library_design" / args.biopart
        )


def _validate_runtime_args(args: argparse.Namespace) -> None:
    import torch

    if args.oversample_factor < 1.0:
        raise ValueError("oversample_factor must be at least 1")
    if not 0.0 <= args.conflict_probe_fraction <= 1.0:
        raise ValueError("conflict_probe_fraction must be between 0 and 1")
    if not args.generator_dir.exists():
        raise FileNotFoundError(f"generator directory not found: {args.generator_dir}")
    if not args.predictor_dir.exists():
        raise FileNotFoundError(f"predictor directory not found: {args.predictor_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")


def _seed_runtime(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args: argparse.Namespace) -> int:
    import torch

    _resolve_paths(args)
    _validate_runtime_args(args)
    _seed_runtime(args.seed)

    device = torch.device(args.device)
    generator = FeedbackAwareDiffusionGenerator(
        checkpoint_dir=args.generator_dir,
        sequence_length=SEQUENCE_LENGTHS[args.biopart],
        device=device,
        sample_batch_size=args.sample_batch_size,
        temperature=args.temperature,
        oversample_factor=args.oversample_factor,
        conflict_probe_fraction=args.conflict_probe_fraction,
    )
    predictor = CheckpointActivityPredictor(
        model_dir=args.predictor_dir,
        biopart=args.biopart,
        device=device,
        batch_size=args.predict_batch_size,
    )
    config = DesignConfig(
        sequence_length=generator.sequence_length,
        target_library_size=args.target_library_size,
        lmax=args.lmax,
        generation_batch=args.generation_batch,
        max_rounds=args.max_rounds,
        max_swaps_per_round=args.max_swaps_per_round,
        min_coverage_gain=args.min_coverage_gain,
        max_stagnant_rounds=args.max_stagnant_rounds,
    )
    config.validate()

    with NrpSequenceConstraints(config.lmax) as constraints:
        result = design_library(generator, predictor, config, constraints)

    write_design_outputs(
        output_dir=args.output_dir,
        result=result,
        config=config,
        run_metadata={
            "generator_dir": str(args.generator_dir),
            "predictor_dir": str(args.predictor_dir),
            "nrpcalc_source": str(PROJECT_ROOT / "src" / "nrpcalc"),
            "device": args.device,
            "seed": args.seed,
        },
    )

    for record in result.history:
        print(
            f"round={record.round_index} size={record.library_size} "
            f"accepted={record.accepted} swapped={record.swapped} "
            f"coverage_loss={record.coverage_loss}"
        )
    print(f"complete={result.complete} output={args.output_dir}")
    return 0 if result.complete else 2


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))
