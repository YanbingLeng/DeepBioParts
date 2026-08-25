"""Output serialization for iterative library-design runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from .domain import DesignConfig, DesignResult


def write_design_outputs(
    output_dir: Path,
    result: DesignResult,
    config: DesignConfig,
    run_metadata: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "designed_library.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sequence",
                "predicted_activity",
                "activity_quantile",
                "accepted_round",
            ),
        )
        writer.writeheader()
        for part in result.library:
            writer.writerow(
                {
                    "sequence": part.sequence,
                    "predicted_activity": part.activity,
                    "activity_quantile": part.activity_quantile,
                    "accepted_round": part.accepted_round,
                }
            )

    with (output_dir / "round_history.json").open("w", encoding="utf-8") as handle:
        json.dump([record.to_dict() for record in result.history], handle, indent=2)

    summary = {
        "complete": result.complete,
        "library_size": len(result.library),
        "retired_sequences": len(result.retired_sequences),
        "coverage_loss": result.coverage_loss,
        "coverage_radius": result.coverage_radius,
        "cumulative_predictions": result.cumulative_predictions,
        "design_config": asdict(config),
        **run_metadata,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
