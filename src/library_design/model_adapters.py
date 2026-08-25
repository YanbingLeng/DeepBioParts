"""Adapters for the trained DeepBioParts generation and prediction models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from .constraints import has_internal_repeat, nrp_kmers
from .domain import GenerationFeedback


class FeedbackAwareDiffusionGenerator:
    """Rank DDPM samples using internal and current-library repeat constraints."""

    def __init__(
        self,
        checkpoint_dir: Path,
        sequence_length: int,
        device,
        sample_batch_size: int,
        temperature: float,
        oversample_factor: float,
        conflict_probe_fraction: float,
    ) -> None:
        from src.evaluation.generative_evaluator import (
            load_direct_diffusion_model as load_diffusion_model,
            p_sample,
        )

        if temperature != 1.0:
            raise ValueError("DDPM sampling currently requires temperature=1.0")

        loaded = load_diffusion_model(
            str(checkpoint_dir),
            sequence_length,
            device,
        )
        self.model, self.noise_schedule, self.model_config = loaded
        self.device = device
        self.sample_batch_size = sample_batch_size
        self.oversample_factor = oversample_factor
        self.conflict_probe_fraction = conflict_probe_fraction
        self.sequence_length = sequence_length
        self.num_timesteps = int(self.model_config.get("num_timesteps", 1_000))
        self._p_sample = p_sample

    def _sample(self, n_sequences: int) -> list[str]:
        import torch

        nucleotide_mapping = ("T", "C", "G", "A")
        sequences: list[str] = []
        for start in range(0, n_sequences, self.sample_batch_size):
            batch_size = min(self.sample_batch_size, n_sequences - start)
            samples = self._p_sample(
                denoiser=self.model,
                noise_schedule=self.noise_schedule,
                shape=(batch_size, 4, self.sequence_length),
                device=self.device,
                num_inference_steps=self.num_timesteps,
            )
            indices = torch.argmax(samples, dim=1).detach().cpu().numpy()
            sequences.extend(
                "".join(nucleotide_mapping[index] for index in row)
                for row in indices
            )
        return sequences

    def generate(
        self,
        n_sequences: int,
        feedback: GenerationFeedback,
    ) -> Sequence[str]:
        n_raw = max(n_sequences, math.ceil(n_sequences * self.oversample_factor))
        raw = self._sample(n_raw)

        unique = list(
            dict.fromkeys(sequence.upper().replace("U", "T") for sequence in raw)
        )

        def conflict_count(sequence: str) -> int:
            internal_conflict = int(
                has_internal_repeat(sequence, feedback.kmer_size)
            )
            if not feedback.forbidden_kmers:
                return internal_conflict
            return internal_conflict + len(
                nrp_kmers(sequence, feedback.kmer_size)
                & feedback.forbidden_kmers
            )

        ranked = sorted(unique, key=conflict_count)
        compatible = [sequence for sequence in ranked if conflict_count(sequence) == 0]
        blocked = [sequence for sequence in ranked if conflict_count(sequence) > 0]

        probe_count = math.ceil(n_sequences * self.conflict_probe_fraction)
        proposals = compatible[:n_sequences]
        if len(proposals) < n_sequences:
            probe_count = max(probe_count, n_sequences - len(proposals))
        proposals.extend(blocked[:probe_count])
        return proposals


class CheckpointActivityPredictor:
    """Load a regression ensemble once and reuse it across design rounds."""

    def __init__(
        self,
        model_dir: Path,
        biopart: str,
        device,
        batch_size: int,
    ) -> None:
        import numpy as np
        import torch

        from src.config import load_config
        from src.evaluation.predictor_inference import (
            detect_task_type,
            find_fold_checkpoints,
            load_activity_transform,
            load_label_transform,
            load_predictor,
        )

        self.np = np
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.checkpoints = find_fold_checkpoints(model_dir)
        self.label_transform = load_label_transform(model_dir)
        self.activity_transform = load_activity_transform(biopart)
        self.activity_config = load_config(biopart).get("activity_transform", {})

        first = torch.load(
            self.checkpoints[0],
            map_location="cpu",
            weights_only=False,
        )
        first_state = first.get("model_state_dict", first)
        task_type, _ = detect_task_type(first_state, self.checkpoints[0])
        if task_type != "regression":
            raise ValueError("library design requires a regression activity predictor")

        self.is_evo = any(key.startswith("evo.") for key in first_state)
        if self.is_evo:
            self.models = [
                load_predictor(
                    self.checkpoints[0],
                    biopart,
                    str(device),
                    task_type="regression",
                    num_classes=1,
                )
            ]
        else:
            self.models = [
                load_predictor(
                    checkpoint,
                    biopart,
                    str(device),
                    task_type="regression",
                    num_classes=1,
                )
                for checkpoint in self.checkpoints
            ]

    def _predict_model(self, model, encoded):
        predictions = []
        with self.torch.inference_mode():
            for start in range(0, encoded.shape[0], self.batch_size):
                batch = encoded[start : start + self.batch_size].to(self.device)
                output = model(batch)
                predictions.append(output.detach().cpu().numpy().reshape(-1))
        return self.np.concatenate(predictions)

    def _encode(self, sequences: Sequence[str]):
        if self.is_evo:
            from src.evo.lora_finetune_evo import dna_sequence_to_tokens

            seq_len = len(sequences[0])
            tokens = [dna_sequence_to_tokens(sequence, seq_len) for sequence in sequences]
            return self.torch.tensor(tokens, dtype=self.torch.long)

        from src.utils.data import seq2onehot

        onehot = seq2onehot(list(sequences))
        return self.torch.tensor(onehot, dtype=self.torch.float32).permute(0, 2, 1)

    def predict(self, sequences: Sequence[str]) -> Sequence[float]:
        if not sequences:
            return []
        encoded = self._encode(sequences)
        fold_predictions = []

        if self.is_evo:
            model = self.models[0]
            for fold_index, checkpoint_path in enumerate(self.checkpoints):
                if len(self.checkpoints) > 1 or fold_index > 0:
                    checkpoint = self.torch.load(
                        checkpoint_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    state = checkpoint.get("model_state_dict", checkpoint)
                    model.load_state_dict(state, strict=False)
                    model.eval()
                fold_predictions.append(self._predict_model(model, encoded))
        else:
            for model in self.models:
                fold_predictions.append(self._predict_model(model, encoded))

        mean_prediction = self.np.stack(fold_predictions, axis=0).mean(axis=0)

        inverse_log = False
        if self.label_transform is not None:
            inverse_log = self.label_transform.get("transform") == "log10"
        elif self.activity_config.get("inverse_log10", False):
            inverse_log = True
        if inverse_log:
            shift = 1.0
            if self.label_transform is not None:
                shift = float(self.label_transform.get("shift", 1.0))
            mean_prediction = self.np.power(10.0, mean_prediction) - shift

        if self.activity_transform is not None:
            mean_prediction = (
                float(self.activity_transform.get("slope", 1.0)) * mean_prediction
                + float(self.activity_transform.get("intercept", 0.0))
            )
        return mean_prediction.astype(float).tolist()
