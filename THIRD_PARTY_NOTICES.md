# Third-Party Notices

This repository includes vendored third-party software under their respective
licenses. The original copyright notices and license terms are preserved in
place alongside the vendored code.

## `src/evo` — Evo DNA foundation model package

- Upstream: <https://github.com/evo-design/evo> (Together AI / Laboratory of
  Evolutionary Design; `evo-model` 0.4 on PyPI).
- License: **Apache License 2.0** — see [`src/evo/LICENSE`](src/evo/LICENSE).
  The upstream distribution carries no NOTICE file.
- Scope: a partial vendored copy of the `evo` Python package
  (`src/evo/evo/`), used for local checkpoint loading, LoRA fine-tuning and
  integration with the DeepBioParts predictors.
- Modifications (per Apache-2.0 §4(b), also flagged in file headers):
  - `src/evo/evo/models.py` — added a `local_model_path` constructor option
    and `load_local_checkpoint` support so the model can be loaded from a
    local directory instead of the Hugging Face hub.
  - `src/evo/evo/generation.py` — added a robust cached-generation fallback:
    a test forward pass detects inference-cache incompatibilities of
    LoRA-adapted models and falls back to uncached generation.
  - All other vendored files are unmodified from `evo-model` 0.4.

## `src/nrpcalc` — NRP Calculator

- Upstream: <https://github.com/ayaanhossain/nrpcalc> by Ayaan Hossain
  (Salis lab, Pennsylvania State University).
- License: **MIT License** — see [`src/nrpcalc/LICENSE`](src/nrpcalc/LICENSE).
- Scope: the `nrpcalc` package (`src/nrpcalc/nrpcalc/`), unmodified, used as
  the embedded backend enforcing the pairwise non-repetitive sequence
  constraint (Lmax) during library design.

All other code in this repository is original DeepBioParts code released
under the repository's MIT license.
