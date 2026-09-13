# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-09-14

### Fixed
- Sampling probes no longer force `temperature=0.0` in the probe request: sampling probes now run under the server's configured default sampler, so a mis-set server-side temperature is observable instead of masked by the probe itself. Chat-template probes still pin `temperature=0.0` for determinism.
- Served-default detection: GapProbe now reads llama.cpp's `/props` `default_generation_settings` and diffs temperature / top_p / repeat_penalty against the per-model canonical sampler registry, naming the mismatched parameter in the layer evidence. Servers without `/props` (e.g. vLLM) fall back to behavioral probes only.
- `--layer` is validated: an unknown value (e.g. `--layer tokenizer`) exits with code 2 and a clear message instead of silently running zero probes and blaming the endpoint.
- Verdict consistency: estimated capability loss is 0 below the misconfiguration threshold (was: a nonzero "~N% loss" alongside "no fix needed"), and a clean configuration reports "no significant capability gap detected" instead of naming a dominant layer at ~0% loss.
- The registry no longer routes Qwen2.5 model ids to the Qwen3 entry, which prescribed the wrong (`qwen3`) chat template; Qwen2.5 now reports "not in registry" until it gets a real entry.

### Added
- `gapprobe --version` flag.
- Version-consistency tests: the pyproject version, `gapprobe.__version__`, `web/site.json` `content_version`, the CHANGELOG section, and the CLI `--version` output must all agree.

## [0.1.0] — 2026-08-25

### Added
- Cross-layer attribution primitive: maps a local CN model's felt "dumbness" to a specific misconfiguration layer (chat-template / sampling) with a quantified capability-loss estimate.
- Chat-template layer probe: 6 behavioral probes (tool-call format, multi-turn memory, system-prompt adherence, structured JSON, role recognition, stop sequence) diffed against a per-model known-good registry.
- Sampling layer probe: 6 behavioral probes (instruction-following, repetition, creative constraint, factual recall, JSON format, code block) sensitive to temperature / top_p / repeat_penalty misconfiguration.
- Per-model registry for Qwen3 + DeepSeek: canonical chat-template name, canonical sampler defaults, known-good probe response patterns, per-layer loss calibration.
- 20-case attribution bench: 10 mis-set + 10 correctly-set cases, validates that 2-layer attribution beats random guess (>50% correct dominant-layer call).
- CLI: `gapprobe probe --endpoint <url>` prints a screenshot-able one-line verdict; `gapprobe bench` runs the 20-case bench.
- Bilingual README (中文 + English).

[Unreleased]: https://github.com/SuperMarioYL/GapProbe/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/SuperMarioYL/GapProbe/releases/tag/v0.2.0
[0.1.0]: https://github.com/SuperMarioYL/GapProbe/releases/tag/v0.1.0
