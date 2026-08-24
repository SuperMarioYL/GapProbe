# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-25

### Added
- Cross-layer attribution primitive: maps a local CN model's felt "dumbness" to a specific misconfiguration layer (chat-template / sampling) with a quantified capability-loss estimate.
- Chat-template layer probe: 6 behavioral probes (tool-call format, multi-turn memory, system-prompt adherence, structured JSON, role recognition, stop sequence) diffed against a per-model known-good registry.
- Sampling layer probe: 6 behavioral probes (instruction-following, repetition, creative constraint, factual recall, JSON format, code block) sensitive to temperature / top_p / repeat_penalty misconfiguration.
- Per-model registry for Qwen3 + DeepSeek: canonical chat-template name, canonical sampler defaults, known-good probe response patterns, per-layer loss calibration.
- 20-case attribution bench: 10 mis-set + 10 correctly-set cases, validates that 2-layer attribution beats random guess (>50% correct dominant-layer call).
- CLI: `gapprobe probe --endpoint <url>` prints a screenshot-able one-line verdict; `gapprobe bench` runs the 20-case bench.
- Bilingual README (中文 + English).
