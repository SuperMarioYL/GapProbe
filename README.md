# GapProbe

<div align="center">

**本地大模型"感觉变笨了"？GapProbe 告诉你是哪一层配错了。**

Capability-gap diagnostician for locally-served CN open-weights models.

[Features](#-功能) · [Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Bench](#-attribution-bench) · [中文](#中文)

</div>

---

Your local Qwen3 / DeepSeek feels dumber than the benchmark. You've tried
everything — different quants, more context, new samplers — but you can't tell
*which* misconfiguration is stealing capability. GapProbe runs ~12 behavioral
probes across two layers (chat-template + sampling), diffs them against a
per-model known-good registry, and tells you in one line:

```
qwen3-27b: ~25% capability loss — dominant layer=chat_template (ChatML not applied). Fix: llama-server --chat-template qwen3
```

## 🎯 功能

- **Cross-layer attribution** — maps felt "dumbness" to a *named layer* with a *quantified capability loss*, not just "it's slow"
- **Chat-template probe** — 6 behavioral probes that only work correctly under the canonical template (tool-call format, multi-turn memory, system-prompt adherence, structured JSON, role recognition, stop sequence)
- **Sampling probe** — 6 probes sensitive to temperature / top_p / repeat_penalty misconfiguration
- **Per-model registry** — Qwen3 + DeepSeek known-good reference data + loss calibration
- **20-case bench** — validates 2-layer attribution beats random guess

## 🚀 Quick Start

```bash
# Install
pip install gapprobe
# or: uvx gapprobe probe --endpoint http://localhost:8080

# Point at your local llama.cpp / vllm server
gapprobe probe --endpoint http://localhost:8080

# Probe only one layer
gapprobe probe --endpoint http://localhost:8080 --layer chat-template

# Run the 20-case attribution bench
gapprobe bench
```

**Output:**
```
✓ Detected: Qwen3-27B-Instruct (registry: qwen3)
Probing chat-template layer…
  divergence_score: 0.500  est_capability_loss: 24.5%
  evidence: ct_tool_call_format: expected '"name"' not found

Probing sampling layer…
  divergence_score: 0.000  est_capability_loss: 0.0%
  evidence: all sampling probes passed

━━━ Verdict ━━━
qwen3-27b: ~25% capability loss — dominant layer=chat_template (ChatML not applied). Fix: llama-server --chat-template qwen3
```

## 🔬 How It Works

```
┌──────────┐    ┌────────────┐    ┌──────────────────────┐
│ CLI typer│───▶│ Probe Engine│───▶│ Local OpenAI-compat  │
│ gapprobe │    │ (httpx)     │    │ llama.cpp / vllm     │
└──────────┘    └─────┬──────┘    └──────────────────────┘
                      │
               ┌──────▼──────┐
               │ Layer probes│  chat_template  sampling
               └──────┬──────┘
                      │
               ┌──────▼──────┐    ┌──────────────────────┐
               │ Verdict     │◀───│ Per-Model Registry    │
               │ Assembler   │    │ known-good refs +     │
               │ (rich)      │    │ loss calibration      │
               └─────────────┘    │ Qwen3 / DeepSeek      │
                                  └──────────────────────┘
```

GapProbe's new primitive is **cross-layer attribution**: it doesn't fix any
single layer — it *attributes* the benchmark-vs-served delta to a specific
layer, so you fix one thing instead of the whole stack.

## 📊 Attribution Bench

The 20-case bench validates that GapProbe's 2-layer attribution beats random guess:

| Case Type | Count | Expected |
|-----------|-------|----------|
| Chat-template mis-set | 4 | dominant=chat_template |
| Sampling mis-set | 4 | dominant=sampling |
| Both mis-set (CT dominant) | 1 | dominant=chat_template |
| Both mis-set (SP dominant) | 1 | dominant=sampling |
| Correctly configured | 10 | no dominant layer |

Run it: `gapprobe bench`

## ⚙️ Supported Models

| Model | Chat Template | Canonical Sampler |
|-------|--------------|-------------------|
| Qwen3 (27B, 32B, etc.) | `qwen3` | temp=0.7, top_p=0.8, repeat_penalty=1.05 |
| DeepSeek (V3, R1) | `deepseek` | temp=0.7, top_p=0.95, repeat_penalty=1.0 |

v0.2+ will add Kimi K3, GLM, and more layers (tokenizer, MoE routing, MTP).

## 📋 Scope (v0.1)

**In scope:** chat-template + sampling attribution for Qwen3 + DeepSeek, CLI only, diagnosis + suggested fix.

**Out of scope (v0.2+):** tokenizer / context-format / MoE-routing / MTP layers, auto-fix, Kimi K3 / GLM, Web UI, daemon mode.

## 📄 License

MIT © SuperMarioYL

---

## 中文

GapProbe 是一个本地大模型能力差距诊断工具。当你的本地 Qwen3 或 DeepSeek
"感觉变笨了"，GapProbe 通过 12 个行为探针（覆盖 chat-template 和 sampling 两层），
与每模型的已知正确配置注册表对比，定位到底是哪一层配置出错导致能力损失。

**一句话输出：**
```
qwen3-27b: ~25% 能力损失 — 主因层=chat_template（ChatML 模板未应用）。修复: llama-server --chat-template qwen3
```

**核心创新：** 跨层归因（cross-layer attribution）——不修复任何单层，
而是将"感觉变笨"的差距归因到一个具体的层，让你只修一个地方，
而不是改重量化、换上下文、调采样器全试一遍。

**安装：** `pip install gapprobe` 或 `uvx gapprobe probe --endpoint http://localhost:8080`

**支持模型：** Qwen3 系列、DeepSeek V3/R1（v0.2 将增加 Kimi K3 / GLM）

**许可证：** MIT © SuperMarioYL
