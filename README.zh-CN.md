[English](README.md) | **简体中文**

<picture>
  <source media="(max-width: 640px) and (prefers-color-scheme: dark)" srcset="assets/presentation/hero-mobile-dark.svg">
  <source media="(max-width: 640px)" srcset="assets/presentation/hero-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="assets/presentation/hero-dark.svg">
  <img src="assets/presentation/hero-light.svg" width="1000" alt="对本地模型服务运行两组行为探针，分别查看 chat-template 与 sampling 的失败信号。">
</picture>

**对本地模型服务运行两组行为探针，分别查看 chat-template 与 sampling 的失败信号。**

`v0.1.0` · `Python 3.12+` · [MIT](LICENSE)

[Website](https://gapprobe.lei6393.com) · [Demo record](docs/demo-results.json)

## 为什么使用

当本地模型输出不符合预期时，同时换模板、采样参数和量化方式很难知道哪一步有帮助。GapProbe 将固定行为检查分成两组，保留失败探针的响应片段，并按内置规则给出排查方向。

## 架构

<picture>
  <source media="(max-width: 640px) and (prefers-color-scheme: dark)" srcset="assets/presentation/architecture-mobile-dark.svg">
  <source media="(max-width: 640px)" srcset="assets/presentation/architecture-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="assets/presentation/architecture-dark.svg">
  <img src="assets/presentation/architecture-light.svg" width="1000" alt="ProbeEngine 从 /v1/models 识别注册表条目，再向 /v1/chat/completions 发固定请求。两个 layer 根据预期子串检查计算失败比例，registry 的线性系数生成估计值，verdict 汇总结果。离线 bench 直接使用模拟 ProbeResult，不连接模型。">
</picture>

ProbeEngine 从 /v1/models 识别注册表条目，再向 /v1/chat/completions 发固定请求。两个 layer 根据预期子串检查计算失败比例，registry 的线性系数生成估计值，verdict 汇总结果。离线 bench 直接使用模拟 ProbeResult，不连接模型。

探针定义与参考参数在 [registry.py](gapprobe/registry.py)，模拟输入在 [cases.py](gapprobe/cases.py)。当前注册表为 Qwen3 与 DeepSeek。

## 安装

需要 Python 3.12+。下面的 bench 与注册表检查完全离线；实时 probe 需要你已经启动的兼容端点。

```bash
git clone https://github.com/SuperMarioYL/GapProbe.git
cd GapProbe
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## 快速开始

实际执行内置 20 个模拟归因案例和注册表检查，结果符合 20/20 预设期望。它检验规则实现，不是模型性能、真实配置诊断准确率或能力损失测量。

```bash
python -m gapprobe.cli bench
python examples/presentation_registry.py
```

bench 的完整输入随 cases.py 提供，注册表检查脚本位于 [examples/presentation_registry.py](examples/presentation_registry.py)。

## 使用

`python -m gapprobe.cli probe --endpoint http://localhost:8080` 运行两组探针；`--layer chat-template` 或 `--layer sampling` 可只运行一组。它不会启动服务器或自动修改参数。

## 实际 Demo

<picture>
  <source media="(max-width: 640px) and (prefers-color-scheme: dark)" srcset="assets/presentation/process-mobile-dark.svg">
  <source media="(max-width: 640px)" srcset="assets/presentation/process-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="assets/presentation/process-dark.svg">
  <img src="assets/presentation/process-light.svg" width="1000" alt="实际执行内置 20 个模拟归因案例和注册表检查，结果符合 20/20 预设期望。它检验规则实现，不是模型性能、真实配置诊断准确率或能力损失测量。">
</picture>

### 运行模拟案例

20 个构造案例全部符合其预设标签。

```text
$ python -m gapprobe.cli bench
           GapProbe 20-Case Attribution Bench
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━┓
┃ Case            ┃ Predicted     ┃ Expected      ┃ ✓/✗ ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━┩
│ mis_ct_1        │ chat_template │ chat_template │ ✓   │
│ mis_ct_2        │ chat_template │ chat_template │ ✓   │
│ mis_ct_3        │ chat_template │ chat_template │ ✓   │
│ mis_ct_4        │ chat_template │ chat_template │ ✓   │
│ mis_sp_1        │ sampling      │ sampling      │ ✓   │
│ mis_sp_2        │ sampling      │ sampling      │ ✓   │
│ mis_sp_3        │ sampling      │ sampling      │ ✓   │
│ mis_sp_4        │ sampling      │ sampling      │ ✓   │
│ mis_both_ct_dom │ chat_template │ chat_template │ ✓   │
│ mis_both_sp_dom │ sampling      │ sampling      │ ✓   │
│ clean_1         │ none          │ none          │ ✓   │
│ clean_2         │ none          │ none          │ ✓   │
│ clean_3         │ none          │ none          │ ✓   │
│ clean_4         │ none          │ none          │ ✓   │
│ clean_5         │ none          │ none          │ ✓   │
│ clean_6         │ none          │ none          │ ✓   │
│ clean_7         │ none          │ none          │ ✓   │
│ clean_8         │ none          │ none          │ ✓   │
│ clean_9         │ none          │ none          │ ✓   │
│ clean_10        │ none          │ none          │ ✓   │
└─────────────────┴───────────────┴───────────────┴─────┘

Attribution accuracy: 20/20 (100%) — beats random guess (>50%)
```

### 检查参考注册表

查看实际模型匹配、探针数量与参考参数。

```text
$ python examples/presentation_registry.py
{
  "matched_model": "qwen3",
  "registry": {
    "qwen3": {
      "template": "qwen3",
      "chat_probes": 6,
      "sampling_probes": 6,
      "temperature": 0.7,
      "top_p": 0.8
    },
    "deepseek": {
      "template": "deepseek",
      "chat_probes": 6,
      "sampling_probes": 6,
      "temperature": 0.7,
      "top_p": 0.95
    }
  }
}
```

## 能力与接入

<picture>
  <source media="(max-width: 640px) and (prefers-color-scheme: dark)" srcset="assets/presentation/integrations-mobile-dark.svg">
  <source media="(max-width: 640px)" srcset="assets/presentation/integrations-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="assets/presentation/integrations-dark.svg">
  <img src="assets/presentation/integrations-light.svg" width="1000" alt="实时端点提供响应，内置注册表定义检查与估计系数，离线案例可验证汇总规则。三者不要混作同一类性能证据。">
</picture>

实时端点提供响应，内置注册表定义检查与估计系数，离线案例可验证汇总规则。三者不要混作同一类性能证据。



## 配置

注册表每组各 6 个探针。实际 HTTP 请求固定 temperature=0.0、max_tokens=128，逐个发送并暂停 0.3 秒；不是对 sampler 参数进行扫描实验。端点超时或 HTTP 失败也会进入失败信号，排查时需先确认服务健康。

## 路线图与范围

当前为两层、两组模型注册表的 CLI。更多模型、tokenizer、MoE 路由、MTP 与自动修复仍不在实现范围。

- 模拟 bench 的 100% 不能推广为真实模型归因准确率。
- “能力损失”是内置规则估计，没有在此任务中进行校准验证。
- 子串匹配和固定请求不能唯一确定配置故障原因。

## 许可证

[MIT](LICENSE)
