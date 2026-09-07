import json
from gapprobe.registry import REGISTRY, match_model
print(json.dumps({"matched_model": match_model("Qwen3-27B-Instruct"), "registry": {key: {"template": entry.canonical_chat_template, "chat_probes": len(entry.chat_template_probes), "sampling_probes": len(entry.sampling_probes), "temperature": entry.canonical_sampler.temperature, "top_p": entry.canonical_sampler.top_p} for key, entry in REGISTRY.items()}}, indent=2))
