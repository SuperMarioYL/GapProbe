"""gapprobe/layers/__init__.py — probe layer registry."""

from gapprobe.layers.chat_template import ChatTemplateLayer
from gapprobe.layers.sampling import SamplingLayer

LAYER_REGISTRY = {
    "chat_template": ChatTemplateLayer,
    "sampling": SamplingLayer,
}
