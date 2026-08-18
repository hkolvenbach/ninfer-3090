"""Persistent-object contract for the complete Qwen3.8-27B artifact.

The graph and all non-vocabulary storage roles are identical to the registered
Qwen3.8-27B groupwise artifact.  The embedding and full output head use the W8
format already supported by the 27B runtime.
"""

from __future__ import annotations

from tools.convert.qwen3_8_27b import base_inventory


MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "qwen3_8_27b"

BF16 = base_inventory.BF16
FP32 = base_inventory.FP32
I32 = base_inventory.I32
Q4 = base_inventory.Q4
Q5 = base_inventory.Q5
Q6 = base_inventory.Q6
W8 = base_inventory.W8
CONTIGUOUS_LAYOUT = base_inventory.CONTIGUOUS_LAYOUT
ROW_SPLIT_LAYOUT = base_inventory.ROW_SPLIT_LAYOUT

FORMAT_NAMES = base_inventory.FORMAT_NAMES
LAYOUT_NAMES = base_inventory.LAYOUT_NAMES
ResourceSpec = base_inventory.ResourceSpec
StoredObjectSpec = base_inventory.StoredObjectSpec
TensorSpec = base_inventory.TensorSpec

FULL_ATTENTION_LAYERS = base_inventory.FULL_ATTENTION_LAYERS
GDN_LAYERS = base_inventory.GDN_LAYERS
RESOURCE_SPECS = base_inventory.RESOURCE_SPECS


def _w8_vocabulary_endpoint(spec: TensorSpec) -> TensorSpec:
    if spec.name in ("text/token_embedding", "text/output_head"):
        return base_inventory.tensor_spec(spec.name, spec.shape, W8)
    return spec


TEXT_CORE_TENSOR_SPECS = tuple(
    _w8_vocabulary_endpoint(spec)
    for spec in base_inventory.TEXT_CORE_TENSOR_SPECS
)
DRAFT_HEAD_TENSOR_SPECS = base_inventory.DRAFT_HEAD_TENSOR_SPECS
MTP_TENSOR_SPECS = base_inventory.MTP_TENSOR_SPECS
VISION_TENSOR_SPECS = base_inventory.VISION_TENSOR_SPECS

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

LOGICAL_ROW_VIEW_SPECS = base_inventory.LOGICAL_ROW_VIEW_SPECS
ALIAS_SPECS = base_inventory.ALIAS_SPECS
