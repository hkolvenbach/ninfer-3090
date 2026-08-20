"""Independent LoRA application for the reference model.

This reads a **PEFT adapter directory** rather than a converted `.lora.ninfer`.  That is the point:
if the reference consumed the converted artifact, a fault in the converter would be inherited by
both implementations and stay invisible.  Reading the PEFT weights and deriving the row layout from
HuggingFace semantics makes the whole chain — converter, binder and runtime application — testable
against this path.

For a HuggingFace `nn.Linear(in, out)` carrying weight `W` of shape `[out, in]`, PEFT computes

    y = x @ W^T + (alpha / r) * (x @ A^T) @ B^T,     A: [r, in],  B: [out, r]

`q_proj` is the only site whose delta does not land on a single artifact tensor.  Its `[12288, in]`
output is 24 heads of 512 rows, the first 256 of each head being query rows and the last 256 being
output-gate rows, so its delta is split across the query half of `query_key` and the gate half of
`gate_value`.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open

from .config import CFG

# `q_proj` packs 256 query rows and 256 output-gate rows for each of the 24 heads.
_HF_HEAD_ROWS = 2 * CFG.head_dim

_KEY = re.compile(r"layers\.(\d+)\.(.+)\.lora_([AB])\.weight$")


class LoraAdapter:
    """Per-layer LoRA factors for the registered site table, applied in HF semantics."""

    def __init__(self, path: str | Path, *, device: str | torch.device):
        path = Path(path)
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        self.rank = int(config["r"])
        alpha = float(config["lora_alpha"])
        # rsLoRA divides by sqrt(r); ordinary LoRA by r. The converter folds this into `B`, so it
        # must be reproduced here rather than assumed to be 1.
        self.scale = alpha / (math.sqrt(self.rank) if config.get("use_rslora") else self.rank)

        factors: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        weights = path / "adapter_model.safetensors"
        with safe_open(str(weights), framework="pt", device="cpu") as source:
            for key in source.keys():
                match = _KEY.search(key)
                if match is None:
                    raise ValueError(f"unrecognized adapter key: {key}")
                layer, module, side = int(match.group(1)), match.group(2), match.group(3)
                tensor = source.get_tensor(key).to(device=device, dtype=torch.bfloat16)
                factors.setdefault((layer, module), {})[side] = tensor

        self.factors: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for site, pair in factors.items():
            if set(pair) != {"A", "B"}:
                raise ValueError(f"adapter site {site} is missing an A or B factor")
            self.factors[site] = (pair["A"], pair["B"])
        self.modules = sorted({module for _, module in self.factors})

    def delta(self, layer: int, module: str, x: torch.Tensor) -> torch.Tensor | None:
        pair = self.factors.get((layer, module))
        if pair is None:
            return None
        a, b = pair
        return (self.scale * ((x.to(torch.bfloat16) @ a.t()) @ b.t())).to(torch.bfloat16)

    def apply_attention_input(self, layer: int, h: torch.Tensor, qk: torch.Tensor,
                              gatev: torch.Tensor) -> None:
        """Correct the four attention input projections in place."""
        query_gate = self.delta(layer, "self_attn.q_proj", h)
        if query_gate is not None:
            heads = query_gate.reshape(-1, CFG.q_heads, _HF_HEAD_ROWS)
            qk[:, :CFG.q_size] += heads[:, :, :CFG.head_dim].reshape(-1, CFG.q_size)
            gatev[:, :CFG.q_size] += heads[:, :, CFG.head_dim:].reshape(-1, CFG.q_size)
        key = self.delta(layer, "self_attn.k_proj", h)
        if key is not None:
            qk[:, CFG.q_size:] += key
        value = self.delta(layer, "self_attn.v_proj", h)
        if value is not None:
            gatev[:, CFG.q_size:] += value

    def apply_attention_output(self, layer: int, mixed: torch.Tensor,
                               out: torch.Tensor) -> torch.Tensor:
        delta = self.delta(layer, "self_attn.o_proj", mixed)
        return out if delta is None else out + delta

    def apply_gdn_output(self, layer: int, mixed: torch.Tensor,
                         out: torch.Tensor) -> torch.Tensor:
        delta = self.delta(layer, "linear_attn.out_proj", mixed)
        return out if delta is None else out + delta

    def apply_mlp_down(self, layer: int, activated: torch.Tensor,
                       out: torch.Tensor) -> torch.Tensor:
        delta = self.delta(layer, "mlp.down_proj", activated)
        return out if delta is None else out + delta
