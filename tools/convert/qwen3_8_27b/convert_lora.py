"""Convert a trained PEFT LoRA adapter into a Qwen3.8-27B ``.ninfer`` adapter.

    python3 -m tools.convert.qwen3_8_27b.convert_lora \
        --adapter out/adapters/math --out models/adapters/math.lora.ninfer

The conversion needs no base checkpoint: a LoRA ``A``/``B`` pair is independent
of the weights it corrects, and every shape this target accepts is a registered
constant in :mod:`tools.convert.qwen3_8_27b.lora_inventory`.

Three transforms happen here and nowhere else:

* ``alpha / r`` (or ``alpha / sqrt(r)`` under rsLoRA) is folded into ``B``, so
  the engine carries no scale and the artifact is self-contained;
* the fused query/gate attention parent is de-interleaved into its two row
  selections, which is exact because ``delta = B @ A`` makes a row slice of
  ``B`` a row slice of ``delta``;
* everything is materialised as BF16, the only numeric format an adapter uses.

An adapter that targets a module NInfer cannot correct after the fact is
rejected by name, never silently dropped.

``--zero-sites`` writes a registered site's ``B`` plane as zeros while keeping
its full inventory.  ``delta = B @ A`` is then exactly zero, so the site is
inert rather than absent, and the artifact keeps the object count and slab size
of its unmasked peers -- which is what lets a whole ablation series load into
one adapter bank.  ``A`` is left as trained: the fused query/gate parent shares
one ``A`` plane, so zeroing ``A`` would corrupt the unmasked twin.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from tools.artifact import ArtifactIdentity, ArtifactWriter
from tools.artifact.container import TensorSpec as ArtifactTensorSpec
from tools.artifact.layouts import encode_direct
from tools.convert.qwen3_8.common import conversion as family_conversion
from tools.convert.qwen3_8_27b import lora_inventory as inventory


TARGET_KEY = "qwen3_8_27b"

_ADAPTER_CONFIG = "adapter_config.json"
_ADAPTER_WEIGHTS = "adapter_model.safetensors"

# PEFT stores `base_model.model.<module path>.lora_{A,B}.weight`. The module
# path differs between the multimodal wrapper and a text-only load, so the
# layer index and the module tail are matched as a suffix instead.
_KEY_PATTERN = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.(?P<module>[A-Za-z0-9_.]+?)\.lora_(?P<side>[AB])\.weight$"
)


class LoraConversionError(ValueError):
    """A trained adapter does not satisfy the registered adapter contract."""


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    rank: int
    alpha: float
    scale: float
    use_rslora: bool
    target_modules: tuple[str, ...]
    base_model_name: str | None


def _reject(message: str) -> None:
    raise LoraConversionError(message)


def load_adapter_config(adapter_dir: Path) -> AdapterConfig:
    path = adapter_dir / _ADAPTER_CONFIG
    if not path.is_file():
        _reject(f"{path} is missing; --adapter must be a PEFT adapter directory")
    raw = json.loads(path.read_text())

    peft_type = raw.get("peft_type")
    if peft_type != "LORA":
        _reject(f"peft_type {peft_type!r} is not supported; only 'LORA' converts")

    for field, message in (
        ("use_dora", "DoRA carries a per-row magnitude that no registered site stores"),
        ("use_qalora", "QALoRA is not a registered adapter form"),
        ("lora_bias", "LoRA bias terms are not a registered adapter form"),
    ):
        if raw.get(field):
            _reject(f"{field} is set: {message}")

    bias = raw.get("bias", "none")
    if bias != "none":
        _reject(f"bias={bias!r} is not supported; train with bias='none'")

    for field in ("modules_to_save", "trainable_token_indices", "target_parameters"):
        if raw.get(field):
            _reject(f"{field} is set; only plain low-rank linear sites convert")

    for field in ("rank_pattern", "alpha_pattern"):
        if raw.get(field):
            _reject(
                f"{field} is non-empty; a per-module rank or alpha would break the uniform "
                "adapter bank. Train with one rank and one alpha."
            )

    for field in ("layers_to_transform", "layers_pattern", "layer_replication"):
        if raw.get(field):
            _reject(
                f"{field} is set; a registered site must exist on every layer of its group"
            )

    rank = raw.get("r")
    if not isinstance(rank, int):
        _reject(f"adapter rank r={rank!r} is not an integer")
    try:
        inventory.require_rank(rank)
    except ValueError as error:
        _reject(str(error))

    alpha = raw.get("lora_alpha")
    if not isinstance(alpha, (int, float)):
        _reject(f"lora_alpha={alpha!r} is not a number")
    alpha = float(alpha)

    use_rslora = bool(raw.get("use_rslora", False))
    scale = alpha / (rank**0.5) if use_rslora else alpha / rank

    target_modules = raw.get("target_modules")
    if isinstance(target_modules, str):
        _reject(
            "target_modules is a regex; convert with an explicit module-name list so the "
            "registered site set is unambiguous"
        )
    if not target_modules:
        _reject("target_modules is empty")
    modules = tuple(sorted(str(item) for item in target_modules))

    for module in modules:
        if module in inventory.REJECTED_HF_MODULES:
            _reject(
                f"target module {module!r} has no registered NInfer site: "
                f"{inventory.REJECTED_HF_MODULES[module]}. "
                f"Retrain with target_modules limited to "
                f"{sorted(inventory.SUPPORTED_HF_MODULES)}."
            )
        if module not in inventory.SUPPORTED_HF_MODULES:
            _reject(
                f"target module {module!r} is not a registered adapter site; supported modules "
                f"are {sorted(inventory.SUPPORTED_HF_MODULES)}"
            )

    return AdapterConfig(
        rank=rank,
        alpha=alpha,
        scale=scale,
        use_rslora=use_rslora,
        target_modules=modules,
        base_model_name=raw.get("base_model_name_or_path"),
    )


def load_adapter_tensors(adapter_dir: Path) -> dict[str, torch.Tensor]:
    path = adapter_dir / _ADAPTER_WEIGHTS
    if not path.is_file():
        _reject(f"{path} is missing; only safetensors adapters convert")
    return load_file(str(path))


def parse_adapter_state(
    state: dict[str, torch.Tensor],
) -> dict[tuple[int, str], dict[str, torch.Tensor]]:
    """Group ``lora_A``/``lora_B`` pairs by ``(layer, module tail)``."""

    entries: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    unmatched: list[str] = []
    for key, value in state.items():
        match = _KEY_PATTERN.search(key)
        if match is None:
            unmatched.append(key)
            continue
        layer = int(match.group("layer"))
        module = match.group("module")
        entries.setdefault((layer, module), {})[match.group("side")] = value
    if unmatched:
        _reject(
            f"{len(unmatched)} adapter tensors are outside the decoder layer stack, "
            f"for example {sorted(unmatched)[:3]}"
        )
    for (layer, module), sides in entries.items():
        missing = {"A", "B"} - set(sides)
        if missing:
            _reject(f"layer {layer} module {module} is missing lora_{sorted(missing)[0]}")
    return entries


def _module_leaf(module: str) -> str:
    return module.rsplit(".", 1)[-1]


def resolve_sites(
    entries: dict[tuple[int, str], dict[str, torch.Tensor]],
    config: AdapterConfig,
) -> frozenset[str]:
    """Determine the registered site set and verify per-layer completeness."""

    present: dict[str, set[int]] = {}
    for (layer, module), _sides in entries.items():
        leaf = _module_leaf(module)
        site_keys = inventory.SUPPORTED_HF_MODULES.get(leaf)
        if site_keys is None:
            reason = inventory.REJECTED_HF_MODULES.get(leaf)
            detail = f": {reason}" if reason else ""
            _reject(f"adapter targets module {leaf!r}, which has no registered site{detail}")
        for site_key in site_keys:
            present.setdefault(site_key, set()).add(layer)

    for site_key, layers in sorted(present.items()):
        expected = set(inventory.SITE_BY_KEY[site_key].layers())
        if layers != expected:
            missing = sorted(expected - layers)
            extra = sorted(layers - expected)
            _reject(
                f"site {site_key} covers {len(layers)} layers but its group has "
                f"{len(expected)}; missing={missing[:6]} unexpected={extra[:6]}"
            )

    targeted_leaves = {leaf for leaf in config.target_modules}
    resolved_leaves = {_module_leaf(module) for (_layer, module) in entries}
    if targeted_leaves != resolved_leaves:
        _reject(
            f"adapter_config target_modules {sorted(targeted_leaves)} does not match the "
            f"stored tensors {sorted(resolved_leaves)}"
        )
    return frozenset(present)


def _site_source(
    entries: dict[tuple[int, str], dict[str, torch.Tensor]],
    layer: int,
    site: inventory.LoraSiteSpec,
) -> dict[str, torch.Tensor]:
    key = (layer, site.hf_module)
    if key in entries:
        return entries[key]
    leaf = _module_leaf(site.hf_module)
    matches = [
        value
        for (entry_layer, module), value in entries.items()
        if entry_layer == layer and _module_leaf(module) == leaf
    ]
    if len(matches) != 1:
        _reject(f"layer {layer} site {site.key}: expected one {leaf} module, found {len(matches)}")
    return matches[0]


def build_site_tensors(
    entries: dict[tuple[int, str], dict[str, torch.Tensor]],
    layer: int,
    site: inventory.LoraSiteSpec,
    config: AdapterConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(A, B)`` for one site: exact shapes, scale folded, BF16."""

    sides = _site_source(entries, layer, site)
    a = sides["A"].to(torch.float32)
    b = sides["B"].to(torch.float32)

    expected_a = (config.rank, site.in_features)
    if tuple(a.shape) != expected_a:
        _reject(
            f"layer {layer} site {site.key}: lora_A has shape {tuple(a.shape)}, "
            f"expected {expected_a}"
        )
    expected_b = (site.hf_out_features, config.rank)
    if tuple(b.shape) != expected_b:
        _reject(
            f"layer {layer} site {site.key}: lora_B has shape {tuple(b.shape)}, "
            f"expected {expected_b}"
        )

    if site.is_row_selection:
        b = b.reshape(site.hf_heads, site.hf_head_rows, config.rank)
        b = b[:, site.hf_row_begin : site.hf_row_end, :]
        b = b.reshape(site.out_features, config.rank)
    b = (b * config.scale).contiguous()

    return a.to(torch.bfloat16).contiguous(), b.to(torch.bfloat16).contiguous()


def convert(
    adapter_dir: Path, out_path: Path, zero_sites: frozenset[str] = frozenset()
) -> dict[str, object]:
    started = time.perf_counter()
    config = load_adapter_config(adapter_dir)
    state = load_adapter_tensors(adapter_dir)
    entries = parse_adapter_state(state)
    site_keys = resolve_sites(entries, config)

    if zero_sites:
        try:
            inventory.require_sites(zero_sites)
        except ValueError as error:
            _reject(str(error))
        absent = sorted(zero_sites - site_keys)
        if absent:
            _reject(
                f"--zero-sites names {absent}, which this adapter does not target; a masked "
                "site must be present in the source adapter for the inventory to be preserved"
            )

    specs = inventory.build_tensor_specs(config.rank, site_keys)
    container_specs = tuple(
        ArtifactTensorSpec(spec.name, spec.shape, spec.format, spec.layout) for spec in specs
    )
    identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(out_path, identity, container_specs)

    emitted: set[str] = set()
    for layer in range(inventory.LAYERS):
        for site in inventory.SITE_SPECS:
            if site.key not in site_keys or layer not in site.layers():
                continue
            a, b = build_site_tensors(entries, layer, site, config)
            if site.key in zero_sites:
                b = torch.zeros_like(b)
            a_name = inventory.a_object_name(layer, site)
            if a_name not in emitted:
                writer.write(a_name, encode_direct(a, "BF16"))
                emitted.add(a_name)
            writer.write(inventory.b_object_name(layer, site), encode_direct(b, "BF16"))
    writer.finish()  # raises unless every planned payload was written

    statistics = family_conversion.object_statistics(writer.objects)
    # Storage is deliberately unchanged by masking, so `parameters` stays the inventory count and
    # `effective_parameters` reports how much of it can still move an activation.
    live_sites = site_keys - zero_sites
    report: dict[str, object] = {
        "identity": {"model_id": identity.model_id, "weights_id": identity.weights_id},
        "target_key": TARGET_KEY,
        "recipe_id": inventory.RECIPE_ID,
        "source": {
            "adapter_path": str(adapter_dir),
            "base_model_name_or_path": config.base_model_name,
        },
        "adapter": {
            "rank": config.rank,
            "lora_alpha": config.alpha,
            "use_rslora": config.use_rslora,
            "folded_scale": config.scale,
            "target_modules": list(config.target_modules),
            "sites": sorted(site_keys),
            "zeroed_sites": sorted(zero_sites),
            "parameters": inventory.parameter_count(config.rank, site_keys),
            "effective_parameters": (
                inventory.parameter_count(config.rank, live_sites) if live_sites else 0
            ),
        },
        "converter": {
            "revision": family_conversion.converter_revision(Path(__file__).resolve().parents[3]),
            "environment": family_conversion.environment(torch.device("cpu")),
        },
        "objects": statistics,
        "elapsed_seconds": time.perf_counter() - started,
        "artifact": {"path": str(out_path), "bytes": out_path.stat().st_size},
    }
    report_path = out_path.with_suffix(out_path.suffix + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path, help="PEFT adapter directory")
    parser.add_argument("--out", required=True, type=Path, help="output .lora.ninfer path")
    parser.add_argument(
        "--zero-sites",
        nargs="+",
        default=(),
        metavar="SITE",
        help="registered sites to write as an all-zero B plane. The site keeps its full "
             "inventory, so the result stays loadable in one bank alongside its unmasked peers.",
    )
    args = parser.parse_args()

    report = convert(args.adapter, args.out, frozenset(args.zero_sites))
    adapter = report["adapter"]
    objects = report["objects"]
    print(f"wrote {report['artifact']['path']}")
    print(f"  rank {adapter['rank']}  alpha {adapter['lora_alpha']}  scale {adapter['folded_scale']}")
    print(f"  sites {', '.join(adapter['sites'])}")
    if adapter["zeroed_sites"]:
        print(f"  zeroed {', '.join(adapter['zeroed_sites'])}  "
              f"({adapter['effective_parameters']} of {adapter['parameters']} parameters live)")
    print(f"  {objects['count']} objects, {adapter['parameters']} parameters, "
          f"{report['artifact']['bytes']} bytes")


if __name__ == "__main__":
    main()
