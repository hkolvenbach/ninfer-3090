"""Measure whether a LoRA adapter survives the training/serving base-quantization change.

An adapter is fitted with the frozen base held at bitsandbytes NF4 — a 4-bit normal-float codebook
whose per-64 scales are themselves quantized — but NInfer serves it on top of the `groupwise-int`
artifact, which stores symmetric uniform Q4/Q5 codes against exactly-stored FP16 scales.  The
question is whether that difference costs the adapter its behaviour.

Answered, for the 120-step math adapter: it does not.  The first run of this driver said otherwise,
but the cause was an engine defect that suppressed the adapter through all of prefill, not the base
difference; see the base-quantization section of docs/maintainer/qwen3.8-27b-lora-adapters.md.
Note the failure mode that implies for this tool: it compares two *runners*, so any defect on the
NInfer side is indistinguishable here from a property of the quantization.  A disagreement it
reports is a starting point for attribution, never a conclusion.  The control that settles one is
`tools/reference/qwen3_8_27b/lora.py`, which applies the adapter inside the reference on the same
`groupwise-int` base and so varies the application while holding the base fixed.

The measurement is a *normalized* cross-base agreement.  Two greedy decodes of the same prompt on
the two bases already diverge without any adapter, simply because the codecs differ, so the raw
agreement of the adapted pair means nothing on its own.  What is informative is whether attaching
the adapter makes the two bases agree *less* than they already did:

    agree_base    = agreement(ninfer base,          bnb nf4 base)
    agree_adapter = agreement(ninfer base+adapter,  bnb nf4 base+adapter)
    retention     = agree_adapter / agree_base

`retention` near 1 means the adapter rides on top of the quantization difference without amplifying
it, and the mismatch costs nothing observable.  A retention well below 1 means the adapter's
behaviour is partly specific to the base it was fitted against.

Agreement is the mean length of the common greedy token prefix, normalized by the decode budget,
which is a graded signal rather than the all-or-nothing exact match.  Greedy decoding is required
on every arm; a sampled arm would measure the sampler instead.

The NInfer arms are served by a running `ninfer-serve` with the adapter registered, so both come
from one load and exercise the per-request routing path.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path


def load_prompts(path: Path, limit: int) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [row["problem"] for row in rows[:limit]]


def strip_thinking(text: str) -> str:
    """Remove a leading reasoning block so both runners are compared on the same surface.

    NInfer parses reasoning out of `message.content`, while `transformers` returns the raw decode.
    An adapter fitted on chain-of-thought data emits `<think>` spontaneously, so without this the
    comparison measures the two runners' output conventions instead of the adapter.
    """
    body = text.lstrip()
    if not body.startswith("<think>"):
        return text.strip()
    end = body.find("</think>")
    return (body[end + len("</think>"):] if end != -1 else "").strip()


def common_prefix_tokens(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def reasoning_rate(texts: list[str]) -> float:
    """Fraction of replies that open a reasoning block.

    For an adapter fitted on chain-of-thought data this is the single most legible behaviour it
    learned, and it needs no normalization to interpret.
    """
    return sum(1 for t in texts if "<think>" in t) / len(texts) if texts else 0.0


def agreement(left: list[list[int]], right: list[list[int]], budget: int) -> dict:
    """Mean normalized common greedy prefix, plus the exact-match rate for reference."""
    prefixes = [common_prefix_tokens(a, b) for a, b in zip(left, right)]
    exact = sum(1 for a, b in zip(left, right) if a == b)
    denominators = [min(len(a), len(b), budget) or 1 for a, b in zip(left, right)]
    normalized = [p / d for p, d in zip(prefixes, denominators)]
    return {
        "n": len(prefixes),
        "mean_common_prefix_tokens": sum(prefixes) / len(prefixes) if prefixes else 0.0,
        "mean_normalized_prefix": sum(normalized) / len(normalized) if normalized else 0.0,
        "exact_match_rate": exact / len(prefixes) if prefixes else 0.0,
    }


def ninfer_tokens(endpoint: str, model: str, prompt: str, budget: int) -> list[int]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": budget,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.load(response)
    return payload["choices"][0]["message"]["content"]


def run_ninfer_arm(endpoint: str, model: str, prompts: list[str], budget: int) -> list[str]:
    outputs = []
    for index, prompt in enumerate(prompts):
        try:
            outputs.append(ninfer_tokens(endpoint, model, prompt, budget))
        except urllib.error.HTTPError as error:
            raise SystemExit(f"{model}: HTTP {error.code} on prompt {index}: "
                             f"{error.read().decode('utf-8', 'replace')[:200]}") from error
        if (index + 1) % 10 == 0:
            print(f"  {model}: {index + 1}/{len(prompts)}", flush=True)
    return outputs


def run_bnb_arms(base: Path, adapter: Path, prompts: list[str], budget: int,
                 seq_length: int) -> tuple[list[str], list[str]]:
    """Both bitsandbytes arms from one load; the adapter is toggled, not reloaded."""
    import unsloth  # noqa: F401  isort:skip
    from unsloth import FastModel  # isort:skip
    import torch
    from peft import PeftModel

    # `text_only=True` must match training: it collapses `model.language_model.layers.N.*` to
    # `model.layers.N.*`, which is the prefix the saved adapter keys carry. Loading without it
    # leaves PEFT nothing to match and silently attaches no adapter at all.
    model, processor = FastModel.from_pretrained(
        str(base), max_seq_length=seq_length, load_in_4bit=True, dtype=None,
        full_finetuning=False, text_only=True,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    inner_class = type(model).__name__
    # transformers' `load_adapter` runs `caching_allocator_warmup`, which reserves memory sized for
    # the whole model and OOMs against an already-resident 4-bit base.
    model = PeftModel.from_pretrained(model, str(adapter))
    # Unsloth's patched `generate` iterates `config.architectures`, which `text_only=True` leaves
    # unset on this checkpoint. Name it after the concrete class actually loaded.
    if getattr(model.config, "architectures", None) is None:
        model.config.architectures = [inner_class]
    model.eval()

    def generate(prompt: str) -> str:
        text = processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        batch = tokenizer(text=text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=budget, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        return tokenizer.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True)

    def sweep(label: str) -> list[str]:
        outputs = []
        for index, prompt in enumerate(prompts):
            outputs.append(generate(prompt))
            if (index + 1) % 10 == 0:
                print(f"  {label}: {index + 1}/{len(prompts)}", flush=True)
        return outputs

    with model.disable_adapter():
        base_outputs = sweep("bnb_base")
    return base_outputs, sweep("bnb_adapter")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", required=True, type=Path, help="JSONL with a `problem` field")
    parser.add_argument("--base", required=True, type=Path, help="BF16 checkpoint directory")
    parser.add_argument("--adapter", required=True, type=Path, help="PEFT adapter directory")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8231")
    parser.add_argument("--served-model", required=True, help="base model id, e.g. qwen3.8-27b")
    parser.add_argument("--served-adapter-model", required=True,
                        help="adapter model id, e.g. qwen3.8-27b-math")
    # The 24 GB card cannot hold the served artifact and the bitsandbytes base at once, so the
    # arms run in separate phases against a shared cache.
    parser.add_argument("--phase", choices=("ninfer", "bnb", "report", "all"), default="all")
    parser.add_argument("--cache", type=Path, default=Path("lora_transfer_cache.json"))
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--budget", type=int, default=48, help="greedy decode budget in tokens")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    prompts = load_prompts(arguments.eval, arguments.limit)
    print(f"{len(prompts)} evaluation prompts, {arguments.budget}-token greedy budget")

    cache: dict[str, list[str]] = {}
    if arguments.cache.exists():
        cache = json.loads(arguments.cache.read_text(encoding="utf-8"))

    if arguments.phase in ("ninfer", "all"):
        print("arms A/B: groupwise-int through ninfer-serve")
        cache["ninfer_base"] = run_ninfer_arm(arguments.endpoint, arguments.served_model, prompts,
                                              arguments.budget)
        cache["ninfer_adapter"] = run_ninfer_arm(arguments.endpoint,
                                                 arguments.served_adapter_model, prompts,
                                                 arguments.budget)
        arguments.cache.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    if arguments.phase in ("bnb", "all"):
        print("arms C/D: bitsandbytes NF4 through transformers")
        cache["bnb_base"], cache["bnb_adapter"] = run_bnb_arms(
            arguments.base, arguments.adapter, prompts, arguments.budget,
            arguments.max_seq_length)
        arguments.cache.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    missing = [k for k in ("ninfer_base", "ninfer_adapter", "bnb_base", "bnb_adapter")
               if k not in cache]
    if missing:
        print(f"cached arms so far: {sorted(cache)}; still needed: {missing}")
        return
    ninfer_base = [strip_thinking(t) for t in cache["ninfer_base"]]
    ninfer_adapter = [strip_thinking(t) for t in cache["ninfer_adapter"]]
    bnb_base = [strip_thinking(t) for t in cache["bnb_base"]]
    bnb_adapter = [strip_thinking(t) for t in cache["bnb_adapter"]]
    dropped = sum(1 for t in cache["bnb_adapter"] if t.lstrip().startswith("<think>")
                  and "</think>" not in t)
    if dropped:
        print(f"note: {dropped}/{len(bnb_adapter)} bnb_adapter replies were still inside an "
              f"unterminated reasoning block at the decode budget and compare as empty")

    # Compare on characters: the two runners do not share a detokenizer, and the token ids of one
    # are not observable through the other's API.
    def as_units(values: list[str]) -> list[list[int]]:
        return [[ord(c) for c in text] for text in values]

    budget_chars = arguments.budget * 8
    agree_base = agreement(as_units(ninfer_base), as_units(bnb_base), budget_chars)
    agree_adapter = agreement(as_units(ninfer_adapter), as_units(bnb_adapter), budget_chars)
    # How much the adapter changes its own base, as a sanity check that it does anything at all.
    effect_ninfer = agreement(as_units(ninfer_base), as_units(ninfer_adapter), budget_chars)
    effect_bnb = agreement(as_units(bnb_base), as_units(bnb_adapter), budget_chars)

    base_value = agree_base["mean_normalized_prefix"]
    retention = (agree_adapter["mean_normalized_prefix"] / base_value) if base_value > 0 else None

    report = {
        "format": "ninfer_lora_transfer_v1",
        "metric": "normalized common greedy prefix between the two bases, in characters",
        "prompts": len(prompts),
        "budget_tokens": arguments.budget,
        "adapter": str(arguments.adapter.resolve()),
        "cross_base_agreement_without_adapter": agree_base,
        "cross_base_agreement_with_adapter": agree_adapter,
        "adapter_effect_on_groupwise_int": effect_ninfer,
        "adapter_effect_on_bnb_nf4": effect_bnb,
        "retention": retention,
        "reasoning_block_rate": {
            "ninfer_base": reasoning_rate(cache["ninfer_base"]),
            "ninfer_adapter": reasoning_rate(cache["ninfer_adapter"]),
            "bnb_base": reasoning_rate(cache["bnb_base"]),
            "bnb_adapter": reasoning_rate(cache["bnb_adapter"]),
        },
        "samples": {
            "ninfer_base": ninfer_base[0][:200] if ninfer_base else "",
            "ninfer_adapter": ninfer_adapter[0][:200] if ninfer_adapter else "",
            "bnb_base": bnb_base[0][:200] if bnb_base else "",
            "bnb_adapter": bnb_adapter[0][:200] if bnb_adapter else "",
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if arguments.output:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
