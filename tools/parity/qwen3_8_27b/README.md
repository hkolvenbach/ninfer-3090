# Qwen3.8-27B parity tools

## Vision

This tool compares the independent `.ninfer` Python reference with the source BF16 Vision tower at
matching semantic boundaries.

Compare quantized artifact Vision activations with the source BF16 tower:

```bash
python -m tools.parity.qwen3_8_27b.vision \
  --weights out/qwen3_8_27b.ninfer \
  --model-dir /path/to/Qwen3.8-27B/base-hf-bf16 \
  --messages messages.json
```

The diagnostic reports numerical differences directly; it does not materialize an activation dump
or require exact generated-token equality between independent implementations.

## LoRA transfer

An adapter is fitted against a bitsandbytes NF4 frozen base but served on top of the
`groupwise-int` artifact. `lora_transfer.py` measures whether the adapter keeps its behaviour
across that change, using a loud probe adapter trained to answer with a single JSON object.

It compares adapter **lift** on each base rather than the two served outputs, because the bases
already differ with no adapter attached:

```text
retention = [compliance(ninfer+adapter) - compliance(ninfer)]
          / [compliance(nf4+adapter)    - compliance(nf4)]
```

The two NInfer arms are served by one running `ninfer-serve` with the adapter registered, so they
share a single load and exercise per-request routing.

```bash
./build-sm89/apps/ninfer-serve models/qwen3_8_27b.ninfer --port 8231 --greedy --no-thinking \
  --lora loud=lora/loud.lora.ninfer &

python -m tools.parity.qwen3_8_27b.lora_transfer \
  --eval /tmp/opencode/loud/eval.jsonl \
  --base models/Qwen3.8-27B --adapter lora/peft_loud \
  --served-model qwen3.8-27b --served-adapter-model qwen3.8-27b-loud \
  --output lora_transfer.json
```

Compliance is measured on the response prefix, so a reply truncated by the token budget still
counts. The result characterizes transfer of the mechanism; it does not establish that a subtler
adapter's task gain survives, which remains a per-adapter question.
