"""QLoRA training for the registered Qwen3.8-27B LoRA site table.

Trains an adapter that `tools/convert/qwen3_8_27b/convert_lora.py` can convert directly.  The
target module list is a structural limit rather than a shortcut: `gate_proj`/`up_proj` and the
Gated DeltaNet input projections are excluded because their deltas would have to land before
``silu(gate) * up`` and before the fused causal convolution, which no post-hoc additive pass can
express.  See ``docs/maintainer/qwen3.8-27b-lora-adapters.md`` section 3.

``--steps 0`` performs no optimizer step and writes the freshly initialized adapter.  PEFT
initializes ``lora_B`` to exactly zero, so that adapter is the exact tier-0 identity and is the
seam test for Unsloth's on-disk key naming.
"""

from __future__ import annotations

# Unsloth must be imported before transformers, peft and trl so its patches apply.
import unsloth  # noqa: F401  isort:skip
from unsloth import FastModel  # isort:skip

import argparse
import json
import os
from pathlib import Path


# The registered site table, expressed as the HuggingFace leaf modules that feed it.
# `q_proj` feeds two sites: the converter de-interleaves query and output-gate rows.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj"]

# `linear_attn.out_proj` is a registered site but shares the `o_proj` leaf name only on the
# attention layers; PEFT matches on the leaf, so it is named explicitly.
LINEAR_ATTENTION_OUTPUT = "out_proj"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="BF16 checkpoint directory")
    parser.add_argument("--out", required=True, type=Path, help="adapter output directory")
    parser.add_argument("--dataset", default="", help="HF dataset id; empty uses --steps 0")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-field", default="text")
    parser.add_argument("--prompt-column", default="", help="with --response-column, render a "
                        "two-turn chat into --dataset-field using the model chat template")
    parser.add_argument("--response-column", default="")
    parser.add_argument("--rank", type=int, default=16, choices=(8, 16, 32, 64))
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=0, help="0 writes the initialized adapter")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--include-linear-attention-output", action="store_true",
                        help="also target linear_attn.out_proj")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if not arguments.base.is_dir():
        raise SystemExit(f"base checkpoint directory does not exist: {arguments.base}")

    targets = list(TARGET_MODULES)
    if arguments.include_linear_attention_output:
        targets.append(LINEAR_ATTENTION_OUTPUT)

    # bfloat16 throughout: fp16 is rejected for this architecture by Unsloth's loader.
    model, tokenizer = FastModel.from_pretrained(
        str(arguments.base),
        max_seq_length=arguments.max_seq_length,
        load_in_4bit=True,
        text_only=True,
        dtype=None,
    )
    model = FastModel.get_peft_model(
        model,
        r=arguments.rank,
        lora_alpha=arguments.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        use_gradient_checkpointing="unsloth",
        use_rslora=False,
        loftq_config=None,
        random_state=arguments.seed,
    )

    if arguments.steps > 0:
        if not arguments.dataset:
            raise SystemExit("--steps > 0 requires --dataset")
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer

        dataset = load_dataset(arguments.dataset, split=arguments.dataset_split)
        if arguments.prompt_column and arguments.response_column:
            # The adapter must learn the same surface the engine serves, so the training text is
            # the model's own chat template rather than a bare concatenation.
            def render(batch: dict) -> dict:
                rendered = []
                for prompt, response in zip(batch[arguments.prompt_column],
                                            batch[arguments.response_column]):
                    rendered.append(tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": response}],
                        tokenize=False))
                return {arguments.dataset_field: rendered}

            dataset = dataset.map(render, batched=True, remove_columns=dataset.column_names)
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=SFTConfig(
                dataset_text_field=arguments.dataset_field,
                max_length=arguments.max_seq_length,
                per_device_train_batch_size=arguments.batch_size,
                gradient_accumulation_steps=arguments.accumulate,
                max_steps=arguments.steps,
                learning_rate=arguments.learning_rate,
                optim="adamw_8bit",
                lr_scheduler_type="linear",
                warmup_steps=min(10, max(1, arguments.steps // 10)),
                logging_steps=1,
                seed=arguments.seed,
                output_dir=str(arguments.out / "trainer"),
                report_to="none",
            ),
        )
        trainer.train()

    arguments.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(arguments.out))
    tokenizer.save_pretrained(str(arguments.out))

    report = {
        "base": str(arguments.base),
        "rank": arguments.rank,
        "alpha": arguments.alpha,
        "target_modules": targets,
        "steps": arguments.steps,
        "max_seq_length": arguments.max_seq_length,
        "seed": arguments.seed,
    }
    (arguments.out / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote adapter to {arguments.out}")
    for name in sorted(os.listdir(arguments.out)):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
