"""QLoRA training for the registered Qwen3.8-27B LoRA site table.

Trains an adapter that `tools/convert/qwen3_8_27b/convert_lora.py` can convert directly, targeting
the complete registered site table: all seven sites, 42,205,184 parameters and 368 objects at
``--rank 16``.  What is absent from that table is a structural limit rather than a shortcut.
``gate_proj``/``up_proj`` and the Gated DeltaNet input projections are excluded because their
deltas would have to land before ``silu(gate) * up`` and before the fused causal convolution,
which no post-hoc additive pass can express.  See
``docs/maintainer/qwen3.8-27b-lora-adapters.md`` section 3.

Every adapter registered in one engine must share one rank and one site inventory, so an adapter
trained against an earlier, narrower table cannot be served alongside one trained here.

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


# The complete registered site table, expressed as the HuggingFace leaf modules that feed it.
# `q_proj` feeds two sites: the converter de-interleaves query and output-gate rows.
#
# `out_proj` is `linear_attn.out_proj`, and it is what makes this table cover the whole model.
# Only 16 of the 64 layers are full attention; the other 48 are Gated DeltaNet. Omitting it adapts
# the token-mixing path in 16 layers and leaves the remaining 48 carrying `mlp.down_proj` alone,
# so three quarters of the model would get a single adapted projection. PEFT matches on the leaf
# name, and `linear_attn.out_proj` does not collide with `self_attn.o_proj`, so naming it here
# targets exactly the 48 GDN layers and nothing else.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "out_proj"]


def frozen_base_quantization(model) -> dict:
    """Describe the quantization of the frozen base the adapter was fitted against.

    The adapter is served on top of the `groupwise-int` artifact, not on top of this base, so the
    two codecs differ.  Neither the PEFT config nor the model config records what bitsandbytes
    actually applied, and without it the training-time base cannot be reconstructed later.
    """
    for _, module in model.named_modules():
        weight = getattr(module, "weight", None)
        state = getattr(weight, "quant_state", None)
        if state is None:
            continue
        nested = getattr(state, "state2", None)
        return {
            "kind": type(weight).__name__,
            "quant_type": getattr(state, "quant_type", None),
            "blocksize": getattr(state, "blocksize", None),
            "compute_dtype": str(getattr(state, "dtype", None)),
            "double_quant": bool(getattr(state, "nested", False)),
            "double_quant_blocksize": getattr(nested, "blocksize", None),
        }
    return {"kind": "unquantized"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="BF16 checkpoint directory")
    parser.add_argument("--out", required=True, type=Path, help="adapter output directory")
    parser.add_argument("--dataset", default="", help="HF dataset id, or 'json' with --data-files; "
                        "empty uses --steps 0")
    parser.add_argument("--data-files", default="", help="local file for --dataset json")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-field", default="text")
    parser.add_argument("--prompt-column", default="", help="with --response-column, render a "
                        "two-turn chat into --dataset-field using the model chat template")
    parser.add_argument("--response-column", default="")
    parser.add_argument("--messages-column", default="", help="render a message list column into "
                        "the TRL prompt/completion schema with --completion-column; keeps "
                        "multi-turn history and always takes the loss on the completion only")
    parser.add_argument("--completion-column", default="completion")
    parser.add_argument("--completion-only", action="store_true",
                        help="mask the prompt and take the loss on the response only")
    # The rendered prompt must match the surface the engine serves. With thinking enabled the
    # template leaves `<think>` open; the server's --no-thinking closes it immediately. An adapter
    # trained against one context does not fire in the other.
    parser.add_argument("--thinking", action="store_true",
                        help="render prompts with the thinking block open (default: closed, "
                             "matching ninfer-serve --no-thinking)")
    parser.add_argument("--rank", type=int, default=16, choices=(8, 16, 32, 64))
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=0, help="0 writes the initialized adapter")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if not arguments.base.is_dir():
        raise SystemExit(f"base checkpoint directory does not exist: {arguments.base}")

    targets = list(TARGET_MODULES)

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

        if arguments.data_files:
            dataset = load_dataset(arguments.dataset, data_files=arguments.data_files,
                                   split=arguments.dataset_split)
        else:
            dataset = load_dataset(arguments.dataset, split=arguments.dataset_split)
        text_field: str | None = arguments.dataset_field
        if arguments.messages_column:
            # A message list rather than a single user string, so a multi-turn row keeps its prior
            # assistant turns - which are already in the target register - as context. The loss is
            # always completion-only here: this schema exists to teach the response side.
            #
            # The list is rendered to a string in this process for the same reason as the
            # prompt/response path below: TRL's conversational branch routes a `messages` column to
            # the multimodal processor's `apply_chat_template`, which expects typed content parts
            # and fails on plain text. Rendering here also keeps one owner for the
            # training-surface-equals-serving-surface guarantee.
            def split_messages(batch: dict) -> dict:
                prompts, completions = [], []
                for messages, completion in zip(batch[arguments.messages_column],
                                                batch[arguments.completion_column]):
                    prompts.append(tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=arguments.thinking))
                    completions.append(completion)
                return {"prompt": prompts, "completion": completions}

            dataset = dataset.map(split_messages, batched=True,
                                  remove_columns=dataset.column_names)
            text_field = None
        elif arguments.prompt_column and arguments.response_column:
            if arguments.completion_only:
                # Emit TRL's prompt/completion schema so the loss is taken on the response only.
                # Training on the concatenation instead lets a long prompt dominate the gradient:
                # a ~30-token answer inside a ~600-token problem receives a few percent of the
                # signal, which is far too little to teach a response-side behaviour.
                #
                # The columns are rendered strings rather than message lists on purpose. TRL's
                # conversational branch would hand them to the multimodal processor's
                # `apply_chat_template`, which expects typed content parts and fails on plain
                # text. Rendering here also pins the training surface to the served one.
                def split(batch: dict) -> dict:
                    prompts, completions = [], []
                    for prompt, response in zip(batch[arguments.prompt_column],
                                                batch[arguments.response_column]):
                        prompts.append(tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False, add_generation_prompt=True,
                            enable_thinking=arguments.thinking))
                        completions.append(response)
                    return {"prompt": prompts, "completion": completions}

                dataset = dataset.map(split, batched=True, remove_columns=dataset.column_names)
                text_field = None
            else:
                # The adapter must learn the same surface the engine serves, so the training text
                # is the model's own chat template rather than a bare concatenation.
                def render(batch: dict) -> dict:
                    rendered = []
                    for prompt, response in zip(batch[arguments.prompt_column],
                                                batch[arguments.response_column]):
                        rendered.append(tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": response}],
                            tokenize=False, enable_thinking=arguments.thinking))
                    return {arguments.dataset_field: rendered}

                dataset = dataset.map(render, batched=True, remove_columns=dataset.column_names)
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=SFTConfig(
                dataset_text_field=text_field,
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
        "dataset": arguments.dataset,
        "data_files": arguments.data_files,
        "dataset_split": arguments.dataset_split,
        "messages_column": arguments.messages_column,
        # The messages schema is prompt/completion by construction, so its loss is completion-only
        # whether or not the flag was passed.
        "completion_only_loss": bool(arguments.completion_only or arguments.messages_column),
        "thinking": arguments.thinking,
        "frozen_base_quantization": frozen_base_quantization(model),
    }
    (arguments.out / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote adapter to {arguments.out}")
    for name in sorted(os.listdir(arguments.out)):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
