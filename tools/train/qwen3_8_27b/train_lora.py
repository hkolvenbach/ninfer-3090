"""QLoRA training for the Qwen3.8-27B LoRA module table.

By default this trains the complete registered site table: the six HF leaf modules in
``TARGET_MODULES``, which `tools/convert/qwen3_8_27b/convert_lora.py` converts into seven sites,
42,205,184 parameters and 368 objects at ``--rank 16``.

``MODULE_NOTES`` documents every module this architecture exposes, registered or not: what each
one computes, where a low-rank delta would have to land, what it costs, and what it is expected
to buy.  Four modules are trainable through ``--extra-modules`` but are **not yet convertible**.
They are documented and reachable so the training-side experiment can run before the engine work
does; `convert_lora.py` rejects the resulting adapter by name.

One property of such a run is worth knowing in advance.  Naming either vocabulary endpoint puts
PEFT's ``save_embedding_layers`` into effect, because ``EMBEDDING_LAYER_NAMES`` is
``["embed_tokens", "lm_head"]`` and either one triggers it, so the adapter additionally stores
*both* full ``[248320, 5120]`` base matrices.  A measured ``--steps 0`` run with all four extra
modules wrote 5,471,486,944 bytes: roughly 5.1 GB of base copy around 193 MB of low-rank planes.
Registering these two sites therefore means skipping ``base_layer.weight`` keys at conversion,
not only adding the planes.

The module names do not map onto a dense transformer, so read the shape of the model first.  Of
64 layers, 16 are full attention (3, 7, 11 ... 63) and 48 are Gated DeltaNet.  ``q_proj`` carries
the query and the per-head output gate interleaved, 24 heads x (256 query rows + 256 gate rows),
so it feeds two sites off one shared ``A``.  ``out_proj`` is ``linear_attn.out_proj``, GDN's
analogue of ``o_proj``; PEFT matches on the leaf name and it does not collide with
``self_attn.o_proj``.  Unsloth's published recipes target dense Llama/Mistral and have no
equivalent of it, which is why their module list is not this one.

Continued pretraining already works with the registered six.  CPT is raw text under full-token
loss, which is ``--dataset-field text`` without ``--completion-only``; no part of the module set
blocks it.  Two limits are real and are not removed by adding modules.  The rank ceiling is
``ops::kMaximumLoraRank`` = 64, below what CPT usually wants.  And the vocabulary is fixed at
248320 rows with the tokenizer SHA-256 pinned into the base artifact, so the canonical CPT case -
teaching a new language by adding tokens - needs a new base artifact, not an adapter.  Weigh both
against the published finding that LoRA loses the most ground to full finetuning in exactly this
regime; it is strongest on style, format and instruction following, weakest on knowledge
injection.

Every adapter registered in one engine must share one rank and one site inventory, so an adapter
trained against a narrower table cannot be served alongside one trained here.

``--steps 0`` performs no optimizer step and writes the freshly initialized adapter, which is the
seam test for Unsloth's on-disk key naming.  It is an exact tier-0 identity, but the zero plane
differs by site kind: PEFT zero-initializes ``lora_B`` for a Linear and ``lora_embedding_A`` for
an Embedding, so an ``--extra-modules embed_tokens`` run is an identity through its ``A`` plane
rather than its ``B``.

Check the GDN fast path before a long run.  ``fla`` and ``causal_conv1d`` can both import cleanly
while 48 of the 64 layers still run their PyTorch fallback, and the only symptom is one line at
startup: "The fast path is not available because one of the required library is not installed".
Training is a full-sequence forward and backward rather than a decode loop, so it pays more for
that fallback than generation does::

    python3 -c "import unsloth, transformers.models.qwen3_5.modeling_qwen3_5 as m; \\
        print(m.is_fast_path_available)"

Importing unsloth first is required -- the veto is Unsloth's import-time patch, not transformers'.
`docs/maintainer/qwen3.8-27b-lora-adapters.md` section 2.3 has the diagnosis and the repair.
"""

from __future__ import annotations

# Unsloth must be imported before transformers, peft and trl so its patches apply.
import unsloth  # noqa: F401  isort:skip
from unsloth import FastModel  # isort:skip

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path


# The complete registered site table, as the HuggingFace leaf modules that feed it. Together these
# cover the attention block in all 16 full-attention layers, the GDN output projection in all 48
# GDN layers, and the down projection in all 64. `MODULE_NOTES` below records what each one
# computes and costs; changing this list changes which adapters can share one bank.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "out_proj"]


@dataclass(frozen=True, slots=True)
class ModuleNote:
    """What one HF leaf module computes, and what adapting it costs and buys.

    `parameters` is the rank-16 count across every layer the module occupies.  It scales exactly
    with rank, because every plane is `rank x in_features` or `out_features x rank`; use
    `parameters_at_rank` rather than the raw field when reporting a run.
    """

    role: str
    destination: str
    layers: int
    parameters: int
    registered: bool
    gain: str
    blocker: str = ""


_REFERENCE_RANK = 16

# hidden 5120, intermediate 17408, 64 layers (16 full attention + 48 GDN), vocabulary 248320.
# Query and output gate are 24 heads x 256 rows each, so both `B` planes are [6144, r] over one
# shared [r, 5120] `A`.  Counts below are `layers * (rank * in_features + out_features * rank)`.
MODULE_NOTES: dict[str, ModuleNote] = {
    "q_proj": ModuleNote(
        role="Query and per-head output gate, interleaved. The query sets what a position looks "
             "for; the gate scales that head's contribution on the way out.",
        destination="q_flat and gate_flat, two plain BF16 destinations of ops::attn_input_proj",
        layers=16,
        parameters=4_456_448,       # 16 * (16*5120 + 2 * 6144*16)
        registered=True,
        gain="Feeds two registered sites off one shared A. Zeroing that A would corrupt the "
             "unmasked twin, which is why site masking zeroes B only.",
    ),
    "k_proj": ModuleNote(
        role="Keys - what each past token advertises to a query. Sets the retrieval matching "
             "criterion.",
        destination="k_flat",
        layers=16,
        parameters=1_572_864,       # 16 * (16*5120 + 1024*16)
        registered=True,
        gain="Smallest attention site. Matters for which tokens are found, not what is done "
             "with them.",
    ),
    "v_proj": ModuleNote(
        role="Values - the content actually retrieved and mixed across positions.",
        destination="v_flat",
        layers=16,
        parameters=1_572_864,       # 16 * (16*5120 + 1024*16)
        registered=True,
        gain="Changes what retrieval returns rather than what it selects.",
    ),
    "o_proj": ModuleNote(
        role="Projects mixed head outputs back to the residual stream: how retrieved content is "
             "written back.",
        destination="the residual x, through ops::linear_add",
        layers=16,
        parameters=2_883_584,       # 16 * (16*6144 + 5120*16)
        registered=True,
        gain="The attention block's write port. Reaches only the 16 full-attention layers.",
    ),
    "out_proj": ModuleNote(
        role="linear_attn.out_proj - Gated DeltaNet's analogue of o_proj. Writes the recurrent "
             "state readout back to the residual in the 48 GDN layers.",
        destination="the residual x, through ops::linear_add",
        layers=48,
        parameters=8_650_752,       # 48 * (16*6144 + 5120*16)
        registered=True,
        gain="Absent from Unsloth's dense recipes. Two paired A/B runs at r=16 measured no "
             "significant quality change from adding it; it is registered because the "
             "token-mixing path should not be economized on in three quarters of the layers.",
    ),
    "down_proj": ModuleNote(
        role="Reads the SwiGLU activation back to the residual: the value side of FFN key-value "
             "memory - what the addressed channels emit.",
        destination="the residual x, through ops::linear_add, applied inside the package "
                    "post-mixer leaf because the activation never leaves it",
        layers=64,
        parameters=23_068_672,      # 64 * (16*17408 + 5120*16)
        registered=True,
        gain="55% of the registered table and the only site spanning all 64 layers. Measured to "
             "carry per-item lexical behaviour, acquired in frequency order.",
    ),
    "gate_proj": ModuleNote(
        role="Produces g in h = silu(g) * u. After SiLU it is a soft switch over 17408 channels: "
             "the addressing (key) side of FFN key-value memory - which memories fire.",
        destination="the pre-activation g inside ops::linear_swiglu. The gate and up weights are "
                    "one packed [34816, 5120] tensor and the intermediate is not materialized on "
                    "the decode or MmaSplitHalfPair routes, so no post-hoc additive pass reaches "
                    "it; silu(g+dg)*(u+du) does not decompose into silu(g)*u + f(dg,du).",
        layers=64,
        parameters=23_068_672,      # 64 * (16*5120 + 17408*16)
        registered=False,
        gain="Highest of the four, on two grounds. Published placement studies concentrate LoRA "
             "benefit in the MLP, and this target's own site ablation found the adapter's "
             "learned behaviour concentrated in mlp/down - the value side - leaving no lever on "
             "addressing. Both are inferences from dense transformers; no published placement "
             "ablation exists for a GDN hybrid.",
        blocker="needs an optional pre-activation addend in every linear_swiglu route",
    ),
    "up_proj": ModuleNote(
        role="Produces u in h = silu(g) * u: the content the gated channels carry.",
        destination="the pre-activation u inside ops::linear_swiglu, same packed tensor and same "
                    "obstruction as gate_proj",
        layers=64,
        parameters=23_068_672,      # 64 * (16*5120 + 17408*16)
        registered=False,
        gain="Paired with gate_proj; the literature's MLP arm is both together with down_proj. "
             "Adding both doubles the adapter, which is the main cost to weigh.",
        blocker="needs an optional pre-activation addend in every linear_swiglu route",
    ),
    "lm_head": ModuleNote(
        role="Final projection from the hidden state to logits over 248320 rows: what the model "
             "is willing to say.",
        destination="the logits tensor - already a plain contiguous BF16 [248320, T] that "
                    "satisfies the ops::lora_delta_add destination contract",
        layers=0,
        parameters=4_055_040,       # 16*5120 + 248320*16
        registered=False,
        gain="Moderate and narrow. It reshapes the output distribution directly, which suits "
             "register, refusal style and vocabulary preference, and it is cheap. It cannot "
             "change what the model computes, only how that is read out. One hazard: the "
             "speculative draft head is a row gather of lm_head, so a delta applied to one and "
             "not the other desynchronizes proposals from the target and costs MTP acceptance.",
        blocker="policy, not an Op limit - the site table and artifact object names are "
                "layer-indexed and have no slot for a non-layer site",
    ),
    "embed_tokens": ModuleNote(
        role="Token id to initial hidden state, a lookup over 248320 rows: what a token means on "
             "entry.",
        destination="none today - ops::embedding is a gather, not a matmul, so there is no "
                    "[N, T] destination for B @ (A @ x). It needs a new Op computing "
                    "out[:, t] += B @ A[:, ids[t]].",
        layers=0,
        parameters=4_055_040,       # 16*248320 + 5120*16
        registered=False,
        gain="Lowest of the four. Unsloth recommends it for continued pretraining, but that "
             "recommendation is about learning new tokens for a new language, and this product "
             "cannot add tokens: the vocabulary is fixed and the tokenizer is pinned into the "
             "base artifact. What remains is re-weighting the meaning of existing tokens.",
        blocker="needs a new Op and the same non-layer site slot as lm_head",
    ),
}

OPTIONAL_MODULES: dict[str, ModuleNote] = {
    name: note for name, note in MODULE_NOTES.items() if not note.registered
}

# `peft.utils.constants.EMBEDDING_LAYER_NAMES`. Naming either one sets `save_embedding_layers`,
# which writes both full base matrices into the adapter alongside the low-rank planes.
_PEFT_EMBEDDING_LAYER_NAMES = ("embed_tokens", "lm_head")

REGISTERED_PARAMETERS = sum(note.parameters for note in MODULE_NOTES.values() if note.registered)


def parameters_at_rank(parameters: int, rank: int) -> int:
    """Exact parameter count at `rank`; every plane is `rank x in` or `out x rank`."""

    return parameters * rank // _REFERENCE_RANK


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
    parser.add_argument(
        "--extra-modules",
        nargs="+",
        default=(),
        choices=sorted(OPTIONAL_MODULES),
        metavar="MODULE",
        help="also train modules that are not registered sites. They train here, but "
             "convert_lora.py rejects the resulting adapter by name, so the run is a "
             "training-side experiment only. See MODULE_NOTES for what each one costs and is "
             f"expected to buy. One or more of: {', '.join(sorted(OPTIONAL_MODULES))}",
    )
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

    extra_modules = sorted(set(arguments.extra_modules))
    targets = list(TARGET_MODULES) + extra_modules
    if extra_modules:
        extra_parameters = sum(MODULE_NOTES[name].parameters for name in extra_modules)
        print("training modules outside the registered site table; convert_lora.py will reject "
              "this adapter:")
        for name in extra_modules:
            note = MODULE_NOTES[name]
            print(f"  {name}: +{parameters_at_rank(note.parameters, arguments.rank):,} "
                  f"parameters - {note.blocker}")
        print(f"  {parameters_at_rank(REGISTERED_PARAMETERS, arguments.rank):,} registered -> "
              f"{parameters_at_rank(REGISTERED_PARAMETERS + extra_parameters, arguments.rank):,} "
              f"trainable at rank {arguments.rank}")
        if any(name in _PEFT_EMBEDDING_LAYER_NAMES for name in extra_modules):
            print("  a vocabulary endpoint is named, so PEFT will also write both full "
                  "[248320, 5120] base matrices into the adapter - about 5.1 GB")

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
        "extra_modules": extra_modules,
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
