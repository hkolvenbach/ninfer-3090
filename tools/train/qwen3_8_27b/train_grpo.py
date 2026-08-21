"""GRPO training for the Qwen3.8-27B LoRA module table, against a verifiable reward.

This is the reinforcement-learning peer of `train_lora.py`.  It trains the same registered site
table -- the six HF leaf modules in ``TARGET_MODULES``, which
`tools/convert/qwen3_8_27b/convert_lora.py` converts into seven sites -- so an adapter produced
here converts and serves exactly like an SFT adapter.  Only the objective differs: instead of
maximizing the likelihood of a written answer, it samples ``--group`` completions per problem,
scores each with a programmatic verifier, and pushes the policy toward the ones that scored well.

Rollouts run through the training model itself (``fast_inference=False``).  That is Unsloth's
documented route for this architecture, and it is also the only correct one available here: a
separate inference engine would have to hold the same weights, and no vLLM that knows ``qwen3_5``
is co-installable with Unsloth's ``transformers<=5.5.0`` pin.  It buys a property worth more than
throughput -- the behaviour policy *is* the target policy, so the importance ratio is exact and
the adapter is applied at the same seven sites during generation and during the gradient step.

Three properties of the reward are load-bearing, and none of them is accuracy.

GRPO's advantage is a within-group z-score, so a problem whose ``--group`` samples all score the
same contributes exactly zero gradient no matter how right or wrong they are.  What predicts
trainability is the fraction of groups that are *not* unanimous.  Measured on this base model at
``--group 4``, ``mini_sudoku`` at 6-8 empty cells scores 0.266 mean with 93.8% of groups mixed,
while ``n_queens`` scores 0.000 at every difficulty tried, including removing only one or two
queens.  A task the model never solves cannot be learned, and one it always solves teaches
nothing; ``--calibrate`` reports both numbers before committing a run to a task.

Prompts are rendered thinking-off, matching ``ninfer-serve --no-thinking``.  This is the same
invariant `train_lora.py` documents: an adapter fitted against one prompt surface does not fire
on the other.  It also keeps completions short, which is what makes the memory envelope fit.

Scoring calls ``reasoning_gym``'s ``score_answer`` in process.  OpenEnv's ``reasoning_gym_env``
wraps that identical call, but its episode is stateful and single-problem -- ``reset()`` selects
one entry and ``step()`` scores against it -- and a GRPO step scores tens of completions spanning
many different problems, each of which must be scored against the problem its own prompt carried.
That correspondence cannot be expressed through the env's interface, so the HTTP and container
layers would add a mid-run failure mode to a multi-hour job without changing a single reward.

Memory is the binding constraint, not time.  Generation alone peaks near 20.6 GiB of the card's
23.5 GiB at 32 concurrent sequences, so ``--generation-batch`` is the first thing to reduce, and
``--beta 0`` (the default) drops the reference-model pass entirely.

Check the GDN fast path before a long run.  ``fla`` and ``causal_conv1d`` can both import cleanly
while the recurrence still runs its PyTorch fallback, and the only symptom is one line at startup:
"The fast path is not available because one of the required library is not installed".  Rollouts
still produce correct rewards, just slower, so a multi-hour run can complete before anyone notices::

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
import re
import time
from pathlib import Path

import torch


# The complete registered site table, identical to `train_lora.py`. An adapter trained against a
# different table cannot share an engine's adapter bank with one trained here.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "out_proj"]

# Difficulty knobs per reasoning-gym task. These are the only keys `--task-config` may set; naming
# anything else is a typo that would otherwise be silently accepted by the dataset constructor.
TASK_KNOBS = {
    "mini_sudoku": ("min_empty", "max_empty"),
    "maze": ("min_dist", "max_dist", "min_grid_size", "max_grid_size"),
    "n_queens": ("n", "min_remove", "max_remove"),
    "sudoku": ("min_empty", "max_empty"),
    "rush_hour": ("min_moves", "max_moves"),
    "sokoban": ("min_w", "max_w", "min_h", "max_h", "min_boxes", "max_boxes"),
}

_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove a reasoning block before scoring.

    The run is thinking-off, so a block should not appear; if the policy emits one anyway it is
    surface, not answer, and the verifier would score it as a malformed grid.
    """
    return _THINKING.sub("", text).strip()


def parse_task_config(task: str, pairs: list[str]) -> dict:
    """Parse ``key=value`` difficulty overrides, rejecting keys the task does not define."""
    known = TASK_KNOBS.get(task)
    if known is None:
        raise SystemExit(f"task {task!r} has no registered difficulty knobs; add it to TASK_KNOBS")
    config: dict[str, int] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--task-config expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if key not in known:
            raise SystemExit(f"{task!r} has no knob {key!r}; known knobs are {', '.join(known)}")
        config[key] = int(raw)
    return config


def name_text_architecture(model) -> None:
    """Give the text-only config an ``architectures`` entry.

    ``text_only=True`` collapses Qwen3.8 onto ``Qwen3_5TextConfig``, which carries no
    ``architectures``.  `unsloth/models/vision.py` iterates that field unconditionally on the
    generate path to decide whether the model is a VLM, so without this every rollout raises
    ``TypeError: 'NoneType' object is not iterable``.  SFT never reaches it because SFT never
    generates.  Naming the text-only causal LM also resolves that VLM test to False, which is
    correct here.
    """
    inner = getattr(model, "base_model", model)
    configs = {id(c): c for c in (model.config, getattr(inner, "config", model.config))}
    for config in configs.values():
        if getattr(config, "architectures", None) is None:
            config.architectures = ["Qwen3_5ForCausalLM"]


def clear_generation_max_length(model) -> None:
    """Drop ``max_length`` from every generation config the rollout path can see.

    transformers decides whether the two length controls clash with::

        has_default_max_length = (
            kwargs.get("max_length") is None
            and (generation_config is None or generation_config.max_length is None)
            and self.generation_config.max_length is None
        )

    in `generation/utils.py`.  Qwen3.8 carries ``max_length`` = 262144 -- its full native context --
    on the runtime generation config, so the third clause is false and every single rollout logs
    "Both `max_new_tokens` and `max_length` seem to have been set", even though ``max_new_tokens``
    is the only length TRL ever sets.  Clearing it restores the intended path, in which transformers
    derives ``max_length`` from ``max_new_tokens`` plus the measured prompt length.  This is
    cosmetic for correctness -- ``max_new_tokens`` already took precedence -- but at one warning per
    generate() call it buries the reward log.
    """
    seen: dict[int, object] = {}
    candidate = model
    while candidate is not None and id(candidate) not in seen:
        seen[id(candidate)] = candidate
        config = getattr(candidate, "generation_config", None)
        if config is not None and getattr(config, "max_length", None) is not None:
            config.max_length = None
        candidate = getattr(candidate, "base_model", None) or getattr(candidate, "model", None)


def build_problems(task: str, config: dict, size: int, seed: int):
    import reasoning_gym

    dataset = reasoning_gym.create_dataset(task, size=size, seed=seed, **config)
    return dataset, [dataset[i] for i in range(size)]


def render(tokenizer, question: str, thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def calibrate(model, tokenizer, arguments) -> None:
    """Report mean reward and the fraction of groups that produce a nonzero advantage.

    The second number is the one that decides whether a task is trainable at all.
    """
    dataset, entries = build_problems(
        arguments.task, parse_task_config(arguments.task, arguments.task_config),
        arguments.calibrate_problems, arguments.seed,
    )
    prompts = [render(tokenizer, e["question"], arguments.thinking) for e in entries]
    expanded = [p for p in prompts for _ in range(arguments.group)]

    completions: list[str] = []
    started = time.time()
    for start in range(0, len(expanded), arguments.generation_batch):
        batch = expanded[start : start + arguments.generation_batch]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=arguments.max_completion_length,
                do_sample=True,
                temperature=arguments.temperature,
                top_p=arguments.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        width = encoded["input_ids"].shape[1]
        completions += tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)

    rewards = [
        [
            dataset.score_answer(strip_thinking(c), entry=entry)
            for c in completions[i * arguments.group : (i + 1) * arguments.group]
        ]
        for i, entry in enumerate(entries)
    ]
    flat = [r for group in rewards for r in group]
    mean = sum(flat) / len(flat)
    solved = sum(1 for r in flat if r >= 1.0) / len(flat)
    informative = sum(1 for g in rewards if len(set(g)) > 1) / len(rewards)
    print(
        f"task={arguments.task} config={parse_task_config(arguments.task, arguments.task_config)}\n"
        f"  mean reward        {mean:.3f}\n"
        f"  fully solved       {solved:.3f}\n"
        f"  informative groups {informative:.3f}   <- fraction contributing a nonzero advantage\n"
        f"  elapsed            {time.time() - started:.0f}s\n"
        f"  peak memory        {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GiB"
    )
    if informative < 0.20:
        print("  WARNING: nearly every group is unanimous; this task will not train as configured.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, type=Path, help="BF16 checkpoint directory")
    parser.add_argument("--out", type=Path, help="adapter output directory (required unless --calibrate)")
    parser.add_argument("--task", default="mini_sudoku", choices=sorted(TASK_KNOBS))
    parser.add_argument(
        "--task-config", nargs="*", default=["min_empty=6", "max_empty=8"],
        help="difficulty overrides as key=value; the default is the calibrated mini_sudoku band",
    )
    parser.add_argument("--problems", type=int, default=512, help="distinct problems in the training set")
    parser.add_argument("--group", type=int, default=4, help="completions sampled per problem (GRPO G)")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--rank", type=int, choices=(8, 16, 32, 64), default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.0,
                        help="KL coefficient; 0 drops the reference-model pass and its memory")
    parser.add_argument("--loss-type", default="grpo", choices=("grpo", "dr_grpo", "bnpo"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4, help="completions per device step")
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--generation-batch", type=int, default=16,
                        help="sequences per generate() call; the first knob to reduce on OOM")
    parser.add_argument(
        "--mask-truncated", action="store_true",
        help="drop completions that hit --max-completion-length from the loss. Off by default: a "
             "correct grid terminates in about 32 tokens, so a completion that runs to the cap has "
             "failed the task's format instruction and its zero reward is real signal. Masking it "
             "measured 50-75%% of samples discarded, which collapses the effective group size.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--thinking", action="store_true",
                        help="render with the thinking block open; default closed to match --no-thinking")
    parser.add_argument("--calibrate", action="store_true",
                        help="measure base reward and informative-group fraction, then exit")
    parser.add_argument("--calibrate-problems", type=int, default=16)
    arguments = parser.parse_args()

    if not arguments.calibrate and arguments.out is None:
        parser.error("--out is required unless --calibrate is given")
    if arguments.batch_size % arguments.group != 0:
        parser.error(f"--batch-size {arguments.batch_size} must be a multiple of --group {arguments.group}")

    task_config = parse_task_config(arguments.task, arguments.task_config)

    model, tokenizer = FastModel.from_pretrained(
        str(arguments.base),
        max_seq_length=arguments.max_seq_length,
        load_in_4bit=True,
        text_only=True,
        dtype=None,
    )
    name_text_architecture(model)
    clear_generation_max_length(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if arguments.calibrate:
        FastModel.for_inference(model)
        calibrate(model, tokenizer, arguments)
        return

    model = FastModel.get_peft_model(
        model,
        r=arguments.rank,
        lora_alpha=arguments.alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        use_rslora=False,
        loftq_config=None,
        random_state=arguments.seed,
    )
    name_text_architecture(model)
    clear_generation_max_length(model)

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    dataset, entries = build_problems(
        arguments.task, task_config, arguments.problems, arguments.seed
    )
    rows = {
        "prompt": [render(tokenizer, e["question"], arguments.thinking) for e in entries],
        "entry_index": list(range(len(entries))),
    }
    train_dataset = Dataset.from_dict(rows)

    def verifier_reward(completions, entry_index, **_) -> list[float]:
        """Score each completion against the problem its own prompt carried.

        TRL forwards surplus dataset columns as keyword arguments aligned with ``completions``,
        which is what preserves the completion-to-problem correspondence across a batch that
        spans many problems and repeats each one ``--group`` times.
        """
        return [
            float(dataset.score_answer(strip_thinking(text), entry=entries[index]))
            for text, index in zip(completions, entry_index)
        ]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=verifier_reward,
        train_dataset=train_dataset,
        args=GRPOConfig(
            output_dir=str(arguments.out / "trainer"),
            num_generations=arguments.group,
            generation_batch_size=arguments.generation_batch,
            max_prompt_length=arguments.max_prompt_length,
            max_completion_length=arguments.max_completion_length,
            temperature=arguments.temperature,
            top_p=arguments.top_p,
            beta=arguments.beta,
            loss_type=arguments.loss_type,
            mask_truncated_completions=arguments.mask_truncated,
            # TRL constructs GenerationConfig(**kwargs); its max_length default is 20, which is
            # also non-None and would re-trigger the clash warning. None is the only quiet value.
            generation_kwargs={"max_length": None},
            per_device_train_batch_size=arguments.batch_size,
            gradient_accumulation_steps=arguments.accumulate,
            max_steps=arguments.steps,
            learning_rate=arguments.learning_rate,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            warmup_steps=min(10, max(1, arguments.steps // 10)),
            logging_steps=1,
            seed=arguments.seed,
            report_to="none",
        ),
    )
    trainer.train()

    arguments.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(arguments.out))
    tokenizer.save_pretrained(str(arguments.out))
    history = [
        {k: v for k, v in entry.items() if k in ("step", "loss", "reward", "reward_std", "kl")}
        for entry in trainer.state.log_history
    ]
    (arguments.out / "training_report.json").write_text(
        json.dumps(
            {
                "objective": "grpo",
                "base": str(arguments.base),
                "task": arguments.task,
                "task_config": task_config,
                "problems": arguments.problems,
                "group": arguments.group,
                "rank": arguments.rank,
                "alpha": arguments.alpha,
                "target_modules": TARGET_MODULES,
                "steps": arguments.steps,
                "beta": arguments.beta,
                "loss_type": arguments.loss_type,
                "learning_rate": arguments.learning_rate,
                "temperature": arguments.temperature,
                "top_p": arguments.top_p,
                "max_prompt_length": arguments.max_prompt_length,
                "max_completion_length": arguments.max_completion_length,
                "batch_size": arguments.batch_size,
                "accumulate": arguments.accumulate,
                "generation_batch": arguments.generation_batch,
                "mask_truncated": arguments.mask_truncated,
                "thinking": arguments.thinking,
                "seed": arguments.seed,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "reward_history": history,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
