#!/usr/bin/env python3
"""Drive ninfer-serve prefill and report measured prefill throughput.

Sends one chat completion per target prompt size with max_tokens=1 so the
measurement isolates prefill. Reads the structured request log for the
server-side prefill rate rather than timing the HTTP round trip.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

WORDS = None


def corpus():
    global WORDS
    if WORDS is None:
        # Deterministic pseudo-text with a realistic token/word ratio and no
        # long repeated span, so nothing upstream can shortcut the prefill.
        src = "/usr/share/dict/words"
        if os.path.exists(src):
            WORDS = [w.strip() for w in open(src) if w.strip().isalpha()]
        else:
            WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
                     "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
                     "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
                     "victor", "whiskey", "xray", "yankee", "zulu"]
    return WORDS


def make_prompt(words, seed):
    rng = random.Random(seed)
    w = corpus()
    return " ".join(rng.choice(w) for _ in range(words))


def post(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.time() - t0


def tail_log(path, since):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            out.append(rec)
    return out[since:]


def find(rec, *names):
    """Locate the first matching key anywhere in a nested record."""
    stack = [rec]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k in names:
                    return v
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            stack.extend(x for x in cur if isinstance(x, (dict, list)))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--log", default="/tmp/ninfer-reqlog.jsonl")
    ap.add_argument("--targets", default="8192,32768,102400")
    ap.add_argument("--words-per-token", type=float, default=0.75)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--dmon", action="store_true", help="sample SM clock/power during the run")
    args = ap.parse_args()

    seen = len(tail_log(args.log, 0))
    print(f"{'label':10s} {'target':>8s} {'prompt_tok':>10s} {'prefill_tok_s':>14s} "
          f"{'ttft_ms':>10s} {'wall_s':>8s} {'sm_mhz':>7s} {'watt':>6s}")

    for target in [int(x) for x in args.targets.split(",")]:
        words = int(target * args.words_per_token)
        prompt = make_prompt(words, args.seed * 100003 + target)
        dmon = None
        if args.dmon:
            dmon = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader,nounits",
                 "-l", "1"], stdout=subprocess.PIPE, text=True)
        payload = {"model": args.model, "max_tokens": args.max_tokens,
                   "temperature": 0.0,
                   "messages": [{"role": "user", "content": prompt}]}
        try:
            body, wall = post(args.url, payload, args.timeout)
        except Exception as exc:
            print(f"{args.label:10s} {target:8d}  REQUEST FAILED: {exc}")
            if dmon:
                dmon.kill()
            continue
        clk = pw = float("nan")
        if dmon:
            dmon.kill()
            rows = [l.split(",") for l in dmon.stdout.read().splitlines() if "," in l]
            if rows:
                clk = max(int(r[0]) for r in rows)
                pw = max(float(r[1]) for r in rows)
        ptok = body.get("usage", {}).get("prompt_tokens", 0)
        time.sleep(0.5)
        recs = tail_log(args.log, seen)
        seen += len(recs)
        rate = ttft = float("nan")
        for rec in recs:
            r = find(rec, "prefill_tok_s", "prefill_tokens_per_second")
            if r:
                rate = r
                ttft = find(rec, "server_ttft_ms", "ttft_ms") or float("nan")
        if rate != rate and wall > 0:
            rate = ptok / wall
        print(f"{args.label:10s} {target:8d} {ptok:10d} {rate:14.1f} {ttft:10.1f} "
              f"{wall:8.2f} {clk:7.0f} {pw:6.1f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
