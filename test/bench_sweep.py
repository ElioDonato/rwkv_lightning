#!/usr/bin/env python3
"""Concurrency sweep for a running server.

Runs bench_throughput.run() at each N in --ns and prints a markdown table, so a
per-config scaling curve (tokens/s, p50 TTFT, p50 per-token ms, errors) can be
captured across N=1..128 and compared between configs/models.

Usage:
  .venv/bin/python test/bench_sweep.py --url http://127.0.0.1:PORT \
      --ns 1,2,4,8,16,32,64,128 --max-tokens 256
"""
import argparse
import asyncio

from bench_throughput import run  # sibling module in test/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--ns", default="1,2,4,8,16,32")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--password", default=None)
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",") if x]

    print(f"| N | reqs | ok | err | wall_s | tok/s | p50 TTFT ms | p50 ms/tok |")
    print(f"|---|------|----|-----|--------|-------|-------------|------------|")
    for n in ns:
        r = asyncio.run(run(args.url, n, n, args.max_tokens, args.password))
        print(
            f"| {n} | {r['completed']} | {r['completed']} | {r['errors']} | "
            f"{r['wall_s']} | {r['tokens_per_s']} | {r['p50_ttft_ms']} | "
            f"{r['p50_per_token_ms']} |"
        )


if __name__ == "__main__":
    main()