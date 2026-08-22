#!/usr/bin/env python3
from __future__ import annotations

import argparse

import psutil


def main() -> None:
    parser = argparse.ArgumentParser(description="Check memory headroom before long MPS training.")
    parser.add_argument("--min-available-gb", type=float, default=5.0)
    parser.add_argument("--max-swap-used-gb", type=float, default=4.0)
    args = parser.parse_args()

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    gib = 1024**3
    available_gb = memory.available / gib
    swap_used_gb = swap.used / gib
    print(f"memory_available_gb={available_gb:.2f}")
    print(f"memory_used_percent={memory.percent:.1f}")
    print(f"swap_used_gb={swap_used_gb:.2f}")
    unsafe = available_gb < args.min_available_gb or swap_used_gb > args.max_swap_used_gb
    if unsafe:
        raise SystemExit(
            "Insufficient clean memory headroom for SKU-110K training. "
            "Restart macOS, close memory-heavy applications, and run this command again."
        )
    print("memory_preflight=passed")


if __name__ == "__main__":
    main()
