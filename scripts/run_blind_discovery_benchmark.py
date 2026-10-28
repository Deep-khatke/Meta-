#!/usr/bin/env python3
"""
RoboForge CLI Runner: Blind Scientific Discovery Benchmark (Milestone 3D)

Executes the fair, blind discovery benchmark comparing:
1. Default APF controller
2. Random parameter search
3. Grid / coordinate search
4. RoboForge Autonomous Research Agent

Usage:
    python scripts/run_blind_discovery_benchmark.py [--budget 4] [--output benchmarks/discovery/benchmark_summary.md]
"""

import sys
import os
import argparse
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.research.blind_discovery_evaluator import blind_discovery_evaluator
from backend.research.reasoning_provider import get_reasoning_provider


def main():
    parser = argparse.ArgumentParser(
        description="RoboForge Blind Scientific Discovery Benchmark Runner"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=4,
        help="Trial candidate budget per strategy (default: 4 candidates x 3 seeds = 12 runs)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick mode with single seed per candidate (faster for testing)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of markdown"
    )
    args = parser.parse_args()

    provider = get_reasoning_provider()
    print("=" * 72)
    print(" RoboForge — Blind Scientific Discovery Benchmark (Milestone 3D)")
    print("=" * 72)
    print(f"Active Reasoning Provider : {provider.name} ({provider.provider_type})")
    print(f"Candidate Budget per Method: {args.budget} candidates")
    print(f"Tuning Seeds               : [100, 101, 102]" if not args.quick else "[100]")
    print(f"Held-Out Seeds             : [200, 201, 202]" if not args.quick else "[200]")
    print(f"Anti-Leakage Isolation     : STRICT (Held-out environments excluded from memory)")
    print("=" * 72)
    print("Running simulation benchmarks across all 4 strategies...")

    tuning_seeds = [100] if args.quick else [100, 101, 102]
    held_out_seeds = [200] if args.quick else [200, 201, 202]

    t0 = time.time()
    report = blind_discovery_evaluator.run_benchmark(
        candidate_budget=args.budget,
        tuning_seeds=tuning_seeds,
        held_out_seeds=held_out_seeds,
        save_artifacts=True
    )
    elapsed = time.time() - t0

    print(f"\nBenchmark completed successfully in {elapsed:.1f}s.")
    print("=" * 72)

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        summary_md = blind_discovery_evaluator.generate_markdown_summary(report)
        print(summary_md)

    print("\n" + "=" * 72)
    print(f"Artifacts saved to: benchmarks/discovery/")
    print(f"  - Scenarios   : benchmarks/discovery/scenarios.json")
    print(f"  - Full Report : benchmarks/discovery/benchmark_report.json")
    print(f"  - Summary     : benchmarks/discovery/benchmark_summary.md")
    print("=" * 72)


if __name__ == "__main__":
    main()
