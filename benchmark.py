#!/usr/bin/env python3
"""
Quick benchmark script to test funcfind_30.py optimizations.
Runs a shorter version of the genetic algorithm to measure performance.
"""

import sys
import time
import os
from funcfind_30 import GeneticApproximator, GRID

# Reduced parameters for faster benchmarking
BENCHMARK_CONFIG = {
    "population_size": 100,      # Reduced from 1000
    "mutation_rate": 0.3,
    "survival_rate": 0.5,
    "max_depth": 6,              # Reduced from 7
    "complexity_weight": 1.0,
    "diversity_weight": 4.5,
    "generations": 100,          # Reduced from 10000
}

def print_system_info():
    """Print relevant system information."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"CPU cores: {os.cpu_count()}")

    # Check for free-threading
    if hasattr(sys, '_is_gil_enabled'):
        try:
            gil_enabled = sys._is_gil_enabled()
            if gil_enabled:
                print("Mode: Standard Python (GIL enabled)")
                print("Executor: ProcessPoolExecutor")
            else:
                print("Mode: Free-threaded Python 3.14t (no GIL!)")
                print("Executor: ThreadPoolExecutor with true parallelism")
        except:
            print("Mode: Standard Python")
            print("Executor: ProcessPoolExecutor")
    else:
        print("Mode: Python < 3.13 (standard GIL)")
        print("Executor: ProcessPoolExecutor")

    print()

def run_benchmark():
    """Run a benchmark of the genetic algorithm."""
    print("=" * 60)
    print("BENCHMARK CONFIGURATION")
    print("=" * 60)
    for key, value in BENCHMARK_CONFIG.items():
        print(f"{key}: {value}")
    print()

    print("=" * 60)
    print("RUNNING BENCHMARK")
    print("=" * 60)
    print("Starting genetic algorithm...")
    print()

    # Create approximator
    ga = GeneticApproximator(
        grid=GRID,
        **BENCHMARK_CONFIG
    )

    # Time the run
    start_time = time.time()

    # Run evolution
    best = ga.run(generations=BENCHMARK_CONFIG["generations"])

    end_time = time.time()
    elapsed = end_time - start_time

    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Generations: {BENCHMARK_CONFIG['generations']}")
    print(f"Time per generation: {elapsed / BENCHMARK_CONFIG['generations']:.3f} seconds")
    print(f"Best fitness: {best.fitness}")
    print()

    # Calculate throughput
    total_evaluations = BENCHMARK_CONFIG["population_size"] * BENCHMARK_CONFIG["generations"]
    evaluations_per_second = total_evaluations / elapsed
    print(f"Total evaluations: {total_evaluations:,}")
    print(f"Evaluations per second: {evaluations_per_second:,.0f}")
    print()

    # Performance rating
    if evaluations_per_second > 10000:
        rating = "EXCELLENT"
        emoji = "🚀"
    elif evaluations_per_second > 5000:
        rating = "VERY GOOD"
        emoji = "✨"
    elif evaluations_per_second > 2000:
        rating = "GOOD"
        emoji = "👍"
    elif evaluations_per_second > 1000:
        rating = "FAIR"
        emoji = "⚡"
    else:
        rating = "SLOW"
        emoji = "🐌"

    print(f"Performance: {rating} {emoji}")
    print()

    # Expected improvements with Python 3.14t
    if hasattr(sys, '_is_gil_enabled'):
        try:
            if sys._is_gil_enabled():
                print("💡 TIP: With Python 3.14t (free-threaded), expect:")
                estimated_3_14t = evaluations_per_second * 3  # Conservative 3x
                print(f"   ~{estimated_3_14t:,.0f} evaluations/second (3-5x faster)")
        except:
            pass

    print("=" * 60)
    return elapsed, best.fitness

def main():
    """Main benchmark entry point."""
    print()
    print_system_info()

    try:
        elapsed, fitness = run_benchmark()

        print("\n✅ Benchmark completed successfully!\n")

        # Save results
        with open("benchmark_results.txt", "a") as f:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | Python {sys.version_info.major}.{sys.version_info.minor} | "
                   f"Time: {elapsed:.2f}s | Fitness: {fitness}\n")

        print("Results appended to benchmark_results.txt")

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
