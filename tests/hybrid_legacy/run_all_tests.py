"""
Master Test Runner for HT2ML
==============================

Runs all test suites and generates comprehensive report.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import subprocess
import time


def run_test_suite(name, script_path):
    """Run a single test suite."""
    print(f"\n{'='*70}")
    print(f"Running {name}")
    print(f"{'='*70}\n")

    start_time = time.time()

    result = subprocess.run(
        ['python3', script_path],
        capture_output=True,
        text=True
    )

    elapsed = time.time() - start_time

    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Parse results
    output = result.stdout + result.stderr
    lines = output.split('\n')

    total = 0
    for line in lines:
        if 'Ran' in line and 'test' in line:
            parts = line.split()
            try:
                total = int(parts[1])
            except (ValueError, IndexError):
                pass

    success = result.returncode == 0

    return success, total, elapsed


def main():
    """Run all test suites."""
    print("\n" + "#"*70)
    print("# HT2ML Test Suite")
    print("# Comprehensive Testing for Hybrid HE/TEE System")
    print("#"*70)

    test_suites = [
        ("HE Operations", "tests/test_he_operations.py"),
        ("TEE Operations", "tests/test_tee_operations.py"),
        ("Protocol", "tests/test_protocol.py"),
        ("Inference", "tests/test_inference.py"),
    ]

    results = []
    total_tests = 0
    total_time = 0

    for suite_name, script_path in test_suites:
        success, total, elapsed = run_test_suite(suite_name, script_path)
        results.append((suite_name, success, total, elapsed))
        total_tests += total
        total_time += elapsed

    # Print summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    print(f"\n{'Suite':<20} {'Status':<12} {'Tests':<10} {'Time (s)':<10}")
    print("-"*70)

    for suite_name, success, total, elapsed in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{suite_name:<20} {status:<12} {total:<10} {elapsed:<10.2f}")

    print("-"*70)
    print(f"{'TOTAL':<20} {'':<12} {total_tests:<10} {total_time:<10.2f}")
    print()

    # Overall success
    all_passed = all(r[1] for r in results if r[2] > 0)
    passed_count = sum(1 for r in results if r[1])

    print(f"Results: {passed_count}/{len(results)} test suites passed")
    print(f"Total tests: {total_tests}")

    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n✗ SOME TESTS FAILED")

    print("\n" + "#"*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
