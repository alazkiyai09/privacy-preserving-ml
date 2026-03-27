#!/usr/bin/env python3
"""
Master Test Runner for Robust Verifiable FL

Runs all unit tests for the project.
"""

import sys
import os
# Add parent directory to path to access src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest

# Set working directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules (relative to tests directory)
from test_attacks.test_label_flip import TestLabelFlipAttack
from test_attacks.test_backdoor import TestBackdoorAttack
from test_attacks.test_model_poisoning import TestModelPoisoningAttack
from test_defenses.test_byzantine import TestKrumAggregator, TestMultiKrumAggregator, TestTrimmedMeanAggregator
from test_defenses.test_anomaly_detection import TestZScoreDetector, TestClusteringDetector, TestCombinedAnomalyDetector
from test_defenses.test_reputation import TestClientReputationSystem
from test_interactions.test_zk_label_flip import TestZKLabelFlipInteraction
from test_interactions.test_combined_defenses import TestCombinedDefenses
from test_adaptive.test_adaptive_attacker import TestAdaptiveAttacker


def run_all_tests(verbose=True):
    """
    Run all tests and return results.

    Args:
        verbose: Whether to print detailed output

    Returns:
        Dictionary with test results
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestLabelFlipAttack))
    suite.addTests(loader.loadTestsFromTestCase(TestBackdoorAttack))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPoisoningAttack))
    suite.addTests(loader.loadTestsFromTestCase(TestKrumAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiKrumAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestTrimmedMeanAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestZScoreDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestClusteringDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestCombinedAnomalyDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestClientReputationSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestZKLabelFlipInteraction))
    suite.addTests(loader.loadTestsFromTestCase(TestCombinedDefenses))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveAttacker))

    # Run tests
    if verbose:
        runner = unittest.TextTestRunner(verbosity=2)
    else:
        runner = unittest.TextTestRunner(verbosity=0)

    result = runner.run(suite)

    # Compile results
    test_results = {
        "tests_run": result.testsRun,
        "successes": result.testsRun - len(result.failures) - len(result.errors),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
    }

    return test_results


def run_specific_test_category(category):
    """
    Run tests for a specific category.

    Args:
        category: One of 'attacks', 'defenses', 'interactions', 'adaptive', 'all'

    Returns:
        Test results
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if category == "attacks":
        suite.addTests(loader.loadTestsFromTestCase(TestLabelFlipAttack))
        suite.addTests(loader.loadTestsFromTestCase(TestBackdoorAttack))
        suite.addTests(loader.loadTestsFromTestCase(TestModelPoisoningAttack))
    elif category == "defenses":
        suite.addTests(loader.loadTestsFromTestCase(TestKrumAggregator))
        suite.addTests(loader.loadTestsFromTestCase(TestMultiKrumAggregator))
        suite.addTests(loader.loadTestsFromTestCase(TestTrimmedMeanAggregator))
        suite.addTests(loader.loadTestsFromTestCase(TestZScoreDetector))
        suite.addTests(loader.loadTestsFromTestCase(TestClusteringDetector))
        suite.addTests(loader.loadTestsFromTestCase(TestCombinedAnomalyDetector))
        suite.addTests(loader.loadTestsFromTestCase(TestClientReputationSystem))
    elif category == "interactions":
        suite.addTests(loader.loadTestsFromTestCase(TestZKLabelFlipInteraction))
        suite.addTests(loader.loadTestsFromTestCase(TestCombinedDefenses))
    elif category == "adaptive":
        suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveAttacker))
    elif category == "all":
        # Run everything
        suite.addTests(loader.loadTestsFromTestCase(TestLabelFlipAttack))
        suite.addTests(loader.loadTestsFromTestCase(TestBackdoorAttack))
        suite.addTests(loader.loadTestsFromTestCase(TestModelPoisoningAttack))
        suite.addTests(loader.loadTestsFromTestCase(TestKrumAggregator))
        suite.addTests(loader.loadTestsFromTestCase(TestMultiKrumAggregator))
        suite.addTests(loader.loadTestsFromTestCase(TestTrimmedMeanAggregator))
        suite.addTests(loader.loadTestsFromTestCase(TestZScoreDetector))
        suite.addTests(loader.loadTestsFromTestCase(TestClusteringDetector))
        suite.addTests(loader.loadTestsFromTestCase(TestCombinedAnomalyDetector))
        suite.addTests(loader.loadTestsFromTestCase(TestClientReputationSystem))
        suite.addTests(loader.loadTestsFromTestCase(TestZKLabelFlipInteraction))
        suite.addTests(loader.loadTestsFromTestCase(TestCombinedDefenses))
        suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveAttacker))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return {
        "tests_run": result.testsRun,
        "successes": result.testsRun - len(result.failures) - len(result.errors),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
    }


def print_summary(results):
    """Print test summary."""
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {results['tests_run']}")
    print(f"Successes: {results['successes']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success rate: {results['success_rate']:.1%}")
    print("="*70)


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run unit tests")
    parser.add_argument("--category", type=str, default="all",
                       choices=["all", "attacks", "defenses", "interactions", "adaptive"])
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce verbosity")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("  ROBUST VERIFIABLE FL - UNIT TESTS")
    print("="*70)

    if args.category == "all":
        results = run_all_tests(verbose=not args.quiet)
        print_summary(results)

        # Exit with appropriate code
        sys.exit(0 if results['failures'] == 0 and results['errors'] == 0 else 1)
    else:
        print(f"\nRunning {args.category} tests...")
        results = run_specific_test_category(args.category)
        print_summary(results)

        sys.exit(0 if results['failures'] == 0 and results['errors'] == 0 else 1)


if __name__ == "__main__":
    main()
