#!/usr/bin/env python3
"""
Run all authority evaluation tests for different search providers
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict
import argparse

def get_available_tests() -> Dict[str, str]:
    """Get available test scripts"""
    test_dir = Path(__file__).parent
    tests = {}
    
    for test_file in test_dir.glob("test_*.py"):
        if test_file.name != "run_all_authority_tests.py":
            provider = test_file.stem.replace("test_", "").replace("_", " ").title()
            tests[provider] = str(test_file)
    
    return tests

def run_test(test_path: str, provider: str) -> bool:
    """Run a single test script"""
    print(f"\n{'='*80}")
    print(f"RUNNING {provider.upper()} AUTHORITY EVALUATION")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            [sys.executable, test_path],
            cwd=Path(test_path).parent,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✓ {provider} evaluation completed successfully")
            return True
        else:
            print(f"\n✗ {provider} evaluation failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error running {provider} test: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run authority evaluation tests')
    parser.add_argument('--providers', nargs='+', help='Specific providers to test')
    parser.add_argument('--list', action='store_true', help='List available tests')
    args = parser.parse_args()
    
    # Get available tests
    available_tests = get_available_tests()
    
    if args.list:
        print("Available authority evaluation tests:")
        for provider in sorted(available_tests.keys()):
            print(f"  - {provider}")
        return 0
    
    # Determine which tests to run
    if args.providers:
        providers_to_run = []
        for provider in args.providers:
            # Try exact match first
            if provider in available_tests:
                providers_to_run.append(provider)
            else:
                # Try case-insensitive partial match
                matches = [p for p in available_tests.keys() 
                          if provider.lower() in p.lower()]
                if matches:
                    providers_to_run.extend(matches)
                else:
                    print(f"Warning: No test found for provider '{provider}'")
        
        if not providers_to_run:
            print("No valid providers specified. Use --list to see available tests.")
            return 1
    else:
        providers_to_run = list(available_tests.keys())
    
    print(f"Authority Evaluation Test Runner")
    print(f"Will run tests for: {', '.join(providers_to_run)}")
    
    # Run tests
    results = {}
    for provider in providers_to_run:
        test_path = available_tests[provider]
        success = run_test(test_path, provider)
        results[provider] = success
    
    # Print summary
    print(f"\n{'='*80}")
    print("AUTHORITY EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    successful = [p for p, success in results.items() if success]
    failed = [p for p, success in results.items() if not success]
    
    print(f"\nSuccessful evaluations ({len(successful)}):")
    for provider in successful:
        print(f"  ✓ {provider}")
    
    if failed:
        print(f"\nFailed evaluations ({len(failed)}):")
        for provider in failed:
            print(f"  ✗ {provider}")
    
    print(f"\nOverall: {len(successful)}/{len(results)} tests passed")
    
    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
