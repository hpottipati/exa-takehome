#!/usr/bin/env python3
"""
Run all evaluations for the search provider benchmarking system.
Usage: python3 run_all_evals.py
"""

import subprocess
import sys
import os
from datetime import datetime
import dotenv

def run_command(cmd, description):
    """Run a command with real-time output"""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    try:
        # Use Popen for real-time output
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')  # Line already has newline
            sys.stdout.flush()
        
        # Wait for process to complete
        return_code = process.wait(timeout=600)  # 10 minute timeout
        
        return return_code == 0
    except subprocess.TimeoutExpired:
        print(f"\n❌ Timeout after 10 minutes: {description}")
        process.kill()
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    """Run all evaluations"""
    
    # Load environment variables
    dotenv.load_dotenv()
    
    # Check environment
    required_vars = ['GROQ_API_KEY', 'EXA_API_KEY', 'GOOGLE_API_KEY', 
                     'GOOGLE_SEARCH_ENGINE_ID', 'SERP_API_KEY']
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print("❌ Missing environment variables:")
        for var in missing:
            print(f"   export {var}='your-key-here'")
        print("\nLoad from .env file or set manually")
        sys.exit(1)
    
    print(f"\n🚀 SEARCH PROVIDER BENCHMARKING SYSTEM")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # All providers to test
    providers = "exa google serp_google serp_bing serp_duckduckgo"
    
    # Run evaluations
    evaluations = [
        # Authority evaluation (faster, no LLM)
        (
            f"python3 tests/authority/run_all_authority_tests.py",
            "Authority Evaluation - Tests domain authority and source quality"
        ),
        
        # Context evaluation (slower, uses LLM + semantic similarity)
        (
            f"python3 tests/context/run_batch_context_eval.py --providers {providers}",
            "Context Evaluation - Tests search relevance, answer quality, and support ratio with semantic similarity"
        )
    ]
    
    results = []
    for cmd, desc in evaluations:
        success = run_command(cmd, desc)
        results.append((desc.split(' - ')[0], success))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    print(f"\n✨ Results saved in: data/")
    print(f"   - Authority: data/authority_results/latest_batch/")
    print(f"   - Context:   data/context_results/latest_batch/")
    
    print(f"\n⏱️  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    sys.exit(0 if all(r[1] for r in results) else 1)

if __name__ == "__main__":
    main()