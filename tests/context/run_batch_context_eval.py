#!/usr/bin/env python3
"""
Run batch context evaluation across all scenarios
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.search.search_manager import SearchManager
from src.evaluators.context_evaluator import ContextEvaluator, generate_context_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_scenarios(file_path: str = None) -> List[Dict]:
    """Load test scenarios from JSON file"""
    if file_path is None:
        file_path = Path(__file__).parent.parent.parent / "data" / "scenarios.json"
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data['questions']


def save_results(results: Dict, output_path: Path):
    """Save results to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def run_batch_evaluation(
    providers: List[str] = None,
    query_ids: List[str] = None,
    max_queries: int = None,
    output_dir: str = None
):
    """
    Run context evaluation on multiple queries
    
    Args:
        providers: List of providers to test (default: exa, google)
        query_ids: Specific query IDs to test (default: all)
        max_queries: Maximum number of queries to test
        output_dir: Output directory for results
    """
    # Default providers
    if providers is None:
        providers = ['exa', 'google', 'serp_google', 'serp_duckduckgo']
    
    # Load scenarios
    scenarios = load_scenarios()
    
    # Filter scenarios if specific IDs provided
    if query_ids:
        scenarios = [q for q in scenarios if q['id'] in query_ids]
    
    # Limit number of queries if specified
    if max_queries:
        scenarios = scenarios[:max_queries]
    
    if not scenarios:
        logger.error("No scenarios to evaluate")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info(f"BATCH CONTEXT EVALUATION")
    logger.info(f"Queries: {len(scenarios)}")
    logger.info(f"Providers: {providers}")
    logger.info(f"{'='*80}\n")
    
    # Initialize components
    search_manager = SearchManager()
    evaluator = ContextEvaluator(ollama_model="qwen2.5:3b")
    
    # Create output directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        output_base = Path(__file__).parent.parent.parent / "data" / "context_results" / f"batch_{timestamp}"
    else:
        output_base = Path(output_dir)
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Store all evaluation results
    all_evaluations = []
    failed_queries = []
    
    # Process each query
    for i, query_data in enumerate(scenarios, 1):
        query_id = query_data['id']
        logger.info(f"\n{'='*60}")
        logger.info(f"Query {i}/{len(scenarios)}: {query_id}")
        logger.info(f"Question: {query_data['question'][:100]}...")
        logger.info(f"{'='*60}")
        
        try:
            # Run searches
            logger.info("Running searches...")
            providers_results = {}
            search_latencies = {}
            
            for provider in providers:
                try:
                    provider_start = time.time()
                    client = search_manager.get_client(provider)
                    results = client.search(query_data['search_query'], num_results=10)
                    search_latencies[provider] = (time.time() - provider_start) * 1000
                    providers_results[provider] = results
                    logger.info(f"  ✓ {provider}: {len(results)} results")
                except Exception as e:
                    logger.error(f"  ✗ {provider}: {e}")
                    providers_results[provider] = []
                    search_latencies[provider] = 0
            
            # Skip if no results
            if not any(providers_results.values()):
                logger.warning(f"No search results for {query_id}, skipping")
                failed_queries.append(query_id)
                continue
            
            # Run context evaluation
            logger.info("Running context evaluation...")
            eval_start = time.time()
            
            context_results = evaluator.evaluate_all_metrics(
                query=query_data['question'],
                providers_results=providers_results
            )
            
            # Add metadata
            context_results['query_id'] = query_id
            context_results['scenario'] = query_data['scenario']
            context_results['search_query'] = query_data['search_query']
            context_results['search_latencies_ms'] = search_latencies
            
            eval_time = (time.time() - eval_start) * 1000
            logger.info(f"  Evaluation completed in {eval_time:.1f}ms")
            
            # Save individual query results
            query_output = output_base / "individual_queries" / f"{query_id}.json"
            save_results(context_results, query_output)
            
            # Add to all evaluations
            all_evaluations.append(context_results)
            
            # Print quick summary
            if 'llm_win_rate' in context_results:
                win_rates = context_results['llm_win_rate'].get('win_rates', {})
                if win_rates:
                    winner = max(win_rates, key=win_rates.get)
                    logger.info(f"  Winner: {winner} ({win_rates[winner]:.1%} win rate)")
            
        except Exception as e:
            logger.error(f"Error evaluating {query_id}: {e}")
            failed_queries.append(query_id)
            continue
    
    # Generate aggregate report
    logger.info(f"\n{'='*80}")
    logger.info("Generating aggregate report...")
    
    if all_evaluations:
        report = generate_context_report(all_evaluations)
        
        # Add metadata
        report['metadata'] = {
            'timestamp': timestamp,
            'total_queries': len(scenarios),
            'successful_queries': len(all_evaluations),
            'failed_queries': failed_queries,
            'providers': providers
        }
        
        # Save report
        report_path = output_base / "aggregate_report.json"
        save_results(report, report_path)
        
        # Save all evaluations
        all_results_path = output_base / "all_evaluations.json"
        save_results(all_evaluations, all_results_path)
        
        # Print summary
        print_aggregate_summary(report)
        
        # Create symlink to latest batch
        latest_link = Path(__file__).parent.parent.parent / "data" / "context_results" / "latest_batch"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(output_base.name)
        
        logger.info(f"\n✓ All results saved to {output_base}")
        logger.info(f"  View latest batch at: data/context_results/latest_batch/")
        
    else:
        logger.error("No successful evaluations to report")
    
    return report if all_evaluations else None


def print_aggregate_summary(report: Dict):
    """Print summary of aggregate report"""
    print("\n" + "="*80)
    print("AGGREGATE CONTEXT EVALUATION REPORT")
    print("="*80)
    
    # Metadata
    metadata = report.get('metadata', {})
    print(f"\n📊 Evaluation Summary:")
    print(f"  Total Queries: {metadata.get('total_queries', 0)}")
    print(f"  Successful: {metadata.get('successful_queries', 0)}")
    print(f"  Failed: {len(metadata.get('failed_queries', []))}")
    
    # Provider rankings
    print(f"\n🏆 Provider Rankings:")
    
    summary = report.get('summary', {})
    if summary:
        # Sort by composite score
        composite_scores = report.get('composite_scores', {})
        sorted_providers = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (provider, score) in enumerate(sorted_providers, 1):
            metrics = summary[provider]
            print(f"\n  {rank}. {provider} (Composite: {score:.3f})")
            print(f"     Win Rate: {metrics['avg_win_rate']:.1%}")
            print(f"     Authority Cited: {metrics['avg_authority_cited']:.3f}")
            print(f"     Support Ratio: {metrics['avg_support_ratio']:.1%}")
            print(f"     Avg Generation Time: {metrics['avg_generation_time_ms']:.1f}ms")
    
    # Recommended provider
    if 'recommended_provider' in report:
        rec = report['recommended_provider']
        print(f"\n✅ Recommended Provider: {rec['name']}")
        print(f"   Composite Score: {rec['composite_score']:.3f}")
        print(f"   {rec['reasoning']}")
    
    print("\n" + "="*80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run batch context evaluation')
    parser.add_argument('--providers', nargs='+', 
                       default=['exa', 'google', 'serp_google', 'serp_duckduckgo'],
                       help='Providers to test')
    parser.add_argument('--query-ids', nargs='+', 
                       help='Specific query IDs to test (default: all)')
    parser.add_argument('--max-queries', type=int,
                       help='Maximum number of queries to test')
    parser.add_argument('--output-dir',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Test Ollama connection
    logger.info("Testing Ollama connection...")
    try:
        import ollama
        response = ollama.list()
        # Handle both dict and object responses
        if hasattr(response, 'models'):
            models = response.models
        elif isinstance(response, dict) and 'models' in response:
            models = response['models']
        else:
            models = []
        
        model_names = []
        for model in models:
            if isinstance(model, dict):
                model_names.append(model.get('name', ''))
            elif hasattr(model, 'name'):
                model_names.append(model.name)
        
        if not any('qwen2.5:3b' == name for name in model_names if name):
            logger.warning("qwen2.5:3b not found. Please run: ollama pull qwen2.5:3b")
            if model_names:
                logger.info(f"Available models: {model_names}")
            logger.info("Trying to continue anyway...")
        else:
            logger.info("✓ Ollama is running with qwen2.5:3b")
    except Exception as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        logger.error("Please ensure Ollama is running: ollama serve")
        logger.info("Trying to continue anyway...")
    
    # Run batch evaluation
    run_batch_evaluation(
        providers=args.providers,
        query_ids=args.query_ids,
        max_queries=args.max_queries,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()