#!/usr/bin/env python3
"""
Test RAG evaluation for a single query
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.search.search_manager import SearchManager
from src.evaluators.rag_evaluator import RAGEvaluator, generate_rag_report
from src.evaluators.authority_scorer import SearchResult

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


def convert_to_search_results(provider_results: List[Dict]) -> List[SearchResult]:
    """Convert provider results to SearchResult format for authority scoring"""
    search_results = []
    for i, result in enumerate(provider_results, 1):
        search_results.append(SearchResult(
            url=result.get('url', ''),
            rank=i,
            title=result.get('title', ''),
            snippet=result.get('text', result.get('snippet', ''))[:200]
        ))
    return search_results


def save_results(results: Dict, output_dir: Path, filename: str):
    """Save results to JSON file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    return output_path


def test_single_query(query_id: str = "Q1", providers: List[str] = None):
    """
    Test RAG evaluation on a single query
    """
    # Default providers
    if providers is None:
        providers = ['exa', 'google']  # Start with just 2 for testing
    
    # Load scenarios
    scenarios = load_scenarios()
    query_data = next((q for q in scenarios if q['id'] == query_id), None)
    
    if not query_data:
        logger.error(f"Query {query_id} not found in scenarios")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing RAG Evaluation for Query {query_id}")
    logger.info(f"Question: {query_data['question']}")
    logger.info(f"Search Query: {query_data['search_query']}")
    logger.info(f"Providers: {providers}")
    logger.info(f"{'='*80}\n")
    
    # Initialize search manager
    search_manager = SearchManager()
    
    # Run searches across providers
    logger.info("Running searches across providers...")
    search_start = time.time()
    
    providers_results = {}
    search_latencies = {}
    
    for provider in providers:
        try:
            provider_start = time.time()
            client = search_manager.get_client(provider)
            results = client.search(query_data['search_query'], num_results=10)
            search_latencies[provider] = (time.time() - provider_start) * 1000
            providers_results[provider] = results
            logger.info(f"✓ {provider}: {len(results)} results in {search_latencies[provider]:.1f}ms")
        except Exception as e:
            logger.error(f"✗ {provider}: Error - {e}")
            providers_results[provider] = []
            search_latencies[provider] = 0
    
    total_search_time = (time.time() - search_start) * 1000
    logger.info(f"Total search time: {total_search_time:.1f}ms\n")
    
    # Check if we have results
    if not any(providers_results.values()):
        logger.error("No search results from any provider. Exiting.")
        return
    
    # Initialize RAG evaluator
    logger.info("Initializing RAG evaluator with llama3.1:latest...")
    evaluator = RAGEvaluator(ollama_model="llama3.1:latest")
    
    # Run RAG evaluation
    logger.info("Running RAG evaluation (this may take a few minutes)...\n")
    eval_start = time.time()
    
    rag_results = evaluator.evaluate_all_metrics(
        query=query_data['question'],
        providers_results=providers_results
    )
    
    # Add search latencies to results
    rag_results['search_latencies_ms'] = search_latencies
    rag_results['query_id'] = query_id
    rag_results['scenario'] = query_data['scenario']
    
    eval_time = (time.time() - eval_start) * 1000
    logger.info(f"\nRAG evaluation completed in {eval_time:.1f}ms")
    
    # Create output directory structure
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_base = Path(__file__).parent.parent.parent / "data" / "rag_results" / f"{timestamp}_evaluation"
    
    # Save raw results
    raw_dir = output_base / "raw_answers"
    for provider in rag_results.get('providers', {}):
        provider_data = rag_results['providers'][provider]
        save_results(
            {
                'query_id': query_id,
                'provider': provider,
                'answer': provider_data.get('answer', ''),
                'generation_latency_ms': provider_data.get('generation_latency_ms', 0)
            },
            raw_dir,
            f"{query_id}_{provider}.json"
        )
    
    # Save pairwise judgments
    if 'pairwise_judgments' in rag_results:
        judgments_dir = output_base / "pairwise_judgments"
        for i, judgment in enumerate(rag_results['pairwise_judgments']):
            save_results(
                judgment,
                judgments_dir,
                f"{query_id}_judgment_{i+1}.json"
            )
    
    # Save metrics summary
    metrics_dir = output_base / "metrics"
    save_results(
        {
            'query_id': query_id,
            'llm_win_rate': rag_results.get('llm_win_rate', {}),
            'provider_metrics': {
                provider: {
                    'authority_of_cited': data.get('authority_of_cited', {}),
                    'support_ratio': data.get('support_ratio', {})
                }
                for provider, data in rag_results.get('providers', {}).items()
            }
        },
        metrics_dir,
        f"{query_id}_metrics.json"
    )
    
    # Save complete results
    complete_path = save_results(rag_results, output_base, f"{query_id}_complete.json")
    
    # Create symlink to latest results
    latest_link = Path(__file__).parent.parent.parent / "data" / "rag_results" / "latest"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(output_base.name)
    
    # Print summary
    print_results_summary(rag_results)
    
    logger.info(f"\n✓ All results saved to {output_base}")
    logger.info(f"  View latest results at: data/rag_results/latest/")
    
    return rag_results


def print_results_summary(results: Dict):
    """Print a summary of the evaluation results"""
    print("\n" + "="*80)
    print("RAG EVALUATION SUMMARY")
    print("="*80)
    
    # Win rates
    if 'llm_win_rate' in results:
        print("\n📊 LLM Win Rates:")
        win_rates = results['llm_win_rate'].get('win_rates', {})
        for provider, rate in sorted(win_rates.items(), key=lambda x: x[1], reverse=True):
            print(f"  {provider}: {rate:.1%}")
    
    # Provider metrics
    print("\n📈 Provider Metrics:")
    for provider, data in results.get('providers', {}).items():
        print(f"\n  {provider}:")
        
        # Authority of cited
        if 'authority_of_cited' in data:
            auth = data['authority_of_cited']
            print(f"    Authority of Cited: {auth['authority_score']:.3f} ({auth['num_citations']} citations)")
        
        # Support ratio
        if 'support_ratio' in data:
            support = data['support_ratio']
            print(f"    Support Ratio: {support['support_ratio']:.1%} ({support['supported_sentences']}/{support['total_sentences']} sentences)")
        
        # Generation time
        if 'generation_latency_ms' in data:
            print(f"    Generation Time: {data['generation_latency_ms']:.1f}ms")
    
    # Latencies
    if 'search_latencies_ms' in results:
        print("\n⏱️  Search Latencies:")
        for provider, latency in results['search_latencies_ms'].items():
            print(f"  {provider}: {latency:.1f}ms")
    
    print("\n" + "="*80)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RAG evaluation on a single query')
    parser.add_argument('--query-id', default='Q1', help='Query ID from scenarios.json (default: Q1)')
    parser.add_argument('--providers', nargs='+', default=['exa', 'google'],
                       help='Providers to test (default: exa google)')
    
    args = parser.parse_args()
    
    # Test Ollama connection first
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
        
        if not any('llama3.1' in name.lower() for name in model_names if name):
            logger.warning("llama3.1:latest not found. Please run: ollama pull llama3.1:latest")
            if model_names:
                logger.info(f"Available models: {model_names}")
        else:
            logger.info("✓ Ollama is running with llama3.1:latest")
    except Exception as e:
        logger.error(f"Cannot connect to Ollama: {e}")
        logger.error("Please ensure Ollama is running: ollama serve")
        logger.info("Trying to continue anyway...")
        # Don't return, try to continue
    
    # Run test
    test_single_query(query_id=args.query_id, providers=args.providers)


if __name__ == "__main__":
    main()