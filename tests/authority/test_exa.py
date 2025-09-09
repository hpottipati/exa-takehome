#!/usr/bin/env python3
"""
Test script to evaluate Exa search results using the Authority Scorer
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from search.exa_client import ExaClient
from evaluators.authority_scorer import (
    AuthorityScorer, 
    SearchResult,
    evaluate_query_authority,
    generate_authority_report
)

def load_test_scenarios(file_path: str = "../../data/scenarios.json") -> List[Dict]:
    """Load test scenarios from JSON file"""
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, file_path)
    with open(full_path, 'r') as f:
        data = json.load(f)
    return data['questions']

def exa_results_to_search_results(exa_results: List[Dict]) -> List[SearchResult]:
    """Convert Exa API results to SearchResult format"""
    search_results = []
    for idx, result in enumerate(exa_results, 1):
        search_results.append(SearchResult(
            url=result.get('url', ''),
            rank=idx,
            title=result.get('title', ''),
            snippet=result.get('text', '')
        ))
    return search_results

def test_single_query(query_data: Dict, client: ExaClient) -> Dict:
    """Test a single query and return results"""
    import time
    
    print(f"\n{'='*60}")
    print(f"Testing Query {query_data['id']}: {query_data['scenario'][:50]}...")
    print(f"Search Query: {query_data['search_query']}")
    
    # Get Exa search results
    try:
        search_start = time.time()
        exa_results = client.search(
            query=query_data['search_query'],
            num_results=10,
        )
        search_latency_ms = (time.time() - search_start) * 1000
        print(f"Retrieved {len(exa_results)} results from Exa in {search_latency_ms:.1f}ms")
    except Exception as e:
        print(f"Error fetching Exa results: {e}")
        return None
    
    # Convert to SearchResult format
    search_results = exa_results_to_search_results(exa_results)
    
    # Evaluate with authority scorer
    providers_results = {
        'exa': search_results
    }
    
    search_latencies = {
        'exa': search_latency_ms
    }
    
    evaluation = evaluate_query_authority(
        query_id=query_data['id'],
        query_text=query_data['search_query'],
        providers_results=providers_results,
        search_latencies=search_latencies
    )
    
    # Print summary metrics
    if 'exa' in evaluation['providers']:
        provider_data = evaluation['providers']['exa']
        metrics = provider_data['metrics']
        
        print(f"\nMetrics for Exa:")
        print(f"  Authority@3: {metrics['authority_at_3']:.3f}")
        print(f"  HighAuthHit@3: {metrics['high_auth_hit_at_3']}")
        print(f"  UtilityScore@10: {metrics['utility_score_at_10']:.3f}")
        
        # Print latency information
        print(f"\nLatency Information:")
        print(f"  Search API Latency: {provider_data['search_api_latency_ms']:.1f}ms")
        print(f"  Evaluation Time: {provider_data['evaluation_time_ms']:.1f}ms")
        print(f"  Total Time: {provider_data['total_time_ms']:.1f}ms")
        
        if 'latency_stats' in metrics:
            latency_stats = metrics['latency_stats']
            print(f"  Page Fetch Stats:")
            print(f"    Avg per URL: {latency_stats['avg_fetch_time_ms']:.1f}ms")
            print(f"    Min/Max: {latency_stats['min_fetch_time_ms']:.1f}ms / {latency_stats['max_fetch_time_ms']:.1f}ms")
            print(f"    Total Fetch Time: {latency_stats['total_fetch_time_ms']:.1f}ms")
        
        # Print top 3 results details
        print(f"\nTop 3 Results:")
        for result in provider_data['detailed_results'][:3]:
            domain = result['url'].split('/')[2] if '/' in result['url'] else result['url']
            print(f"  Rank {result['rank']}: {domain[:50]}")
            print(f"    Authority Score: {result['authority_score']:.3f}")
            print(f"    Domain Score: {result['domain_score']:.2f}, Citations: {result['citation_count']}")
            print(f"    Fetch Time: {result['fetch_latency_ms']:.1f}ms")
    
    return evaluation

def main():
    """Main test function"""
    import time
    
    overall_start = time.time()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize Exa client
    client = ExaClient()
    
    # Verify API key is loaded
    if not client.api_key:
        print("Error: Exa API key not found in environment variables")
        print("Please set EXA_API_KEY in your .env file")
        sys.exit(1)
    
    print(f"Exa API Key loaded: {client.api_key[:10]}...")
    
    # Load test scenarios
    scenarios = load_test_scenarios()
    print(f"\nLoaded {len(scenarios)} test scenarios")
    
    # Test each query
    all_results = []
    num_queries = min(3, len(scenarios))  # Test first 3 queries to start
    
    for scenario in scenarios[:num_queries]:
        result = test_single_query(scenario, client)
        if result:
            all_results.append(result)
    
    overall_time_ms = (time.time() - overall_start) * 1000
    
    # Generate aggregate report
    if all_results:
        print(f"\n{'='*60}")
        print("AGGREGATE REPORT - EXA")
        print('='*60)
        
        report = generate_authority_report(all_results)
        
        if 'exa' in report.get('summary', {}):
            summary = report['summary']['exa']
            print(f"\nExa Search - Average Metrics:")
            print(f"  Authority@3: {summary['authority_at_3']:.3f}")
            print(f"  HighAuthHit Rate: {summary['high_auth_hit_rate']:.1%}")
            print(f"  Avg Utility Score@10: {summary['avg_utility_score_at_10']:.3f}")
            print(f"  Queries Evaluated: {summary['queries_evaluated']}")
            
            # Print latency summary
            if 'latency_summary' in summary:
                latency_summary = summary['latency_summary']
                print(f"\nLatency Summary:")
                print(f"  Avg Search API Latency: {latency_summary['avg_search_api_latency_ms']:.1f}ms")
                print(f"  Avg Evaluation Time: {latency_summary['avg_evaluation_time_ms']:.1f}ms")
                print(f"  Avg Total Time per Query: {latency_summary['avg_total_time_ms']:.1f}ms")
                
                fetch_stats = latency_summary['fetch_time_stats']
                print(f"  Page Fetch Performance:")
                print(f"    Avg per URL: {fetch_stats['avg_per_url_ms']:.1f}ms")
                print(f"    Min/Max: {fetch_stats['min_fetch_time_ms']:.1f}ms / {fetch_stats['max_fetch_time_ms']:.1f}ms")
                print(f"    Total URLs Evaluated: {fetch_stats['total_urls_evaluated']}")
        
        print(f"\nOverall Test Duration: {overall_time_ms:.1f}ms ({overall_time_ms/1000:.2f}s)")
        
        # Save results to file
        results_dir = Path("../../data/results")
        results_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"exa_authority_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'provider': 'exa',
                'queries': all_results,
                'aggregate_report': report
            }, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
