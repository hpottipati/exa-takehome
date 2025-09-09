#!/usr/bin/env python3
"""
Test script to evaluate SERP Bing search results using the Authority Scorer
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

from search.serp_bing import SerpBingClient
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

def serp_results_to_search_results(serp_results: List[Dict]) -> List[SearchResult]:
    """Convert SERP API results to SearchResult format"""
    search_results = []
    for idx, result in enumerate(serp_results, 1):
        search_results.append(SearchResult(
            url=result.get('url', ''),
            rank=idx,
            title=result.get('title', ''),
            snippet=result.get('text', '')
        ))
    return search_results

def test_single_query(query_data: Dict, client: SerpBingClient) -> Dict:
    """Test a single query and return results"""
    print(f"\n{'='*60}")
    print(f"Testing Query {query_data['id']}: {query_data['scenario'][:50]}...")
    print(f"Search Query: {query_data['search_query']}")
    
    # Get SERP Bing search results
    try:
        serp_results = client.search(
            query=query_data['search_query'],
            num_results=10
        )
        print(f"Retrieved {len(serp_results)} results from SERP Bing")
    except Exception as e:
        print(f"Error fetching SERP Bing results: {e}")
        return None
    
    # Convert to SearchResult format
    search_results = serp_results_to_search_results(serp_results)
    
    # Evaluate with authority scorer
    providers_results = {
        'serp_bing': search_results
    }
    
    evaluation = evaluate_query_authority(
        query_id=query_data['id'],
        query_text=query_data['search_query'],
        providers_results=providers_results
    )
    
    # Print summary metrics
    if 'serp_bing' in evaluation['providers']:
        metrics = evaluation['providers']['serp_bing']['metrics']
        print(f"\nMetrics for SERP Bing:")
        print(f"  Authority@3: {metrics['authority_at_3']:.3f}")
        print(f"  HighAuthHit@3: {metrics['high_auth_hit_at_3']}")
        print(f"  UtilityScore@10: {metrics['utility_score_at_10']:.3f}")
        
        # Print top 3 results details
        print(f"\nTop 3 Results:")
        for result in evaluation['providers']['serp_bing']['detailed_results'][:3]:
            domain = result['url'].split('/')[2] if '/' in result['url'] else result['url']
            print(f"  Rank {result['rank']}: {domain[:50]}")
            print(f"    Authority Score: {result['authority_score']:.3f}")
            print(f"    Domain Score: {result['domain_score']:.2f}, Citations: {result['citation_count']}")
    
    return evaluation

def main():
    """Main test function"""
    # Load environment variables
    load_dotenv()
    
    # Initialize SERP Bing client
    client = SerpBingClient()
    
    # Verify API key is loaded
    if not client.api_key:
        print("Error: SERP API key not found in environment variables")
        print("Please set SERP_API_KEY in your .env file")
        sys.exit(1)
    
    print(f"SERP API Key loaded: {client.api_key[:10]}...")
    
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
    
    # Generate aggregate report
    if all_results:
        print(f"\n{'='*60}")
        print("AGGREGATE REPORT - SERP BING")
        print('='*60)
        
        report = generate_authority_report(all_results)
        
        if 'serp_bing' in report.get('summary', {}):
            summary = report['summary']['serp_bing']
            print(f"\nSERP Bing Search - Average Metrics:")
            print(f"  Authority@3: {summary['authority_at_3']:.3f}")
            print(f"  HighAuthHit Rate: {summary['high_auth_hit_rate']:.1%}")
            print(f"  Avg Utility Score@10: {summary['avg_utility_score_at_10']:.3f}")
            print(f"  Queries Evaluated: {summary['queries_evaluated']}")
        
        # Save results to file
        results_dir = Path("../../data/results")
        results_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = results_dir / f"serp_bing_authority_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'provider': 'serp_bing',
                'queries': all_results,
                'aggregate_report': report
            }, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()