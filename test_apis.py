#!/usr/bin/env python3
"""Test which search APIs are working"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.search.search_manager import SearchManager

def test_apis():
    """Test each API individually"""
    
    providers = ['exa', 'google', 'serp_google', 'serp_bing', 'serp_duckduckgo']
    test_query = "test query"
    
    print("API Key Status:")
    print(f"EXA_API_KEY: {'✓' if os.getenv('EXA_API_KEY') else '✗'}")
    print(f"GOOGLE_API_KEY: {'✓' if os.getenv('GOOGLE_API_KEY') else '✗'}")
    print(f"GOOGLE_SEARCH_ENGINE_ID: {'✓' if os.getenv('GOOGLE_SEARCH_ENGINE_ID') else '✗'}")
    print(f"SERP_API_KEY: {'✓' if os.getenv('SERP_API_KEY') else '✗'}")
    print()
    
    print("Testing each provider:")
    print("-" * 50)
    
    working_providers = []
    
    for provider in providers:
        try:
            client = SearchManager.get_client(provider)
            results = client.search(test_query, num_results=1)
            print(f"✓ {provider}: Working! Got {len(results)} results")
            working_providers.append(provider)
        except Exception as e:
            error_msg = str(e)
            if '403' in error_msg:
                print(f"✗ {provider}: 403 Forbidden (invalid key or quota exceeded)")
            elif '401' in error_msg:
                print(f"✗ {provider}: 401 Unauthorized (invalid API key)")
            elif '400' in error_msg:
                print(f"✗ {provider}: 400 Bad Request (malformed request or invalid key)")
            else:
                print(f"✗ {provider}: {error_msg[:100]}")
    
    print("\n" + "=" * 50)
    print(f"Working providers: {working_providers}")
    print(f"Failed providers: {[p for p in providers if p not in working_providers]}")
    
    if not working_providers:
        print("\n⚠️  No providers are working!")
        print("Please check your API keys in the .env file")
    
    return working_providers

if __name__ == "__main__":
    test_apis()