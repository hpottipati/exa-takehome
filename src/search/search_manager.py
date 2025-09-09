from typing import Dict, Any, List, Optional
from .exa_client import ExaClient
from .google_client import GoogleClient
from .serp_google import SerpGoogleClient
from .serp_bing import SerpBingClient
from .serp_duckduckgo import SerpDuckDuckGoClient


class SearchManager:
    """Manager for search API clients and operations"""
    
    # Provider registry
    PROVIDERS = {
        'exa': ExaClient,
        'google': GoogleClient,
        'serp_google': SerpGoogleClient,
        'serp_bing': SerpBingClient,
        'serp_duckduckgo': SerpDuckDuckGoClient,
    }
    
    # Provider categories
    CORE_PROVIDERS = ['exa', 'google']
    SERP_PROVIDERS = ['serp_google', 'serp_bing', 'serp_duckduckgo']
    
    @classmethod
    def get_client(cls, provider: str, **kwargs):   
        if provider.lower() not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(cls.PROVIDERS.keys())}")
        
        return cls.PROVIDERS[provider.lower()](**kwargs)
    
    @classmethod
    def get_initialized_clients(cls, providers: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        if providers is None:
            providers = cls.get_available_providers()
        
        clients = {}
        for provider in providers:
            try:
                clients[provider] = cls.get_client(provider, **kwargs)
            except Exception as e:
                print(f"Warning: Could not initialize {provider} client: {e}")
        
        return clients
    
    @classmethod
    def search_single_provider(cls, provider: str, query: str, num_results: int = 10, **kwargs) -> List[Dict]:
        client = cls.get_client(provider)
        return client.search(query, num_results)
    
    @classmethod
    def search_multiple_providers(cls, providers: List[str], query: str, num_results: int = 10, 
                                  provider_kwargs: Optional[Dict[str, Dict]] = None) -> Dict[str, List[Dict]]:
        # Ignore provider_kwargs to ensure identical requests across providers
        
        clients = cls.get_initialized_clients(providers)
        results = {}
        
        for provider, client in clients.items():
            try:
                provider_results = client.search(query, num_results)
                results[provider] = provider_results
                print(f"✓ {provider}: {len(provider_results)} results")
            except Exception as e:
                print(f"✗ {provider}: Error - {e}")
                results[provider] = []
        
        return results
 