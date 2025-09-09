import requests
from typing import List, Dict, Optional
import os


class GoogleClient:
    def __init__(self, api_key: Optional[str] = None, search_engine_id: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.search_engine_id = search_engine_id or os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        self.base_url = "https://customsearch.googleapis.com/customsearch/v1"
        
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': min(num_results, 10)  # Google Custom Search API max is 10 per request
        }
            
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        for item in data.get('items', []):
            results.append({
                'url': item.get('link', ''),
                'title': item.get('title', ''),
                'text': item.get('snippet', ''),
                'score': 1.0,  # Google doesn't provide relevance scores
                'published_date': '',  # Not available in snippet
                'provider': 'google'
            })
            
        return results