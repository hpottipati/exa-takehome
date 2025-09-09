import requests
from typing import List, Dict, Optional
import os


class SerpGoogleClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('SERP_API_KEY')
        self.base_url = "https://serpapi.com/search"
        
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        
        params = {
            'api_key': self.api_key,
            'engine': 'google',
            'q': query,
            'num': min(num_results, 100)
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        for item in data.get('organic_results', []):
            results.append({
                'url': item.get('link', ''),
                'title': item.get('title', ''),
                'text': item.get('snippet', ''),
                'score': 1.0 / (item.get('position', 1)),  # Convert position to relevance score
                'published_date': item.get('date', ''),
                'provider': 'serp_google'
            })
            
        return results