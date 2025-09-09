import requests
from typing import List, Dict, Optional
import os


class ExaClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('EXA_API_KEY')
        self.base_url = "https://api.exa.ai"
        
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'query': query,
            'numResults': num_results,
            'contents': {'text': True}
        }
        
            
        response = requests.post(f'{self.base_url}/search', headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        for result in data.get('results', []):
            results.append({
                'url': result.get('url', ''),
                'title': result.get('title', ''),
                'text': result.get('text', ''),
                'score': result.get('score', 0),
                'published_date': result.get('publishedDate', ''),
                'provider': 'exa'
            })
            
        return results