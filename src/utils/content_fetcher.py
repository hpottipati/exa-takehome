"""
Content fetcher for retrieving full page content from URLs.
Ensures fair comparison across all search providers by using the same content.
"""

import requests
from bs4 import BeautifulSoup
import time
import hashlib
import json
import os
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class ContentFetcher:
    """Fetches and caches full page content from URLs for fair evaluation"""
    
    def __init__(self, cache_dir: str = ".content_cache", timeout: int = 10, max_workers: int = 5):
        """
        Initialize the content fetcher.
        
        Args:
            cache_dir: Directory to cache fetched content
            timeout: Request timeout in seconds
            max_workers: Maximum concurrent fetch workers
        """
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_workers = max_workers
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Headers to avoid being blocked
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def _get_cache_path(self, url: str) -> str:
        """Generate cache file path for a URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{url_hash}.json")
    
    def _load_from_cache(self, url: str) -> Optional[Dict]:
        """Load cached content if available and fresh (24 hours)"""
        cache_path = self._get_cache_path(url)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                
                # Check if cache is fresh (24 hours)
                if time.time() - cached.get('timestamp', 0) < 86400:
                    logger.debug(f"Cache hit for {url}")
                    return cached
            except Exception as e:
                logger.warning(f"Error loading cache for {url}: {e}")
        
        return None
    
    def _save_to_cache(self, url: str, content: Dict):
        """Save fetched content to cache"""
        cache_path = self._get_cache_path(url)
        content['timestamp'] = time.time()
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error saving cache for {url}: {e}")
    
    def _extract_text_from_html(self, html: str) -> str:
        """Extract clean text from HTML content"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'meta', 'link']):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ""
    
    def fetch_content(self, url: str, use_cache: bool = True) -> Dict:
        """
        Fetch full page content from a URL.
        
        Args:
            url: The URL to fetch
            use_cache: Whether to use cached content if available
            
        Returns:
            Dict with keys: url, title, text, success, error, cached
        """
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(url)
            if cached:
                cached['cached'] = True
                return cached
        
        result = {
            'url': url,
            'title': '',
            'text': '',
            'success': False,
            'error': None,
            'cached': False
        }
        
        try:
            # Add delay to be polite
            time.sleep(0.5)
            
            # Make request
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True,
                verify=False  # Some sites have cert issues
            )
            response.raise_for_status()
            
            # Extract content
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            
            # Get title
            title_tag = soup.find('title')
            if title_tag:
                result['title'] = title_tag.get_text(strip=True)
            
            # Extract text
            extracted_text = self._extract_text_from_html(html)
            
            # Limit text length to avoid huge documents
            max_length = 10000  # 10KB max
            if len(extracted_text) > max_length:
                # Take first part and last part to get intro and conclusion
                result['text'] = extracted_text[:max_length//2] + "\n...\n" + extracted_text[-max_length//2:]
            else:
                result['text'] = extracted_text
            
            result['success'] = True
            
            # Cache successful fetch
            if use_cache:
                self._save_to_cache(url, result)
            
            logger.info(f"Successfully fetched {url} ({len(result['text'])} chars)")
            
        except requests.exceptions.Timeout:
            result['error'] = 'Timeout'
            logger.warning(f"Timeout fetching {url}")
        except requests.exceptions.RequestException as e:
            result['error'] = str(e)
            logger.warning(f"Error fetching {url}: {e}")
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Unexpected error fetching {url}: {e}")
        
        return result
    
    def batch_fetch(self, urls: List[str], use_cache: bool = True) -> List[Dict]:
        """
        Fetch content for multiple URLs in parallel.
        
        Args:
            urls: List of URLs to fetch
            use_cache: Whether to use cached content
            
        Returns:
            List of fetch results in the same order as input URLs
        """
        results = [None] * len(urls)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.fetch_content, url, use_cache): idx
                for idx, url in enumerate(urls)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error in batch fetch: {e}")
                    results[idx] = {
                        'url': urls[idx],
                        'title': '',
                        'text': '',
                        'success': False,
                        'error': str(e),
                        'cached': False
                    }
        
        return results
    
    def deduplicate_urls(self, provider_results: Dict[str, List[Dict]]) -> Dict:
        """
        Extract unique URLs from all providers and map them back.
        
        Args:
            provider_results: Dict of provider -> list of search results
            
        Returns:
            Dict with:
                - unique_urls: List of unique URLs
                - url_to_providers: Mapping of URL to list of providers that returned it
                - provider_url_indices: Mapping to find results
        """
        unique_urls = []
        url_to_providers = {}
        provider_url_indices = {}
        seen_urls = set()
        
        for provider, results in provider_results.items():
            provider_url_indices[provider] = []
            
            for result in results:
                url = result.get('url', '')
                if not url:
                    continue
                
                if url not in seen_urls:
                    unique_urls.append(url)
                    seen_urls.add(url)
                    url_to_providers[url] = []
                
                url_to_providers[url].append(provider)
                # Store the index of this URL in unique_urls for this provider's result
                provider_url_indices[provider].append(unique_urls.index(url))
        
        return {
            'unique_urls': unique_urls,
            'url_to_providers': url_to_providers,
            'provider_url_indices': provider_url_indices
        }