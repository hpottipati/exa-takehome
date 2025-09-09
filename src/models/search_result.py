"""
Standardized search result model for fair comparison across providers.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StandardizedSearchResult:
    """
    Standardized search result that separates provider content from fetched content.
    This enables fair comparison by using the same fetched content for all providers.
    """
    
    # Basic metadata (from provider)
    url: str
    title: str
    provider: str
    score: float
    position: int  # Position in provider's results (1-based)
    
    # Provider's original content (for reference, not evaluation)
    original_snippet: str  # What the provider returned (snippet for Google, full text for Exa)
    original_text_length: int  # Length of provider's text
    
    # Fetched content (for fair evaluation)
    fetched_content: Optional[str] = None  # Full content we fetched
    fetched_title: Optional[str] = None  # Title from fetched page
    fetch_success: bool = False
    fetch_error: Optional[str] = None
    
    # Evaluation content (what we actually use)
    @property
    def evaluation_text(self) -> str:
        """
        Get the text to use for evaluation.
        Prioritizes fetched content for fairness, falls back to original if fetch failed.
        """
        if self.fetch_success and self.fetched_content:
            return self.fetched_content
        return self.original_snippet
    
    @property
    def evaluation_title(self) -> str:
        """
        Get the title to use for evaluation.
        Prioritizes fetched title, falls back to original.
        """
        if self.fetch_success and self.fetched_title:
            return self.fetched_title
        return self.title
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'url': self.url,
            'title': self.title,
            'provider': self.provider,
            'score': self.score,
            'position': self.position,
            'original_snippet': self.original_snippet,
            'original_text_length': self.original_text_length,
            'fetched_content_length': len(self.fetched_content) if self.fetched_content else 0,
            'fetch_success': self.fetch_success,
            'fetch_error': self.fetch_error,
            'uses_fetched_content': self.fetch_success and self.fetched_content is not None
        }
    
    @classmethod
    def from_provider_result(cls, result: dict, provider: str, position: int) -> 'StandardizedSearchResult':
        """
        Create a StandardizedSearchResult from a provider's search result.
        
        Args:
            result: Raw result from search provider
            provider: Name of the provider
            position: Position in search results (1-based)
        """
        # Get the original text/snippet
        original_text = result.get('text', '') or result.get('snippet', '')
        
        return cls(
            url=result.get('url', ''),
            title=result.get('title', ''),
            provider=provider,
            score=result.get('score', 1.0),
            position=position,
            original_snippet=original_text,
            original_text_length=len(original_text)
        )
    
    def add_fetched_content(self, fetched_data: dict):
        """
        Add fetched content to this result.
        
        Args:
            fetched_data: Dict from ContentFetcher with keys:
                         url, title, text, success, error, cached
        """
        if fetched_data.get('success'):
            self.fetched_content = fetched_data.get('text', '')
            self.fetched_title = fetched_data.get('title', '')
            self.fetch_success = True
            self.fetch_error = None
        else:
            self.fetch_success = False
            self.fetch_error = fetched_data.get('error', 'Unknown error')