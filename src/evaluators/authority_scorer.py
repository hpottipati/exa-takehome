# authority_scorer.py
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from difflib import SequenceMatcher
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    url: str
    rank: int
    title: str = ""
    snippet: str = ""

@dataclass
class AuthorityEvaluation:
    url: str
    rank: int
    domain_score: float
    citation_bonus: float
    authority_score: float
    page_length: int
    citation_count: int
    fetch_latency_ms: float = 0.0  # Time to fetch page content in milliseconds

class AuthorityScorer:
    def __init__(self):
        # Domain authority mapping
        self.domain_weights = {
            # Tier 1: Government sources
            '.gov': 1.0,
            'supremecourt.gov': 1.0,
            'congress.gov': 1.0,
            'govinfo.gov': 1.0,
            'uscourts.gov': 1.0,
            'eeoc.gov': 1.0,
            'doj.gov': 1.0,
            'sec.gov': 1.0,
            'treasury.gov': 1.0,
            'nlrb.gov': 1.0,
            
            # Tier 2: Educational/Authoritative legal sources
            'law.cornell.edu': 0.9,
            'oyez.org': 0.9,
            
            # Tier 3: Established legal publishers
            'justia.com': 0.7,
            'findlaw.com': 0.7,
            'case.law': 0.7,
            'law360.com': 0.7,
            'lexisnexis.com': 0.7,
            
            # Tier 4: Everything else
            'default': 0.5
        }
        
        # Legal citation templates for fuzzy matching
        self.citation_templates = [
            # U.S. Reports patterns
            {"pattern": "U.S.", "type": "us_reports", "min_similarity": 0.7},
            {"pattern": "US", "type": "us_reports", "min_similarity": 0.8},
            {"pattern": "United States Reports", "type": "us_reports", "min_similarity": 0.6},
            
            # Federal Reporter patterns
            {"pattern": "F.2d", "type": "federal_reporter", "min_similarity": 0.8},
            {"pattern": "F.3d", "type": "federal_reporter", "min_similarity": 0.8},
            {"pattern": "F.4th", "type": "federal_reporter", "min_similarity": 0.7},
            {"pattern": "Fed", "type": "federal_reporter", "min_similarity": 0.7},
            
            # Supreme Court Reporter
            {"pattern": "S.Ct.", "type": "supreme_court", "min_similarity": 0.8},
            {"pattern": "S. Ct.", "type": "supreme_court", "min_similarity": 0.8},
            {"pattern": "Sup. Ct.", "type": "supreme_court", "min_similarity": 0.7},
            
            # U.S. Code patterns
            {"pattern": "U.S.C.", "type": "us_code", "min_similarity": 0.8},
            {"pattern": "USC", "type": "us_code", "min_similarity": 0.8},
            {"pattern": "United States Code", "type": "us_code", "min_similarity": 0.6},
            
            # Code of Federal Regulations
            {"pattern": "C.F.R.", "type": "cfr", "min_similarity": 0.8},
            {"pattern": "CFR", "type": "cfr", "min_similarity": 0.8},
            {"pattern": "Code of Federal Regulations", "type": "cfr", "min_similarity": 0.6},
            
            # Federal Supplement
            {"pattern": "F.Supp.", "type": "federal_supplement", "min_similarity": 0.8},
            {"pattern": "F. Supp.", "type": "federal_supplement", "min_similarity": 0.8},
            {"pattern": "Federal Supplement", "type": "federal_supplement", "min_similarity": 0.6},
        ]
        
        # Numeric pattern for finding potential citations - more flexible
        self.numeric_pattern = re.compile(r'\b\d+\s+[A-Za-z\.\s§]+\s*\d+[a-z]*\b')
        # Additional pattern for citations with periods and special chars
        self.citation_pattern = re.compile(r'\b\d+\s+[A-Za-z]+\.?\d*[a-z]*\s+\d+\b')
    
    def fetch_page_text(self, url: str, timeout: int = 10) -> Tuple[str, float]:
        """
        Fetch and clean text content from URL
        Returns: (text_content, latency_in_ms)
        """
        start_time = time.time()
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Remove noise elements
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
                tag.decompose()
            
            # Extract main text
            text = " ".join(soup.stripped_strings)
            
            latency_ms = (time.time() - start_time) * 1000
            return text, latency_ms
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.warning(f"Failed to fetch {url}: {str(e)} (took {latency_ms:.1f}ms)")
            return "", latency_ms
    
    
    def fuzzy_match_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings using SequenceMatcher
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def extract_potential_citations(self, text: str) -> List[str]:
        """
        Extract potential citation patterns from text using multiple patterns
        """
        potential_citations = []
        
        # Use multiple patterns to catch different citation formats
        patterns = [
            self.numeric_pattern,
            self.citation_pattern,
            re.compile(r'\b\d+\s*§\s*\d+[a-z]*\b', re.IGNORECASE),  # Section symbols
            re.compile(r'\b\d+\s+[A-Z][a-z]*\.?\s*\d*[a-z]*\s+\d+\b'),  # Capitalized legal terms
        ]
        
        for pattern in patterns:
            matches = pattern.findall(text)
            potential_citations.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in potential_citations:
            if citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)
        
        return unique_citations
    
    def count_legal_citations(self, text: str) -> int:
        """
        Count legal citations in text using fuzzy matching
        """
        if not text.strip():
            return 0
        
        # Extract potential citation candidates
        potential_citations = self.extract_potential_citations(text)
        
        citation_count = 0
        found_citations = set()  # Avoid double counting
        
        for candidate in potential_citations:
            candidate_clean = candidate.strip()
            if len(candidate_clean) < 3:  # Skip very short matches
                continue
                
            # Check against each template using fuzzy matching
            for template in self.citation_templates:
                pattern = template["pattern"]
                min_similarity = template["min_similarity"]
                
                # Check if the candidate contains the pattern with fuzzy matching
                if pattern.lower() in candidate_clean.lower():
                    # Direct substring match - high confidence
                    citation_key = f"{candidate_clean}_{template['type']}"
                    if citation_key not in found_citations:
                        found_citations.add(citation_key)
                        citation_count += 1
                        break
                else:
                    # Fuzzy match for variations
                    similarity = self.fuzzy_match_similarity(candidate_clean, pattern)
                    if similarity >= min_similarity:
                        citation_key = f"{candidate_clean}_{template['type']}"
                        if citation_key not in found_citations:
                            found_citations.add(citation_key)
                            citation_count += 1
                            break
        
        # Also look for standalone legal terms using fuzzy matching
        legal_terms = [
            {"term": "statute", "min_similarity": 0.8},
            {"term": "regulation", "min_similarity": 0.8},
            {"term": "code", "min_similarity": 0.9},
            {"term": "act", "min_similarity": 0.9},
            {"term": "law", "min_similarity": 0.9},
            {"term": "rule", "min_similarity": 0.8},
            {"term": "order", "min_similarity": 0.8},
            {"term": "decision", "min_similarity": 0.8},
            {"term": "opinion", "min_similarity": 0.8},
            {"term": "case", "min_similarity": 0.9},
            {"term": "holding", "min_similarity": 0.8},
            {"term": "precedent", "min_similarity": 0.8},
        ]
        
        legal_context_terms = ["federal", "state", "supreme", "circuit", "court", "judicial", "constitutional"]
        
        words = text.lower().split()
        for i, word in enumerate(words):
            # Use fuzzy matching for legal terms
            for legal_term in legal_terms:
                similarity = self.fuzzy_match_similarity(word, legal_term["term"])
                if similarity >= legal_term["min_similarity"] and i > 0 and i < len(words) - 1:
                    # Check if surrounded by numbers or legal context
                    prev_word = words[i-1]
                    next_word = words[i+1] if i+1 < len(words) else ""
                    
                    # Check for numeric context
                    has_numeric = bool(re.match(r'\d+', prev_word) or re.match(r'\d+', next_word))
                    
                    # Check for legal context using fuzzy matching
                    has_legal_context = False
                    for context_term in legal_context_terms:
                        if (self.fuzzy_match_similarity(prev_word, context_term) >= 0.8 or
                            self.fuzzy_match_similarity(next_word, context_term) >= 0.8):
                            has_legal_context = True
                            break
                    
                    if has_numeric or has_legal_context:
                        citation_count += 1
                        break
        
        return citation_count
    
    def calculate_citation_bonus(self, text: str) -> float:
        """
        Calculate citation density bonus (0.3 to 1.0)
        Formula: 0.3 + min(0.7, citations_per_100_words * 20)
        """
        if not text.strip():
            return 0.3
        
        word_count = len(text.split())
        if word_count == 0:
            return 0.3
        
        citation_count = self.count_legal_citations(text)
        citations_per_100_words = citation_count / (word_count / 100)
        
        # Scale citations: 20x multiplier, cap at 0.7 bonus
        bonus = min(0.7, citations_per_100_words * 20)
        
        return 0.3 + bonus

    def get_domain_score(self, url: str) -> float:
        """
        Get domain authority score based on URL using fuzzy matching
        """
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Check for exact domain matches first (highest priority)
        for domain_key, score in self.domain_weights.items():
            if domain_key == "default":
                continue
            if domain_key in domain:
                return score
        
        # Use fuzzy matching for partial matches (lower priority)
        best_match_score = self.domain_weights["default"]
        best_similarity = 0.0
        
        for domain_key, score in self.domain_weights.items():
            if domain_key == "default":
                continue
            
            # Calculate similarity between domain and domain_key
            similarity = self.fuzzy_match_similarity(domain, domain_key)
            
            # Only consider matches with reasonable similarity (0.7+)
            if similarity >= 0.7 and similarity > best_similarity:
                best_similarity = similarity
                best_match_score = score
        
        return best_match_score

    def evaluate_url(self, url: str, rank: int) -> AuthorityEvaluation:
        """
        Evaluate a single URL for authority
        """
        logger.info(f"Evaluating rank {rank}: {url}")
        
        # Get domain authority
        domain_score = self.get_domain_score(url)
        
        # Fetch page content with timing
        page_text, fetch_latency_ms = self.fetch_page_text(url)
        page_length = len(page_text.split())
        
        # Calculate citation bonus
        citation_bonus = self.calculate_citation_bonus(page_text)
        citation_count = self.count_legal_citations(page_text)
        
        # Final authority score - hybrid model to avoid penalizing high-authority domains
        if domain_score >= 0.9:  # Government, edu, and other high-authority domains
            # Additive model: high domains get citation bonuses, not penalties
            # Base score preserved, citations add modest bonus
            authority_score = min(1.0, domain_score + (citation_bonus - 0.3) * 0.5)
        else:  # Regular domains (legal firms, blogs, etc.)
            # Multiplicative model: citation density affects overall score
            authority_score = domain_score * citation_bonus
        
        return AuthorityEvaluation(
            url=url,
            rank=rank,
            domain_score=domain_score,
            citation_bonus=citation_bonus,
            authority_score=authority_score,
            page_length=page_length,
            citation_count=citation_count,
            fetch_latency_ms=fetch_latency_ms
        )
    
    def evaluate_search_results(self, results: List[SearchResult]) -> Dict:
        """
        Evaluate a list of search results and compute authority metrics
        """
        evaluations = []
        
        # Evaluate each result
        for result in results[:10]:  # Top 10 only
            eval_result = self.evaluate_url(result.url, result.rank)
            evaluations.append(eval_result)
        
        # Calculate metrics
        metrics = self.calculate_authority_metrics(evaluations)
        
        return {
            'evaluations': evaluations,
            'metrics': metrics
        }
    
    def calculate_authority_metrics(self, evaluations: List[AuthorityEvaluation]) -> Dict:
        """
        Calculate the three authority metrics plus latency statistics
        """
        if not evaluations:
            return {
                'authority_at_3': 0.0,
                'high_auth_hit_at_3': False,
                'utility_score_at_10': 0.0,
                'latency_stats': {
                    'total_fetch_time_ms': 0.0,
                    'avg_fetch_time_ms': 0.0,
                    'min_fetch_time_ms': 0.0,
                    'max_fetch_time_ms': 0.0,
                    'urls_evaluated': 0
                }
            }
        
        # Authority@3: Average authority score of top-3 results
        top_3 = evaluations[:3]
        authority_at_3 = sum(e.authority_score for e in top_3) / len(top_3) if top_3 else 0.0
        
        # HighAuthHit@3: Any result in top-3 with authority >= 0.85
        high_auth_hit_at_3 = any(e.authority_score >= 0.85 for e in top_3)
        
        # UtilityScore@10: Sum of (authority * 1/rank) for top-10
        utility_score_at_10 = sum(e.authority_score * (1.0 / e.rank) for e in evaluations)
        
        # Latency statistics
        fetch_times = [e.fetch_latency_ms for e in evaluations]
        total_fetch_time = sum(fetch_times)
        avg_fetch_time = total_fetch_time / len(fetch_times) if fetch_times else 0.0
        
        return {
            'authority_at_3': round(authority_at_3, 3),
            'high_auth_hit_at_3': high_auth_hit_at_3,
            'utility_score_at_10': round(utility_score_at_10, 3),
            'latency_stats': {
                'total_fetch_time_ms': round(total_fetch_time, 1),
                'avg_fetch_time_ms': round(avg_fetch_time, 1),
                'min_fetch_time_ms': round(min(fetch_times) if fetch_times else 0.0, 1),
                'max_fetch_time_ms': round(max(fetch_times) if fetch_times else 0.0, 1),
                'urls_evaluated': len(evaluations)
            }
        }

def evaluate_query_authority(query_id: str, query_text: str, providers_results: Dict[str, List[SearchResult]], 
                           search_latencies: Optional[Dict[str, float]] = None) -> Dict:
    """
    Evaluate authority metrics for a single query across multiple providers
    
    Args:
        query_id: Unique query identifier
        query_text: The search query text
        providers_results: Dict mapping provider names to search results
        search_latencies: Optional dict mapping provider names to search API latency in ms
    """
    scorer = AuthorityScorer()
    query_results = {
        'query_id': query_id,
        'query_text': query_text,
        'providers': {}
    }
    
    for provider_name, results in providers_results.items():
        logger.info(f"Evaluating {provider_name} for query {query_id}")
        
        evaluation_start = time.time()
        evaluation = scorer.evaluate_search_results(results)
        evaluation_time_ms = (time.time() - evaluation_start) * 1000
        
        # Add search API latency if provided
        search_api_latency = search_latencies.get(provider_name, 0.0) if search_latencies else 0.0
        
        query_results['providers'][provider_name] = {
            'metrics': evaluation['metrics'],
            'search_api_latency_ms': round(search_api_latency, 1),
            'evaluation_time_ms': round(evaluation_time_ms, 1),
            'total_time_ms': round(search_api_latency + evaluation_time_ms, 1),
            'detailed_results': [
                {
                    'url': e.url,
                    'rank': e.rank,
                    'authority_score': e.authority_score,
                    'domain_score': e.domain_score,
                    'citation_bonus': e.citation_bonus,
                    'citation_count': e.citation_count,
                    'page_length': e.page_length,
                    'fetch_latency_ms': e.fetch_latency_ms
                }
                for e in evaluation['evaluations']
            ]
        }
    
    return query_results

def generate_authority_report(all_results: List[Dict]) -> Dict:
    """
    Generate aggregate authority report across all queries and providers
    """
    if not all_results:
        return {}
    
    # Get all provider names
    provider_names = set()
    for query_result in all_results:
        provider_names.update(query_result['providers'].keys())
    
    provider_names = sorted(list(provider_names))
    
    # Aggregate metrics
    report = {
        'summary': {},
        'detailed_comparison': []
    }
    
    for provider in provider_names:
        authority_at_3_scores = []
        high_auth_hits = []
        utility_scores = []
        search_api_latencies = []
        evaluation_times = []
        total_times = []
        fetch_time_stats = []
        
        for query_result in all_results:
            if provider in query_result['providers']:
                provider_data = query_result['providers'][provider]
                metrics = provider_data['metrics']
                
                authority_at_3_scores.append(metrics['authority_at_3'])
                high_auth_hits.append(metrics['high_auth_hit_at_3'])
                utility_scores.append(metrics['utility_score_at_10'])
                
                # Collect latency data
                search_api_latencies.append(provider_data.get('search_api_latency_ms', 0.0))
                evaluation_times.append(provider_data.get('evaluation_time_ms', 0.0))
                total_times.append(provider_data.get('total_time_ms', 0.0))
                
                # Collect fetch time stats if available
                if 'latency_stats' in metrics:
                    fetch_stats = metrics['latency_stats']
                    fetch_time_stats.append({
                        'total': fetch_stats['total_fetch_time_ms'],
                        'avg': fetch_stats['avg_fetch_time_ms'],
                        'min': fetch_stats['min_fetch_time_ms'],
                        'max': fetch_stats['max_fetch_time_ms'],
                        'count': fetch_stats['urls_evaluated']
                    })
        
        # Calculate averages
        avg_authority_at_3 = sum(authority_at_3_scores) / len(authority_at_3_scores) if authority_at_3_scores else 0
        high_auth_hit_rate = sum(high_auth_hits) / len(high_auth_hits) if high_auth_hits else 0
        avg_utility_score = sum(utility_scores) / len(utility_scores) if utility_scores else 0
        
        # Calculate latency statistics
        avg_search_latency = sum(search_api_latencies) / len(search_api_latencies) if search_api_latencies else 0
        avg_evaluation_time = sum(evaluation_times) / len(evaluation_times) if evaluation_times else 0
        avg_total_time = sum(total_times) / len(total_times) if total_times else 0
        
        # Aggregate fetch time statistics
        # total_fetch_times not used; omit to avoid linter warnings
        all_avg_fetch_times = [stat['avg'] for stat in fetch_time_stats]
        all_min_fetch_times = [stat['min'] for stat in fetch_time_stats if stat['min'] > 0]
        all_max_fetch_times = [stat['max'] for stat in fetch_time_stats]
        total_urls_evaluated = sum(stat['count'] for stat in fetch_time_stats)
        
        report['summary'][provider] = {
            'authority_at_3': round(avg_authority_at_3, 3),
            'high_auth_hit_rate': round(high_auth_hit_rate, 3),
            'avg_utility_score_at_10': round(avg_utility_score, 3),
            'queries_evaluated': len(authority_at_3_scores),
            'latency_summary': {
                'avg_search_api_latency_ms': round(avg_search_latency, 1),
                'avg_evaluation_time_ms': round(avg_evaluation_time, 1),
                'avg_total_time_ms': round(avg_total_time, 1),
                'fetch_time_stats': {
                    'avg_per_url_ms': round(sum(all_avg_fetch_times) / len(all_avg_fetch_times) if all_avg_fetch_times else 0, 1),
                    'min_fetch_time_ms': round(min(all_min_fetch_times) if all_min_fetch_times else 0, 1),
                    'max_fetch_time_ms': round(max(all_max_fetch_times) if all_max_fetch_times else 0, 1),
                    'total_urls_evaluated': total_urls_evaluated
                }
            }
        }
    
    # Add per-query comparison
    for query_result in all_results:
        query_comparison = {
            'query_id': query_result['query_id'],
            'query_text': query_result['query_text'],
            'provider_metrics': {}
        }
        
        for provider in provider_names:
            if provider in query_result['providers']:
                query_comparison['provider_metrics'][provider] = query_result['providers'][provider]['metrics']
        
        report['detailed_comparison'].append(query_comparison)
    
    return report

