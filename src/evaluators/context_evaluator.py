"""
Context Evaluation Module for Legal Search Benchmarking
Implements three core metrics: LLMWinRate, AuthorityOfCited, SupportRatio
"""

import json
import logging
import re
import time
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
from groq import Groq
from sentence_transformers import SentenceTransformer

from .authority_scorer import AuthorityScorer
from ..utils.content_fetcher import ContentFetcher
from ..models.search_result import StandardizedSearchResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContextAnswer:
    """Container for generated answer with metadata"""
    query: str
    provider: str
    answer: str
    generation_latency_ms: float
    documents_used: List[Dict]
    
@dataclass
class PairwiseJudgment:
    """Result of pairwise comparison between two answers"""
    query: str
    provider_a: str
    provider_b: str
    answer_a: str
    answer_b: str
    winner: str  # "A", "B", or "tie"
    confidence: int  # 1-3
    reasoning: str
    judgment_latency_ms: float


class GroqClient:
    """Client for interacting with Groq LLM"""
    
    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        self.client = Groq(api_key=api_key)
        
    def generate_answer(self, query: str, documents: List[Dict], max_tokens: int = 500) -> Tuple[str, float]:
        """
        Generate an answer using context-based approach
        Returns: (answer, latency_ms)
        """
        # Format documents for context
        formatted_docs = []
        for i, doc in enumerate(documents[:10], 1):  # Use top 10 documents
            formatted_docs.append(
                f"[{i}] Title: {doc.get('title', 'N/A')}\n"
                f"URL: {doc.get('url', 'N/A')}\n"
                f"Content: {doc.get('text', doc.get('snippet', ''))[:500]}..."
            )
        
        context = "\n\n".join(formatted_docs)
        
        prompt = f"""You are a legal research assistant. Based on the search results below, provide a comprehensive answer to the legal question. 
Cite specific sources using [number] notation when making claims.

Question: {query}

Search Results:
{context}

Instructions:
- Provide a clear, accurate answer addressing the legal question
- Cite sources using [1], [2], etc. format
- Focus on legal accuracy and completeness
- Include relevant statutes, regulations, or case law mentioned in the sources
"""
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            answer = response.choices[0].message.content
            latency_ms = (time.time() - start_time) * 1000
            
            return answer, latency_ms
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return f"Error generating answer: {str(e)}", latency_ms
    
    def evaluate_legal_accuracy(self, query: str, answer_a: str, answer_b: str) -> Dict:
        """Evaluate only legal accuracy and correctness"""
        prompt = f"""You are evaluating ONLY legal accuracy. Ignore citations, writing style, and length.

SCORING CRITERIA:
- 10: Perfect legal accuracy, all statements correct
- 8-9: Generally accurate with minor imprecisions
- 6-7: Mostly correct with some notable errors
- 4-5: Mix of correct and incorrect legal information
- 2-3: Multiple significant legal errors
- 0-1: Fundamentally wrong legal understanding

Evaluate each answer and return JSON:
{{
    "score_a": [0-10],
    "score_b": [0-10],
    "errors_a": ["list specific legal errors, or empty if none"],
    "errors_b": ["list specific legal errors, or empty if none"],
    "explanation": "brief explanation of the scores"
}}

QUESTION: {query}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Return ONLY the JSON evaluation:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in legal accuracy evaluation: {e}")
            return {"score_a": 5, "score_b": 5, "errors_a": [], "errors_b": [], "explanation": str(e)}
    
    def evaluate_citation_quality(self, query: str, answer_a: str, answer_b: str) -> Dict:
        """Evaluate only citation quality and source relevance"""
        prompt = f"""You are evaluating ONLY citation quality. Focus on sources cited and their relevance.

SCORING CRITERIA:
- 10: Excellent citations from authoritative primary sources (statutes, regulations, cases)
- 8-9: Good mix of primary and quality secondary sources
- 6-7: Some relevant sources but missing key authorities
- 4-5: Few citations or mostly weak secondary sources
- 2-3: Poor quality sources or irrelevant citations
- 0-1: No citations or completely irrelevant sources

Count and evaluate citations. Return JSON:
{{
    "score_a": [0-10],
    "score_b": [0-10],
    "primary_sources_a": 0,
    "primary_sources_b": 0,
    "quality_assessment": "brief assessment of citation quality"
}}

QUESTION: {query}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Return ONLY the JSON evaluation:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in citation quality evaluation: {e}")
            return {"score_a": 5, "score_b": 5, "primary_sources_a": 0, "primary_sources_b": 0, "quality_assessment": str(e)}
    
    def evaluate_completeness(self, query: str, answer_a: str, answer_b: str) -> Dict:
        """Evaluate completeness of coverage"""
        prompt = f"""You are evaluating ONLY completeness. Does the answer address all parts of the question?

SCORING CRITERIA:
- 10: Comprehensive, addresses every aspect of the question
- 8-9: Most aspects covered with minor omissions
- 6-7: About 70% of important points covered
- 4-5: About half the important points covered
- 2-3: Significant gaps, many aspects missing
- 0-1: Barely addresses the question

Return JSON:
{{
    "score_a": [0-10],
    "score_b": [0-10],
    "missing_aspects_a": ["list what's missing"],
    "missing_aspects_b": ["list what's missing"],
    "coverage_assessment": "brief assessment"
}}

QUESTION: {query}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Return ONLY the JSON evaluation:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in completeness evaluation: {e}")
            return {"score_a": 5, "score_b": 5, "missing_aspects_a": [], "missing_aspects_b": [], "coverage_assessment": str(e)}
    
    def evaluate_jurisdictional_relevance(self, query: str, answer_a: str, answer_b: str) -> Dict:
        """Evaluate correct identification of applicable jurisdictions and laws"""
        prompt = f"""You are evaluating ONLY jurisdictional relevance. Focus on proper identification of applicable laws.

SCORING CRITERIA:
- 10: All relevant jurisdictions correctly identified with specific laws
- 8-9: Most jurisdictions correct with minor issues
- 6-7: Some jurisdictions correct, some missed
- 4-5: Mixed - some right, some wrong jurisdictions
- 2-3: Major jurisdictional errors or confusion
- 0-1: Wrong jurisdictions or none identified

Evaluate jurisdictional accuracy. Return JSON:
{{
    "score_a": [0-10],
    "score_b": [0-10],
    "jurisdictions_a": ["list identified jurisdictions"],
    "jurisdictions_b": ["list identified jurisdictions"],
    "jurisdiction_assessment": "brief assessment"
}}

QUESTION: {query}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Return ONLY the JSON evaluation:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in jurisdictional evaluation: {e}")
            return {"score_a": 5, "score_b": 5, "jurisdictions_a": [], "jurisdictions_b": [], "jurisdiction_assessment": str(e)}
    
    def evaluate_legal_reasoning(self, query: str, answer_a: str, answer_b: str) -> Dict:
        """Evaluate logical flow and legal argumentation"""
        prompt = f"""You are evaluating ONLY legal reasoning quality. Focus on logical flow and argumentation.

SCORING CRITERIA:
- 10: Exceptional logical flow, clear legal methodology
- 8-9: Good reasoning with minor logical gaps
- 6-7: Decent reasoning but some confusion
- 4-5: Basic reasoning with notable flaws
- 2-3: Poor logical structure, hard to follow
- 0-1: No clear reasoning or illogical

Evaluate reasoning quality. Return JSON:
{{
    "score_a": [0-10],
    "score_b": [0-10],
    "strengths_a": ["list reasoning strengths"],
    "strengths_b": ["list reasoning strengths"],
    "reasoning_assessment": "brief assessment"
}}

QUESTION: {query}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Return ONLY the JSON evaluation:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error in reasoning evaluation: {e}")
            return {"score_a": 5, "score_b": 5, "strengths_a": [], "strengths_b": [], "reasoning_assessment": str(e)}
    
    def judge_pairwise(self, query: str, answer_a: str, answer_b: str, 
                      provider_a: str, provider_b: str) -> Tuple[Dict, float]:
        """
        Enhanced pairwise comparison using multiple specialized evaluations.
        Returns: (judgment_dict, latency_ms)
        """
        start_time = time.time()
        
        try:
            # Run all 5 category evaluations
            logger.debug(f"Running multi-category evaluation for {provider_a} vs {provider_b}")
            
            evaluations = {
                'legal_accuracy': self.evaluate_legal_accuracy(query, answer_a, answer_b),
                'citation_quality': self.evaluate_citation_quality(query, answer_a, answer_b),
                'completeness': self.evaluate_completeness(query, answer_a, answer_b),
                'jurisdictional_relevance': self.evaluate_jurisdictional_relevance(query, answer_a, answer_b),
                'legal_reasoning': self.evaluate_legal_reasoning(query, answer_a, answer_b)
            }
            
            # Configurable weights for different aspects (sum doesn't need to be 1)
            weights = {
                'legal_accuracy': 2.0,           # Most important for legal research
                'citation_quality': 1.5,         # Very important for credibility
                'completeness': 1.0,             # Standard importance
                'jurisdictional_relevance': 1.5, # Critical for legal applicability
                'legal_reasoning': 1.0           # Standard importance
            }
            
            # Calculate weighted scores
            total_weight = sum(weights.values())
            
            # Extract scores and calculate weighted totals
            score_a = 0
            score_b = 0
            
            for category, eval_result in evaluations.items():
                # Get scores with fallback to 5 if missing
                cat_score_a = eval_result.get('score_a', 5)
                cat_score_b = eval_result.get('score_b', 5)
                
                # Apply weights
                score_a += cat_score_a * weights[category]
                score_b += cat_score_b * weights[category]
            
            # Normalize to 0-100 scale
            score_a = (score_a / (total_weight * 10)) * 100
            score_b = (score_b / (total_weight * 10)) * 100
            
            # Determine winner with 2% buffer for ties
            score_diff = abs(score_a - score_b)
            
            if score_a > score_b + 2:  # 2% buffer
                winner = "A"
            elif score_b > score_a + 2:
                winner = "B"
            else:
                winner = "tie"
            
            # Calculate confidence based on score difference
            # Confidence levels with clear definitions:
            # 1 = Low: Very close (≤5% difference)
            # 2 = Medium: Clear winner (6-15% difference)  
            # 3 = High: Decisive victory (>15% difference)
            if score_diff <= 5:
                confidence = 1
            elif score_diff <= 15:
                confidence = 2
            else:
                confidence = 3
            
            # Build detailed reasoning from category evaluations
            reasoning_parts = []
            for category, eval_result in evaluations.items():
                cat_score_a = eval_result.get('score_a', 5)
                cat_score_b = eval_result.get('score_b', 5)
                if cat_score_a > cat_score_b:
                    reasoning_parts.append(f"{category.replace('_', ' ').title()}: A better ({cat_score_a} vs {cat_score_b})")
                elif cat_score_b > cat_score_a:
                    reasoning_parts.append(f"{category.replace('_', ' ').title()}: B better ({cat_score_b} vs {cat_score_a})")
            
            detailed_reasoning = f"Overall: A={score_a:.1f}, B={score_b:.1f} (diff={score_diff:.1f}%). " + "; ".join(reasoning_parts[:3])
            
            latency_ms = (time.time() - start_time) * 1000
            
            judgment = {
                "winner": winner,
                "confidence": confidence,
                "reasoning": detailed_reasoning,
                "detailed_scores": {
                    "answer_a": {cat: evaluations[cat].get('score_a', 5) for cat in evaluations},
                    "answer_b": {cat: evaluations[cat].get('score_b', 5) for cat in evaluations},
                    "weighted_total_a": round(score_a, 1),
                    "weighted_total_b": round(score_b, 1),
                    "score_difference": round(score_diff, 1)
                },
                "category_evaluations": evaluations,
                "weights_used": weights
            }
            
            logger.debug(f"Judgment complete: {winner} (confidence: {confidence}, diff: {score_diff:.1f}%)")
            
            return judgment, latency_ms
            
        except Exception as e:
            logger.error(f"Error in multi-prompt pairwise judgment: {e}")
            latency_ms = (time.time() - start_time) * 1000
            
            # Fallback to simple judgment
            return {
                "winner": "tie",
                "confidence": 1,
                "reasoning": f"Evaluation error: {str(e)}",
                "detailed_scores": {},
                "category_evaluations": {},
                "weights_used": {}
            }, latency_ms


class ContextEvaluator:
    """Main context evaluation class implementing the three metrics"""
    
    def __init__(self, groq_model: str = "llama-3.1-8b-instant", use_fetched_content: bool = True, use_semantic_similarity: bool = True):
        self.groq_client = GroqClient(model=groq_model)
        self.authority_scorer = AuthorityScorer()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.content_fetcher = ContentFetcher()
        self.use_fetched_content = use_fetched_content
        self.use_semantic_similarity = use_semantic_similarity
        
        # Initialize semantic similarity model if requested
        if self.use_semantic_similarity:
            logger.info("Loading semantic similarity model...")
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Semantic similarity model loaded")
    
    def standardize_and_fetch_content(self, providers_results: Dict[str, List[Dict]]) -> Dict[str, List[StandardizedSearchResult]]:
        """
        Standardize search results and optionally fetch full content for fair comparison.
        
        Args:
            providers_results: Raw results from each provider
            
        Returns:
            Dict of provider -> list of StandardizedSearchResult
        """
        standardized = {}
        
        # First, convert all results to StandardizedSearchResult
        for provider, results in providers_results.items():
            standardized[provider] = []
            for i, result in enumerate(results[:10], 1):  # Top 10 results
                std_result = StandardizedSearchResult.from_provider_result(result, provider, i)
                standardized[provider].append(std_result)
        
        # If using fetched content, fetch all unique URLs
        if self.use_fetched_content:
            logger.info("Fetching full content for fair comparison...")
            
            # Deduplicate URLs
            dedup_info = self.content_fetcher.deduplicate_urls(providers_results)
            unique_urls = dedup_info['unique_urls']
            
            logger.info(f"Found {len(unique_urls)} unique URLs across all providers")
            
            # Batch fetch content
            fetched_content = self.content_fetcher.batch_fetch(unique_urls)
            
            # Map fetched content back to results
            url_to_content = {url: content for url, content in zip(unique_urls, fetched_content)}
            
            for provider, std_results in standardized.items():
                for std_result in std_results:
                    if std_result.url in url_to_content:
                        std_result.add_fetched_content(url_to_content[std_result.url])
            
            # Log fetch statistics
            successful_fetches = sum(1 for content in fetched_content if content.get('success'))
            logger.info(f"Successfully fetched {successful_fetches}/{len(unique_urls)} URLs")
        
        return standardized
    
    def calculate_semantic_support_ratio(self, answer: str, documents: List[Dict]) -> Dict:
        """
        Calculate support ratio using semantic similarity instead of TF-IDF.
        Much better for long documents with mixed content.
        """
        if not answer or not documents:
            return {
                'support_ratio': 0.0,
                'supported_sentences': 0,
                'total_sentences': 0
            }
        
        # Split answer into sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if not sentences:
            return {
                'support_ratio': 0.0,
                'supported_sentences': 0,
                'total_sentences': 0
            }
        
        # Prepare document texts with intelligent chunking
        doc_chunks = []
        for doc in documents[:10]:  # Use top 10 documents
            text = doc.get('text', '').strip() or doc.get('snippet', '').strip() or doc.get('content', '').strip()
            if not text:
                continue
            
            # For semantic similarity, we can handle longer chunks but still need to be reasonable
            # Break very long documents into overlapping chunks
            if len(text) > 3000:
                # Split into 1500-char chunks with 300-char overlap
                words = text.split()
                chunk_size = 250  # ~1500 chars
                overlap = 50      # ~300 chars
                
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[i:i + chunk_size]
                    if len(chunk_words) < 30:  # Skip very short chunks
                        break
                    doc_chunks.append(' '.join(chunk_words))
            else:
                doc_chunks.append(text)
        
        if not doc_chunks:
            return {
                'support_ratio': 0.0,
                'supported_sentences': 0,
                'total_sentences': len(sentences)
            }
        
        try:
            # Encode sentences and document chunks
            logger.info(f"Encoding {len(sentences)} sentences and {len(doc_chunks)} document chunks for semantic similarity...")
            sentence_embeddings = self.semantic_model.encode(sentences, show_progress_bar=False)
            doc_embeddings = self.semantic_model.encode(doc_chunks, show_progress_bar=False)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(sentence_embeddings, doc_embeddings)
            
            # Count supported sentences with semantic similarity threshold
            # Semantic similarities are typically higher than TF-IDF, so we can use a higher threshold
            semantic_threshold = 0.6  # Higher threshold for semantic similarity
            supported_count = 0
            sentence_support = []
            
            for i, sentence in enumerate(sentences):
                max_similarity = similarities[i].max() if similarities[i].size > 0 else 0
                is_supported = max_similarity >= semantic_threshold
                supported_count += int(is_supported)
                
                sentence_support.append({
                    'sentence': sentence,
                    'max_similarity': float(max_similarity),
                    'is_supported': bool(is_supported),
                    'method': 'semantic'
                })
            
            support_ratio = supported_count / len(sentences) if sentences else 0.0
            
            logger.info(f"Semantic similarity: {supported_count}/{len(sentences)} sentences supported")
            
            return {
                'support_ratio': round(support_ratio, 3),
                'supported_sentences': supported_count,
                'total_sentences': len(sentences),
                'sentence_details': sentence_support,
                'method': 'semantic',
                'threshold': semantic_threshold,
                'num_doc_chunks': len(doc_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {e}")
            return {
                'support_ratio': 0.0,
                'supported_sentences': 0,
                'total_sentences': len(sentences),
                'error': str(e),
                'method': 'semantic'
            }
        
    def evaluate_llm_win_rate(self, query: str, providers_results: Dict[str, List[Dict]]) -> Dict:
        """
        Metric 1: LLMWinRate - Pairwise comparison of answers
        Returns win rates and all pairwise judgments
        """
        # Generate answers for each provider
        answers = {}
        generation_times = {}
        
        for provider, results in providers_results.items():
            if not results:
                logger.warning(f"No results for {provider}, skipping")
                continue
                
            answer, latency = self.groq_client.generate_answer(query, results)
            answers[provider] = answer
            generation_times[provider] = latency
            logger.info(f"Generated answer for {provider} in {latency:.1f}ms")
        
        # Perform pairwise comparisons
        providers = list(answers.keys())
        judgments = []
        win_counts = {p: 0 for p in providers}
        total_comparisons = {p: 0 for p in providers}
        
        for i, provider_a in enumerate(providers):
            for provider_b in providers[i+1:]:
                # Randomize order to avoid position bias
                if np.random.random() > 0.5:
                    # A vs B
                    judgment, judge_latency = self.groq_client.judge_pairwise(
                        query, answers[provider_a], answers[provider_b],
                        provider_a, provider_b
                    )
                    actual_a, actual_b = provider_a, provider_b
                    swap = False
                else:
                    # B vs A (swapped)
                    judgment, judge_latency = self.groq_client.judge_pairwise(
                        query, answers[provider_b], answers[provider_a],
                        provider_b, provider_a
                    )
                    actual_a, actual_b = provider_b, provider_a
                    swap = True
                
                # Adjust winner based on swap
                winner = judgment['winner']
                if swap:
                    if winner == 'A':
                        winner = 'B'
                    elif winner == 'B':
                        winner = 'A'
                
                # Update win counts
                if winner == 'A':
                    win_counts[actual_a] += 1
                elif winner == 'B':
                    win_counts[actual_b] += 1
                else:  # tie
                    win_counts[actual_a] += 0.5
                    win_counts[actual_b] += 0.5
                    
                total_comparisons[actual_a] += 1
                total_comparisons[actual_b] += 1
                
                # Store judgment
                judgments.append(PairwiseJudgment(
                    query=query,
                    provider_a=actual_a,
                    provider_b=actual_b,
                    answer_a=answers[actual_a],
                    answer_b=answers[actual_b],
                    winner=actual_a if winner == 'A' else (actual_b if winner == 'B' else 'tie'),
                    confidence=judgment['confidence'],
                    reasoning=judgment['reasoning'],
                    judgment_latency_ms=judge_latency
                ))
                
                logger.info(f"Judged {actual_a} vs {actual_b}: {winner} (confidence: {judgment['confidence']})")
        
        # Calculate win rates
        win_rates = {}
        for provider in providers:
            if total_comparisons[provider] > 0:
                win_rates[provider] = win_counts[provider] / total_comparisons[provider]
            else:
                win_rates[provider] = 0.0
        
        return {
            'win_rates': win_rates,
            'win_counts': win_counts,
            'total_comparisons': total_comparisons,
            'judgments': [asdict(j) for j in judgments],
            'answers': answers,
            'generation_times_ms': generation_times
        }
    
    def calculate_authority_of_cited(self, answer: str, search_results: List[Dict]) -> Dict:
        """
        Metric 2: AuthorityOfCited - Authority score of sources actually cited in answer
        """
        # Extract citation patterns from answer
        # Look for [1], [2], etc. or direct URL mentions
        citation_pattern = r'\[(\d+)\]'
        cited_indices = re.findall(citation_pattern, answer)
        
        # Also look for direct URL mentions
        url_pattern = r'https?://[^\s\)]+|www\.[^\s\)]+'
        mentioned_urls = re.findall(url_pattern, answer)
        
        # Map citations to search results
        cited_sources = []
        
        # Handle numbered citations
        for idx_str in cited_indices:
            idx = int(idx_str) - 1  # Convert to 0-based index
            if 0 <= idx < len(search_results):
                cited_sources.append(search_results[idx])
        
        # Handle direct URL mentions
        for url in mentioned_urls:
            for result in search_results:
                if url in result.get('url', ''):
                    if result not in cited_sources:
                        cited_sources.append(result)
                    break
        
        if not cited_sources:
            return {
                'authority_score': 0.0,
                'num_citations': 0,
                'cited_urls': []
            }
        
        # Calculate authority scores for cited sources
        authority_scores = []
        cited_urls = []
        
        for source in cited_sources:
            url = source.get('url', '')
            if url:
                domain_score = self.authority_scorer.get_domain_score(url)
                authority_scores.append(domain_score)
                cited_urls.append(url)
        
        avg_authority = np.mean(authority_scores) if authority_scores else 0.0
        
        return {
            'authority_score': round(avg_authority, 3),
            'num_citations': len(cited_sources),
            'cited_urls': cited_urls,
            'individual_scores': [round(s, 3) for s in authority_scores]
        }
    
    def calculate_support_ratio(self, answer: str, documents: List[Dict]) -> Dict:
        """
        Metric 3: SupportRatio - How much of the answer is backed by retrieved documents
        Uses semantic similarity (preferred) or TF-IDF similarity
        """
        # Use semantic similarity if available and enabled
        if hasattr(self, 'use_semantic_similarity') and self.use_semantic_similarity and hasattr(self, 'semantic_model'):
            return self.calculate_semantic_support_ratio(answer, documents)
        
        # Fall back to TF-IDF approach
        if not answer or not documents:
            return {
                'support_ratio': 0.0,
                'supported_sentences': 0,
                'total_sentences': 0
            }
        
        # Split answer into sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {
                'support_ratio': 0.0,
                'supported_sentences': 0,
                'total_sentences': 0
            }
        
        # Prepare document texts and track their lengths
        doc_texts = []
        doc_lengths = []
        for i, doc in enumerate(documents[:10]):  # Use top 10 documents
            # Fix: Use 'or' to properly fallback when 'text' is empty string
            text = doc.get('text', '').strip() or doc.get('snippet', '').strip() or doc.get('content', '').strip()
            if text:
                doc_texts.append(text)
                doc_lengths.append(len(text))
        
        if not doc_texts:
            return {
                'support_ratio': 0.0,
                'supported_sentences': 0,
                'total_sentences': len(sentences)
            }
        
        # Vectorize documents and sentences together
        all_texts = doc_texts + sentences
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Split back into documents and sentences
            doc_vectors = tfidf_matrix[:len(doc_texts)]
            sentence_vectors = tfidf_matrix[len(doc_texts):]
            
            # Calculate cosine similarity between each sentence and all documents
            similarities = cosine_similarity(sentence_vectors, doc_vectors)
            
            # Use standard threshold when fetched content is enabled (fair comparison)
            # Dynamic threshold only when using provider's original content
            if hasattr(self, 'use_fetched_content') and self.use_fetched_content:
                threshold = 0.5  # Standard threshold for fetched content
            else:
                # Dynamic threshold based on average document length
                # Long documents (like Exa's full articles) need lower thresholds
                avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
                if avg_doc_length > 2000:  # Full articles (Exa)
                    threshold = 0.3
                elif avg_doc_length > 500:  # Medium content
                    threshold = 0.45
                else:  # Short snippets (Google/SERP)
                    threshold = 0.6
            
            # Count supported sentences with dynamic threshold
            supported_count = 0
            sentence_support = []
            
            for i, sentence in enumerate(sentences):
                max_similarity = similarities[i].max() if similarities[i].size > 0 else 0
                is_supported = max_similarity >= threshold
                supported_count += int(is_supported)
                
                sentence_support.append({
                    'sentence': sentence,
                    'max_similarity': float(max_similarity),
                    'is_supported': bool(is_supported)
                })
            
            support_ratio = supported_count / len(sentences) if sentences else 0.0
            
            return {
                'support_ratio': round(support_ratio, 3),
                'supported_sentences': supported_count,
                'total_sentences': len(sentences),
                'sentence_details': sentence_support
            }
            
        except Exception as e:
            logger.error(f"Error calculating support ratio: {e}")
            return {
                'support_ratio': 0.0,
                'supported_sentences': 0,
                'total_sentences': len(sentences),
                'error': str(e)
            }
    
    def evaluate_all_metrics(self, query: str, providers_results: Dict[str, List[Dict]]) -> Dict:
        """
        Run all three context evaluation metrics
        """
        logger.info(f"Starting context evaluation for query: {query[:100]}...")
        
        evaluation_start = time.time()
        results = {
            'query': query,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'providers': {}
        }
        
        # Standardize and optionally fetch content for fair comparison
        if self.use_fetched_content:
            logger.info("Standardizing results and fetching content for fair comparison...")
            fetch_start = time.time()
            standardized_results = self.standardize_and_fetch_content(providers_results)
            
            # Convert StandardizedSearchResult back to dict format for compatibility
            # But use evaluation_text instead of original text
            providers_results_fair = {}
            for provider, std_results in standardized_results.items():
                providers_results_fair[provider] = []
                for std_result in std_results:
                    providers_results_fair[provider].append({
                        'url': std_result.url,
                        'title': std_result.evaluation_title,
                        'text': std_result.evaluation_text,  # Uses fetched content if available
                        'score': std_result.score,
                        'provider': std_result.provider,
                        '_original_length': std_result.original_text_length,
                        '_fetched_length': len(std_result.fetched_content) if std_result.fetched_content else 0,
                        '_fetch_success': std_result.fetch_success
                    })
            
            fetch_latency = (time.time() - fetch_start) * 1000
            results['content_fetch_latency_ms'] = fetch_latency
            
            # Use fair results for evaluation
            providers_results = providers_results_fair
        
        # Metric 1: LLM Win Rate (includes answer generation)
        logger.info("Evaluating LLM Win Rate...")
        win_rate_start = time.time()
        win_rate_results = self.evaluate_llm_win_rate(query, providers_results)
        win_rate_latency = (time.time() - win_rate_start) * 1000
        
        results['llm_win_rate'] = {
            'win_rates': win_rate_results['win_rates'],
            'win_counts': win_rate_results['win_counts'],
            'judgments_count': len(win_rate_results['judgments']),
            'evaluation_latency_ms': win_rate_latency
        }
        
        # Store judgments separately for detailed analysis
        results['pairwise_judgments'] = win_rate_results['judgments']
        
        # For each provider, calculate other metrics using generated answers
        for provider, search_results in providers_results.items():
            if provider not in win_rate_results['answers']:
                continue
                
            answer = win_rate_results['answers'][provider]
            provider_metrics = {
                'answer': answer,
                'generation_latency_ms': win_rate_results['generation_times_ms'][provider]
            }
            
            # Metric 2: Authority of Cited
            logger.info(f"Calculating Authority of Cited for {provider}...")
            auth_start = time.time()
            authority_results = self.calculate_authority_of_cited(answer, search_results)
            authority_results['latency_ms'] = (time.time() - auth_start) * 1000
            provider_metrics['authority_of_cited'] = authority_results
            
            # Metric 3: Support Ratio
            logger.info(f"Calculating Support Ratio for {provider}...")
            support_start = time.time()
            support_results = self.calculate_support_ratio(answer, search_results)
            support_results['latency_ms'] = (time.time() - support_start) * 1000
            provider_metrics['support_ratio'] = support_results
            
            results['providers'][provider] = provider_metrics
        
        # Total evaluation time
        results['total_evaluation_latency_ms'] = (time.time() - evaluation_start) * 1000
        
        logger.info(f"Context evaluation completed in {results['total_evaluation_latency_ms']:.1f}ms")
        
        return results


def generate_context_report(evaluations: List[Dict]) -> Dict:
    """
    Generate aggregate report across all query evaluations
    """
    if not evaluations:
        return {}
    
    # Extract all providers
    all_providers = set()
    for eval_result in evaluations:
        all_providers.update(eval_result.get('providers', {}).keys())
    
    providers = sorted(list(all_providers))
    
    report = {
        'summary': {},
        'detailed_metrics': {
            'llm_win_rate': {},
            'authority_of_cited': {},
            'support_ratio': {}
        },
        'queries_evaluated': len(evaluations)
    }
    
    # Aggregate metrics for each provider
    for provider in providers:
        # Collect metrics across all queries
        win_rates = []
        authority_scores = []
        support_ratios = []
        generation_times = []
        
        for eval_result in evaluations:
            # Win rate
            if 'llm_win_rate' in eval_result and provider in eval_result['llm_win_rate'].get('win_rates', {}):
                win_rates.append(eval_result['llm_win_rate']['win_rates'][provider])
            
            # Provider-specific metrics
            if provider in eval_result.get('providers', {}):
                provider_data = eval_result['providers'][provider]
                
                # Authority score
                if 'authority_of_cited' in provider_data:
                    authority_scores.append(provider_data['authority_of_cited']['authority_score'])
                
                # Support ratio
                if 'support_ratio' in provider_data:
                    support_ratios.append(provider_data['support_ratio']['support_ratio'])
                
                # Generation time
                if 'generation_latency_ms' in provider_data:
                    generation_times.append(provider_data['generation_latency_ms'])
        
        # Calculate averages
        report['summary'][provider] = {
            'avg_win_rate': round(np.mean(win_rates), 3) if win_rates else 0.0,
            'avg_authority_cited': round(np.mean(authority_scores), 3) if authority_scores else 0.0,
            'avg_support_ratio': round(np.mean(support_ratios), 3) if support_ratios else 0.0,
            'avg_generation_time_ms': round(np.mean(generation_times), 1) if generation_times else 0.0,
            'queries_with_data': len(win_rates)
        }
        
        # Store detailed metrics
        report['detailed_metrics']['llm_win_rate'][provider] = win_rates
        report['detailed_metrics']['authority_of_cited'][provider] = authority_scores
        report['detailed_metrics']['support_ratio'][provider] = support_ratios
    
    # Calculate overall winner based on composite score
    composite_scores = {}
    for provider in providers:
        summary = report['summary'][provider]
        # Weight: 40% win rate, 30% authority, 30% support
        composite = (
            summary['avg_win_rate'] * 0.4 +
            summary['avg_authority_cited'] * 0.3 +
            summary['avg_support_ratio'] * 0.3
        )
        composite_scores[provider] = round(composite, 3)
    
    report['composite_scores'] = composite_scores
    
    if composite_scores:
        best_provider = max(composite_scores, key=composite_scores.get)
        report['recommended_provider'] = {
            'name': best_provider,
            'composite_score': composite_scores[best_provider],
            'reasoning': f"{best_provider} achieved the highest composite score across all three context metrics"
        }
    
    return report