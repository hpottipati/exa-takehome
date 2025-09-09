"""
Context Evaluation Module for Legal Search Benchmarking
Implements three core metrics: LLMWinRate, AuthorityOfCited, SupportRatio
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from urllib.parse import urlparse

from .authority_scorer import AuthorityScorer

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


class OllamaClient:
    """Client for interacting with Ollama LLM"""
    
    def __init__(self, model: str = "qwen2.5:3b", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        
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
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": max_tokens
                }
            )
            answer = response['response']
            latency_ms = (time.time() - start_time) * 1000
            
            return answer, latency_ms
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return f"Error generating answer: {str(e)}", latency_ms
    
    def judge_pairwise(self, query: str, answer_a: str, answer_b: str, 
                      provider_a: str, provider_b: str) -> Tuple[Dict, float]:
        """
        Compare two answers and determine which is better
        Returns: (judgment_dict, latency_ms)
        """
        prompt = f"""You are an impartial legal research evaluator comparing two answers.

EVALUATION CRITERIA:
1. Legal accuracy and correctness
2. Quality and relevance of citations
3. Completeness of legal analysis
4. Clear legal reasoning
5. Proper identification of applicable laws/regulations

SCORING:
- Winner A: Answer A is clearly superior (more accurate, better citations, legally sound)
- Winner B: Answer B is clearly superior
- Tie: Both answers are equally good or equally poor

IMPORTANT: 
- Ignore answer length - focus on legal quality
- Base judgment on legal merit, not writing style
- Consider citation quality over quantity

Output ONLY valid JSON in this exact format:
{{
  "winner": "A" or "B" or "tie",
  "confidence": 1 or 2 or 3,
  "reasoning": "step-by-step explanation of why this answer wins"
}}

LEGAL QUESTION: {query}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Provide your evaluation as JSON:"""

        start_time = time.time()
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistency
                    "num_predict": 300
                }
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse JSON response
            response_text = response['response'].strip()
            
            # Clean the response text to handle control characters
            # Replace unescaped control characters with escaped versions
            def clean_json_string(text):
                """Clean a string that should contain JSON by escaping control characters"""
                # First, try to find the JSON object boundaries
                start_idx = text.find('{')
                end_idx = -1
                
                if start_idx != -1:
                    # Count braces to find the matching closing brace
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    
                    for i in range(start_idx, len(text)):
                        char = text[i]
                        
                        # Track if we're inside a string
                        if char == '"' and not escape_next:
                            in_string = not in_string
                        elif char == '\\':
                            escape_next = not escape_next
                            continue
                        
                        escape_next = False
                        
                        # Count braces only outside of strings
                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i
                                    break
                    
                    if end_idx > start_idx:
                        # Extract just the JSON part
                        json_str = text[start_idx:end_idx+1]
                        
                        # Replace unescaped control characters within string values
                        # This regex finds string values and processes them
                        def escape_string_value(match):
                            value = match.group(1)
                            # Replace actual newlines, tabs, etc. with escaped versions
                            value = value.replace('\n', '\\n')
                            value = value.replace('\r', '\\r')
                            value = value.replace('\t', '\\t')
                            # Remove other control characters
                            value = ''.join(char if ord(char) >= 32 or char in '\n\r\t' else ' ' for char in value)
                            return f'"{value}"'
                        
                        # Process string values in the JSON
                        json_str = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', escape_string_value, json_str)
                        return json_str
                
                return text
            
            # Try to parse the response directly first
            try:
                judgment = json.loads(response_text)
            except json.JSONDecodeError:
                # Clean the response and try again
                cleaned_text = clean_json_string(response_text)
                try:
                    judgment = json.loads(cleaned_text)
                except json.JSONDecodeError as e:
                    # If still failing, raise the error
                    raise json.JSONDecodeError(f"Failed after cleaning: {str(e)}", cleaned_text, 0)
            
            # Validate required fields
            if 'winner' not in judgment or 'confidence' not in judgment or 'reasoning' not in judgment:
                raise ValueError("Missing required fields in judgment")
            
            # Normalize winner value
            winner = judgment['winner'].upper()
            if winner not in ['A', 'B', 'TIE']:
                winner = 'TIE'
            judgment['winner'] = winner
            
            # Ensure confidence is int 1-3
            try:
                confidence = int(judgment['confidence'])
                judgment['confidence'] = max(1, min(3, confidence))
            except:
                judgment['confidence'] = 2
                
            return judgment, latency_ms
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse judge response as JSON: {str(e)}")
            logger.debug(f"Response text (first 500 chars): {response_text[:500]}")
            # Return default judgment on parse error
            return {
                "winner": "tie",
                "confidence": 1,
                "reasoning": f"Parse error: {str(e)}"
            }, latency_ms
                
        except Exception as e:
            logger.error(f"Error in pairwise judgment: {e}")
            latency_ms = (time.time() - start_time) * 1000
            return {
                "winner": "tie",
                "confidence": 1,
                "reasoning": f"Judgment error: {str(e)}"
            }, latency_ms


class ContextEvaluator:
    """Main context evaluation class implementing the three metrics"""
    
    def __init__(self, ollama_model: str = "qwen2.5:3b"):
        self.ollama_client = OllamaClient(model=ollama_model)
        self.authority_scorer = AuthorityScorer()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
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
                
            answer, latency = self.ollama_client.generate_answer(query, results)
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
                    judgment, judge_latency = self.ollama_client.judge_pairwise(
                        query, answers[provider_a], answers[provider_b],
                        provider_a, provider_b
                    )
                    actual_a, actual_b = provider_a, provider_b
                    swap = False
                else:
                    # B vs A (swapped)
                    judgment, judge_latency = self.ollama_client.judge_pairwise(
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
        """
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
        
        # Prepare document texts
        doc_texts = []
        for doc in documents[:10]:  # Use top 10 documents
            text = doc.get('text', doc.get('snippet', ''))
            if text:
                doc_texts.append(text)
        
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
            
            # Count supported sentences (max similarity >= 0.6)
            supported_count = 0
            sentence_support = []
            
            for i, sentence in enumerate(sentences):
                max_similarity = similarities[i].max() if similarities[i].size > 0 else 0
                is_supported = max_similarity >= 0.6
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