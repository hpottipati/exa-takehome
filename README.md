# Legal Search Benchmark

**TLDR**: A set of evals to answer the question — *Which search provider delivers the most authoritative and useful results for legal research?*


**VIDEO** : https://youtu.be/9BzsxnOtWO4
---
## The Problem

Traditional search engines optimize for broad relevance, not authority. Lawyers need **precise answers and credible sources**

## The Solution

A benchmark that measures what matters:

1. **Authority of Sources**: Are results from .gov, case law, or trusted publishers?
2. **Context Quality**: Do the retrieved results actually support accurate answers?

This framework makes search APIs directly comparable on dimensions customers care about: **credibility and answer quality**. If Exa performs better, that's strong evidence it delivers superior value for legal research and compliance workflows.

---

## How It Works

**Drop in any provider**:
- Exa (semantic search)
- Google Programmable Search
- SerpAPI (Bing, DuckDuckGo, Google)

**Two Core Metrics**:
- **Authority Score**: Domain + citation-based heuristic (e.g. `.gov` = 1.0, Cornell/Justia = 0.9, general domains = 0.5, etc).
- **Context Quality**: LLM-as-a-judge grades answers based on legal accuracy, citations, and completeness, grounded in retrieved documents. The context will differ depending on the top results of each search API. 

**Fair Comparison**: We fetch **THE FULL PAGE** to minimze bias

---

## Test Scenarios

10 realistic research queries lawyers face, spanning employment law, securities regulation, data privacy, corporate law, and federal compliance.

---

## Setup

```bash
pip install -r requirements.txt
```

Add API keys to `.env`:
```
EXA_API_KEY=your_key
GOOGLE_API_KEY=your_key  
GOOGLE_SEARCH_ENGINE_ID=your_id
SERP_API_KEY=your_key
GROQ_API_KEY=your_key
```

Run all evals:
```bash
python3 run_all_evals.py
```

Run specific tests:
```bash
# Authority scoring
python3 tests/authority/run_all_authority_tests.py

# Context eval
python3 tests/context/run_batch_context_eval.py --providers exa google
```

**Authority Metrics** (`data/results/`):
- `authority_at_3`: Avg. authority of top 3 results
- `high_auth_hit_rate`: % queries w/ ≥1 authoritative source in top 3
- `avg_utility_score_at_10`: Weighted authority across top 10

**Context Metrics** (`data/context_results/`):
- `avg_win_rate`: How often provider's AI answers beat others
- `avg_authority_cited`: Avg. authority of cited sources
- `avg_support_ratio`: % of answer sentences backed by retrieved content

---

## How Calculations Work

**Authority Score**: Domain weight × citation bonus. `.gov` = 1.0, legal publishers = 0.7-0.9, others = 0.5. Citation bonus = 0.3 + min(0.7, citations_per_100_words × 20).

**Context Quality**: Groq Llama 3.1 judges answers on 5 criteria (legal accuracy, citations, completeness, jurisdiction, reasoning) with weighted scoring. Win rate from pairwise comparisons across all providers.

**Support Ratio**: Sentence transformers calculate semantic similarity between each answer sentence and retrieved documents. Sentences above threshold count as "supported".

## Results

<img width="2234" height="755" alt="context_evaluation_dashboard" src="https://github.com/user-attachments/assets/511ccab1-c317-42e4-a56b-4d6a56f7a409" />
<img width="2380" height="887" alt="combined_summary" src="https://github.com/user-attachments/assets/615a617b-f0f5-472d-a4ca-f7aa8162d2c3" />
<img width="1784" height="742" alt="authority_evaluation_dashboard" src="https://github.com/user-attachments/assets/eaa37b1d-953f-4117-b98e-f775fbaeb62d" />


### Time Usage
<img width="256" height="591" alt="image" src="https://github.com/user-attachments/assets/24763d4e-4486-4840-9aa0-77f3a76e2876" />
