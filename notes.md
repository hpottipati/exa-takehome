# Start 7:50 PM



### Brain dump
"""

MTEB - 

The eval can be either general or specific. We’re always looking for novel applications, so it could be a good idea to focus on a particular (possibly niche!) vertical. 
Initial ideas: Better coding agent when retrieve appropriate docs, Law firm research citations or some other law related shit,

Based on research
1. semantic search for case docs
2. Hidden precedents (semantic gaps) - seems the best, lots of customer pain points here. "Does Exa api find non-obvious but legally relevant precedents that normal search misses(like keywords or BM25)?"
3. search across siloed info in cross domain retreival
""

# Planning
### high level
Compare Exa vs 5 Competitors
Dataset (queries + cases)
Evaluation harness (run + eval)
Possibly something to run queries in parallel with live results and evaluation

# "Can someone use the results to answer their legal research question correctly and efficiently


I want to make a solution where we are evaluating two things:
1. The authority levels of each of the top k results
2. A simple context-based pipeline and evaluating the answers from each search and figuring out if Exa is better than competitors, etc.

weights and evals for now:

```json
"evaluation_framework": {
    "authority_scoring": {
      "domains": {
        ".gov": 1.0,
        "law.cornell.edu": 0.9,
        "oyez.org": 0.9,
        "justia.com": 0.7,
        "findlaw.com": 0.7,
        "other": 0.5
      },
      "citation_bonus": "0.3 + min(0.7, legal_citations_per_100_words * 20)"
    },
    "context_evaluation": {
      "support_ratio": "how much of the generated answer is actually backed by the retrieved documents",
      "authority_of_cited": "how authoritative are the sources that the LLM actually references in its answer", 
      "llm_win_rate": "how often an independent LLM judge prefers the answer generated using Provider A vs Provider B",
    },
    "metrics": {
        "authority_metrics": ["Authority@k", "HighAuthHit@k", "UtilityScore@k"],
        "context_metrics": ["LLMWinRate", "AuthorityOfCited", "SupportRatio"]
    },
    "latency":{
      "retrival_time": seconds for api call
    }
}```