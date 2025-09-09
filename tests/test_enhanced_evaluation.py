#!/usr/bin/env python3
"""
Test script for the enhanced multi-prompt evaluation system
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluators.context_evaluator import OllamaClient

def test_enhanced_evaluation():
    """Test the enhanced multi-prompt evaluation system"""
    
    print("="*60)
    print("TESTING ENHANCED MULTI-PROMPT EVALUATION SYSTEM")
    print("="*60)
    
    # Initialize the client
    client = OllamaClient(model="qwen2.5:3b", temperature=0.1)
    
    # Sample legal question
    query = "What are the compliance requirements for using AI in hiring in NYC?"
    
    # Sample answers (simplified for testing)
    answer_a = """Based on NYC Local Law 144 (effective January 1, 2023), employers using AI in hiring must:

1. Conduct annual bias audits by independent auditors [1]
2. Publish audit results on company website [2]
3. Provide notice to candidates about AI use [3]
4. Include data on impact ratios by race/ethnicity and gender [1]

The law applies to employers and employment agencies in NYC using automated employment decision tools (AEDT) for screening candidates or employees for hiring or promotion [2].

Penalties include civil penalties up to $1,500 per violation [3].

[1] NYC Local Law 144, Section 20-871
[2] NYC DCWP Rules, Chapter 5
[3] NYC Admin Code § 20-870"""
    
    answer_b = """AI hiring in NYC requires following some rules:

Companies need to be careful when using AI tools for hiring. There are requirements about fairness and transparency. 

The city wants to make sure AI doesn't discriminate. Some audits might be needed.

Check with legal counsel for specific requirements as regulations are evolving."""
    
    print("\nQuestion:", query)
    print("\nTesting pairwise judgment...")
    print("-"*40)
    
    try:
        # Run the enhanced pairwise judgment
        judgment, latency = client.judge_pairwise(
            query, answer_a, answer_b, 
            "Provider_A", "Provider_B"
        )
        
        print(f"\n✓ Evaluation completed in {latency:.1f}ms")
        print("\nRESULTS:")
        print("-"*40)
        print(f"Winner: {judgment['winner']}")
        print(f"Confidence: {judgment['confidence']} ", end="")
        
        # Explain confidence level
        if judgment['confidence'] == 1:
            print("(Low - very close call)")
        elif judgment['confidence'] == 2:
            print("(Medium - clear winner)")
        else:
            print("(High - decisive victory)")
        
        print(f"\nReasoning: {judgment['reasoning']}")
        
        if 'detailed_scores' in judgment:
            print("\nDETAILED SCORES:")
            print("-"*40)
            scores = judgment['detailed_scores']
            
            print("Answer A scores:")
            for cat, score in scores['answer_a'].items():
                print(f"  {cat.replace('_', ' ').title()}: {score}/10")
            print(f"  Weighted Total: {scores['weighted_total_a']}/100")
            
            print("\nAnswer B scores:")
            for cat, score in scores['answer_b'].items():
                print(f"  {cat.replace('_', ' ').title()}: {score}/10")
            print(f"  Weighted Total: {scores['weighted_total_b']}/100")
            
            print(f"\nScore Difference: {scores['score_difference']}%")
        
        if 'weights_used' in judgment:
            print("\nWEIGHTS USED:")
            print("-"*40)
            for cat, weight in judgment['weights_used'].items():
                print(f"  {cat.replace('_', ' ').title()}: {weight}x")
        
        # Show some category details if available
        if 'category_evaluations' in judgment:
            print("\nCATEGORY INSIGHTS:")
            print("-"*40)
            evals = judgment['category_evaluations']
            
            # Legal accuracy details
            if 'legal_accuracy' in evals:
                legal = evals['legal_accuracy']
                if 'errors_a' in legal and legal['errors_a']:
                    print(f"Legal errors in A: {', '.join(legal['errors_a'][:2])}")
                if 'errors_b' in legal and legal['errors_b']:
                    print(f"Legal errors in B: {', '.join(legal['errors_b'][:2])}")
            
            # Citation quality details
            if 'citation_quality' in evals:
                citations = evals['citation_quality']
                if 'quality_assessment' in citations:
                    print(f"Citation assessment: {citations['quality_assessment']}")
        
        print("\n" + "="*60)
        print("✅ Enhanced evaluation system working correctly!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check if Ollama is running
    print("Checking Ollama connection...")
    try:
        import ollama
        models = ollama.list()
        print("✓ Ollama is running")
        
        # Check for the model
        model_names = []
        if hasattr(models, 'models'):
            for model in models.models:
                if hasattr(model, 'name'):
                    model_names.append(model.name)
        
        if 'qwen2.5:3b' not in model_names:
            print("⚠️  Warning: qwen2.5:3b model not found")
            print("   Run: ollama pull qwen2.5:3b")
            print("   Continuing anyway...")
        else:
            print("✓ qwen2.5:3b model available")
            
    except Exception as e:
        print(f"⚠️  Ollama connection issue: {e}")
        print("   Make sure Ollama is running: ollama serve")
        print("   Continuing anyway...")
    
    print()
    
    # Run the test
    success = test_enhanced_evaluation()
    
    if success:
        print("\n🎉 All tests passed! The enhanced evaluation system is ready.")
    else:
        print("\n⚠️  Tests failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)