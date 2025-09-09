#!/usr/bin/env python3
"""
Visualize evaluation results from both authority and context evaluations
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.patches as mpatches
from datetime import datetime

def load_latest_results(results_dir: Path, pattern: str) -> Optional[Dict]:
    """Load the most recent results file matching pattern"""
    files = list(results_dir.glob(pattern))
    if not files:
        return None
    
    # Get most recent file
    latest = max(files, key=lambda f: f.stat().st_mtime)
    
    with open(latest, 'r') as f:
        return json.load(f)

def load_context_results() -> Optional[Dict]:
    """Load latest context evaluation results"""
    context_dir = Path("results/context_results")
    
    # Try to find latest batch
    latest_batch = context_dir / "latest_batch"
    if latest_batch.exists():
        report_file = latest_batch / "aggregate_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                return json.load(f)
    
    # Otherwise find most recent batch
    batches = [d for d in context_dir.glob("batch_*") if d.is_dir()]
    if batches:
        latest = max(batches, key=lambda d: d.stat().st_mtime)
        report_file = latest / "aggregate_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                return json.load(f)
    
    return None

def load_authority_results() -> Dict[str, Dict]:
    """Load all authority evaluation results"""
    authority_dir = Path("results/authority_results")
    if not authority_dir.exists():
        return {}
    
    results = {}
    
    # Load results for each provider with exact matching
    provider_patterns = {
        'exa': 'exa_authority_results*.json',
        'google': 'google_authority_results*.json',  # This will match exactly 'google_' not 'serp_google_'
        'serp_google': 'serp_google_authority_results*.json',
        'serp_bing': 'serp_bing_authority_results*.json',
        'serp_duckduckgo': 'serp_duckduckgo_authority_results*.json'
    }
    
    for provider, pattern in provider_patterns.items():
        # Find files matching the exact pattern
        matching_files = list(authority_dir.glob(pattern))
        # For google, exclude serp_google files
        if provider == 'google':
            matching_files = [f for f in matching_files if not f.name.startswith('serp_')]
        
        if matching_files:
            # Get most recent file
            latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
            with open(latest_file, 'r') as f:
                results[provider] = json.load(f)
    
    return results

def plot_context_evaluation(context_data: Dict):
    """Create visualizations for context evaluation results"""
    if not context_data or 'summary' not in context_data:
        print("No context evaluation data available")
        return
    
    summary = context_data['summary']
    composite_scores = context_data.get('composite_scores', {})
    
    # Create figure with 3 key charts - simplified for README
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('Context Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Composite Scores (Main overall metric)
    ax1 = plt.subplot(1, 3, 1)
    if composite_scores:
        # Sort by score (highest first)
        sorted_items = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        providers = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]
        colors = [plt.cm.viridis((0.9 - 0.3) * i / max(1, len(providers) - 1) + 0.3) for i in range(len(providers))]
        
        bars = ax1.bar(providers, scores, color=colors)
        ax1.set_title('Composite Scores (Higher = Better)', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Composite Score', fontsize=12)
        ax1.set_ylim(0, max(scores) * 1.2 if scores else 1)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Win Rate Comparison (Key performance indicator)
    ax2 = plt.subplot(1, 3, 2)
    win_rates = {p: data.get('avg_win_rate', 0) for p, data in summary.items()}
    if win_rates:
        # Sort by win rate (highest first)
        sorted_items = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)
        providers = [item[0] for item in sorted_items]
        rates = [item[1] * 100 for item in sorted_items]
        colors = [plt.cm.coolwarm((0.9 - 0.3) * i / max(1, len(providers) - 1) + 0.3) for i in range(len(providers))]
        
        bars = ax2.bar(providers, rates, color=colors)
        ax2.set_title('Win Rate (Higher = Better)', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Win Rate (%)', fontsize=12)
        ax2.set_ylim(0, 100)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Support Ratio (Evidence backing)
    ax3 = plt.subplot(1, 3, 3)
    support_ratios = {p: data.get('avg_support_ratio', 0) for p, data in summary.items()}
    if support_ratios:
        # Sort by support ratio (highest first)
        sorted_items = sorted(support_ratios.items(), key=lambda x: x[1], reverse=True)
        providers = [item[0] for item in sorted_items]
        ratios = [item[1] * 100 for item in sorted_items]
        colors = [plt.cm.plasma((0.9 - 0.3) * i / max(1, len(providers) - 1) + 0.3) for i in range(len(providers))]
        
        bars = ax3.bar(providers, ratios, color=colors)
        ax3.set_title('Support Ratio (Higher = Better)', fontweight='bold', fontsize=14)
        ax3.set_ylabel('Support Ratio (%)', fontsize=12)
        ax3.set_ylim(0, 100)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{ratio:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Add metadata
    metadata = context_data.get('metadata', {})
    fig.text(0.02, 0.02, f"Queries: {metadata.get('successful_queries', 0)}/{metadata.get('total_queries', 0)}", 
             fontsize=10, ha='left')
    
    if 'timestamp' in metadata:
        fig.text(0.98, 0.02, f"Generated: {metadata['timestamp']}", fontsize=10, ha='right')
    
    return fig

def plot_authority_evaluation(authority_data: Dict[str, Dict]):
    """Create visualizations for authority evaluation results"""
    if not authority_data:
        print("No authority evaluation data available")
        return
    
    # Extract aggregated metrics
    providers = []
    authority_at_3 = []
    high_auth_rates = []
    utility_scores = []
    
    for provider, data in authority_data.items():
        if 'aggregate_report' in data and 'summary' in data['aggregate_report']:
            for prov_name, metrics in data['aggregate_report']['summary'].items():
                providers.append(prov_name)
                authority_at_3.append(metrics.get('authority_at_3', 0))
                high_auth_rates.append(metrics.get('high_auth_hit_rate', 0))
                utility_scores.append(metrics.get('avg_utility_score_at_10', 0))
    
    if not providers:
        print("No valid authority data to plot")
        return
    
    # Create figure - simplified to 2 key charts
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('Authority Evaluation Results', fontsize=16, fontweight='bold')
    
    # Sort all data by authority@3 (highest first)
    sorted_data = sorted(zip(providers, authority_at_3, high_auth_rates, utility_scores), key=lambda x: x[1], reverse=True)
    providers = [x[0] for x in sorted_data]
    authority_at_3 = [x[1] for x in sorted_data]
    high_auth_rates = [x[2] for x in sorted_data]
    utility_scores = [x[3] for x in sorted_data]
    
    # 1. Authority@3 Scores (Main metric)
    ax1 = plt.subplot(1, 2, 1)
    colors = [plt.cm.viridis((0.9 - 0.3) * i / max(1, len(providers) - 1) + 0.3) for i in range(len(providers))]
    bars = ax1.bar(providers, authority_at_3, color=colors)
    ax1.set_title('Authority@3 Scores (Higher = Better)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Authority Score', fontsize=12)
    ax1.set_ylim(0, max(authority_at_3) * 1.2 if authority_at_3 else 1)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, score in zip(bars, authority_at_3):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. High Authority Hit Rate (Key reliability metric)
    ax2 = plt.subplot(1, 2, 2)
    colors = [plt.cm.coolwarm((0.9 - 0.3) * i / max(1, len(providers) - 1) + 0.3) for i in range(len(providers))]
    bars = ax2.bar(providers, [r * 100 for r in high_auth_rates], color=colors)
    ax2.set_title('High Authority Hit Rate (Higher = Better)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Hit Rate (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, rate in zip(bars, high_auth_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    return fig

def create_combined_dashboard():
    """Create a combined dashboard with both evaluations"""
    context_data = load_context_results()
    authority_data = load_authority_results()
    
    if not context_data and not authority_data:
        print("No evaluation results found in results/ directory")
        print("Please run evaluations first with: python3 run_all_evals.py")
        return
    
    # Create visualizations directory
    viz_dir = Path("visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    # Create context evaluation plot
    if context_data:
        print("Creating context evaluation visualizations...")
        context_fig = plot_context_evaluation(context_data)
        if context_fig:
            context_fig.savefig(viz_dir / 'context_evaluation_dashboard.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {viz_dir}/context_evaluation_dashboard.png")
    
    # Create authority evaluation plot
    if authority_data:
        print("Creating authority evaluation visualizations...")
        authority_fig = plot_authority_evaluation(authority_data)
        if authority_fig:
            authority_fig.savefig(viz_dir / 'authority_evaluation_dashboard.png', dpi=150, bbox_inches='tight')
            print(f"Saved: {viz_dir}/authority_evaluation_dashboard.png")
    
    # Create combined summary figure
    if context_data and authority_data:
        print("Creating combined summary...")
        
        fig = plt.figure(figsize=(16, 6))
        fig.suptitle('Search Provider Benchmarking - Combined Results', fontsize=18, fontweight='bold')
        
        # Context scores on left
        ax1 = plt.subplot(1, 2, 1)
        if 'composite_scores' in context_data:
            scores = context_data['composite_scores']
            # Sort by score (highest first)
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            providers = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            colors = [plt.cm.viridis((0.9 - 0.3) * i / max(1, len(providers) - 1) + 0.3) for i in range(len(providers))]
            bars = ax1.barh(providers, values, color=colors)
            ax1.set_title('Context Evaluation - Composite Scores (Best→Worst)', fontweight='bold')
            ax1.set_xlabel('Score')
            
            for bar, val in zip(bars, values):
                width = bar.get_width()
                ax1.text(width, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center')
        
        # Authority scores on right
        ax2 = plt.subplot(1, 2, 2)
        
        # Extract authority scores
        auth_scores = {}
        for provider, data in authority_data.items():
            if 'aggregate_report' in data and 'summary' in data['aggregate_report']:
                for prov_name, metrics in data['aggregate_report']['summary'].items():
                    auth_scores[prov_name] = metrics.get('authority_at_3', 0)
        
        if auth_scores:
            # Sort by authority score (highest first)
            sorted_items = sorted(auth_scores.items(), key=lambda x: x[1], reverse=True)
            providers = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            colors = [plt.cm.plasma((0.9 - 0.3) * i / max(1, len(providers) - 1) + 0.3) for i in range(len(providers))]
            bars = ax2.barh(providers, values, color=colors)
            ax2.set_title('Authority Evaluation - Authority@3 Scores (Best→Worst)', fontweight='bold')
            ax2.set_xlabel('Score')
            
            for bar, val in zip(bars, values):
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        fig.savefig(viz_dir / 'combined_summary.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {viz_dir}/combined_summary.png")
    
    print(f"\nAll visualizations created successfully in {viz_dir}/ folder!")

if __name__ == "__main__":
    print("Search Provider Evaluation Visualizer")
    print("=" * 50)
    create_combined_dashboard()