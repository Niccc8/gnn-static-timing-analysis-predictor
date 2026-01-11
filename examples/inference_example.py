"""
Inference Examples

Demonstrates various ways to use the GNN timing predictor for circuit analysis.
Shows threshold-based, rank-based, and adaptive prediction methods.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.predict import TimingPredictor
from src.data.dataset import TimingDataset


def main():
    """Run comprehensive inference examples."""
    
    # ============================================
    # Example 1: Load Predictor and Data
    # ============================================
    print("="*60)
    print("Loading Model and Data")
    print("="*60)
    
    predictor = TimingPredictor(
        checkpoint_path='experiments/checkpoints/best_model.pth',
        device='cuda'  # Use 'cpu' if CUDA unavailable
    )
    
    # Load test circuit
    dataset = TimingDataset(root='data/processed/timing_predict/', split='test')
    circuit = dataset[1]  # Circuit with highest violation rate
    
    true_violations = (circuit.y == 1).cpu().numpy()
    num_true_violations = true_violations.sum()
    
    print(f"\nCircuit: {circuit.num_nodes:,} nodes")
    print(f"True violations: {num_true_violations} ({num_true_violations/circuit.num_nodes*100:.2f}%)")
    
    # ============================================
    # Example 2: Threshold-Based Prediction
    # ============================================
    print("\n" + "="*60)
    print("Method 1: Threshold-Based Prediction")
    print("="*60)
    
    threshold_results = predictor.predict_threshold(data=circuit, threshold=0.58)
    
    threshold_preds = threshold_results['predictions']
    violations_caught = true_violations[threshold_preds == 1].sum()
    recall = violations_caught / num_true_violations if num_true_violations > 0 else 0
    precision = violations_caught / threshold_results['num_flagged'] if threshold_results['num_flagged'] > 0 else 0
    
    print(f"Flagged: {threshold_results['num_flagged']} / {circuit.num_nodes} nodes ({threshold_results['flagged_ratio']*100:.2f}%)")
    print(f"Recall: {recall*100:.1f}%")
    print(f"Precision: {precision*100:.1f}%")
    
    if violations_caught < num_true_violations:
        print(f"⚠️  Missed {num_true_violations - violations_caught} violations!")
    
    # ============================================
    # Example 3: Rank-Based Prediction (RECOMMENDED)
    # ============================================
    print("\n" + "="*60)
    print("Method 2: Rank-Based Prediction (Recommended)")
    print("="*60)
    
    ranking_results = predictor.predict_top_k(data=circuit, top_percent=5.0)
    
    top_k_indices = np.array(ranking_results['top_k_indices'])
    violations_caught_rank = true_violations[top_k_indices].sum()
    recall_rank = violations_caught_rank / num_true_violations
    precision_rank = violations_caught_rank / len(top_k_indices)
    
    print(f"Inspecting top {ranking_results['k']:,} nodes ({ranking_results['top_percent']:.1f}%)")
    print(f"Risk range: {ranking_results['min_top_k_prob']:.4f} - {ranking_results['max_top_k_prob']:.4f}")
    print(f"Recall: {recall_rank*100:.1f}%")
    print(f"Precision: {precision_rank*100:.1f}%")
    
    # ============================================
    # Example 4: Adaptive Ranking
    # ============================================
    print("\n" + "="*60)
    print("Method 3: Adaptive Ranking")
    print("="*60)
    
    adaptive_results = predictor.predict_adaptive(data=circuit, sensitivity_threshold=0.01)
    
    adaptive_indices = np.array(adaptive_results['top_k_indices'])
    violations_caught_adaptive = true_violations[adaptive_indices].sum()
    recall_adaptive = violations_caught_adaptive / num_true_violations
    precision_adaptive = violations_caught_adaptive / len(adaptive_indices)
    
    print(f"Adaptive K: {adaptive_results['k']:,} nodes ({adaptive_results['top_percent']:.2f}%)")
    print(f"Recall: {recall_adaptive*100:.1f}%")
    print(f"Precision: {precision_adaptive*100:.1f}%")
    
    if violations_caught_adaptive == num_true_violations:
        print("✅ Caught ALL violations!")
    
    # ============================================
    # Example 5: Probability Distribution Analysis
    # ============================================
    print("\n" + "="*60)
    print("Probability Distribution Analysis")
    print("="*60)
    
    probs = predictor.predict(circuit)
    
    violation_probs = probs[true_violations == 1]
    clean_probs = probs[true_violations == 0]
    
    print(f"\nTrue Violations ({len(violation_probs)} nodes):")
    print(f"  Min: {violation_probs.min():.4f}, Max: {violation_probs.max():.4f}, Mean: {violation_probs.mean():.4f}")
    
    print(f"\nClean Nodes ({len(clean_probs)} nodes):")
    print(f"  Min: {clean_probs.min():.4f}, Max: {clean_probs.max():.4f}, Mean: {clean_probs.mean():.4f}")
    
    print(f"\nOverall Statistics:")
    print(f"  Min: {probs.min():.4f}")
    print(f"  Max: {probs.max():.4f}")
    print(f"  Mean: {probs.mean():.4f}")
    print(f"  Median: {np.median(probs):.4f}")
    
    high_risk = (probs > 0.9).sum()
    print(f"\nHigh-risk nodes (>0.9): {high_risk}")
    
    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    print("\nMethod Comparison:")
    print(f"  {'Method':<20} {'Inspect':<12} {'Recall':<10} {'Precision':<10}")
    print("-" * 60)
    print(f"  {'Threshold (0.58)':<20} {threshold_results['num_flagged']:<12,} {recall*100:>6.1f}%     {precision*100:>6.1f}%")
    print(f"  {'Rank (Top 5%)':<20} {ranking_results['k']:<12,} {recall_rank*100:>6.1f}%     {precision_rank*100:>6.1f}%")
    print(f"  {'Adaptive (0.01)':<20} {adaptive_results['k']:<12,} {recall_adaptive*100:>6.1f}%     {precision_adaptive*100:>6.1f}%")
    
    print("\n💡 Recommendation: Use adaptive ranking for best recall-precision trade-off")
    print(f"   Inspects {adaptive_results['top_percent']:.1f}% of nodes to catch {recall_adaptive*100:.0f}% of violations")


if __name__ == '__main__':
    main()
