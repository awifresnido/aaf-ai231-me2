"""
Benchmark Comparison & Reporting Utilities
Compares VCM baseline vs tuned results against GSC reference benchmarks
"""

import json
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BenchmarkComparison:
    """Compare your VCM model against benchmarks and reference data"""
    
    def __init__(self, gsc_reference_file: str = 'configs/gsc_benchmark_reference.json'):
        self.gsc_ref_path = Path(gsc_reference_file)
        self.gsc_data = self._load_gsc_reference()
    
    def _load_gsc_reference(self) -> Dict:
        """Load GSC reference benchmark data"""
        if not self.gsc_ref_path.exists():
            logger.warning(f"GSC reference file not found: {self.gsc_ref_path}")
            return {}
        
        with open(self.gsc_ref_path, 'r') as f:
            return json.load(f)
    
    def update_baseline_results(
        self,
        accuracy: float,
        latency_ms: float,
        model_size_kb: float,
        f1_score: float,
    ):
        """Update GSC reference file with your baseline results"""
        if not self.gsc_data:
            return
        
        self.gsc_data['your_vcm_benchmark']['baseline']['accuracy'] = accuracy
        self.gsc_data['your_vcm_benchmark']['baseline']['latency_ms'] = latency_ms
        self.gsc_data['your_vcm_benchmark']['baseline']['model_size_kb'] = model_size_kb
        self.gsc_data['your_vcm_benchmark']['baseline']['f1_score_weighted'] = f1_score
        
        self._save_updated_reference()
    
    def update_tuned_results(
        self,
        trial_number: int,
        accuracy: float,
        latency_ms: float,
        model_size_kb: float,
        f1_score: float,
    ):
        """Update GSC reference file with your best tuned results"""
        if not self.gsc_data:
            return
        
        self.gsc_data['your_vcm_benchmark']['tuned_best']['trial_number'] = trial_number
        self.gsc_data['your_vcm_benchmark']['tuned_best']['accuracy'] = accuracy
        self.gsc_data['your_vcm_benchmark']['tuned_best']['latency_ms'] = latency_ms
        self.gsc_data['your_vcm_benchmark']['tuned_best']['model_size_kb'] = model_size_kb
        self.gsc_data['your_vcm_benchmark']['tuned_best']['f1_score_weighted'] = f1_score
        
        # Compute improvements
        baseline = self.gsc_data['your_vcm_benchmark']['baseline']
        if baseline['accuracy'] and baseline['accuracy'] > 0:
            acc_delta = accuracy - baseline['accuracy']
            acc_delta_pct = (acc_delta / baseline['accuracy']) * 100
            
            self.gsc_data['your_vcm_benchmark']['improvement']['accuracy_delta'] = round(acc_delta, 4)
            self.gsc_data['your_vcm_benchmark']['improvement']['accuracy_delta_pct'] = round(acc_delta_pct, 2)
            self.gsc_data['your_vcm_benchmark']['improvement']['latency_delta_ms'] = round(latency_ms - baseline['latency_ms'], 2)
            self.gsc_data['your_vcm_benchmark']['improvement']['latency_constraint_met'] = latency_ms < 500
        
        self._save_updated_reference()
    
    def _save_updated_reference(self):
        """Save updated reference data"""
        with open(self.gsc_ref_path, 'w') as f:
            json.dump(self.gsc_data, f, indent=2)
        logger.info(f"✓ Updated benchmark reference: {self.gsc_ref_path}")
    
    def generate_comparison_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate markdown comparison report
        
        Args:
            output_file: Path to save report (optional)
        
        Returns:
            Markdown report string
        """
        if not self.gsc_data:
            return "# Error: Could not load benchmark reference data\n"
        
        report = []
        report.append("# VCM Hyperparameter Tuning: Benchmark Comparison Report\n")
        
        # ===== YOUR RESULTS =====
        report.append("## Your VCM Model Results\n")
        
        baseline = self.gsc_data['your_vcm_benchmark']['baseline']
        tuned = self.gsc_data['your_vcm_benchmark']['tuned_best']
        
        if baseline['accuracy'] and baseline['accuracy'] > 0:
            report.append("### Baseline vs Tuned Comparison\n")
            report.append("|Metric|Baseline|Tuned|Change|Change %|\n")
            report.append("|------|--------|-----|------|--------|\n")
            
            # Accuracy
            if tuned['accuracy'] and tuned['accuracy'] > 0:
                acc_delta = tuned['accuracy'] - baseline['accuracy']
                acc_delta_pct = (acc_delta / baseline['accuracy']) * 100
                report.append(f"|Accuracy|{baseline['accuracy']:.4f}|{tuned['accuracy']:.4f}|")
                report.append(f"+{acc_delta:.4f}|+{acc_delta_pct:.2f}%|\n")
            
            # Latency
            if tuned['latency_ms'] and tuned['latency_ms'] > 0:
                latency_delta = tuned['latency_ms'] - baseline['latency_ms']
                report.append(f"|Latency (ms)|{baseline['latency_ms']:.2f}|{tuned['latency_ms']:.2f}|")
                report.append(f"{latency_delta:+.2f}|")
                if latency_delta < 0:
                    report.append(f"{(latency_delta/baseline['latency_ms'])*100:.1f}%|\n")
                else:
                    report.append(f"+{(latency_delta/baseline['latency_ms'])*100:.1f}%|\n")
            
            # Model Size
            if tuned['model_size_kb'] and tuned['model_size_kb'] > 0:
                size_delta = tuned['model_size_kb'] - baseline['model_size_kb']
                report.append(f"|Model Size (KB)|{baseline['model_size_kb']:.2f}|{tuned['model_size_kb']:.2f}|")
                report.append(f"{size_delta:+.2f}|")
                if size_delta < 0:
                    report.append(f"{(size_delta/baseline['model_size_kb'])*100:.1f}%|\n")
                else:
                    report.append(f"+{(size_delta/baseline['model_size_kb'])*100:.1f}%|\n")
            
            # F1-Score
            if tuned['f1_score_weighted'] and tuned['f1_score_weighted'] > 0:
                f1_delta = tuned['f1_score_weighted'] - baseline['f1_score_weighted']
                report.append(f"|F1-Score (weighted)|{baseline['f1_score_weighted']:.4f}|{tuned['f1_score_weighted']:.4f}|")
                report.append(f"+{f1_delta:.4f}|+{(f1_delta/baseline['f1_score_weighted'])*100:.2f}%|\n")
        
        # ===== CONTEXT: GSC BENCHMARKS =====
        report.append("\n## Context: Google Speech Commands V2 Benchmarks\n")
        report.append("*(Your 13-intent task is simpler than GSC 35-word task, but uses less training data)*\n\n")
        
        gsc_baselines = self.gsc_data['gsc_v2_benchmark']['baseline_results']
        
        report.append("### GSC V2 Reference Models\n")
        report.append("|Model|Classes|Accuracy|Latency (ms)|Model Size (KB)|\n")
        report.append("|-----|--------|----------|------------|---------------|\n")
        
        for model_name, model_data in gsc_baselines.items():
            report.append(f"|{model_data['model']}|35|")
            report.append(f"{model_data['accuracy']:.1%}|")
            report.append(f"{model_data['latency_ms']}|")
            report.append(f"{model_data['model_size_kb']}|\n")
        
        # ===== INTERPRETATION =====
        report.append("\n## Interpretation\n")
        
        report.append("\n### Task Complexity\n")
        report.append("- **Your VCM task:** 13-intent commands (domain-specific, smaller)\n")
        report.append("- **GSC task:** 35-word recognition (general-purpose, larger)\n")
        report.append("- **Your advantage:** Simpler task (13 vs 35 classes) = potentially higher accuracy possible\n")
        report.append("- **Your challenge:** Smaller dataset (488 vs 70K samples) = lower ceiling\n")
        report.append("- **Result:** Your accuracy should be in the 85-95% range (reasonable for domain-specific task)\n")
        
        report.append("\n### Latency & Deployment\n")
        if tuned['latency_ms'] and tuned['latency_ms'] > 0:
            constraint_met = tuned['latency_ms'] < 500
            report.append(f"- **Jetson latency:** {tuned['latency_ms']:.2f}ms ")
            report.append(f"({'✓ Within <500ms constraint' if constraint_met else '✗ Exceeds constraint'})\n")
            report.append(f"- **Expected RPi4 latency:** {tuned['latency_ms'] * 3:.0f}-{tuned['latency_ms'] * 5:.0f}ms ")
            report.append("(3-5x slower on CPU)\n")
            report.append(f"- **Feasibility for RPi4:** {'✓ Deployable' if tuned['latency_ms'] * 5 < 2000 else '✗ May need optimization'}\n")
        
        report.append("\n### Hardware Considerations\n")
        report.append("- **Jetson Orin Nano (tuning):** GPU acceleration, low power\n")
        report.append("- **RPi4 (deployment):** CPU-only, 3-5x slower than Jetson\n")
        report.append("- **Your model:** 14K params, suitable for edge deployment\n")
        report.append("- **Quantization:** int8 can reduce size 75% with <2% accuracy loss\n")
        
        report.append("\n### Tuning Success Criteria\n")
        if baseline['accuracy'] and tuned['accuracy'] and tuned['accuracy'] > baseline['accuracy']:
            acc_improvement = ((tuned['accuracy'] - baseline['accuracy']) / baseline['accuracy']) * 100
            report.append(f"✓ Accuracy improved {acc_improvement:.2f}%\n")
        if tuned['latency_ms'] and tuned['latency_ms'] < 500:
            report.append(f"✓ Latency within <500ms soft constraint\n")
        if tuned['model_size_kb'] and tuned['model_size_kb'] < 100:
            report.append(f"✓ Model size <100KB (efficient deployment)\n")
        
        report_text = "".join(report)
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"✓ Report saved to {output_path}")
        
        return report_text
    
    def print_quick_summary(self):
        """Print quick summary to stdout/logs"""
        if not self.gsc_data:
            logger.warning("Could not load benchmark data")
            return
        
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK COMPARISON SUMMARY")
        logger.info("="*70)
        
        baseline = self.gsc_data['your_vcm_benchmark']['baseline']
        tuned = self.gsc_data['your_vcm_benchmark']['tuned_best']
        
        if baseline['accuracy'] and baseline['accuracy'] > 0:
            logger.info("\n✓ Your VCM Results:")
            logger.info(f"  Baseline Accuracy: {baseline['accuracy']:.4f}")
            logger.info(f"  Best Tuned Accuracy: {tuned['accuracy']:.4f}")
            if tuned['accuracy'] > baseline['accuracy']:
                improvement = ((tuned['accuracy'] - baseline['accuracy']) / baseline['accuracy']) * 100
                logger.info(f"  Improvement: +{improvement:.2f}%")
            
            if tuned['latency_ms'] > 0:
                logger.info(f"  Latency: {tuned['latency_ms']:.2f}ms")
                logger.info(f"  Constraint (<500ms): {'✓ Pass' if tuned['latency_ms'] < 500 else '✗ Fail'}")
            
            if tuned['model_size_kb'] > 0:
                logger.info(f"  Model Size: {tuned['model_size_kb']:.2f}KB")
        
        logger.info("\n✓ GSC Reference (for context):")
        gsc_cnn = self.gsc_data['gsc_v2_benchmark']['baseline_results']['cnn_small']
        logger.info(f"  CNN Small (35-class): {gsc_cnn['accuracy']:.1%} acc, {gsc_cnn['latency_ms']}ms")
        
        logger.info("\n→ Your task (13-class) is simpler than GSC (35-class)")
        logger.info("→ Your dataset (488 samples) is smaller than GSC (70K samples)")
        logger.info("→ Accuracy 85-95% is reasonable for domain-specific task")
        
        logger.info("="*70 + "\n")


def create_comparison_table(
    baseline: Dict,
    tuned: Dict,
    gsc_reference: Optional[Dict] = None,
) -> str:
    """
    Create a comparison table as markdown string
    
    Args:
        baseline: Your baseline model metrics
        tuned: Your tuned model metrics
        gsc_reference: Optional GSC reference metrics
    
    Returns:
        Markdown table string
    """
    table = []
    table.append("|Metric|Baseline|Tuned Best|Improvement|GSC Ref|\n")
    table.append("|------|--------|----------|-----------|--------|\n")
    
    # Accuracy
    baseline_acc = baseline.get('accuracy', 0)
    tuned_acc = tuned.get('accuracy', 0)
    gsc_acc = gsc_reference.get('accuracy', 0) if gsc_reference else None
    
    if baseline_acc > 0 and tuned_acc > 0:
        delta = tuned_acc - baseline_acc
        table.append(f"|Accuracy|{baseline_acc:.4f}|{tuned_acc:.4f}|+{delta:.4f}|")
        if gsc_acc:
            table.append(f"{gsc_acc:.1%}|\n")
        else:
            table.append("N/A|\n")
    
    # Latency
    baseline_lat = baseline.get('latency_ms', 0)
    tuned_lat = tuned.get('latency_ms', 0)
    gsc_lat = gsc_reference.get('latency_ms', 0) if gsc_reference else None
    
    if baseline_lat > 0 and tuned_lat > 0:
        delta = tuned_lat - baseline_lat
        table.append(f"|Latency (ms)|{baseline_lat:.2f}|{tuned_lat:.2f}|{delta:+.2f}|")
        if gsc_lat:
            table.append(f"{gsc_lat}|\n")
        else:
            table.append("N/A|\n")
    
    # Model Size
    baseline_size = baseline.get('model_size_kb', 0)
    tuned_size = tuned.get('model_size_kb', 0)
    gsc_size = gsc_reference.get('model_size_kb', 0) if gsc_reference else None
    
    if baseline_size > 0 and tuned_size > 0:
        delta = tuned_size - baseline_size
        table.append(f"|Model Size (KB)|{baseline_size:.2f}|{tuned_size:.2f}|{delta:+.2f}|")
        if gsc_size:
            table.append(f"{gsc_size}|\n")
        else:
            table.append("N/A|\n")
    
    return "".join(table)
