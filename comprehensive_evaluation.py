#!/usr/bin/env python3
"""
Comprehensive Style-Compress evaluation following paper specifications.
This script provides detailed analysis and comparison with baselines.
"""

import os
import sys
import json
import time
import pandas as pd
from typing import List, Dict, Tuple, Optional
import numpy as np
import random
from datetime import datetime

# Add current directory to path
sys.path.append('.')
from style_compress_v2 import StyleCompress, CompressionResult

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class StyleCompressEvaluator:
    """Comprehensive evaluator for Style-Compress method"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Style-Compress with available models
        self.style_compress = StyleCompress(
            compression_model_name="llama3.2:1b",     # Fast compression model
            evaluation_model_name="llama3.1:latest"   # Better evaluation model
        )
        
        self.results = {}
        
    def create_baseline_compressions(self, texts: List[str], target_ratio: float) -> List[str]:
        """Create baseline compressions for comparison"""
        
        baseline_compressions = []
        
        for text in texts:
            original_tokens = self.style_compress.evaluator.count_tokens(text)
            target_tokens = max(1, int(original_tokens * target_ratio))
            
            # Simple truncation baseline
            truncated = self.style_compress.evaluator.truncate_to_target_tokens(text, target_tokens)
            baseline_compressions.append(truncated)
            
        return baseline_compressions
    
    def evaluate_compression_quality(self, original_texts: List[str], 
                                   compressed_texts: List[str],
                                   task_type: str) -> Dict[str, float]:
        """Evaluate compression quality using paper metrics"""
        
        all_metrics = []
        
        for original, compressed in zip(original_texts, compressed_texts):
            metrics = self.style_compress.evaluate_compression(
                original, compressed, task_type
            )
            all_metrics.append(metrics)
        
        # Calculate average metrics
        avg_metrics = {}
        if all_metrics:
            metric_keys = all_metrics[0].keys()
            for key in metric_keys:
                values = [m[key] for m in all_metrics if key in m]
                avg_metrics[key] = np.mean(values) if values else 0.0
        
        return avg_metrics
    
    def run_ablation_study(self, prompts: List[str], task_type: str, 
                          target_ratio: float = 0.25) -> Dict[str, Dict]:
        """Run ablation study comparing different Style-Compress configurations"""
        
        print(f"\n{'='*60}")
        print(f"ABLATION STUDY: {task_type}")
        print(f"Target ratio: {target_ratio}")
        print(f"{'='*60}")
        
        configurations = {
            'style_only': {'use_icl': False, 'use_style': True},
            'icl_only': {'use_icl': True, 'use_style': False},
            'full_style_compress': {'use_icl': True, 'use_style': True},
            'baseline_truncation': {'use_icl': False, 'use_style': False}
        }
        
        results = {}
        
        for config_name, config in configurations.items():
            print(f"\nTesting configuration: {config_name}")
            print("-" * 40)
            
            try:
                if config_name == 'baseline_truncation':
                    # Simple baseline: truncation
                    compressed_texts = self.create_baseline_compressions(prompts[10:12], target_ratio)
                    compression_ratios = [target_ratio] * len(compressed_texts)
                else:
                    # Style-Compress variants
                    adaptation_prompts = prompts[:10]
                    test_prompts = prompts[10:12]
                    
                    # Modify Style-Compress behavior for ablation
                    if not config['use_style']:
                        # Force single style for all compressions
                        original_styles = self.style_compress.styles.copy()
                        self.style_compress.styles = {'default': 'Compress the text.'}
                    
                    # Run adaptation with modified parameters
                    N_style = 2 if config['use_style'] else 0
                    N_icl = 3 if config['use_icl'] else 0
                    N_total = max(N_style + N_icl, 1)
                    
                    if N_total > 0:
                        self.style_compress.adaptation_stage(
                            prompts=adaptation_prompts,
                            task_type=task_type,
                            target_ratio=target_ratio,
                            M=min(8, len(adaptation_prompts)),
                            N=N_total,
                            ca_setting="CAmid" if task_type == "cot_gsm8k" else "CAmin"
                        )
                    
                    # Run inference
                    compressed_texts = []
                    compression_ratios = []
                    
                    for prompt in test_prompts:
                        if N_total > 0 and len(self.style_compress.demonstration_pool) > 0:
                            compressed = self.style_compress.inference_stage(
                                prompt=prompt,
                                task_type=task_type,
                                target_ratio=target_ratio,
                                S=1
                            )
                        else:
                            # Fallback to truncation
                            original_tokens = self.style_compress.evaluator.count_tokens(prompt)
                            target_tokens = max(1, int(original_tokens * target_ratio))
                            compressed = self.style_compress.evaluator.truncate_to_target_tokens(prompt, target_tokens)
                        
                        compressed_texts.append(compressed)
                        
                        orig_tokens = self.style_compress.evaluator.count_tokens(prompt)
                        comp_tokens = self.style_compress.evaluator.count_tokens(compressed)
                        ratio = comp_tokens / orig_tokens if orig_tokens > 0 else 0
                        compression_ratios.append(ratio)
                    
                    # Restore original styles if modified
                    if not config['use_style'] and 'original_styles' in locals():
                        self.style_compress.styles = original_styles
                
                # Evaluate quality
                quality_metrics = self.evaluate_compression_quality(
                    prompts[10:12], compressed_texts, task_type
                )
                
                results[config_name] = {
                    'compression_ratios': compression_ratios,
                    'avg_compression_ratio': np.mean(compression_ratios),
                    'quality_metrics': quality_metrics,
                    'num_examples': len(compressed_texts)
                }
                
                print(f"✓ {config_name}:")
                print(f"  Avg compression ratio: {np.mean(compression_ratios):.3f}")
                print(f"  Quality metrics: {quality_metrics}")
                
            except Exception as e:
                print(f"✗ Error in {config_name}: {e}")
                results[config_name] = {'error': str(e)}
        
        return results
    
    def run_compression_ratio_analysis(self, prompts: List[str], task_type: str) -> Dict[str, Dict]:
        """Analyze performance across different compression ratios"""
        
        print(f"\n{'='*60}")
        print(f"COMPRESSION RATIO ANALYSIS: {task_type}")
        print(f"{'='*60}")
        
        ratios = [0.1, 0.25, 0.5]  # Paper ratios
        results = {}
        
        for ratio in ratios:
            print(f"\nTesting ratio: {ratio}")
            print("-" * 30)
            
            try:
                adaptation_prompts = prompts[:8]
                test_prompts = prompts[8:10]
                
                # Run Style-Compress pipeline
                self.style_compress.adaptation_stage(
                    prompts=adaptation_prompts,
                    task_type=task_type,
                    target_ratio=ratio,
                    M=min(6, len(adaptation_prompts)),
                    N=3,
                    ca_setting="CAmid" if task_type == "cot_gsm8k" else "CAmin"
                )
                
                compressed_texts = []
                actual_ratios = []
                
                for prompt in test_prompts:
                    compressed = self.style_compress.inference_stage(
                        prompt=prompt,
                        task_type=task_type,
                        target_ratio=ratio,
                        S=1
                    )
                    
                    compressed_texts.append(compressed)
                    
                    orig_tokens = self.style_compress.evaluator.count_tokens(prompt)
                    comp_tokens = self.style_compress.evaluator.count_tokens(compressed)
                    actual_ratio = comp_tokens / orig_tokens if orig_tokens > 0 else 0
                    actual_ratios.append(actual_ratio)
                
                # Evaluate quality
                quality_metrics = self.evaluate_compression_quality(
                    test_prompts, compressed_texts, task_type
                )
                
                results[str(ratio)] = {
                    'target_ratio': ratio,
                    'actual_ratios': actual_ratios,
                    'avg_actual_ratio': np.mean(actual_ratios),
                    'quality_metrics': quality_metrics,
                    'num_demonstrations': len(self.style_compress.demonstration_pool)
                }
                
                print(f"✓ Ratio {ratio}:")
                print(f"  Avg actual ratio: {np.mean(actual_ratios):.3f}")
                print(f"  Primary metric: {self.style_compress.get_primary_metric_score(quality_metrics, task_type):.3f}")
                
            except Exception as e:
                print(f"✗ Error with ratio {ratio}: {e}")
                results[str(ratio)] = {'error': str(e)}
        
        return results
    
    def run_style_performance_analysis(self, prompts: List[str], task_type: str) -> Dict[str, float]:
        """Analyze individual style performance"""
        
        print(f"\n{'='*60}")
        print(f"STYLE PERFORMANCE ANALYSIS: {task_type}")
        print(f"{'='*60}")
        
        # Run a full adaptation to gather style statistics
        adaptation_prompts = prompts[:min(10, len(prompts)-2)]
        
        self.style_compress.adaptation_stage(
            prompts=adaptation_prompts,
            task_type=task_type,
            target_ratio=0.25,
            M=min(8, len(adaptation_prompts)),
            N=5,
            ca_setting="CAmid" if task_type == "cot_gsm8k" else "CAmin"
        )
        
        style_stats = self.style_compress.get_style_statistics()
        
        print("Style performance ranking:")
        sorted_styles = sorted(style_stats.items(), key=lambda x: x[1], reverse=True)
        for i, (style, performance) in enumerate(sorted_styles[:10]):
            print(f"  {i+1:2d}. {style:20s} {performance:.3f}")
        
        return style_stats
    
    def generate_report(self, task_results: Dict[str, Dict]) -> str:
        """Generate comprehensive evaluation report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"style_compress_report_{timestamp}.json")
        
        # Save detailed results
        with open(report_path, 'w') as f:
            json.dump(task_results, f, indent=2, default=str)
        
        # Generate summary report
        summary_path = os.path.join(self.output_dir, f"style_compress_summary_{timestamp}.txt")
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STYLE-COMPRESS EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            for task_type, results in task_results.items():
                f.write(f"\n{task_type.upper()}:\n")
                f.write("-" * 40 + "\n")
                
                if 'ratio_analysis' in results:
                    f.write("Compression Ratio Analysis:\n")
                    for ratio, data in results['ratio_analysis'].items():
                        if 'avg_actual_ratio' in data:
                            f.write(f"  {ratio}: actual={data['avg_actual_ratio']:.3f}\n")
                
                if 'ablation_study' in results:
                    f.write("\nAblation Study Results:\n")
                    for config, data in results['ablation_study'].items():
                        if 'avg_compression_ratio' in data:
                            f.write(f"  {config}: ratio={data['avg_compression_ratio']:.3f}\n")
        
        print(f"\n✓ Detailed report saved to: {report_path}")
        print(f"✓ Summary saved to: {summary_path}")
        
        return report_path
    
    def run_full_evaluation(self, test_data: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Run complete evaluation suite"""
        
        print("="*80)
        print("STYLE-COMPRESS COMPREHENSIVE EVALUATION")
        print("="*80)
        
        all_results = {}
        
        for task_type, prompts in test_data.items():
            if len(prompts) < 12:
                print(f"Skipping {task_type}: insufficient prompts ({len(prompts)} < 12)")
                continue
            
            print(f"\n{'='*60}")
            print(f"EVALUATING TASK: {task_type}")
            print(f"Prompts available: {len(prompts)}")
            print(f"{'='*60}")
            
            task_results = {}
            
            try:
                # 1. Compression ratio analysis
                task_results['ratio_analysis'] = self.run_compression_ratio_analysis(prompts, task_type)
                
                # 2. Ablation study
                task_results['ablation_study'] = self.run_ablation_study(prompts, task_type)
                
                # 3. Style performance analysis
                task_results['style_analysis'] = self.run_style_performance_analysis(prompts, task_type)
                
                all_results[task_type] = task_results
                
            except Exception as e:
                print(f"✗ Error evaluating {task_type}: {e}")
                all_results[task_type] = {'error': str(e)}
        
        # Generate final report
        report_path = self.generate_report(all_results)
        
        return all_results


def main():
    """Main evaluation function"""
    
    # Define test data for each task
    test_data = {
        'reconstruction': [
            "The recent developments in artificial intelligence have shown remarkable progress in natural language processing, computer vision, and robotics. These advances are transforming industries from healthcare to transportation, enabling new applications and improving existing systems.",
            "Climate change represents one of the most significant challenges of our time, with rising global temperatures causing widespread environmental impacts including sea level rise, extreme weather events, and ecosystem disruptions.",
            "The global economy is experiencing unprecedented changes due to technological innovation, shifting demographics, and evolving consumer preferences. Traditional business models are being disrupted across industries.",
            "Modern education systems are adapting to incorporate digital technologies, personalized learning approaches, and new pedagogical methods to better serve diverse student populations.",
            "Healthcare delivery is being revolutionized through telemedicine, precision medicine, and artificial intelligence applications that enable more accurate diagnoses and personalized treatment plans.",
            "The renewable energy sector has experienced rapid growth with solar and wind power becoming increasingly cost-competitive with fossil fuels, driving a global transition toward sustainable energy sources.",
            "Urban planning faces complex challenges in creating sustainable, livable cities that accommodate growing populations while addressing issues like traffic congestion, housing affordability, and environmental sustainability.",
            "Scientific research collaboration has been enhanced by digital platforms and data sharing initiatives, enabling researchers worldwide to work together on complex problems and accelerate discovery.",
            "Space exploration continues to push the boundaries of human knowledge and capability, with new missions to Mars, the Moon, and beyond providing insights into the origins of our solar system.",
            "Quantum computing represents a paradigm shift in computational capabilities, promising to solve complex problems in cryptography, drug discovery, and optimization that are currently intractable.",
            "The Internet of Things is connecting billions of devices worldwide, creating unprecedented opportunities for automation and data collection while also raising important questions about privacy and security.",
            "Artificial intelligence and machine learning are being integrated into various sectors including finance, automotive, and entertainment, transforming how businesses operate and how consumers interact with technology.",
            "Test prompt for reconstruction evaluation and validation purposes",
            "Additional test prompt for comprehensive analysis of reconstruction capabilities"
        ]
    }
    
    # Initialize evaluator
    evaluator = StyleCompressEvaluator()
    
    # Run full evaluation
    results = evaluator.run_full_evaluation(test_data)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = main()
