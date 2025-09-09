#!/usr/bin/env python3
"""
Performance comparison between Style-Compress and baseline methods.
Reproduces key results from the paper.
"""

import os
import sys
import json
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Add current directory to path
sys.path.append('.')
from style_compress_v2 import StyleCompress

class StyleCompressComparison:
    """Compare Style-Compress with baseline compression methods"""
    
    def __init__(self):
        self.style_compress = StyleCompress()
        self.results = {}
        
    def random_compression(self, texts: List[str], target_ratio: float) -> List[str]:
        """Random compression baseline"""
        compressed = []
        for text in texts:
            original_tokens = self.style_compress.evaluator.count_tokens(text)
            target_tokens = max(1, int(original_tokens * target_ratio))
            
            # Random truncation point
            words = text.split()
            if len(words) > target_tokens:
                start_idx = np.random.randint(0, len(words) - target_tokens + 1)
                selected_words = words[start_idx:start_idx + target_tokens]
                compressed.append(' '.join(selected_words))
            else:
                compressed.append(text)
        return compressed
    
    def selective_compression(self, texts: List[str], target_ratio: float) -> List[str]:
        """Selective compression (keep important sentences)"""
        compressed = []
        for text in texts:
            sentences = text.split('. ')
            original_tokens = self.style_compress.evaluator.count_tokens(text)
            target_tokens = max(1, int(original_tokens * target_ratio))
            
            # Keep first few sentences until we reach target
            result = ""
            current_tokens = 0
            for sentence in sentences:
                sentence_tokens = self.style_compress.evaluator.count_tokens(sentence)
                if current_tokens + sentence_tokens <= target_tokens:
                    result += sentence + ". "
                    current_tokens += sentence_tokens
                else:
                    break
            
            compressed.append(result.strip())
        return compressed
    
    def compare_methods(self, test_prompts: List[str], task_type: str = "reconstruction", 
                       target_ratio: float = 0.25) -> Dict[str, Dict]:
        """Compare all compression methods"""
        
        print(f"\n{'='*60}")
        print(f"COMPARING COMPRESSION METHODS")
        print(f"Task: {task_type}, Target ratio: {target_ratio}")
        print(f"Test prompts: {len(test_prompts)}")
        print(f"{'='*60}")
        
        results = {}
        
        # 1. Style-Compress (our method)
        print("\n1. Running Style-Compress...")
        try:
            # Use test prompts with replication to reach minimum size if needed
            min_required_prompts = 12
            
            if len(test_prompts) < min_required_prompts:
                # For demo purposes, replicate prompts to reach the minimum required count
                test_prompts_expanded = test_prompts * (min_required_prompts // len(test_prompts) + 1)
                test_prompts_expanded = test_prompts_expanded[:min_required_prompts]
                
                adaptation_prompts = test_prompts_expanded[:10]
                inference_prompts = test_prompts # Use original prompts for inference
            else:
                adaptation_prompts = test_prompts[:10]
                inference_prompts = test_prompts[10:]
            
            start_time = time.time()
            self.style_compress.adaptation_stage(
                prompts=adaptation_prompts,
                task_type=task_type,
                target_ratio=target_ratio,
                M=8,
                N=3,
                ca_setting="CAmin"
            )
            
            style_compress_results = []
            for prompt in inference_prompts:
                compressed = self.style_compress.inference_stage(
                    prompt=prompt,
                    task_type=task_type,
                    target_ratio=target_ratio,
                    S=1
                )
                style_compress_results.append(compressed)
            
            style_compress_time = time.time() - start_time
            
            # Evaluate quality
            quality_metrics = {}
            for original, compressed in zip(inference_prompts, style_compress_results):
                metrics = self.style_compress.evaluate_compression(original, compressed, task_type)
                for key, value in metrics.items():
                    if key not in quality_metrics:
                        quality_metrics[key] = []
                    quality_metrics[key].append(value)
            
            avg_quality = {k: np.mean(v) for k, v in quality_metrics.items()}
            
            # Calculate compression ratios
            ratios = []
            for original, compressed in zip(inference_prompts, style_compress_results):
                orig_tokens = self.style_compress.evaluator.count_tokens(original)
                comp_tokens = self.style_compress.evaluator.count_tokens(compressed)
                ratios.append(comp_tokens / orig_tokens if orig_tokens > 0 else 0)
            
            results['style_compress'] = {
                'compressed_texts': style_compress_results,
                'compression_ratios': ratios,
                'avg_compression_ratio': np.mean(ratios),
                'quality_metrics': avg_quality,
                'time_seconds': style_compress_time,
                'num_demonstrations': len(self.style_compress.demonstration_pool)
            }
            
            print(f"✓ Style-Compress completed")
            print(f"  Avg ratio: {np.mean(ratios):.3f}")
            print(f"  Time: {style_compress_time:.1f}s")
                
        except Exception as e:
            print(f"✗ Style-Compress failed: {e}")
            results['style_compress'] = {'error': str(e)}
        
        # 2. Truncation baseline
        print("\n2. Running Truncation baseline...")
        try:
            start_time = time.time()
            truncation_results = []
            for prompt in test_prompts[-2:]:  # Use last 2 prompts
                original_tokens = self.style_compress.evaluator.count_tokens(prompt)
                target_tokens = max(1, int(original_tokens * target_ratio))
                compressed = self.style_compress.evaluator.truncate_to_target_tokens(prompt, target_tokens)
                truncation_results.append(compressed)
            
            truncation_time = time.time() - start_time
            
            # Calculate ratios
            ratios = []
            for original, compressed in zip(test_prompts[-2:], truncation_results):
                orig_tokens = self.style_compress.evaluator.count_tokens(original)
                comp_tokens = self.style_compress.evaluator.count_tokens(compressed)
                ratios.append(comp_tokens / orig_tokens if orig_tokens > 0 else 0)
            
            # Evaluate quality
            quality_metrics = {}
            for original, compressed in zip(test_prompts[-2:], truncation_results):
                metrics = self.style_compress.evaluate_compression(original, compressed, task_type)
                for key, value in metrics.items():
                    if key not in quality_metrics:
                        quality_metrics[key] = []
                    quality_metrics[key].append(value)
            
            avg_quality = {k: np.mean(v) for k, v in quality_metrics.items()}
            
            results['truncation'] = {
                'compressed_texts': truncation_results,
                'compression_ratios': ratios,
                'avg_compression_ratio': np.mean(ratios),
                'quality_metrics': avg_quality,
                'time_seconds': truncation_time
            }
            
            print(f"✓ Truncation completed")
            print(f"  Avg ratio: {np.mean(ratios):.3f}")
            print(f"  Time: {truncation_time:.3f}s")
            
        except Exception as e:
            print(f"✗ Truncation failed: {e}")
            results['truncation'] = {'error': str(e)}
        
        # 3. Random compression
        print("\n3. Running Random compression...")
        try:
            start_time = time.time()
            random_results = self.random_compression(test_prompts[-2:], target_ratio)
            random_time = time.time() - start_time
            
            # Calculate ratios and quality
            ratios = []
            quality_metrics = {}
            for original, compressed in zip(test_prompts[-2:], random_results):
                orig_tokens = self.style_compress.evaluator.count_tokens(original)
                comp_tokens = self.style_compress.evaluator.count_tokens(compressed)
                ratios.append(comp_tokens / orig_tokens if orig_tokens > 0 else 0)
                
                metrics = self.style_compress.evaluate_compression(original, compressed, task_type)
                for key, value in metrics.items():
                    if key not in quality_metrics:
                        quality_metrics[key] = []
                    quality_metrics[key].append(value)
            
            avg_quality = {k: np.mean(v) for k, v in quality_metrics.items()}
            
            results['random'] = {
                'compressed_texts': random_results,
                'compression_ratios': ratios,
                'avg_compression_ratio': np.mean(ratios),
                'quality_metrics': avg_quality,
                'time_seconds': random_time
            }
            
            print(f"✓ Random compression completed")
            print(f"  Avg ratio: {np.mean(ratios):.3f}")
            print(f"  Time: {random_time:.3f}s")
            
        except Exception as e:
            print(f"✗ Random compression failed: {e}")
            results['random'] = {'error': str(e)}
        
        # 4. Selective compression
        print("\n4. Running Selective compression...")
        try:
            start_time = time.time()
            selective_results = self.selective_compression(test_prompts[-2:], target_ratio)
            selective_time = time.time() - start_time
            
            # Calculate ratios and quality
            ratios = []
            quality_metrics = {}
            for original, compressed in zip(test_prompts[-2:], selective_results):
                orig_tokens = self.style_compress.evaluator.count_tokens(original)
                comp_tokens = self.style_compress.evaluator.count_tokens(compressed)
                ratios.append(comp_tokens / orig_tokens if orig_tokens > 0 else 0)
                
                metrics = self.style_compress.evaluate_compression(original, compressed, task_type)
                for key, value in metrics.items():
                    if key not in quality_metrics:
                        quality_metrics[key] = []
                    quality_metrics[key].append(value)
            
            avg_quality = {k: np.mean(v) for k, v in quality_metrics.items()}
            
            results['selective'] = {
                'compressed_texts': selective_results,
                'compression_ratios': ratios,
                'avg_compression_ratio': np.mean(ratios),
                'quality_metrics': avg_quality,
                'time_seconds': selective_time
            }
            
            print(f"✓ Selective compression completed")
            print(f"  Avg ratio: {np.mean(ratios):.3f}")
            print(f"  Time: {selective_time:.3f}s")
            
        except Exception as e:
            print(f"✗ Selective compression failed: {e}")
            results['selective'] = {'error': str(e)}
        
        return results
    
    def create_comparison_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Create comparison table of results"""
        
        table_data = []
        
        for method, data in results.items():
            if 'error' in data:
                continue
                
            # Get primary quality metric (ROUGE-L for reconstruction)
            primary_metric = data['quality_metrics'].get('rougeL', 0.0)
            
            row = {
                'Method': method.replace('_', ' ').title(),
                'Avg Compression Ratio': f"{data['avg_compression_ratio']:.3f}",
                'ROUGE-L': f"{primary_metric:.3f}",
                'Time (s)': f"{data['time_seconds']:.3f}",
                'Demonstrations': data.get('num_demonstrations', 'N/A')
            }
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def generate_comparison_report(self, results: Dict[str, Dict], output_file: str = "comparison_report.txt"):
        """Generate detailed comparison report"""
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STYLE-COMPRESS vs BASELINES COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Summary table
            df = self.create_comparison_table(results)
            f.write("PERFORMANCE COMPARISON:\n")
            f.write("-" * 40 + "\n")
            f.write(df.to_string(index=False) + "\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 40 + "\n")
            
            for method, data in results.items():
                if 'error' in data:
                    f.write(f"\n{method.upper()}: ERROR - {data['error']}\n")
                    continue
                
                f.write(f"\n{method.upper()}:\n")
                f.write(f"  Compression Ratios: {data['compression_ratios']}\n")
                f.write(f"  Average Ratio: {data['avg_compression_ratio']:.3f}\n")
                f.write(f"  Quality Metrics: {data['quality_metrics']}\n")
                f.write(f"  Time: {data['time_seconds']:.3f}s\n")
                
                if 'num_demonstrations' in data:
                    f.write(f"  Demonstrations Used: {data['num_demonstrations']}\n")
                
                # Show example compressions
                if 'compressed_texts' in data and data['compressed_texts']:
                    f.write(f"  Example Compression: {data['compressed_texts'][0][:100]}...\n")
        
        print(f"✓ Comparison report saved to: {output_file}")


def main():
    """Main comparison function"""
    
    # Test data
    test_prompts = [
        "The recent developments in artificial intelligence have shown remarkable progress in natural language processing, computer vision, and robotics. These advances are transforming industries from healthcare to transportation, enabling new applications and improving existing systems. Machine learning algorithms are becoming more sophisticated, allowing for better pattern recognition and decision-making capabilities across various domains.",
        
        "Climate change represents one of the most significant challenges of our time, with rising global temperatures causing widespread environmental impacts including sea level rise, extreme weather events, and ecosystem disruptions that affect billions of people worldwide. The scientific consensus indicates that immediate action is required to reduce greenhouse gas emissions and implement sustainable practices.",
        
        "The global economy is experiencing unprecedented changes due to technological innovation, shifting demographics, and evolving consumer preferences. Traditional business models are being disrupted across industries, forcing companies to adapt or risk obsolescence. Digital transformation has become a necessity rather than an option for survival in the modern marketplace.",
        
        "Modern education systems are adapting to incorporate digital technologies, personalized learning approaches, and new pedagogical methods to better serve diverse student populations and prepare them for future careers in an increasingly complex world. Online learning platforms and AI-powered tutoring systems are revolutionizing how knowledge is transmitted and acquired.",
        
        "Healthcare delivery is being revolutionized through telemedicine, precision medicine, and artificial intelligence applications that enable more accurate diagnoses and personalized treatment plans, improving patient outcomes while reducing costs. Wearable devices and remote monitoring systems are changing how we approach preventive care and chronic disease management.",
        
        "The renewable energy sector has experienced rapid growth with solar and wind power becoming increasingly cost-competitive with fossil fuels, driving a global transition toward sustainable energy sources and reducing carbon emissions. Energy storage technologies and smart grid systems are addressing the intermittency challenges associated with renewable sources.",
        
        "Urban planning faces complex challenges in creating sustainable, livable cities that accommodate growing populations while addressing issues like traffic congestion, housing affordability, and environmental sustainability in the 21st century. Smart city technologies and data-driven approaches are being deployed to optimize resource allocation and improve quality of life.",
        
        "Scientific research collaboration has been enhanced by digital platforms and data sharing initiatives, enabling researchers worldwide to work together on complex problems and accelerate discovery across multiple disciplines. Open science practices and computational tools are democratizing access to research resources and methodologies.",
        
        "Space exploration continues to push the boundaries of human knowledge and capability, with new missions to Mars, the Moon, and beyond providing insights into the origins of our solar system and the potential for extraterrestrial life. Private space companies are making space more accessible and driving down launch costs through innovative technologies.",
        
        "Quantum computing represents a paradigm shift in computational capabilities, promising to solve complex problems in cryptography, drug discovery, and optimization that are currently intractable for classical computers. As quantum hardware matures, we're seeing the development of quantum algorithms and applications that could revolutionize various fields of science and technology.",
        
        "The Internet of Things is connecting billions of devices worldwide, creating unprecedented opportunities for automation and data collection while also raising important questions about privacy, security, and data governance. Smart homes, industrial IoT, and connected vehicles are just the beginning of this technological revolution.",
        
        "Artificial intelligence and machine learning are being integrated into various sectors including finance, automotive, and entertainment, transforming how businesses operate and how consumers interact with technology in their daily lives. Natural language processing and computer vision are enabling more intuitive human-computer interfaces and automated decision-making systems.",
        
        "Test prompt for final comparison and validation purposes in the Style-Compress evaluation framework.",
        
        "Additional test prompt to ensure comprehensive analysis and robust comparison between different compression methodologies."
    ]
    
    # Initialize comparison
    comparison = StyleCompressComparison()
    
    # Run comparison
    results = comparison.compare_methods(test_prompts, "reconstruction", 0.25)
    
    # Generate report
    comparison.generate_comparison_report(results)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON COMPLETED")
    print("="*60)
    
    df = comparison.create_comparison_table(results)
    print("\nRESULTS SUMMARY:")
    print(df.to_string(index=False))
    
    return results


if __name__ == "__main__":
    results = main()
