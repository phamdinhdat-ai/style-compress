import os
import sys
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset, Dataset
import datasets
import matplotlib.pyplot as plt
import re

# Add current directory to path
sys.path.append('.')
from style_compress_v2 import StyleCompress, CompressionResult
from performance_comparison import StyleCompressComparison

# Add baseline compression implementations
class ExtendedStyleCompressComparison(StyleCompressComparison):
    """Extended comparison class with implementations for baseline methods"""
    
    def compress_vanilla(self, text: str, target_ratio: float) -> str:
        """Vanilla (zero-shot) compression using the model"""
        original_tokens = self.style_compress.evaluator.count_tokens(text)
        target_tokens = max(1, int(original_tokens * target_ratio))
        
        # Simple zero-shot compression prompt
        vanilla_prompt = f"""Please compress the following text to approximately {target_tokens} tokens while preserving the most important information:

Text: {text}

Compressed:"""
        
        try:
            response = self.style_compress.compression_model.invoke(vanilla_prompt)
            compressed = response.strip()
            
            # Extract just the compressed text
            patterns = [
                r"Compressed:\s*(.*?)(?:\n\n|\n$|$)",
                r"Output:\s*(.*?)(?:\n\n|\n$|$)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, compressed, re.DOTALL)
                if match:
                    compressed = match.group(1).strip()
                    break
            
            # Post-process to remove explanations and other fluff
            compressed = re.sub(r"^(Here's|The compressed|I've compressed|This is|The).*?:", "", compressed, 
                              flags=re.IGNORECASE | re.DOTALL).strip()
            
            # Truncate to target token count
            return self.style_compress.evaluator.truncate_to_target_tokens(compressed, target_tokens)
        except Exception as e:
            print(f"Error in vanilla compression: {e}")
            # Fallback to selective context if vanilla fails
            return self.compress_selective_context(text, target_ratio)
    
    def compress_selective_context(self, text: str, target_ratio: float) -> str:
        """Selective-Context baseline - selects beginning of text up to token limit"""
        original_tokens = self.style_compress.evaluator.count_tokens(text)
        target_tokens = max(1, int(original_tokens * target_ratio))
        
        # Simply take the first part of the text based on token count
        tokens = self.style_compress.evaluator.tokenizer.encode(text)
        if len(tokens) > target_tokens:
            tokens = tokens[:target_tokens]
            compressed = self.style_compress.evaluator.tokenizer.decode(tokens)
            return compressed
        else:
            return text
    
    def compress_llm_lingua(self, text: str, target_ratio: float) -> str:
        """LLM-Lingua baseline - uses specific prompt as described in paper"""
        original_tokens = self.style_compress.evaluator.count_tokens(text)
        target_tokens = max(1, int(original_tokens * target_ratio))
        
        # LLM-Lingua style prompt as described in the paper
        llm_lingua_prompt = f"""Compress the following text while preserving key information:

        Text: {text}

        Compressed (around {target_tokens} tokens):"""

        try:
            response = self.style_compress.compression_model.invoke(llm_lingua_prompt)
            compressed = response.strip()
            
            # Process the output to extract just the compressed text
            patterns = [
                r"Compressed \(.*?\):\s*(.*?)(?:\n\n|\n$|$)",
                r"Compressed:\s*(.*?)(?:\n\n|\n$|$)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, compressed, re.DOTALL)
                if match:
                    compressed = match.group(1).strip()
                    break
            
            # Truncate to target token count
            return self.style_compress.evaluator.truncate_to_target_tokens(compressed, target_tokens)
        except Exception as e:
            print(f"Error in LLM-Lingua compression: {e}")
            # Fallback to selective context if LLM-Lingua fails
            return self.compress_selective_context(text, target_ratio)

# Methods to evaluate as specified in Table 2
METHODS = [
    "selective-context", # Baseline abbreviated as 'sc' in the paper
    "LLM-Lingua",       # Baseline abbreviated as 'lingua' in the paper 
    "vanilla",          # Direct zero-shot prompt compression
    "style-compress"    # Our method (Style-Compress)
]

# Tasks and their evaluation metrics for Table 2
TASK_METRICS = {
    "reconstruction": ["Rouge-1", "Rouge-2", "Rouge-L", "BERTScore"]
}

class Table2Reproduction:
    """Reproduces Table 2 from the Style-Compress paper (Original Prompt Reconstruction)"""
    
    def __init__(self):
        self.comparison = ExtendedStyleCompressComparison()
        self.style_compress = StyleCompress(
            compression_model_name="llama3.1:latest",
            evaluation_model_name="llama3.1:1b"
        )
        self.tasks = ["reconstruction"]  # Only reconstruction for Table 2
        self.methods = METHODS
        self.compression_ratios = [0.1, 0.25, 0.5]  # Ratios from Table 2
        
        # Models for evaluation as used in Table 2
        # Map friendly names to actual model identifiers
        self.evaluation_models = {
            "LLaMA-2 13B": "llama3.1:13b",  # Using llama3.1 as a substitute
            "GPT-3.5": "gpt3.5"             # This will need to be configured based on available API
        }
        self.results = {}
        
    def load_datasets(self):
        """Load dataset for reconstruction task as specified in the paper"""
        print("\nLoading dataset for reconstruction evaluation...")
        
        # Dictionary to store dataset
        dataset_dict = {}
        
        try:
            # Dataset paths from get_data.py
            reconstruction_path = "style_compress/data/prompt_reconstruction/reconstruction_test"
            
            # BBC News dataset for prompt reconstruction
            print("Loading BBC News dataset for reconstruction task...")
            try:
                # Load from local saved dataset
                bbc_dataset = datasets.load_from_disk(reconstruction_path)
                print(f"✓ Loaded BBC News dataset: {len(bbc_dataset)} samples")
                dataset_dict["reconstruction"] = {
                    "prompts": bbc_dataset["text"][:200],  # 200 test samples as specified
                    "extra_data": {}
                }
            except Exception as e:
                print(f"✗ Error loading BBC News dataset: {e}")
                print("  Using fallback sample prompts for reconstruction")
                dataset_dict["reconstruction"] = {
                    "prompts": self.style_compress.prompts[:20],
                    "extra_data": {}
                }
            
            return dataset_dict
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return {"reconstruction": {"prompts": self.style_compress.prompts[:20], "extra_data": {}}}

    def evaluate_method(self, method: str, task: str, prompts: List[str], 
                       extra_data: Dict = None, ratio: float = 0.25, 
                       eval_model: str = "llama3.1:13b") -> Dict[str, float]:
        """Evaluate a compression method on the given task with specified evaluation model"""
        print(f"Evaluating {method} on {task} with ratio {ratio} using {eval_model}...")
        
        if method == "style-compress":
            # Our method - use Style-Compress with adaptation
            # For reconstruction, set appropriate parameters
            compressed_prompts = self.style_compress.compress_batch(
                prompts=prompts[:10],  # Use first 10 samples for adaptation
                task_type=task,
                target_ratio=ratio,
                adaptation_size=10
            )
            
        elif method == "vanilla":
            # Zero-shot LLM compression
            compressed_prompts = []
            for i, prompt in enumerate(prompts):
                compressed = self.comparison.compress_vanilla(prompt, ratio)
                compressed_prompts.append(compressed)
                if i % 5 == 0:
                    print(f"  Progress: {i+1}/{len(prompts)}")
        
        elif method == "selective-context":
            # Selective-Context baseline
            compressed_prompts = []
            for i, prompt in enumerate(prompts):
                compressed = self.comparison.compress_selective_context(prompt, ratio)
                compressed_prompts.append(compressed)
                if i % 5 == 0:
                    print(f"  Progress: {i+1}/{len(prompts)}")
        
        elif method == "LLM-Lingua":
            # LLM-Lingua baseline
            compressed_prompts = []
            for i, prompt in enumerate(prompts):
                compressed = self.comparison.compress_llm_lingua(prompt, ratio)
                compressed_prompts.append(compressed)
                if i % 5 == 0:
                    print(f"  Progress: {i+1}/{len(prompts)}")
        
        else:
            print(f"Unknown method: {method}")
            return {}
        
        # Evaluate the results for reconstruction
        # Adjust evaluation model for the task (update the style_compress object's model)
        try:
            # Check if we need to set the evaluation model
            self.style_compress = StyleCompress(
                compression_model_name="llama3.1:latest",
                evaluation_model_name=eval_model
            )
            print(f"Updated evaluation model to {eval_model}")
        except Exception as e:
            print(f"Error setting evaluation model: {e}, using default model")
        
        # Calculate metrics for each sample and average
        all_metrics = []
        for i, (original, compressed) in enumerate(zip(prompts, compressed_prompts)):
            metrics = self.style_compress.evaluate_compression(original, compressed, task)
            all_metrics.append(metrics)
            if i % 5 == 0:
                print(f"  Evaluation progress: {i+1}/{len(prompts)}")
        
        # Average metrics across samples
        avg_metrics = {}
        for metric in TASK_METRICS[task]:
            metric_lower = metric.lower()
            values = [m.get(metric_lower, 0) for m in all_metrics if metric_lower in m]
            if values:
                avg_metrics[metric_lower] = sum(values) / len(values)
        
        print(f"  Results: {avg_metrics}")
        return avg_metrics

    def run_table2_reproduction(self, max_samples: int = 20):
        """Run evaluation to reproduce Table 2 - Original Prompt Reconstruction results"""
        print("\n" + "="*80)
        print("STYLE-COMPRESS TABLE 2 REPRODUCTION - ORIGINAL PROMPT RECONSTRUCTION")
        print("="*80)
        
        # Initialize results structure with compression ratios
        self.results = {}
        for model_name, model_id in self.evaluation_models.items():
            self.results[model_name] = {
                ratio: {method: {} for method in self.methods}
                for ratio in self.compression_ratios
            }
        
        # Load datasets
        datasets = self.load_datasets()
        task = "reconstruction"  # Only reconstruction for Table 2
        
        if task not in datasets or not datasets[task]["prompts"]:
            print(f"✗ No dataset available for {task}, exiting...")
            return {}
        
        # Get data for reconstruction task
        prompts = datasets[task]["prompts"][:max_samples]
        extra_data = datasets[task].get("extra_data", {})
        
        print(f"Processing {len(prompts)} samples for {task}")
        
        # For each evaluation model (LLaMA-2 13B and GPT-3.5)
        for model_name, model_id in self.evaluation_models.items():
            print(f"\n{'-'*60}")
            print(f"EVALUATING WITH MODEL: {model_name} ({model_id})")
            print(f"{'-'*60}")
            
            # For each compression ratio
            for ratio in self.compression_ratios:
                print(f"\n{'='*60}")
                print(f"EVALUATING COMPRESSION RATIO: {ratio}")
                print(f"{'='*60}")
                
                # Evaluate each method
                for method in self.methods:
                    metrics = self.evaluate_method(method, task, prompts, extra_data, ratio, model_id)
                    # Store results 
                    self.results[model_name][ratio][method] = metrics
                    # Save after each method to preserve progress
                    self._save_results()
        
        # Generate final table
        self._create_formatted_table()
        
        return self.results
    
    def _save_results(self):
        """Save current results to CSV with model and compression ratio information"""
        rows = []
        for model, model_data in self.results.items():
            for ratio, ratio_data in model_data.items():
                for method, metrics in ratio_data.items():
                    row = {
                        "Model": model,
                        "Compression Ratio": ratio,
                        "Method": method
                    }
                    # Add metrics
                    for metric, value in metrics.items():
                        row[metric] = value
                    
                    rows.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(rows)
        df.to_csv("table2_reproduction_results.csv", index=False)
        print("Results saved to table2_reproduction_results.csv")
    
    def _create_formatted_table(self):
        """Create a formatted table similar to Table 2 in the paper"""
        # First, pivot the data to match the paper's table format
        try:
            df = pd.read_csv("table2_reproduction_results.csv")
            
            # Pivot for each evaluation model
            for model_name in self.evaluation_models.keys():
                model_df = df[df["Model"] == model_name].copy()
                
                # Create pivot table with only the metrics that are available
                # Check available metrics
                available_metrics = [col for col in model_df.columns 
                                    if col not in ["Model", "Compression Ratio", "Method"]]
                
                print(f"Available metrics: {available_metrics}")
                
                pivot_df = pd.pivot_table(
                    model_df,
                    index=["Method"],
                    columns=["Compression Ratio"],
                    values=available_metrics
                )
                
                # Reorder columns to match paper (metrics grouped together)
                new_columns = []
                for metric in ["rouge-1", "rouge-2", "rougel", "bertscore"]:
                    for ratio in self.compression_ratios:
                        new_columns.append((metric, ratio))
                
                # Try to reorder if all columns exist
                try:
                    pivot_df = pivot_df[new_columns]
                except:
                    print(f"Could not reorder columns for {model_name}, using default order")
                
                # Save the pivoted table
                safe_name = model_name.replace(' ', '_').replace('-', '_')
                output_file = f"table2_reproduction_pivoted_{safe_name}.csv"
                pivot_df.to_csv(output_file)
                print(f"Formatted table saved to {output_file}")
                
                # Print the table
                print(f"\nTable 2 Results for {model_name}:")
                print(pivot_df)
                
                # Create a visualization of the results
                self._create_visualization(model_name, pivot_df)
        except Exception as e:
            print(f"Error creating formatted table: {e}")
            
    def _create_visualization(self, model_name: str, pivot_df):
        """Create a visualization of the results for the given model"""
        try:
            # Get available metrics from the pivot table
            metrics = list(pivot_df.columns.levels[0]) if hasattr(pivot_df.columns, 'levels') else [pivot_df.columns[0][0]]
            
            # Create a figure with subplots for each metric
            fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
            fig.suptitle(f'Table 2 Results - {model_name} - Original Prompt Reconstruction', fontsize=16)
            
            # If only one metric is available, axes might not be an array
            if len(metrics) == 1:
                axes = [axes]
            
            # Get methods in proper order for visualization
            methods = pivot_df.index.tolist()
            
            # For each metric, create a grouped bar chart
            for i, metric in enumerate(metrics):
                metric_data = pivot_df.xs(metric, axis=1, level=0)
                
                # Plot bars for each ratio
                bar_width = 0.2
                x = np.arange(len(methods))
                
                for j, ratio in enumerate(self.compression_ratios):
                    offset = (j - 1) * bar_width
                    axes[i].bar(x + offset, metric_data[ratio], width=bar_width, 
                              label=f'Ratio {ratio}')
                
                # Set labels and title
                axes[i].set_title(f'{metric.upper()}')
                axes[i].set_xticks(x)
                axes[i].set_xticklabels(methods, rotation=45, ha='right')
                axes[i].set_ylabel('Score')
                axes[i].grid(True, linestyle='--', alpha=0.6)
            
            # Add legend
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.98))
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            
            # Save figure
            safe_name = model_name.replace(' ', '_').replace('-', '_')
            fig.savefig(f'table2_visualization_{safe_name}.png', dpi=300, bbox_inches='tight')
            print(f"Visualization saved as table2_visualization_{safe_name}.png")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")

def main():
    # Create the reproduction object
    reproduction = Table2Reproduction()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Reproduce Table 2 from Style-Compress paper')
    parser.add_argument('--samples', type=int, default=20, 
                        help='Number of samples to use for evaluation (default: 20)')
    parser.add_argument('--ratios', type=str, default="0.1,0.25,0.5",
                        help='Compression ratios to evaluate, comma separated (default: 0.1,0.25,0.5)')
    args = parser.parse_args()
    
    # Parse ratios
    ratios = [float(r) for r in args.ratios.split(',')]
    
    print(f"Running with {args.samples} samples and ratios: {ratios}")
    
    # Run the reproduction
    results = reproduction.run_table2_reproduction(
        max_samples=args.samples
    )
    
    print("\nTable 2 reproduction completed!")

if __name__ == "__main__":
    main()
