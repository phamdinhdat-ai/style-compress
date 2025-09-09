#!/usr/bin/env python3
"""
Simple dataset inspection script to help debug loading issues.
"""

import os
import sys
from datasets import load_dataset, Dataset
import datasets
import json

def inspect_dataset(dataset, name, limit=3):
    """Print information about a dataset's structure and content"""
    print(f"\n{'='*80}\nINSPECTING {name.upper()} DATASET\n{'='*80}")
    
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset size: {len(dataset)} samples")
    
    # Get column names if available
    try:
        print(f"Columns: {dataset.column_names}")
        print(f"Features: {dataset.features}")
    except:
        print("Could not get column information")
    
    # Print sample items
    print(f"\nSample items (max {limit}):")
    for i, item in enumerate(dataset):
        if i >= limit:
            break
        print(f"\nItem {i+1}:")
        try:
            # Try to pretty print
            print(json.dumps(item, indent=2, default=str)[:1000] + "...")
        except:
            # Fallback to simple print
            print(str(item)[:1000] + "...")

def main():
    """Load and inspect datasets"""
    # Dataset paths
    reconstruction_path = "style_compress/data/prompt_reconstruction/reconstruction_test"
    summarization_path = "style_compress/data/text_summarization/text_summarization_test"
    multi_hop_qa_path = "style_compress/data/multi_hop_QA/multi_hop_qa_test"
    cot_reasoning_path = "style_compress/data/cot_reasoning/cot_gsm8k_test"
    
    # Check if these paths exist
    for path in [reconstruction_path, summarization_path, multi_hop_qa_path, cot_reasoning_path]:
        print(f"Checking if path exists: {path}")
        if os.path.exists(path):
            print(f"  ✓ Path exists")
        else:
            print(f"  ✗ Path does not exist")
    
    # 1. BBC News dataset for reconstruction
    print("\nLoading BBC News dataset...")
    try:
        bbc_dataset = datasets.load_from_disk(reconstruction_path)
        print(f"✓ Successfully loaded BBC News dataset")
        inspect_dataset(bbc_dataset, "BBC News (reconstruction)")
    except Exception as e:
        print(f"✗ Error loading BBC News dataset: {e}")
    
    # 2. CNN/Daily-Mail for text summarization
    print("\nLoading CNN/Daily-Mail dataset...")
    try:
        cnn_dataset = datasets.load_from_disk(summarization_path)
        print(f"✓ Successfully loaded CNN/Daily-Mail dataset")
        inspect_dataset(cnn_dataset, "CNN/Daily-Mail (summarization)")
    except Exception as e:
        print(f"✗ Error loading CNN/Daily-Mail dataset: {e}")
    
    # 3. HotpotQA for multi-hop QA
    print("\nLoading HotpotQA dataset...")
    try:
        hotpot_dataset = datasets.load_from_disk(multi_hop_qa_path)
        print(f"✓ Successfully loaded HotpotQA dataset")
        inspect_dataset(hotpot_dataset, "HotpotQA (multi-hop QA)")
    except Exception as e:
        print(f"✗ Error loading HotpotQA dataset: {e}")
    
    # 4. GSM8k for CoT reasoning
    print("\nLoading GSM8k dataset...")
    try:
        gsm8k_dataset = datasets.load_from_disk(cot_reasoning_path)
        print(f"✓ Successfully loaded GSM8k dataset")
        inspect_dataset(gsm8k_dataset, "GSM8k (CoT reasoning)")
    except Exception as e:
        print(f"✗ Error loading GSM8k dataset: {e}")

if __name__ == "__main__":
    main()
