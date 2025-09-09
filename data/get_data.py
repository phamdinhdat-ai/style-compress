import transformers 
import datasets
from datasets import load_dataset
import torch 
import numpy as np 
import pandas as pd 


cot_reasoning_folder = 'style_compress/data/cot_reasoning/'
multi_hop_qa_folder = 'style_compress/data/multi_hop_QA/'
reconstruction_folder = "style_compress/data/prompt_reconstruction/"
text_summarization_folder = "style_compress/data/text_summarization/"


# cot_gsm8k = load_dataset("openai/gsm8k", split="train") 
# multi_hop_qa  = load_dataset("hotpotqa/hotpot_qa", split="train")
# reconstruction = load_dataset("SetFit/bbc-news", split="train")
# text_summarization = load_dataset("abisee/cnn_dailymail", split="train")

# cot_gsm8k_test = cot_gsm8k.shuffle(seed=42).select(range(200))
# multi_hop_qa_test = multi_hop_qa.shuffle(seed=42).select(range(200))
# reconstruction_test = reconstruction.shuffle(seed=42).select(range(200))
# text_summarization_test = text_summarization.shuffle(seed=42).select(range(200))

# cot_gsm8k_adapt = cot_gsm8k.shuffle(seed=42).select(range(10,100))
# multi_hop_qa_adapt = multi_hop_qa.shuffle(seed=42).select(range(10,100))
# reconstruction_adapt = reconstruction.shuffle(seed=42).select(range(10,100))
# text_summarization_adapt = text_summarization.shuffle(seed=42).select(range(10,100))

# #save to folder
# cot_gsm8k_test.save_to_disk(cot_reasoning_folder + 'cot_gsm8k_test')
# multi_hop_qa_test.save_to_disk(multi_hop_qa_folder + 'multi_hop_qa_test')
# reconstruction_test.save_to_disk(reconstruction_folder + 'reconstruction_test')
# text_summarization_test.save_to_disk(text_summarization_folder + 'text_summarization_test')

# cot_gsm8k_adapt.save_to_disk(cot_reasoning_folder + 'cot_gsm8k_adapt')
# multi_hop_qa_adapt.save_to_disk(multi_hop_qa_folder + 'multi_hop_qa_adapt')
# reconstruction_adapt.save_to_disk(reconstruction_folder + 'reconstruction_adapt')



# load data from disk
cot_gsm8k_test = datasets.load_from_disk(cot_reasoning_folder + 'cot_gsm8k_test')
multi_hop_qa_test = datasets.load_from_disk(multi_hop_qa_folder + 'multi_hop_qa_test')
reconstruction_test = datasets.load_from_disk(reconstruction_folder + 'reconstruction_test')
text_summarization_test = datasets.load_from_disk(text_summarization_folder + 'text_summarization_test')

cot_gsm8k_adapt = datasets.load_from_disk(cot_reasoning_folder + 'cot_gsm8k_adapt')
multi_hop_qa_adapt = datasets.load_from_disk(multi_hop_qa_folder + 'multi_hop_qa_adapt')
reconstruction_adapt = datasets.load_from_disk(reconstruction_folder + 'reconstruction_adapt')

print("Datasets saved and loaded successfully.")
print(f"cot_gsm8k_test: {len(cot_gsm8k_test)} samples")
print(f"multi_hop_qa_test: {len(multi_hop_qa_test)} samples")
print(f"reconstruction_test: {len(reconstruction_test)} samples")
print(f"text_summarization_test: {len(text_summarization_test)} samples")
print(f"cot_gsm8k_adapt: {len(cot_gsm8k_adapt)} samples")
print(f"multi_hop_qa_adapt: {len(multi_hop_qa_adapt)} samples")
print(f"reconstruction_adapt: {len(reconstruction_adapt)} samples")