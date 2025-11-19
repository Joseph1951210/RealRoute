"""
utils/data_load.py

This module loads data from the data/rag directory.
"""

import json
import numpy as np
from typing import List, Dict, Tuple


def load_queries(dataset: str, sample_size: int = None) -> List[Dict[str, str]]:
    """
    Load queries from dataset file.
    
    Args:
        dataset: Dataset name
        sample_size: Number of samples to return. If None, returns all data.
                    If specified, samples are uniformly distributed across the dataset.
    
    Returns:
        List of query dictionaries with 'query' and 'ground_truth' keys
    """
    import os
    
    # Try law-med naming convention first (_qa.json)
    qa_path = f"data/rag/{dataset}_qa.json"
    standard_path = f"data/rag/{dataset}.json"
    
    if os.path.exists(qa_path):
        file_path = qa_path
    elif os.path.exists(standard_path):
        file_path = standard_path
    else:
        raise FileNotFoundError(f"Neither {qa_path} nor {standard_path} exists")
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    if sample_size is None or sample_size >= len(data):
        # Return all data
        return [
            {"query": item["question"], "ground_truth": item["answer"]}
            for item in data
        ]
    
    # Uniformly sample indices across the dataset
    total_size = len(data)
    # Generate evenly spaced indices
    indices = np.linspace(0, total_size - 1, sample_size, dtype=int)
    
    return [
        {"query": data[i]["question"], "ground_truth": data[i]["answer"]}
        for i in indices
    ]


# utils/data_utils.py
def load_corpus_and_profiles(dataset: str) -> Tuple[List[str], List[str], str, str]:
    with open(f"data/rag/{dataset}_corpus_local.json", "r") as f:
        local = [f"{x['title']}. {x['text']}" for x in json.load(f)]
    with open(f"data/rag/{dataset}_corpus_global.json", "r") as f:
        global_ = [f"{x['title']}. {x['text']}" for x in json.load(f)]
    with open(f"data/rag/{dataset}_corpus_profiles.json", "r") as f:
        profiles = json.load(f)
    return local, global_, profiles["local_profile"], profiles["global_profile"]
