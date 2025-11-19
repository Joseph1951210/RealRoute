"""
rag/naive_rag.py

This module implements the NaiveRAG system for document retrieval and question answering.
"""

import os
import sys
import json
import requests
import numpy as np
from typing import List, Dict, Union, Optional
from sklearn.metrics.pairwise import cosine_similarity
import time
import tiktoken
import networkx as nx
from collections import defaultdict
import re

# 设置环境变量以解决 Unicode 编码问题（必须在导入 sentence_transformers 之前）
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LC_CTYPE'] = 'en_US.UTF-8'

# 清理可能包含非 ASCII 字符的代理环境变量
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    if key in os.environ:
        value = os.environ[key]
        try:
            value.encode('latin-1')
        except UnicodeEncodeError:
            # 如果包含非 ASCII 字符，删除该环境变量
            del os.environ[key]

from sentence_transformers import SentenceTransformer




class NaiveRAG:
    def __init__(self, docs: List[str], embed_model: str = "all-MiniLM-L6-v2"):
        self.docs = docs
        self.embedder = SentenceTransformer(embed_model)
        self.doc_vecs = self.embedder.encode(docs, convert_to_numpy=True)
        print(f"Initialized knowledge base, number of documents: {len(docs)}")

    def rag_qa(self, question: str, k: int = 5) -> Dict:
        start_time = time.time()
        
        # Encode the question
        q_vec = self.embedder.encode([question], convert_to_numpy=True)
        
        # Calculate similarity
        scores = cosine_similarity(q_vec, self.doc_vecs)[0]
        topk_idx = np.argsort(scores)[-k:][::-1]
        topk_docs = [self.docs[i] for i in topk_idx]
        topk_scores = [float(scores[i]) for i in topk_idx]
        
        # Compute retrieval metrics
        retrieval_time = time.time() - start_time
        avg_similarity = np.mean(topk_scores)
        max_similarity = np.max(topk_scores)
        
        return {
            "docs": topk_docs,
            "doc_scores": topk_scores,
            "metrics": {
                "retrieval_time": retrieval_time,
                "avg_similarity": avg_similarity,
                "max_similarity": max_similarity,
                "total_docs_searched": len(self.docs)
            }
        }
