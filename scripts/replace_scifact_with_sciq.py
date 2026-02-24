#!/usr/bin/env python3
"""
Replace SciFact with SciQ in the multi-source dataset.

SciQ (Allen AI) has proper Q&A pairs:
  - question: science exam question
  - correct_answer: short, exact answer (EM-friendly)
  - support: background paragraph (used as corpus)

This script:
1. Downloads SciQ train+validation+test from HuggingFace (parquet)
2. Builds multi_source_corpus_sciq.json
3. Samples 200 QA pairs for multi_source.json
4. Updates multi_source_profiles.json
5. Regenerates multi_source.json (wiki + bioasq + sciq)
"""

import json
import os
import random
import tempfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "rag")
DATA_DIR = os.path.abspath(DATA_DIR)

SCIQ_QA_SAMPLE = 200  # Number of SciQ QA pairs to include

# HuggingFace parquet URLs for SciQ
PARQUET_URLS = {
    "train": "https://huggingface.co/datasets/allenai/sciq/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/allenai/sciq/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
    "test": "https://huggingface.co/datasets/allenai/sciq/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
}


def download_sciq():
    """Download SciQ parquet files and return all rows as list of dicts."""
    import pandas as pd

    all_rows = []
    for split_name, url in PARQUET_URLS.items():
        print(f"  Downloading SciQ {split_name} split...")
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            urllib.request.urlretrieve(url, tmp_path)
            df = pd.read_parquet(tmp_path)
            rows = df.to_dict("records")
            all_rows.extend(rows)
            print(f"    Got {len(rows)} rows from {split_name}")
        finally:
            os.unlink(tmp_path)

    print(f"  Total SciQ rows: {len(all_rows)}")
    return all_rows


def build_corpus(rows):
    """Build corpus from SciQ support text.
    
    Each entry: {"title": <first sentence or question>, "text": <support paragraph>}
    Only includes rows that have non-empty support text.
    """
    corpus = []
    seen_texts = set()

    for row in rows:
        support = (row.get("support") or "").strip()
        if not support or len(support) < 30:
            continue

        # Deduplicate by text content
        text_key = support[:200]
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)

        # Use the first sentence as title (capped at 150 chars)
        first_sentence = support.split(".")[0].strip()
        if len(first_sentence) > 150:
            first_sentence = first_sentence[:147] + "..."

        corpus.append({
            "title": first_sentence,
            "text": support
        })

    return corpus


def build_qa(rows, sample_size=SCIQ_QA_SAMPLE):
    """Build QA pairs from SciQ.
    
    Uses question + correct_answer. Only includes rows with non-empty support
    (so there's matching corpus content).
    """
    qa_candidates = []
    for row in rows:
        question = (row.get("question") or "").strip()
        answer = (row.get("correct_answer") or "").strip()
        support = (row.get("support") or "").strip()

        if not question or not answer or not support or len(support) < 30:
            continue

        qa_candidates.append({
            "question": question,
            "answer": answer,
            "source": "sciq"
        })

    random.seed(42)
    if len(qa_candidates) > sample_size:
        qa_candidates = random.sample(qa_candidates, sample_size)

    print(f"  SciQ QA pairs: {len(qa_candidates)}")
    return qa_candidates


def update_profiles():
    """Update multi_source_profiles.json: replace scifact with sciq."""
    profiles_path = os.path.join(DATA_DIR, "multi_source_profiles.json")

    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    # Remove scifact if exists
    profiles.pop("scifact", None)

    # Add sciq
    profiles["sciq"] = (
        "This knowledge base contains science educational content covering "
        "physics, chemistry, biology, earth science, and life sciences. "
        "Each entry is an explanatory paragraph from science textbooks or "
        "educational materials, describing scientific concepts, processes, "
        "and phenomena. It is suitable for answering science exam questions "
        "about natural phenomena, chemical reactions, biological processes, "
        "physical laws, and earth science topics."
    )

    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    print(f"  Updated profiles: {list(profiles.keys())}")
    return profiles


def regenerate_mixed_qa(sciq_qa):
    """Regenerate multi_source.json: keep wiki + bioasq, replace scifact with sciq."""
    qa_path = os.path.join(DATA_DIR, "multi_source.json")

    with open(qa_path, "r", encoding="utf-8") as f:
        existing_qa = json.load(f)

    # Keep wiki and bioasq entries
    kept_qa = [item for item in existing_qa if item.get("source") != "scifact"]
    print(f"  Kept {len(kept_qa)} existing QA (wiki + bioasq)")
    print(f"  Removed {len(existing_qa) - len(kept_qa)} scifact QA")

    # Add sciq
    kept_qa.extend(sciq_qa)
    print(f"  Added {len(sciq_qa)} sciq QA")

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(kept_qa)

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(kept_qa, f, indent=2, ensure_ascii=False)

    # Count by source
    counts = {}
    for item in kept_qa:
        src = item.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    print(f"  Final QA distribution: {counts}")
    print(f"  Total QA: {len(kept_qa)}")

    return kept_qa


def main():
    print("=" * 60)
    print("Replacing SciFact with SciQ")
    print("=" * 60)

    # Step 1: Download SciQ
    print("\n[1/5] Downloading SciQ dataset...")
    rows = download_sciq()

    # Step 2: Build corpus
    print("\n[2/5] Building SciQ corpus...")
    corpus = build_corpus(rows)
    corpus_path = os.path.join(DATA_DIR, "multi_source_corpus_sciq.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(corpus)} corpus entries to {corpus_path}")

    # Step 3: Build QA
    print("\n[3/5] Building SciQ QA pairs...")
    sciq_qa = build_qa(rows)

    # Step 4: Update profiles
    print("\n[4/5] Updating profiles...")
    update_profiles()

    # Step 5: Regenerate mixed QA
    print("\n[5/5] Regenerating mixed QA file...")
    regenerate_mixed_qa(sciq_qa)

    # Step 6: Remove old scifact corpus
    old_scifact = os.path.join(DATA_DIR, "multi_source_corpus_scifact.json")
    if os.path.exists(old_scifact):
        os.remove(old_scifact)
        print(f"\n  Deleted old file: {old_scifact}")

    print("\n" + "=" * 60)
    print("Done! SciFact has been replaced with SciQ.")
    print("=" * 60)
    print("\nFiles updated:")
    print(f"  NEW:     {corpus_path}")
    print(f"  UPDATED: {os.path.join(DATA_DIR, 'multi_source_profiles.json')}")
    print(f"  UPDATED: {os.path.join(DATA_DIR, 'multi_source.json')}")
    if os.path.exists(old_scifact):
        pass
    else:
        print(f"  DELETED: {old_scifact}")


if __name__ == "__main__":
    main()
