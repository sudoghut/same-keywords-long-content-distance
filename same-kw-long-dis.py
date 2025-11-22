#!/usr/bin/env python3
"""
Same Keywords Long Distance Analysis

This script analyzes news data with the same keywords to find pairs with the lowest similarity scores.
It reads data from a SQLite database, generates embeddings using sentence transformers,
and identifies news pairs with the same keywords but lowest content similarity.

Installation:
    pip install -r requirements.txt
"""

# ==============================================================================
# STEP 1: Import Required Libraries
# ==============================================================================

import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from itertools import combinations
import os
import json
from typing import List, Union
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("SAME KEYWORDS LONG DISTANCE ANALYSIS")
print("=" * 80)
print()


# ==============================================================================
# STEP 2: Read Database Path from .env File
# ==============================================================================

def read_db_path_from_env(env_file: str = '.env') -> str:
    """
    Read database path from .env file.
    Expected format: db_path: xxx
    
    Args:
        env_file: Path to the .env file
        
    Returns:
        Database path as string
    """
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('db_path:'):
                    db_path = line.split('db_path:')[1].strip()
                    return db_path
        raise ValueError("db_path not found in .env file")
    except FileNotFoundError:
        raise FileNotFoundError(f"{env_file} file not found")


def read_embedded_ids_from_env(env_file: str = '.env') -> List[int]:
    """
    Read embedded IDs from .env file.
    Expected format: embedded_ids: [1, 2, 3, 4]
    
    Args:
        env_file: Path to the .env file
        
    Returns:
        List of embedded IDs as integers. Returns empty list if field is empty or invalid.
    """
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('embedded_ids:'):
                    # Extract the JSON part after the colon
                    json_part = line.split('embedded_ids:')[1].strip()
                    
                    # If empty, return empty list
                    if not json_part:
                        return []
                    
                    # Parse JSON array
                    try:
                        ids = json.loads(json_part)
                        # Ensure it's a list of integers
                        if isinstance(ids, list):
                            return [int(id) for id in ids]
                        else:
                            print(f"Warning: embedded_ids is not a list, returning empty list")
                            return []
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON in embedded_ids, returning empty list")
                        return []
        
        # If embedded_ids line not found, return empty list
        return []
    except FileNotFoundError:
        print(f"Warning: {env_file} file not found, returning empty list")
        return []
    except Exception as e:
        print(f"Warning: Error reading embedded_ids: {str(e)}, returning empty list")
        return []


def write_embedded_ids_to_env(embedded_ids: List[int], env_file: str = '.env'):
    """
    Write embedded IDs to .env file in JSON format.
    Updates the embedded_ids line while preserving all other content.
    
    Args:
        embedded_ids: List of embedded IDs to write
        env_file: Path to the .env file
    """
    try:
        # Read current content
        with open(env_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Convert IDs to JSON format (compact, no extra whitespace)
        ids_json = json.dumps(embedded_ids, separators=(',', ' '))
        
        # Find and update the embedded_ids line
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith('embedded_ids:'):
                lines[i] = f'embedded_ids: {ids_json}\n'
                updated = True
                break
        
        # If embedded_ids line doesn't exist, add it
        if not updated:
            lines.append(f'embedded_ids: {ids_json}\n')
        
        # Write back to file
        with open(env_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
    except Exception as e:
        print(f"Error writing embedded_ids to .env: {str(e)}")
        raise


def add_embedded_ids(new_ids: List[int], env_file: str = '.env'):
    """
    Add new embedded IDs to the .env file.
    Merges with existing IDs, removes duplicates, and sorts the list.
    
    Args:
        new_ids: List of new IDs that were just embedded
        env_file: Path to the .env file
    """
    try:
        # Read current embedded IDs
        current_ids = read_embedded_ids_from_env(env_file)
        
        # Merge with new IDs (use set to remove duplicates)
        combined_ids = list(set(current_ids + new_ids))
        
        # Sort the list
        combined_ids.sort()
        
        # Write back to .env
        write_embedded_ids_to_env(combined_ids, env_file)
        
        print(f"✓ Updated embedded_ids in .env: added {len(new_ids)} new IDs, total now: {len(combined_ids)}")
        
    except Exception as e:
        print(f"Error adding embedded_ids: {str(e)}")
        # Don't raise to allow script to continue


# Read database path
print("Reading database configuration...")
db_path = read_db_path_from_env()
print(f"✓ Database path: {db_path}")
print()


# ==============================================================================
# STEP 3: Query Database and Create Initial Dataframe
# ==============================================================================

def query_news_data(db_path: str) -> pd.DataFrame:
    """
    Query the database to get news data with keywords.
    Joins main_news_data with serpapi_data to get the query field.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        DataFrame with id, serpapi_id, news, date, and keywords columns
    """
    try:
        conn = sqlite3.connect(db_path)
        
        query = """
        SELECT 
            m.id,
            m.serpapi_id,
            m.news,
            m.date,
            s.query as keywords
        FROM main_news_data m
        JOIN serpapi_data s ON m.serpapi_id = s.id
        WHERE m.news IS NOT NULL AND m.news != ''
        ORDER BY s.query, m.date
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"✓ Total records loaded: {len(df)}")
        print(f"✓ Unique keywords: {df['keywords'].nunique()}")
        print()
        
        return df
    
    except Exception as e:
        raise Exception(f"Error querying database: {str(e)}")


# Load data
print("Loading data from database...")
df = query_news_data(db_path)
print(f"First few records:")
print(df.head())
print()

# For testing purposes, limit to first 20 records
# df = df[:20]

# ==============================================================================
# STEP 4: Define Modular Embedding Function
# ==============================================================================

# Initialize the embedding model (global variable for efficiency)
embedding_model = None


def check_gpu_setup():
    """
    Check and display GPU setup information.
    Helps diagnose why GPU might not be available.
    """
    import torch
    
    print("\n" + "="*80)
    print("GPU SETUP DIAGNOSTICS")
    print("="*80)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\n⚠️  CUDA is NOT available. Possible reasons:")
        print("1. PyTorch CPU-only version is installed")
        print("2. CUDA drivers not installed")
        print("3. CUDA version mismatch")
        print("\n📋 To fix this:")
        print("1. Uninstall current PyTorch:")
        print("   pip uninstall torch torchvision torchaudio")
        print("\n2. Install PyTorch with CUDA support:")
        print("   For CUDA 11.8:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n   For CUDA 12.1:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\n   Note: CUDA 11.8 is backward compatible with CUDA 11.6")
        print("\n3. Verify installation:")
        print("   python -c \"import torch; print(torch.cuda.is_available())\"")
    
    print("="*80 + "\n")


def get_embeddings(texts: Union[str, List[str]], model_name: str = "Qwen/Qwen3-Embedding-0.6B") -> np.ndarray:
    """
    Generate embeddings for input text(s) using the specified model.
    This function is modular and can be easily replaced with different embedding models.
    
    Args:
        texts: Single text string or list of text strings to embed
        model_name: Name of the sentence transformer model to use
        
    Returns:
        numpy array of embeddings
    """
    global embedding_model
    
    # Load model only once with GPU support if available
    if embedding_model is None:
        print(f"Loading embedding model: {model_name}...")
        
        # Check if CUDA (NVIDIA GPU) is available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load model with device specification
        embedding_model = SentenceTransformer(model_name, device=device)
        print("✓ Model loaded successfully!")
        print()
    
    # Convert single string to list
    if isinstance(texts, str):
        texts = [texts]
    
    # Generate embeddings
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    
    return embeddings


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score
    """
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


# Test the embedding function with a sample
print("Testing embedding function...")

# Add GPU diagnostics
check_gpu_setup()

print("Loading embedding model: Qwen/Qwen3-Embedding-0.6B...")
test_embedding = get_embeddings("This is a test sentence.")
print(f"✓ Embedding shape: {test_embedding.shape}")
print()

# ==============================================================================
# STEP 4.5: NumPy-based Embedding Storage Functions
# ==============================================================================

def save_embeddings_to_file(ids, embeddings, keywords, filepath='embeddings_storage.npz'):
    """
    Save embeddings to a NumPy compressed file with IDs and keywords.
    
    Args:
        ids: List of news IDs
        embeddings: NumPy array of embeddings
        keywords: String of keywords for this group
        filepath: Path to save the embeddings file
    """
    try:
        # Load existing data if file exists
        if os.path.exists(filepath):
            existing_data = np.load(filepath, allow_pickle=True)
            existing_ids = existing_data['ids'].tolist()
            existing_embeddings = existing_data['embeddings']
            existing_keywords = existing_data['keywords'].tolist()
            
            # Append new data
            all_ids = existing_ids + ids
            all_embeddings = np.vstack([existing_embeddings, embeddings])
            all_keywords = existing_keywords + [keywords] * len(ids)
        else:
            # First time saving
            all_ids = ids
            all_embeddings = embeddings
            all_keywords = [keywords] * len(ids)
        
        # Save to compressed NumPy file
        np.savez_compressed(
            filepath,
            ids=np.array(all_ids),
            embeddings=all_embeddings,
            keywords=np.array(all_keywords)
        )
        print(f"  ✓ Saved {len(ids)} embeddings to {filepath}")
        return True
    except Exception as e:
        print(f"  ✗ Error saving embeddings: {e}")
        return False



def load_embeddings_from_file(filepath='embeddings_storage.npz'):
    """
    Load embeddings from a NumPy compressed file.
    
    Args:
        filepath: Path to the embeddings file
        
    Returns:
        Dictionary mapping news IDs to their embeddings, or empty dict if file doesn't exist
    """
    try:
        if not os.path.exists(filepath):
            print(f"  ℹ No existing embeddings file found at {filepath}")
            return {}
        
        data = np.load(filepath, allow_pickle=True)
        ids = data['ids'].tolist()
        embeddings = data['embeddings']
        
        # Create a dictionary mapping ID to embedding
        embedding_dict = {int(id_val): embeddings[i] for i, id_val in enumerate(ids)}
        
        print(f"  ✓ Loaded {len(embedding_dict)} cached embeddings from {filepath}")
        return embedding_dict
    
    except Exception as e:
        print(f"  ⚠ Error loading embeddings: {e}")
        return {}


# ==============================================================================
# STEP 5: Find Lowest Similarity Pairs for Each Keyword
# ==============================================================================

def find_lowest_similarity_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each keyword with multiple records, find the pair with the lowest similarity.
    The pair consists of later news (news1) and earlier news (news2) based on date.
    Uses cached embeddings when available to avoid regenerating them.
    
    Args:
        df: DataFrame with news data
        
    Returns:
        DataFrame with lowest similarity pairs
    """
    results = []
    
    # Load cached embeddings at the start
    print("Loading cached embeddings...")
    cached_embeddings = load_embeddings_from_file()
    print()
    
    # Group by keywords
    grouped = df.groupby('keywords')

    # Filter out groups with only one record
    df_filtered = df[df.groupby('keywords')['keywords'].transform('size') > 1]
    grouped = df_filtered.groupby('keywords')

    print(f"Processing {len(grouped)} keyword groups...")
    print()

    
    for keyword, group in grouped:
        # Skip keywords with only one record
        if len(group) < 2:
            continue
        
        # Sort by date to ensure proper ordering
        group = group.sort_values('date').reset_index(drop=True)
        
        # Separate IDs into cached and new
        all_ids = group['id'].tolist()
        cached_ids = [id_val for id_val in all_ids if id_val in cached_embeddings]
        new_ids = [id_val for id_val in all_ids if id_val not in cached_embeddings]
        
        print(f"Keyword '{keyword}': {len(cached_ids)} cached, {len(new_ids)} new embeddings needed")
        
        # Build embeddings array for all items in group
        embeddings = []
        new_embeddings_list = []
        new_ids_to_save = []
        
        # Generate embeddings only for new IDs
        if new_ids:
            new_group = group[group['id'].isin(new_ids)]
            news_texts = new_group['news'].tolist()
            new_embeddings = get_embeddings(news_texts)
            new_embeddings_list = new_embeddings
            new_ids_to_save = new_ids
        
        # Build complete embeddings array in the same order as group
        for id_val in all_ids:
            if id_val in cached_embeddings:
                embeddings.append(cached_embeddings[id_val])
            else:
                # Find index in new_ids_to_save
                idx = new_ids_to_save.index(id_val)
                embeddings.append(new_embeddings_list[idx])
        
        embeddings = np.array(embeddings)
        
        # Save new embeddings to file
        if new_ids_to_save:
            keywords_value = group['keywords'].iloc[0]
            save_embeddings_to_file(new_ids_to_save, new_embeddings_list, keywords_value)
            
            # Track embedded IDs in .env file
            add_embedded_ids(new_ids_to_save)
            
            # Update cached_embeddings dict for subsequent groups
            for i, id_val in enumerate(new_ids_to_save):
                cached_embeddings[id_val] = new_embeddings_list[i]
        
        # Find the pair with lowest similarity
        # Only consider pairs where news1 is later than news2
        min_similarity = float('inf')
        best_pair = None
        
        for i in range(len(group)):
            for j in range(i):
                # i is later (news1), j is earlier (news2)
                row1 = group.iloc[i]
                row2 = group.iloc[j]
                
                # Calculate date difference
                date_diff = abs((row1['date'] - row2['date']).days)
                
                # Skip pairs with same date (date_diff = 0)
                if date_diff == 0:
                    continue
                
                similarity = calculate_cosine_similarity(embeddings[i], embeddings[j])
                
                if similarity < min_similarity:
                    min_similarity = similarity
                    best_pair = (i, j)
        
        if best_pair is not None:
            idx1, idx2 = best_pair
            row1 = group.iloc[idx1]
            row2 = group.iloc[idx2]
            
            # Calculate date difference
            date_diff = (row1['date'] - row2['date']).days
            
            results.append({
                'id': row1['id'],
                'keywords': keyword,
                'news1': row1['news'],
                'news2': row2['news'],
                'similarity': min_similarity,
                'serpapi_id': row1['serpapi_id'],
                'date_diff': date_diff,
                'date1': row1['date'],
                'date2': row2['date']
            })
    
    # Create DataFrame from results
    result_df = pd.DataFrame(results)
    
    # Check if we have any results before sorting
    if len(result_df) == 0:
        print("\n" + "="*80)
        print("No new similarity pairs to calculate - all IDs already processed")
        print("="*80)
        return result_df
    
    # Sort by similarity (lowest first)
    result_df = result_df.sort_values('similarity').reset_index(drop=True)
    
    print(f"✓ Found {len(result_df)} keyword pairs with lowest similarity")
    print()
    
    return result_df


# Find lowest similarity pairs
print("=" * 80)
print("FINDING LOWEST SIMILARITY PAIRS")
print("=" * 80)
print()
result_df = find_lowest_similarity_pairs(df)
print("Top 10 results:")
if len(result_df) > 0:
    print(result_df.head(10))
else:
    print("No results - all IDs already processed")
print()


# ==============================================================================
# STEP 6: Display Summary Statistics
# ==============================================================================

print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print()
print(f"Total pairs found: {len(result_df)}")
print()
# Only show statistics if we have results
if len(result_df) > 0:
    print("Similarity score statistics:")
    print(result_df['similarity'].describe())
    print()
    print("Date difference statistics (days):")
    print(result_df['date_diff'].describe())
    print()
    print(f"Lowest similarity score: {result_df['similarity'].min():.4f}")
    print(f"Highest similarity score: {result_df['similarity'].max():.4f}")
else:
    print("No statistics to display - no new pairs were calculated")
print()


# ==============================================================================
# STEP 7: Save Results to CSV
# ==============================================================================

def save_to_csv(df: pd.DataFrame, filename: str = 'same_keywords_lowest_similarity.csv'):
    """
    Save the DataFrame to a CSV file with UTF-8 with BOM encoding.
    
    Args:
        df: DataFrame to save
        filename: Output filename
    """
    try:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✓ Results saved to: {filename}")
        print(f"✓ Total rows saved: {len(df)}")
        print()
    except Exception as e:
        print(f"✗ Error saving CSV: {str(e)}")
        print()


# Save results
print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)
print()
save_to_csv(result_df)


# ==============================================================================
# STEP 8: Display Sample Results
# ==============================================================================

print("=" * 80)
print("SAMPLE RESULTS (Top 5 Lowest Similarity Pairs)")
print("=" * 80)
print()

if len(result_df) > 0:
    for idx, row in result_df.head(5).iterrows():
        print(f"{'─' * 80}")
        print(f"PAIR {idx + 1}")
        print(f"{'─' * 80}")
        print(f"Keywords:        {row['keywords']}")
        print(f"Similarity:      {row['similarity']:.4f}")
        print(f"Date difference: {row['date_diff']} days")
        print(f"Date1 (later):   {row['date1']}")
        print(f"Date2 (earlier): {row['date2']}")
        print()
        print(f"News1 (later):")
        print(f"  {row['news1'][:200]}...")
        print()
        print(f"News2 (earlier):")
        print(f"  {row['news2'][:200]}...")
        print()
else:
    print("No sample results to display - all IDs already processed")
    print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)