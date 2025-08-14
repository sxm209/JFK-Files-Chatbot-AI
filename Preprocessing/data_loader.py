from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
from multiprocessing import Pool
import os
import re
import gc
import torch
import spacy
import numpy as np
import glob
import faiss

# === CONFIG FLAGS ===
SAVE_METADATA_AS_PICKLE = True
SAVE_METADATA_AS_JSON = True

class Embedder:
    """
    Embedding model wrapper for BAAI/bge-base-en-v1.5.

    Attributes:
        device (str): Device to run the embedding model on (e.g., 'cpu', 'cuda').
    """

    def __init__(self, device='cpu'):
        self.device = device

    def embed(self, texts: List[str], batch_size=16) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Parameters:
            texts (List[str]): List of input texts.
            batch_size (int): Batch size for embedding generation.

        Returns:
            np.ndarray: Array of embeddings.
        """
        # Placeholder for embedding logic
        pass

def initializer(device='cpu', spacy_model="en_core_web_sm"):
    """
    Initialize global variables for embedding and NLP processing.

    Parameters:
        device (str): Device to run the embedding model on.
        spacy_model (str): spaCy model to load.
    """
    global embedder, nlp

    # Limit PyTorch to 1 thread per process to avoid CPU thread contention
    torch.set_num_threads(1)
    # Limit OpenMP-based libraries (NumPy, spaCy, etc.) to 1 thread as well
    os.environ["OMP_NUM_THREADS"] = "1"

    print(f"Initializing process with device={device} and spaCy model={spacy_model}")
    embedder = Embedder(device=device)
    nlp = spacy.load(spacy_model)

def extract_pages_and_metadata(file_str: str) -> Tuple[List[str], Optional[str], Optional[int]]:
    """
    Extract pages and metadata from a markdown file string.

    Parameters:
        file_str (str): Content of the markdown file as a string.

    Returns:
        Tuple[List[str], Optional[str], Optional[int]]: List of pages, file name, and release year.
    """
    file_name_match = re.search(r'file_name:\s*(.+)', file_str)
    release_year_match = re.search(r'release_year:\s*(\d+)', file_str)
    file_name = file_name_match.group(1).strip().strip("'\"") if file_name_match else None
    release_year = int(release_year_match.group(1)) if release_year_match else None

    file_str = re.sub(r'^---.*?---\s*', '', file_str, flags=re.DOTALL)
    file_str = re.sub(r'\n---\s*\n(?=## Page \d+)', '', file_str)
    file_str = re.sub(r'\n---\s*$', '', file_str)
    pages = re.split(r'## Page \d+', file_str)
    pages = [page.strip() for page in pages if page.strip()]
    return pages, file_name, release_year

def chunk_text(pages: List[str], chunk_size=1500, overlap_ratio=0.25, use_spacy=False) -> List[dict]:
    """
    Chunk text into smaller segments for processing.

    Parameters:
        pages (List[str]): List of text pages.
        chunk_size (int): Maximum size of each chunk.
        overlap_ratio (float): Overlap ratio between consecutive chunks.
        use_spacy (bool): Whether to use spaCy for chunking.

    Returns:
        List[dict]: List of text chunks with metadata.
    """
    global nlp
    full_text = ""
    page_boundaries = []
    for page in pages:
        full_text += page

    chunks = []

    if use_spacy:
        # Placeholder for spaCy-based chunking logic
        pass
    else:
        # Placeholder for simple chunking logic
        pass

    return chunks

def build_faiss_index(chunks, embedder, file_name=None, release_year=None, batch_size=16):
    """
    Build a FAISS index from text chunks and save metadata.

    Parameters:
        chunks (List[dict]): List of text chunks.
        embedder (Embedder): Embedding model wrapper.
        file_name (str): Name of the file being processed.
        release_year (int): Release year of the file.
        batch_size (int): Batch size for embedding generation.
    """
    texts = [c['chunk'] for c in chunks]
    print(f"Generating embeddings for chunks from {file_name}")
    embeddings = embedder.embed(texts, batch_size=batch_size).astype('float32')
    torch.cuda.empty_cache()

    # Save FAISS index per file
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    index_file = f"indices/{file_name}_index.bin"
    faiss.write_index(index, index_file)
    print(f"  Saved FAISS index: {index_file}")

    # Save metadata per file
    metadata = []
    for i, chunk in enumerate(chunks):
        # Placeholder for metadata logic
        pass

    if SAVE_METADATA_AS_PICKLE:
        # Placeholder for saving metadata as pickle
        pass
    if SAVE_METADATA_AS_JSON:
        # Placeholder for saving metadata as JSON
        pass

    print(f"  Saved metadata for {file_name}")

def main():
    """Main function to process markdown files and build FAISS indices."""
    folder_path = r"Path\to\your\markdown_files"  # Path to the folder containing markdown files
    chunk_size = 1500
    overlap_ratio = 0.25
    use_spacy_flag = True
    BATCH_SIZE = 6
    FILE_BATCH_SIZE = 8
    PROCESS_COUNT = 8

    # Ensure folders exist but don't recreate if already exist
    os.makedirs("indices", exist_ok=True)
    os.makedirs("metadata", exist_ok=True)

    # Gather markdown files
    md_files = sorted(glob.glob(os.path.join(folder_path, "*.md")))
    total_files = len(md_files)
    print(f"Found {total_files} markdown files in folder.\n")

    # Efficient check for already processed files
    processed_file_names = set()  # Placeholder for processed files logic
    unprocessed_files = [
        f for f in md_files 
        if os.path.basename(f) not in processed_file_names
    ]  

    print(f"Skipping {len(processed_file_names)} already processed files.")
    print(f"{len(unprocessed_files)} files left to process.\n")

    if not unprocessed_files:
        print("No files left to process.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    args_list = [(md_file, chunk_size, overlap_ratio, use_spacy_flag, BATCH_SIZE) for md_file in unprocessed_files]

    try:
        # Placeholder for multiprocessing logic
        pass
    except Exception as e:
        print(f"An error occurred during processing: {e}")

    print("Completed current processing batch.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")