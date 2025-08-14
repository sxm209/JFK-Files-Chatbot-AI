import os
import faiss
import pickle

# Constants
INDICES_DIR = r"Path\to\your\indices"
METADATA_DIR = r"metadata\metadata_pkl"
TEMP_BATCH_DIR = os.path.join(INDICES_DIR, "batch_temp")
FINAL_OUTPUT_DIR = os.path.join(INDICES_DIR, "Combined_FAISS")
BATCH_SIZE = 500  # Adjust based on your RAM

# Ensure necessary directories exist
os.makedirs(TEMP_BATCH_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

def load_faiss_index(path):
    """Load a FAISS index from the specified path."""
    return faiss.read_index(path)

def load_metadata(path):
    """Load metadata from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_faiss_index(index, path):
    """Save a FAISS index to the specified path."""
    faiss.write_index(index, path)

def save_metadata(metadata, path):
    """Save metadata to a pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(metadata, f)

def merge_indices(indices):
    """
    Merge multiple FAISS indices into a single index.

    Parameters:
        indices (list): List of FAISS indices to merge.

    Returns:
        faiss.IndexFlatIP: Merged FAISS index.
    """
    dim = indices[0].d
    combined_index = faiss.IndexFlatIP(dim)
    for idx in indices:
        combined_index.add(idx.reconstruct_n(0, idx.ntotal))
    return combined_index

def process_batch(batch_files, batch_num):
    """
    Process a batch of FAISS index files and their corresponding metadata.

    Parameters:
        batch_files (list): List of FAISS index file names.
        batch_num (int): Batch number for identification.

    Returns:
        tuple: Paths to the saved batch index and metadata files.
    """
    batch_indices = []
    batch_metadata = []

    for index_file in batch_files:
        base_name = index_file.replace("_index.bin", "")
        index_path = os.path.join(INDICES_DIR, index_file)
        metadata_path = os.path.join(METADATA_DIR, f"{base_name}_metadata.pkl")

        if not os.path.exists(metadata_path):
            print(f"[WARNING] Metadata missing for {base_name}, skipping.")
            continue

        idx = load_faiss_index(index_path)
        batch_indices.append(idx)

        md = load_metadata(metadata_path)
        batch_metadata.extend(md)

    if not batch_indices:
        print(f"No valid indices in batch {batch_num}, skipping batch save.")
        return None, None

    combined_index = merge_indices(batch_indices)

    batch_index_path = os.path.join(TEMP_BATCH_DIR, f"batch_{batch_num}_index.bin")
    batch_metadata_path = os.path.join(TEMP_BATCH_DIR, f"batch_{batch_num}_metadata.pkl")

    save_faiss_index(combined_index, batch_index_path)
    save_metadata(batch_metadata, batch_metadata_path)

    print(f"Batch {batch_num} saved with {len(batch_indices)} indices.")

    return batch_index_path, batch_metadata_path

def merge_all_batches(batch_index_paths, batch_metadata_paths):
    """
    Merge all batch indices and metadata into final combined files.

    Parameters:
        batch_index_paths (list): List of batch index file paths.
        batch_metadata_paths (list): List of batch metadata file paths.
    """
    print("Merging all batch indices into final combined index...")
    indices = [load_faiss_index(p) for p in batch_index_paths]
    metadata = []
    for md_path in batch_metadata_paths:
        metadata.extend(load_metadata(md_path))

    final_index = merge_indices(indices)

    final_index_path = os.path.join(FINAL_OUTPUT_DIR, "combined_index.bin")
    final_metadata_path = os.path.join(FINAL_OUTPUT_DIR, "combined_metadata.pkl")

    save_faiss_index(final_index, final_index_path)
    save_metadata(metadata, final_metadata_path)

    print(f"Final combined index saved to: {final_index_path}")
    print(f"Final combined metadata saved to: {final_metadata_path}")

def main():
    """Main function to process and merge FAISS indices in batches."""
    index_files = [f for f in os.listdir(INDICES_DIR) if f.endswith("_index.bin")]
    total_files = len(index_files)
    print(f"Total index files found: {total_files}")

    batch_index_paths = []
    batch_metadata_paths = []

    for i in range(0, total_files, BATCH_SIZE):
        batch_files = index_files[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"\nProcessing batch {batch_num} with {len(batch_files)} files...")
        idx_path, md_path = process_batch(batch_files, batch_num)
        if idx_path and md_path:
            batch_index_paths.append(idx_path)
            batch_metadata_paths.append(md_path)

    # Merge all batch files into final combined index
    merge_all_batches(batch_index_paths, batch_metadata_paths)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
