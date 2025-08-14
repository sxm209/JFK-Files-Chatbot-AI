import os
import faiss
import pickle

# === Constants ===
INDICES_DIR = r"path\to\your\indices"  # Path to the directory containing FAISS index files
METADATA_DIR = r"metadata"  # Directory containing metadata files
OUTPUT_DIR = os.path.join(INDICES_DIR, "Combined_FAISS")

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def main():
    """Main function to merge FAISS indices and metadata into combined files."""
    combined_index = None
    combined_metadata = []

    # Iterate through all index files
    index_files = [f for f in os.listdir(INDICES_DIR) if f.endswith("_index.bin")]

    for index_file in index_files:
        index_path = os.path.join(INDICES_DIR, index_file)
        base_name = index_file.replace("_index.bin", "")

        # Load corresponding metadata
        metadata_file = os.path.join(METADATA_DIR, f"{base_name}_metadata.pkl")
        if not os.path.exists(metadata_file):
            print(f"[WARNING] Metadata not found for {base_name}, skipping...")
            continue

        print(f"Processing {index_file} ...")

        # Load FAISS index
        index = load_faiss_index(index_path)

        # Convert to cosine similarity (inner product on normalized vectors)
        if combined_index is None:
            combined_index = faiss.IndexFlatIP(index.d)
        combined_index.add(index.reconstruct_n(0, index.ntotal))

        # Load metadata
        metadata = load_metadata(metadata_file)
        combined_metadata.extend(metadata)

    # Save combined index
    combined_index_path = os.path.join(OUTPUT_DIR, "combined_index.bin")
    save_faiss_index(combined_index, combined_index_path)
    print(f"\nSaved combined FAISS index to: {combined_index_path}")

    # Save combined metadata
    combined_metadata_path = os.path.join(OUTPUT_DIR, "combined_metadata.pkl")
    save_metadata(combined_metadata, combined_metadata_path)
    print(f"Saved combined metadata to: {combined_metadata_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
