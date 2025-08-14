import os
from collections import Counter

def get_base_filenames(directory, valid_extensions):
    """
    Extract base filenames (without extensions) from a directory for specified file extensions.

    Parameters:
        directory (str): Path to the directory to scan.
        valid_extensions (list): List of valid file extensions to consider.

    Returns:
        list: List of base filenames without extensions.
    """
    base_names = []
    for file in os.listdir(directory):
        if any(file.endswith(ext) for ext in valid_extensions):
            base = file
            for ext in valid_extensions:
                if base.endswith(ext):
                    base = base[:-len(ext)]
            base_names.append(base.strip().strip("'\""))
    return base_names

def count_duplicates(name_list):
    """
    Count duplicate names in a list.

    Parameters:
        name_list (list): List of names to check for duplicates.

    Returns:
        tuple: Number of duplicates and a dictionary of duplicate names with their counts.
    """
    counter = Counter(name_list)
    duplicates = {name: count for name, count in counter.items() if count > 1}
    return len(duplicates), duplicates

def main():
    """Main function to check for duplicate index and metadata files."""
    indices_dir = "indices"
    metadata_dir = "metadata"

    # File extensions to check
    index_exts = [".bin"]
    metadata_exts = ["_metadata.pkl", "_metadata.json"]

    # Get base names
    index_bases = get_base_filenames(indices_dir, index_exts)
    metadata_bases = get_base_filenames(metadata_dir, metadata_exts)

    # Count duplicates
    index_dup_count, index_dups = count_duplicates(index_bases)
    metadata_dup_count, metadata_dups = count_duplicates(metadata_bases)

    print(f"Number of duplicate index files: {index_dup_count}")
    if index_dups:
        print("Duplicate index files:")
        for name, count in index_dups.items():
            print(f"  {name}: {count} occurrences")

    print(f"Number of duplicate metadata files: {metadata_dup_count}")
    if metadata_dups:
        print("Duplicate metadata files:")
        for name, count in metadata_dups.items():
            print(f"  {name}: {count} occurrences")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
