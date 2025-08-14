import os
import shutil

# Constants for subdirectory names
JSON_SUBDIR = "metadata_JSON"
PKL_SUBDIR = "metadata_pkl"

def organize_metadata_files(metadata_dir="metadata"):
    """
    Organize metadata files into subdirectories based on their extensions.

    Parameters:
        metadata_dir (str): Path to the directory containing metadata files.
    """
    json_dir = os.path.join(metadata_dir, JSON_SUBDIR)
    pkl_dir = os.path.join(metadata_dir, PKL_SUBDIR)

    # Create subdirectories if they don't exist
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(pkl_dir, exist_ok=True)

    # Iterate through files in the metadata directory
    for filename in os.listdir(metadata_dir):
        full_path = os.path.join(metadata_dir, filename)

        # Skip directories (do not recurse into subdirectories)
        if os.path.isdir(full_path):
            continue

        # Move JSON files to the JSON subdirectory
        if filename.endswith(".json"):
            shutil.move(full_path, os.path.join(json_dir, filename))
            print(f"Moved JSON: {filename}")
        # Move PKL files to the PKL subdirectory
        elif filename.endswith(".pkl"):
            shutil.move(full_path, os.path.join(pkl_dir, filename))
            print(f"Moved PKL: {filename}")

    print("\nMetadata organization complete.")

if __name__ == "__main__":
    try:
        organize_metadata_files()
    except Exception as e:
        print(f"An error occurred: {e}")
