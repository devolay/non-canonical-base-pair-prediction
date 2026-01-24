import csv
from pathlib import Path

def create_idx_csv(base_path, output_csv):
    base_dir = Path(base_path)

    data = []

    search_pattern = "**/raw/idxs/*.idx"
    
    print(f"Scanning directory: {base_dir}...")

    for idx_file in base_dir.glob(search_pattern):
        dataset_name = idx_file.parents[2].name
        file_name = idx_file.name
        
        data.append({
            'evaluation_directory': dataset_name,
            'idx': file_name[:4]
        })

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['evaluation_directory', 'idx'])
        writer.writeheader()
        writer.writerows(data)

    print(f"Done! Created {output_csv} with {len(data)} entries.")

if __name__ == "__main__":
    EVAL_DIR = "/mnt/storage_3/home/dawid.stachowiak/non-canonical-base-pair-prediction/data/evaluation"
    OUTPUT_FILE = "benchmark_idxs.csv"
    
    create_idx_csv(EVAL_DIR, OUTPUT_FILE)