"""Script loads fasta file and training dataset and checks for overlap between the two."""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
from pair_prediction.data.dataset import LinkPredictionDataset
from pair_prediction.data.utils import load_fasta_sequences, save_fasta_sequences

DATA_DIR = Path("data/")

def parse_args():
    parser = argparse.ArgumentParser(description="Check for overlap between training dataset and fasta sequences.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to LinkPredictionDataset data directory")
    return parser.parse_args()

def main(args):
    train_dataset = LinkPredictionDataset(root=DATA_DIR, mode="train")
    valid_dataset = LinkPredictionDataset(root=args.data_dir)

    train_ids = set(train_dataset.id)

    overlap_ids = set()
    for data in tqdm(valid_dataset, desc="Checking for overlap", unit="sequence"):
        pdb_id = data.id.upper()
        for train_id in train_ids:
            train_pdb_id = train_id.split("_")[0]
            train_model_id = train_id.split("_")[1]
            train_string_id = train_id.split("_")[2]
            if pdb_id == train_pdb_id:
                print(f"Overlap found: {data.id} in training dataset {train_id}")
                overlap_ids.add(data.id)

    if len(overlap_ids) == 0:
        print("No overlap found between training dataset and FASTA sequences.")
    else:
        print(f"Found {len(overlap_ids)} overlapping sequences between training dataset and FASTA sequences.")

    # for overlap_id in overlap_ids:
    #     del fasta_data[overlap_id]

    # output_file = os.path.splitext(fasta_file)[0] + "_nonoverlap.txt"
    # with open(output_file, 'w') as f:
    #     for seq_id in fasta_data.keys():
    #         f.write(f"{seq_id}\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)