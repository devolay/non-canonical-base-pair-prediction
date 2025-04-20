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
    parser.add_argument("--fasta-file", type=str, required=True, help="Path to the FASTA file.")
    return parser.parse_args()

def main(args):
    fasta_file = args.fasta_file
    fasta_data = load_fasta_sequences(fasta_file)

    fasta_ids = set(fasta_data.keys())

    train_dataset = LinkPredictionDataset(root=DATA_DIR, validation=False)
    train_ids = set(train_dataset.id)

    overlap_ids = set()
    for fasta_id in tqdm(fasta_ids, desc="Checking for overlap", unit="sequence"):
        pdb_id, model_id, string_id = fasta_id.split("-")
        pdb_id = pdb_id.upper()
        for train_id in train_ids:
            train_pdb_id = train_id.split("_")[0]
            train_model_id = train_id.split("_")[1]
            train_string_id = train_id.split("_")[2]
            if pdb_id == train_pdb_id and model_id == train_model_id and string_id == train_string_id:
                print(f"Overlap found: {fasta_id} in training dataset {train_id}")
                overlap_ids.add(fasta_id)

    if len(overlap_ids) == 0:
        print("No overlap found between training dataset and FASTA sequences.")
    else:
        print(f"Found {len(overlap_ids)} overlapping sequences between training dataset and FASTA sequences.")

    for overlap_id in overlap_ids:
        del fasta_data[overlap_id]

    output_file = os.path.splitext(fasta_file)[0] + "_nonoverlap.txt"
    with open(output_file, 'w') as f:
        for seq_id in fasta_data.keys():
            f.write(f"{seq_id}\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)