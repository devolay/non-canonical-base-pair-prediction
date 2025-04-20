import argparse
from pathlib import Path
from pair_prediction.data.utils import export_dataset_to_fasta
from pair_prediction.data.dataset import LinkPredictionDataset

DATA_DIR = Path("data/")

def parse_args():
    parser = argparse.ArgumentParser(description="Export dataset to FASTA format.")
    parser.add_argument("--batchsize", type=int, default=None, help="Batch size for exporting.")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory.")
    return parser.parse_args()

def main(args):
    output_dir = Path(args.output)
    batchsize = args.batchsize

    val_dataset = LinkPredictionDataset(root=DATA_DIR, validation=True)
    export_dataset_to_fasta(val_dataset, output_dir, batchsize)

    print(f"Exported validation dataset to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)