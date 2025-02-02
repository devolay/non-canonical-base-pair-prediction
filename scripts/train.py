import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from pytorch_lightning.loggers import NeptuneLogger

from pair_prediction.data.dataset import LinkPredictionDataset
from pair_prediction.model.model import LitLinkPredictor

load_dotenv()

DATA_DIR = Path(
    "/Users/dawid/Private/School/Master's Thesuis/non-canonical-base-pair-prediction/data/raw/"
)
IDX_DIR = DATA_DIR / "idxs"
MATRIX_DIR = DATA_DIR / "matrices"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a link prediction model for RNA graphs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of graphs in a batch.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument(
        "--log-neptune", action="store_true", help="Log training metrics to Neptune."
    )
    return parser.parse_args()


def main(args):
    dataset = LinkPredictionDataset(idx_dir=IDX_DIR, matrix_dir=MATRIX_DIR)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = LitLinkPredictor(in_channels=4, hidden_channels=64, num_layers=2, dropout=0.3, lr=1e-3)

    trainer = pl.Trainer(max_epochs=args.epochs, log_every_n_steps=10, accelerator="cpu")

    if args.log_neptune:
        neptune_logger = NeptuneLogger(
            api_key=os.environ["NEPTUNE_KEY"],
            project=os.environ["NEPTUNE_PROJECT"],
            tags=["link-prediction", "rna"],
            log_model_checkpoints=True,
        )
        trainer.logger = neptune_logger

    # Start training.
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
