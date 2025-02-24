import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pair_prediction.data.dataset import LinkPredictionDataset
from pair_prediction.model.lit_wrapper import LitWrapper

load_dotenv()

DATA_DIR = Path(
    "/Users/dawid/Private/School/Master's Thesuis/non-canonical-base-pair-prediction/data/"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a link prediction model for RNA graphs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of graphs in a batch.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GCN layers.")
    parser.add_argument("--hidden-channels", type=int, default=64, help="Number of hidden channels.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--model-type", type=str, default="local", help="Type of model to train.")
    parser.add_argument(
        "--log-neptune", action="store_true", help="Log training metrics to Neptune."
    )
    return parser.parse_args()


def main(args):
    train_dataset = LinkPredictionDataset(root=DATA_DIR, validation=False)
    val_dataset = LinkPredictionDataset(root=DATA_DIR, validation=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = LitWrapper(model_type=args.model_type, in_channels=4, hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=0.1, lr=1e-3)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=10,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
    )

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
