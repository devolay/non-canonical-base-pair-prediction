import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from lightning.pytorch.loggers import NeptuneLogger

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from pair_prediction.data.dataset import LinkPredictionDataset
from pair_prediction.model.lit_wrapper import LitWrapper
from pair_prediction.config import ModelConfig

load_dotenv()

DATA_DIR = Path("data/")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a link prediction model for RNA graphs.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()

def main(args):
    config = ModelConfig.from_yaml(args.config)
    
    train_dataset = LinkPredictionDataset(root=DATA_DIR, mode="train")
    val_dataset = LinkPredictionDataset(root=DATA_DIR, mode="validation")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    model = LitWrapper(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        save_weights_only=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=100,
        verbose=True,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        log_every_n_steps=10,
        accelerator="gpu",
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        gradient_clip_val=config.gradient_clip_value,
        gradient_clip_algorithm=config.gradient_clip_algorithm,
    )

    if config.log_neptune:
        neptune_logger = NeptuneLogger(
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            project=os.environ["NEPTUNE_PROJECT"],
            tags=["link-prediction", "rna"],
            log_model_checkpoints=False,
        )
        trainer.logger = neptune_logger

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    args = parse_args()
    main(args)
