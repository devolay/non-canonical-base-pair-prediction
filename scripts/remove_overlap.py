import shutil
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict

import click
from tqdm import tqdm

from pair_prediction.data.dataset import LinkPredictionDataset


def collect_train_pdb_ids(train_ds_root: str | Path) -> Dict[str, List[Tuple[str, str]]]:
    """Return upper-case 4-character PDB IDs present in the training dataset."""
    train_dataset = LinkPredictionDataset(root=str(train_ds_root), mode="train")
    train_ids = defaultdict(list)
    for data in train_dataset:
        splitted_id = data.id.split("_")
        pdb_id = splitted_id[0].upper()
        model_id = splitted_id[1].upper()
        string_id = splitted_id[2].upper()
        train_ids[pdb_id].append((model_id, string_id))
    return train_ids
        

def copy_if_not_overlapping(
    valid_ds_root: str | Path,
    out_root: Path,
    train_ids: Dict[str, List[Tuple[str, str]]],
) -> None:
    """Copy validation samples that *do not* overlap into *out_root*."""
    valid_dataset = LinkPredictionDataset(root=str(valid_ds_root))

    kept, dropped = 0, 0
    out_raw = out_root / "raw"
    out_raw.mkdir(parents=True, exist_ok=True)
    (out_raw / "idxs").mkdir(parents=True, exist_ok=True)
    (out_raw / "matrices").mkdir(parents=True, exist_ok=True)

    for data in tqdm(valid_dataset, desc="Filtering validation set", unit="sequence"):
        splitted_id = data.id.split("_")
        if len(splitted_id) < 3:
            click.echo(click.style(f"❗ Warning: Checking only PDB ID", fg="yellow"))
            pdb_id = splitted_id[0].upper()
            if pdb_id in train_ids:
                dropped += 1
                continue
        else:
            pdb_id = splitted_id[0].upper()
            model_id = splitted_id[1].upper()
            string_id = splitted_id[2].upper()
            if pdb_id in train_ids and (model_id, string_id) in train_ids[pdb_id]:
                dropped += 1
                continue

        kept += 1
        idx_file = Path(valid_ds_root) / "raw" / "idxs" / f"{data.id}.idx"
        cmt_file = Path(valid_ds_root) / "raw" / "matrices" / f"{data.id}.cmt"
        amt_file = Path(valid_ds_root) / "raw" / "matrices" / f"{data.id}.amt"
        if not idx_file.exists() or not cmt_file.exists() or not amt_file.exists():
            raise FileNotFoundError(f"Expected raw files for {data.id}")
        
        shutil.copy(idx_file, out_raw / "idxs" / f"{data.id}.idx")
        shutil.copy(cmt_file, out_raw / "matrices" / f"{data.id}.cmt")
        shutil.copy(amt_file, out_raw / "matrices" / f"{data.id}.amt")

    click.echo(
        click.style(
            f"✔ Finished - kept {kept} sequences, dropped {dropped} overlaps.", fg="green"
        )
    )

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--train-dir",
    "train_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
    help="Root directory of the training dataset",
)
@click.option(
    "--valid-dir",
    "valid_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
    help="Root directory of the validation dataset",
)
@click.option(
    "--output-dir",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    help="Destination for the filtered validation dataset",
)
def cli(train_dir: Path, valid_dir: Path, output_dir: Path) -> None:  # pragma: no cover
    train_ids = collect_train_pdb_ids(train_dir)
    copy_if_not_overlapping(valid_dir, output_dir, train_ids)


if __name__ == "__main__":
    cli()