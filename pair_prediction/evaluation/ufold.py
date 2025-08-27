import os
import sys
import re
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from typing import List, Dict, Any

from pathlib import Path
from pair_prediction.model.utils import get_negative_edges
from pair_prediction.data.dataset import LinkPredictionDataset

HEADER_RE = re.compile(r"^>seq length:\s*(\d+)\s+seq name:\s+(\S+)")


def split_multi_ct(path: str, outdir: str = ".") -> None:
    """Parse *path*, write one .ct file per sequence into *outdir*."""
    
    os.makedirs(outdir, exist_ok=True)

    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        m = HEADER_RE.match(line)
        if not m:
            i += 1
            continue

        length = int(m.group(1))
        seq_name = m.group(2)

        block = [line + "\n"]
        i += 1
        for _ in range(length):
            if i >= len(lines):
                sys.exit(f"Error: reached EOF while reading sequence '{seq_name}'.")
            block.append(lines[i])
            i += 1

        out_path = os.path.join(outdir, f"{seq_name}.ct")
        with open(out_path, "w", encoding="utf-8") as out_fh:
            out_fh.writelines(block)

        print(f"Wrote {out_path}  ({length} nucleotides)")


def read_ct(path: Path) -> np.ndarray:
    """Return an N×N numpy.int8 matrix of base pairs in *path*."""
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n")
        m = HEADER_RE.match(header)
        if not m:
            sys.exit(f"{path}: first line is not a CT header")

        n = int(m.group(1))
        mat = np.zeros((n, n), dtype=np.int8)

        for k in range(n):
            fields = fh.readline().split()
            if len(fields) < 6:
                sys.exit(f"{path}: line {k+2} is malformed")

            i = int(fields[0])
            j = int(fields[4])

            if j != 0:
                mat[i-1, j-1] = 1
                mat[j-1, i-1] = 1

    return mat

def ufold_eval(
    dataset: LinkPredictionDataset,
    ct_source: str | Path,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Evaluate link-prediction performance when the “predictions” are the base pairs
    contained in CT files (e.g. computed by RNAstructure, ViennaRNA, IPknot, …).

    Returns a list of dicts with the *same* keys as `sincfold_eval`, so any
    downstream metric/plot code keeps working.
    """
    ct_dir = Path(ct_source)
    tmp_dir: Path | None = None        
    try:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        outputs = []

        for data in tqdm(dataloader, desc="Evaluating CT predictions"):
            try:
                seq_id: str = data.id[0]
                seq:    str = data.seq[0]

                ct_file = ct_dir / f"{seq_id}.ct"
                if not ct_file.exists():
                    # mock CT mat for sequences without a CT file
                    ct_mat = torch.zeros((len(seq), len(seq)), dtype=np.int8)
                else:
                    ct_mat = torch.from_numpy(read_ct(ct_file)).float()

                edge_types      = np.concatenate(data.edge_type)
                edge_index      = data.edge_index
                noncanon_mask   = edge_types == "non-canonical"

                pos_edge_index  = edge_index[:, noncanon_mask]
                pos_labels      = torch.ones(pos_edge_index.size(1))

                neg_edge_index  = get_negative_edges(data, validation=True)
                neg_labels      = torch.zeros(neg_edge_index.size(1))

                all_edge_index  = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                all_labels      = torch.cat([pos_labels,    neg_labels],    dim=0)

                rows = all_edge_index[0].to(dtype=torch.long).cpu()
                cols = all_edge_index[1].to(dtype=torch.long).cpu()
                probabilities = ct_mat[rows, cols]
                predictions   = (probabilities > 0.5)

                outputs.append({
                    "data":          data,
                    "preds":         predictions,
                    "labels":        all_labels,
                    "probabilities": probabilities,
                    "edge_types":    edge_types,
                    "pair_types":    data.pair_type,
                })
            except Exception as e:
                print(f"[ERROR] An error occurred during evaluation of sequence {seq_id}, skipping: {e}")
                continue
    finally:
        if tmp_dir is not None and tmp_dir.exists():
            rmtree(tmp_dir)

    return outputs