import os
import torch
import numpy as np
import subprocess
from tqdm import tqdm
from typing import List, Dict, Any

from torch_geometric.loader import DataLoader

from pair_prediction.constants import BASE_DIR
from pair_prediction.data.dataset import LinkPredictionDataset
from pair_prediction.evaluation.utils import rmtree
from pair_prediction.model.utils import get_negative_edges

SPOT_RNA_DIR = BASE_DIR / "external" / "SPOT-RNA"
SPOTRNA_THRESHOLD = 0.335

def run_spotrna_on_fasta(fasta_path, output_dir, device):
    env = os.environ.copy()
    cmd = [
        "python3", "SPOT-RNA.py",
        "--inputs", str(fasta_path),
        "--outputs", str(output_dir),
        "--gpu", "0" if device.type == 'cuda' else '-1',
    ]
    process = subprocess.run(cmd, cwd=SPOT_RNA_DIR, check=True, env=env)
    if process.returncode != 0:
        print("❌ Error running SPOT-RNA prediction:")
        print(process.stderr)
    else:
        print("✅ SPOT-RNA prediction completed.")
        print(process.stdout)


def spotrna_eval(
    dataset: LinkPredictionDataset,
    device: torch.device,
    **kwargs
) -> List[Dict[str, Any]]:
    try:
        tmp_path = BASE_DIR / "tmp"
        temp_fasta_dir = tmp_path / "temp_fastas"
        batch_fasta_path = temp_fasta_dir / "all_sequences.fasta"
        preds_path = tmp_path / "preds"

        temp_fasta_dir.mkdir(exist_ok=True, parents=True)
        preds_path.mkdir(exist_ok=True, parents=True)

        with open(batch_fasta_path, "w") as batch_fasta:
            for data in dataset:
                seq_id = data.id
                seq = data.seq
                batch_fasta.write(f">{seq_id}\n{seq}\n")

        run_spotrna_on_fasta(
            fasta_path=batch_fasta_path,
            output_dir=preds_path,
            device=device, 
        )

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        outputs = []
        for data in tqdm(dataloader, desc="Evaluating predictions"):
            seq_id = data.id[0]

            edge_types = np.concatenate(data.edge_type)
            edge_index = data.edge_index
            non_canonical_mask = edge_types == 'non-canonical'

            bpseq_file = preds_path / f"{seq_id}.prob"
            if not bpseq_file.exists():
                print(f"File {bpseq_file} does not exist. Skipping.")
                continue

            with open(bpseq_file, "r") as file:
                pred = np.loadtxt(file, dtype=np.float32, delimiter="\t")
                pred = pred + pred.T

            pos_edge_index = edge_index[:, non_canonical_mask]
            pos_labels = torch.ones(pos_edge_index.size(1), dtype=torch.float32)

            neg_edge_index = get_negative_edges(data, validation=True)
            neg_labels = torch.zeros(neg_edge_index.size(1), dtype=torch.float32)

            all_edge_index = torch.concatenate([pos_edge_index, neg_edge_index], axis=1)
            all_labels = torch.concatenate([pos_labels, neg_labels], axis=0)

            rows = all_edge_index[0].cpu().numpy()
            cols = all_edge_index[1].cpu().numpy()
            all_scores = pred[rows, cols]
            all_preds = (all_scores >= SPOTRNA_THRESHOLD)  

            outputs.append({
                "id": seq_id,
                "seq": data.seq[0],
                "preds": torch.from_numpy(all_preds.astype(np.float32)),
                "labels": all_labels,
                "probabilities": torch.from_numpy(all_scores.astype(np.float32)),
                "edge_types": edge_types,
                "pair_types": data.pair_type,
            })
    finally:
        rmtree(tmp_path)

    return outputs