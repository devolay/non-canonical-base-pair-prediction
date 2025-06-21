import torch
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from typing import Any, List, Dict

from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader

from sincfold import pred as sincfold_predict
from sincfold.utils import ct2dot

from pair_prediction.data.dataset import LinkPredictionDataset
from pair_prediction.model.utils import get_negative_edges
from pair_prediction.evaluation.utils import rmtree
from pair_prediction.constants import BASE_DIR


def sincfold_eval(
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

        with open(batch_fasta_path, "w") as batch_fasta:
            for data in dataset:
                seq_id = data.id
                seq = data.seq
                batch_fasta.write(f">{seq_id}\n{seq}\n")

        sincfold_predict(
            pred_input=str(batch_fasta_path),
            out_path=str(preds_path),
            logits=True,
            config={
                "device": device,
                "batch_size": kwargs.get("batch_size", 1024),
                "max_len": kwargs.get("max_len", 4096),
            },
            nworkers=kwargs.get("nworkers", 2),
            verbose=kwargs.get("verbose", False),
        )
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        outputs = []
        for data in tqdm(dataloader, desc="Evaluating predictions"):
            seq_id = data.id[0]
            seq = data.seq[0] 
            
            edge_types = np.concatenate(data.edge_type)
            edge_index = data.edge_index
            non_canonical_mask = edge_types == 'non-canonical'

            pred_logits_file = preds_path / "logits" / f"{seq_id}.pk"
            if not pred_logits_file.exists():
                print(f"Prediction for {seq_id} not found. Skipping.")
                continue
            
            with pred_logits_file.open("rb") as f:
                pred_logits = pickle.load(f)[0]

            pos_edge_index = edge_index[:, non_canonical_mask]
            pos_labels = torch.ones(pos_edge_index.size(1), dtype=torch.float32)

            neg_edge_index = get_negative_edges(data, validation=True)
            neg_labels = torch.zeros(neg_edge_index.size(1), dtype=torch.float32)

            all_edge_index = torch.concatenate([pos_edge_index, neg_edge_index], axis=1)
            all_labels = torch.concatenate([pos_labels, neg_labels], axis=0)

            all_edge_index = to_dense_adj(all_edge_index, batch=data.batch).squeeze(0)
            all_logits = pred_logits[all_edge_index.bool()]
            probabilities = torch.sigmoid(all_logits)
            predictions = (probabilities > 0.5)

            outputs.append({
                "id": seq_id,
                "seq": seq,
                "preds": predictions,
                "labels": all_labels,
                "probabilities": probabilities,
                "edge_types": edge_types,
                "pair_types": data.pair_type,
            })
    finally:
        rmtree(tmp_path)

    return outputs
        
