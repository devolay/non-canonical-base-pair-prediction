import subprocess
import sys
import multiprocessing as mp
import torch

from pathlib import Path
from typing import Any, List, Dict

from pair_prediction.constants import BASE_DIR
from pair_prediction.data.dataset import LinkPredictionDataset
from pair_prediction.model.utils import get_negative_edges
from pair_prediction.evaluation.utils import rmtree


UFOLD_DIR = BASE_DIR / "external" / "UFold"

def run_ufold_prediction():
    """Run the UFold prediction script as a subprocess."""
    script_path = UFOLD_DIR / "ufold_predict.py"
    cmd = [sys.executable, str(script_path), "--nc", "True"]
    print(f"Running: {' '.join(cmd)}")

    process = subprocess.run(cmd, cwd=UFOLD_DIR, capture_output=True, text=True)
    if process.returncode != 0:
        print("❌ Error running UFold prediction:")
        print(process.stderr)
    else:
        print("✅ UFold prediction completed.")
        print(process.stdout)


def eval_ufold(
    dataset: LinkPredictionDataset, 
    device: torch.device,
    negative_sample_ratio: int, 
    **kwargs
) -> List[Dict[str, Any]]:
    mp.set_start_method('spawn', force=True)
    
    try:
        batch_fasta_path = UFOLD_DIR / "data" / "input.txt"
        with open(batch_fasta_path, "w") as batch_fasta:
            for data in dataset:
                seq_id = data.id
                seq = data.seq
                batch_fasta.write(f">{seq_id}\n{seq}\n")

        run_ufold_prediction()
    finally:
        # batch_fasta_path.unlink(missing_ok=True)
        pass
