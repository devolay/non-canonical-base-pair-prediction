import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from pair_prediction.data.dataset import LinkPredictionDataset


def export_dataset_to_fasta(dataset: LinkPredictionDataset,  output_dir: str, batchsize: int = 1):
    """
    Iterate over the dataset and save each batch into <output_dir> 
    with a `.fasat` extension. 
    """
    os.makedirs(output_dir, exist_ok=True)
    if batchsize:
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            filename = os.path.join(output_dir, f"batch_{batch_idx}.fasta")
            ids = batch.id
            seqs = batch.seq

            with open(filename, 'w') as f:
                for seq_id, seq in zip(ids, seqs):
                    f.write(f">{seq_id}\n{seq}\n")
    else:
        filename = os.path.join(output_dir, f"batch.fasta")
        with open(filename, 'w') as f:
            for data in dataset:
                seq = data.seq
                id = data.id
                f.write(f">{id}\n{seq}\n")
                    

def load_fasta_sequences(fasta_file: str):
    """
    Load sequences from a FASTA file.
    """
    sequences = {}
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                seq_id = line.strip()[1:]
                sequences[seq_id] = next(f).strip()
    return sequences

def save_fasta_sequences(sequences: dict, output_file: str):
    """
    Save sequences to a FASTA file.
    """
    with open(output_file, 'w') as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n{seq}\n")


def load_dataset(dataset_name: str, root: str = 'data') -> LinkPredictionDataset:
    """
    Load a dataset by name.
    """
    if dataset_name == "validation":
        dataset = LinkPredictionDataset(root=root, mode="validation")
    else:
        dataset = LinkPredictionDataset(root=f"{root}/evaluation/{dataset_name}_clean")
    return dataset