import requests
import pandas as pd


RFAM_API_URL = URL = "https://rfam.org/family/{rfam_id}?content-type=application/json"                  

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


def get_rfam_id(mapping_file: pd.DataFrame, pdb_id: str) -> str:
    """
    Get the Rfam ID for a given PDB ID.
    """
    pdb_id = pdb_id.lower().strip()
    if pdb_id not in mapping_file[1].values:
        raise ValueError(f"PDB ID {pdb_id} not found in mapping.")
    
    rfam_id = mapping_file.loc[mapping_file[1] == pdb_id, 0].values[0]
    return rfam_id


def get_rfam_family(mapping_file: pd.DataFrame, pdb_id: str) -> str:
    """
    Get the Rfam family for a given PDB ID.
    """

    try:
        rfam_id = get_rfam_id(mapping_file, pdb_id)
    except ValueError:
        return None
    
    response = requests.get(RFAM_API_URL.format(rfam_id=rfam_id))
    response.raise_for_status()
    data = response.json()
    return data['rfam']['id']