import csv
import numpy as np

from pathlib import Path
from typing import Union, Tuple, Dict, Any


def read_idx_file(idx_filename: Union[Path, str]) -> Tuple[str, Dict[str, Any]]:
    """
    Reads an IDX file (CSV) containing residue information. Each row has at least two columns:
    (1) residue number
    (2) chain and residue identifier (e.g. "A.ARG12" or "ARG12").

    The function parses each row to extract:
    - res_number (int): The numeric residue index (e.g. 1, 2, 3, ...).
    - chain_id (Optional[str]): The chain identifier if present before a dot ('.'), otherwise None.
    - res_type (str): The single-letter residue code or first letter of the residue name.
    - res_id (int): The numeric part of the residue identifier after the residue type letter.
    """
    seq = []
    details = []

    with open(idx_filename, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if len(row) < 2:
                continue
            res_number = int(row[0].strip())
            chain_and_res = row[1].strip()

            if "." in chain_and_res:
                chain_id, res_full = chain_and_res.split(".", 1)
                res_type = res_full[0]
                res_id = int(res_full[1:])
            else:
                chain_id = None
                res_type = chain_and_res[0]
                res_id = int(chain_and_res[1:])

            seq.append(res_type)
            details.append(
                {
                    "res_number": res_number,
                    "chain_id": chain_id,
                    "res_type": res_type,
                    "res_id": res_id,
                }
            )

        seq = "".join(seq)
    return seq, details


def read_matrix_file(matrix_filename: Union[Path, str]) -> np.ndarray:
    """
    Reads a .cmt or .amt file that contains a matrix of integers and returns it as a NumPy array.

    Each row in the file is expected to contain integers separated by commas.
    Blank entries are ignored.
    """
    matrix = []
    with open(matrix_filename, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            row_int = [int(x.strip()) for x in row if x.strip() != ""]
            if len(row_int) > 0:
                matrix.append(row_int)
    return np.array(matrix)
