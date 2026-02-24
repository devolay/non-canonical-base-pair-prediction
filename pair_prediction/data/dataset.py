import os
import os.path as osp
import pandas as pd
import logging

from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx

from pair_prediction.data.processing import create_rna_graph
from pair_prediction.data.read import read_idx_file, read_matrix_file
from pair_prediction.constants import FAMILY_MAPPING_FILE, TRAIN_FAMILIES, BENCHMARK_IDXS, BASE_DIR, RFAM_FAMILIES_FILE
from pair_prediction.data.utils import get_rfam_family

logger = logging.getLogger(__name__)


class LinkPredictionDataset(InMemoryDataset):
    def __init__(
        self, root, transform=None, pre_transform=None, pre_filter=None, mode: str = None
    ):
        """
        Dataset for RNA graph link prediction as an InMemoryDataset.

        Args:
            root (str): Root directory where the dataset should be saved. This directory should have
                        two subdirectories: `raw` (with your .idx and .amt files) and `processed` (to be generated).
            validation (bool): If True, uses a smaller subset of the data.
            transform, pre_transform, pre_filter: Standard PyG arguments.
        """
        self.mode = mode
        match mode:
            case "train":
                self.prefix = "train"
            case "validation":
                self.prefix = "val"
            case _:
                self.prefix = 'all'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        match self.mode:
            case "train":
                benchmark_idxs = set(pd.read_csv(osp.join(self.raw_dir, BENCHMARK_IDXS))['idx'].tolist())
                family_mapping = pd.read_csv(osp.join(self.raw_dir, FAMILY_MAPPING_FILE))
                family_mapping = family_mapping[
                    family_mapping['rfam_name'].isin(TRAIN_FAMILIES) & 
                    ~family_mapping['pdb_id'].isin(benchmark_idxs)
                ]
                return sorted(family_mapping['id'].values.tolist())
            case "validation":
                benchmark_idxs = set(pd.read_csv(osp.join(self.raw_dir, BENCHMARK_IDXS))['idx'].tolist())
                family_mapping = pd.read_csv(osp.join(self.raw_dir, FAMILY_MAPPING_FILE))
                family_mapping = family_mapping[
                    ~family_mapping['rfam_name'].isin(TRAIN_FAMILIES) & 
                    ~family_mapping['pdb_id'].isin(benchmark_idxs)
                ]
                return sorted(family_mapping['id'].values.tolist())
            case _:
                rfam_mapping = pd.read_csv(osp.join(BASE_DIR, "data", "raw", RFAM_FAMILIES_FILE), sep='\t', header=None)
                raw_dir = osp.join(self.raw_dir, "idxs")
                raw_files = [f for f in os.listdir(raw_dir) if f.endswith(".idx")]
                non_training_faimilies = [f for f in raw_files if get_rfam_family(rfam_mapping, f[:4]) not in TRAIN_FAMILIES]

                if len(non_training_faimilies) < len(raw_files):
                    print(f"Filtered out {len(raw_files) - len(non_training_faimilies)} benchmark sequences belonging to training families.")

                return sorted(non_training_faimilies)        

    @property
    def processed_file_names(self):
        return [f"{self.prefix}_data.pt"]

    def process(self):
        data_list = []
        raw_idx_files = self.raw_file_names
        family_mapping_file = pd.read_csv(osp.join(BASE_DIR, "data", "raw", RFAM_FAMILIES_FILE), sep='\t', header=None)

        for idx_file_name in tqdm(raw_idx_files, desc=f"Processing {self.prefix} data"):
            file_stem, _ = osp.splitext(idx_file_name)
            idx_file_path = osp.join(self.raw_dir, "idxs", f"{file_stem}.idx")
            amt_file_path = osp.join(self.raw_dir, "matrices", f"{file_stem}.amt")

            seq, details = read_idx_file(idx_file_path)
            amt_matrix = read_matrix_file(amt_file_path)

            try:
                graph = create_rna_graph(details, amt_matrix)
            except ValueError as e:
                raise ValueError(
                    f"Error creating graph for {file_stem}: {e}"
                ) from e
            
            data = from_networkx(graph)
            data.seq = seq
            data.id = file_stem

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
