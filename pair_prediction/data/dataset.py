import os
import os.path as osp
from tqdm import tqdm

from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx

from pair_prediction.data.processing import create_rna_graph
from pair_prediction.data.read import read_idx_file, read_matrix_file

class LinkPredictionDataset(InMemoryDataset):
    def __init__(self, root, validation=False, transform=None, pre_transform=None, pre_filter=None):
        """
        Dataset for RNA graph link prediction as an InMemoryDataset.
        
        Args:
            root (str): Root directory where the dataset should be saved. This directory should have
                        two subdirectories: `raw` (with your .idx and .amt files) and `processed` (to be generated).
            validation (bool): If True, uses a smaller subset of the data.
            transform, pre_transform, pre_filter: Standard PyG arguments.
        """
        self.validation = validation
        self.prefix = "val" if validation else "train"
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        raw_dir = osp.join(self.raw_dir, "idxs")
        raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.idx')]
        return sorted(raw_files)

    @property
    def processed_file_names(self):
        return [f'{self.prefix}_data.pt']

    def process(self):
        data_list = []
        raw_idx_files = self.raw_file_names

        if self.validation:
            raw_idx_files = raw_idx_files[: len(raw_idx_files) // 10]
        else:
            raw_idx_files = raw_idx_files[len(raw_idx_files) // 10:]
            
        for idx_file_name in tqdm(raw_idx_files, desc=f"Processing {self.prefix} data"):
            file_stem, _ = osp.splitext(idx_file_name)
            idx_file_path = osp.join(self.raw_dir, "idxs", f"{file_stem}.idx")
            amt_file_path = osp.join(self.raw_dir, "matrices", f"{file_stem}.amt")

            seq, details = read_idx_file(idx_file_path)
            amt_matrix = read_matrix_file(amt_file_path)
            graph = create_rna_graph(seq, amt_matrix)
            data = from_networkx(graph)
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            
        self.save(data_list, self.processed_paths[0])

