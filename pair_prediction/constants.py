from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

FAMILY_MAPPING_FILE = "rfam_mapping.csv"
BENCHMARK_IDXS = "benchmark_idxs.csv"
RFAM_FAMILIES_FILE = "Rfam.pdb"
TRAIN_FAMILIES = ["5S_rRNA", "tRNA"]