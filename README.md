# Non-Canonical Base Pair Prediction

This repository contains code for the Master Thesis's project adressing prediction of non-canonical base pairs in RNA structures using deep learning models.

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Model Weights](#model-weights)
- [Usage](#usage)
- [Reproducing Experiments](#reproducing-experiments)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Overview

This project implements a link prediction framework for RNA graphs, specifically designed to predict non-canonical base pairs. The main components include:

- **RiNALMo Encoder**: Uses RiNALMo embeddings for RNA sequence representation
- **Graph Neural Networks**: Implements GNN-based link prediction models
- **Comprehensive Benchmarking**: Performance evaluation across different models and datasets

## Environment Setup

### Prerequisites
- Python 3.10+
- CUDA>=11.8
- Git with submodule support

### Base Environment

1. **Clone the repository with submodules:**
   ```bash
   git clone --recursive https://github.com/your-username/non-canonical-base-pair-prediction.git
   cd non-canonical-base-pair-prediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install base dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install RiNAlmo:**
   ```bash
   cd external/RiNALMo
   pip install .
   pip install flash-attn==2.3.2 # this step may take some time
   cd ../..
   ```

### Additional Dependencies

Some evaluation tools may require additional setup:

- **SPOT-RNA**: Located in `external/SPOT-RNA/`
- **Ufold**: May require additional installation steps
- **SincFold**: May require additional installation steps

## Model Weights

### Downloading Pre-trained Models

Pre-trained model weights are available on [Google Drive](https://drive.google.com/drive/folders/1iFTqdLND9V-ic00FWZcnqmzm2TMax__F?usp=sharing)

To use the trained model download the model weights from `Models/model.ckpt` on GDrive and put it under the `models/` directory in the repository.

## Data Setup

### Downloading Datasets

The project uses datasets stored on [Google Drive](https://drive.google.com/drive/folders/1iFTqdLND9V-ic00FWZcnqmzm2TMax__F?usp=sharing). To download them go to the `Datasets/` directory in the link provided. There are 2 directories

`train_and_validation` - directory with main dataset. It also contains the `.csv` file with data split for clarity. Nevertheless the split can be reproduced directly using the code in repository as well.

`benchmark` - directory with all benchmark datasets on whcih the results were reported in the thesis. Suffix `_clean` means that these dataset versions are filtered to do not overlap with the training data.

### Dataset Repo Structure

The expected data structure where you should locate the downloaded datasets.
```
data/
├── evaluation/
│   ├──ts1_clean
│   │   └──raw/
│   │       ├── idxs/
│   │       └── matrices/
│   └── ....
├── raw/
│   ├── idxs/
│   ├── matrices/
│   └── rfam_mapping.csv
└── external/
```

### Using Pre-trained Models

```python
import torch
from pair_prediction.model.rinalmo_link_predictor import RiNAlmoLinkPredictionModel

# Load model
model = RiNAlmoLinkPredictionModel(
    in_channels=1280,
    gnn_channels=[1280, 512, 256],
    cnn_head_embed_dim=64,
    cnn_head_num_blocks=3
)

# Load weights
checkpoint = torch.load('models/model.ckpt', map_location='cuda:0')
checkpoint['state_dict'] = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(checkpoint['state_dict'])
```

## Usage

### Training

1. **Default training:**
   ```bash
   python scripts/train.py --config configs/train-config.yaml
   ```

2. **Custom training configuration:**
   ```bash
   python scripts/train.py --config path/to/your/config.yaml
   ```

### Evaluation

1. **Run evaluation on multiple models and datasets:**
   ```bash
   python scripts/run_eval.py \
     --models rinalmo,sincfold,spotrna \
     --datasets ts1,ts2,rnapuzzles \
     --model-path models/model.ckpt \
     --output_dir outputs 
   ```

2. **Benchmark performance:**
   ```bash
   python scripts/benchmark.py \
     --config configs/train-config.yaml \
     --batch-sizes 2 8 32 64 128 \
     --precisions fp16 bf16 \
     --out bench_results.csv
   ```

### Case Studies

```bash
python scripts/case_study_script.py
```

## Project Structure

```
├── configs/                 # Configuration files
│   └── train-config.yaml
├── data/                   # Data directory
│   ├── processed/          # Processed datasets
│   └── raw/               # Raw data files
├── external/              # External tools
│   ├── RiNALMo/          # RiNALMo submodule
│   └── SPOT-RNA/         # SPOT-RNA tool
├── models/                # Model checkpoints
├── notebooks/             # Jupyter notebooks
├── pair_prediction/       # Main package
│   ├── data/             # Data processing
│   ├── evaluation/       # Evaluation tools
│   ├── model/            # Model implementations
│   └── visualization/    # Visualization tools
├── scripts/              # Training and evaluation scripts
├── requirements.txt      # Python dependencies
└── pyproject.toml       # Project configuration
```

## Configuration

The main configuration file `configs/train-config.yaml` contains:

- **Training parameters**: batch size, learning rate, epochs
- **Model architecture**: GNN channels, embedding dimensions
- **Negative sampling**: ratios and strategies
- **Optimization**: gradient clipping, schedulers

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- RiNALMo team for the RNA language model
- SPOT-RNA, SincFold, and Ufold developers
- PyTorch Geometric community

