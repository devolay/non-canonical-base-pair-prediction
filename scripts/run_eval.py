import torch
import click
from pathlib import Path
from typing import List

from pair_prediction.evaluation import EVAL_FUNCTIONS, collect_and_save_metrics
from pair_prediction.evaluation.ufold import split_multi_ct
from pair_prediction.model.rinalmo_link_predictor import RiNAlmoLinkPredictionModel
from pair_prediction.data.utils import load_dataset
from pair_prediction.constants import BASE_DIR

def validate_multi_input(ctx, param, value):
    if not value:
        return []
    inputs = [v.strip() for v in value.split(',')]
    return inputs

@click.command()
@click.option('--models', callback=validate_multi_input, required=True, help='List of model names to evaluate (e.g., rinalmo, sincfold)')
@click.option('--datasets', callback=validate_multi_input, required=True, help='List of dataset names to evaluate (e.g., rinalmo, sincfold)')
@click.option('--output_dir', type=click.Path(), default=(BASE_DIR / "outputs"), required=True, help='Directory to save evaluation results')
@click.option('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
def main(models: List[str], datasets: List[str], output_dir: Path, device: str):
    device = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pbar = click.progressbar(length=len(models) * len(datasets), label='Evaluating models', show_eta=True, show_percent=True)
    for model_name in models:
        for dataset_name in datasets:
            dataset = load_dataset(dataset_name)
            print(f"Evaluating model {model_name} on dataset {dataset_name}...")
            eval_fn = EVAL_FUNCTIONS.get(model_name)
            if eval_fn is None:
                print(f"No evaluation function for model {model_name}. Skipping.")
                continue

            match model_name:
                case 'rinalmo':
                    model = RiNAlmoLinkPredictionModel(
                        in_channels=1280,
                        gnn_channels=[1280, 128, 128],
                        cnn_head_embed_dim=64,
                        cnn_head_num_blocks=3
                    )
                    
                    checkpoint = torch.load(BASE_DIR / "models" / "model.ckpt", map_location=device)
                    checkpoint['state_dict'] = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
                    model.load_state_dict(checkpoint['state_dict'])
                    outputs = eval_fn(
                        model=model,
                        dataset=dataset,
                        device=device,
                    )
                case 'sincfold' | 'spotrna':
                    outputs = eval_fn(
                        dataset=dataset,
                        device=device,
                    )
                case 'ufold':
                    results_file = output_dir / f"{model_name}_{dataset_name}" / f"{model_name}_{dataset_name}.ct"
                    if not results_file.exists():
                        raise FileNotFoundError(f"Expected results file {results_file} not found.")
                    pred_dir = output_dir / f"{model_name}_{dataset_name}" / "preds"
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    split_multi_ct(results_file, outdir=pred_dir)
                    outputs = eval_fn(
                        dataset=dataset,
                        device=device,
                        ct_source=pred_dir,
                    )

            collect_and_save_metrics(outputs, output_dir / f"{model_name}_{dataset_name}")
            print(f"Saved results for {model_name} on {dataset_name}.")
            pbar.update(1)

if __name__ == "__main__":
    main()