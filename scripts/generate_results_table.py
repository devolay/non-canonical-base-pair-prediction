import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from collections import defaultdict, Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import click

from pair_prediction.constants import OUTPUT_DIR


def load_results(model_name: str, dataset_name: str, results_dir: Path) -> List[Dict]:
    """Load results from a pickle file."""
    pkl_path = results_dir / f"{model_name}_{dataset_name}" / "results.pkl"
    
    if not pkl_path.exists():
        raise FileNotFoundError(f"Results file not found: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate aggregated metrics from results."""
    all_labels = []
    all_preds = []
    all_probs = []
    
    for result in results:
        labels = result['labels'].numpy() if hasattr(result['labels'], 'numpy') else result['labels']
        probs = result['probabilities'].numpy() if hasattr(result['probabilities'], 'numpy') else result['probabilities']
        
        all_labels.extend(labels)
        all_probs.extend(probs)
        all_preds.extend((probs > 0.5).astype(int))
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate metrics
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    inf = np.sqrt(precision * recall) if precision > 0 and recall > 0 else 0
    
    return {
        'F1': round(f1, 2),
        'INF': round(inf, 2),
        'PPV': round(precision, 2),  # Positive Predictive Value
        'TPR': round(recall, 2),      # True Positive Rate
    }


def calculate_per_sequence_metrics(results: List[Dict]) -> pd.DataFrame:
    """Calculate metrics for each sequence individually."""
    rows = []
    
    for result in results:
        # Extract sequence ID from data object
        data = result['data']
        seq_id = data.id[0] if hasattr(data.id, '__getitem__') else data.id
        
        # Get labels and probabilities
        labels = result['labels'].numpy() if hasattr(result['labels'], 'numpy') else result['labels']
        probs = result['probabilities'].numpy() if hasattr(result['probabilities'], 'numpy') else result['probabilities']
        
        y_true = labels.astype(bool) if hasattr(labels, 'dtype') and labels.dtype != bool else labels
        y_pred = (probs > 0.5).astype(bool)
        
        # Calculate metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        inf = np.sqrt(precision * recall) if precision > 0 and recall > 0 else 0
        
        rows.append({
            'sequence_id': seq_id,
            'F1': round(f1, 2),
            'PPV': round(precision, 2),
            'TPR': round(recall, 2),
            'INF': round(inf, 2),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
        })
    
    return pd.DataFrame(rows)


def analyze_pairing_performance(results: List[Dict]) -> pd.DataFrame:
    """
    Analyze metrics by nucleotide pair type.
    Returns a DataFrame with pair types as rows and metrics (tp, fp, fn, precision, recall, f1) as columns.
    """
    stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_ones': 0})
    
    for result in results:
        # Extract from unified format
        data = result['data']
        labels = result['labels'].numpy() if hasattr(result['labels'], 'numpy') else result['labels']
        probs = result['probabilities'].numpy() if hasattr(result['probabilities'], 'numpy') else result['probabilities']
        
        labels = np.array(labels).astype(bool)
        preds = (np.array(probs) > 0.5).astype(bool)
        
        # Get sequence
        seq = data.seq[0] if hasattr(data.seq, '__getitem__') else str(data.seq)
        
        # Get edge index and ensure (E, 2) format
        edge_index = data.edge_index.numpy() if hasattr(data.edge_index, 'numpy') else data.edge_index
        if edge_index.shape[0] == 2:
            edge_index = edge_index.T
        
        # Get edge types
        edge_types = data.edge_type if hasattr(data, 'edge_type') else []
        
        # Process each edge
        for i, (idx_a, idx_b) in enumerate(edge_index):
            # Skip phosphodiester bonds (backbone edges)
            if i < len(edge_types) and edge_types[i] == 'phosphodiester':
                continue
            
            idx_a = int(idx_a)
            idx_b = int(idx_b)
            
            # Check bounds
            if idx_a >= len(seq) or idx_b >= len(seq) or i >= len(labels):
                continue
            
            # Get nucleotides and create canonical pair notation (sorted)
            nuc_a = seq[idx_a]
            nuc_b = seq[idx_b]
            pair = "".join(sorted([nuc_a, nuc_b]))
            
            y_true = labels[i]
            y_pred = preds[i]
            
            # Track metrics (divide by 2 since edges are undirected)
            if y_true:
                stats[pair]['gt_ones'] += 0.5
                if y_pred:
                    stats[pair]['tp'] += 0.5
                else:
                    stats[pair]['fn'] += 0.5
            else:
                if y_pred:
                    stats[pair]['fp'] += 0.5
    
    if not stats:
        return pd.DataFrame()
    
    result_df = pd.DataFrame.from_dict(stats, orient='index').rename_axis('pair_type').reset_index()
    
    # Rename gt_ones to GT Count and convert to int
    result_df['GT Count'] = result_df['gt_ones'].astype(int)
    
    # Convert TP, FP, FN to integers
    result_df['TP'] = result_df['tp'].astype(int)
    result_df['FP'] = result_df['fp'].astype(int)
    result_df['FN'] = result_df['fn'].astype(int)
    
    return result_df.sort_values('gt_ones', ascending=False)


def combine_pairing_results(all_results: Dict, models: List[str], dataset: str) -> pd.DataFrame:
    """Combine pairing analysis results from all models into one table with shared GT Count."""
    all_dfs = []
    gt_count_df = None
    
    for model in models:
        if model not in all_results or dataset not in all_results[model]:
            continue
        
        results = all_results[model][dataset]
        df_pairs = analyze_pairing_performance(results)
        
        if df_pairs.empty:
            continue
        
        # Extract GT Count from first model only
        if gt_count_df is None:
            gt_count_df = df_pairs[['pair_type', 'GT Count']].set_index('pair_type')
        
        # Keep only TP, FP, FN for this model
        metrics_df = df_pairs[['pair_type', 'TP', 'FP', 'FN']].set_index('pair_type')
        
        # Create multi-index columns: (method, metric)
        method_name = f"{model}_{dataset}"
        metrics_df.columns = pd.MultiIndex.from_product([[method_name], metrics_df.columns])
        
        all_dfs.append(metrics_df)
    
    # Concatenate all dataframes horizontally
    if all_dfs and gt_count_df is not None:
        combined_df = pd.concat([gt_count_df] + all_dfs, axis=1)
        # Sort by GT Count descending (which was used for sorting originally)
        combined_df = combined_df.sort_values('GT Count', ascending=False)
        return combined_df
    else:
        return pd.DataFrame()


def combine_per_sequence_results(all_results: Dict, models: List[str], dataset: str) -> pd.DataFrame:
    """Combine per-sequence results from all models/datasets into one multi-indexed table."""
    all_dfs = []
    
    for model in models:
        if model not in all_results or dataset not in all_results[model]:
            continue
        
        results = all_results[model][dataset]
        df_seq = calculate_per_sequence_metrics(results)
        
        if df_seq.empty:
            continue
        
        # Keep only metrics columns and set sequence_id as index
        metrics_df = df_seq[['sequence_id', 'F1', 'INF', 'PPV', 'TPR']].set_index('sequence_id')
        
        # Create multi-index columns: (method, metric)
        method_name = f"{model}_{dataset}"
        metrics_df.columns = pd.MultiIndex.from_product([[method_name], metrics_df.columns])
        
        all_dfs.append(metrics_df)
    
    # Concatenate all dataframes horizontally
    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=1)
        # Reorder columns: first level is method, second level is metric
        combined_df = combined_df.sort_index(axis=1, level=[0, 1])
        return combined_df
    else:
        return pd.DataFrame()


@click.command()
@click.option('--models', callback=lambda ctx, param, value: [v.strip() for v in value.split(',')],
              required=True, help='Comma-separated list of model names (e.g., rinalmo,sincfold,spotrna,ufold)')
@click.option('--dataset', type=click.Choice(['casp', 'large', 'rnapuzzles', 'smallmed', 'large']), required=True, help='Comma-separated list of dataset names (e.g., casp,large,rnapuzzles)')
@click.option('--output-dir', type=click.Path(), default=None, help='Directory containing results (default: outputs/)')
def main(models: List[str], dataset: str, output_dir: str):
    """Generate results table from model outputs."""
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    else:         
        output_dir = Path(output_dir)

    latex_per_seq_output = OUTPUT_DIR / f"{dataset}_per_sequence_table.tex"
    latex_pairing_output = OUTPUT_DIR / f"{dataset}_pairing_table.tex"
    latex_output = OUTPUT_DIR / f"{dataset}_summary_table.tex"

    print(f"Loading results from {output_dir}...")
    
    all_metrics = {}
    all_results = {}  # Store raw results for per-sequence metrics
    
    for model in models:
        all_metrics[model] = {}
        all_results[model] = {}
        try:
            results = load_results(model, dataset, output_dir)
            metrics = calculate_metrics(results)
            all_metrics[model][dataset] = metrics
            all_results[model][dataset] = results
            print(f"✓ {model:15} {dataset:15} - F1: {metrics['F1']:.2f}, INF: {metrics['INF']:.2f}, PPV: {metrics['PPV']:.2f}, TPR: {metrics['TPR']:.2f}")
        except FileNotFoundError as e:
            print(f"✗ {model:15} {dataset:15} - Not found")
            continue
    
    # Per-sequence results

    df_seq = combine_per_sequence_results(all_results, models, dataset)
    if not df_seq.empty:
        with open(latex_per_seq_output, 'w') as f:
            f.write(df_seq.to_latex(float_format=lambda x: f'{x:.2f}' if not np.isnan(x) else '-'))
        print(f"Per-sequence LaTeX table saved to {latex_per_seq_output}")
    
    # Pairing type results
    df_pairing = combine_pairing_results(all_results, models, dataset)
    if not df_pairing.empty:
        with open(latex_pairing_output, 'w') as f:
            # Format integers as integers (no decimals), keep GT Count as int
            def format_value(x):
                if pd.isna(x):
                    return '-'
                elif isinstance(x, (int, np.integer)):
                    return str(int(x))
                else:
                    return str(int(x))
            
            f.write(df_pairing.to_latex(float_format=format_value))
        print(f"Pairing type LaTeX table saved to {latex_pairing_output}")
    else:
        print(f"⚠ No pairing type data found for {dataset}")
    
    # Aggregated results
    index_tuples = []
    row_data = []
    
    for model in sorted(all_metrics.keys()):
        for metric in ['F1', 'INF', 'PPV', 'TPR']:
            index_tuples.append((model, metric))
            row_values = []
            for dataset in sorted(all_metrics[model].keys()):
                row_values.append(all_metrics[model][dataset].get(metric, np.nan))
            row_data.append(row_values)
    
    df = pd.DataFrame(row_data, columns=sorted(set(all_metrics[models[0]].keys())))
    df.index = pd.MultiIndex.from_tuples(index_tuples, names=['Method', 'Metric'])
    latex_table = df.to_latex(float_format=lambda x: f'{x:.2f}' if not np.isnan(x) else '-')
    
    if latex_output:
        with open(latex_output, 'w') as f:
            f.write(latex_table)
        print(f"Aggregated LaTeX table saved to {latex_output}")



if __name__ == "__main__":
    main()
