import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import click

from pair_prediction.evaluation.utils import load_dataset
from pair_prediction.constants import BASE_DIR, OUTPUT_DIR


def calculate_dataset_statistics(dataset) -> pd.DataFrame:
    """
    Calculate statistics for each sequence in the dataset.
    
    Returns a DataFrame with the following columns:
    - Seq ID: Sequence identifier
    - ID: Sequence identifier (from data)
    - Length [nt]: Sequence length
    - Unpaired nts: Number of unpaired nucleotides
    - Paired nts: Number of paired nucleotides
    - Nodes: Number of nodes in the graph
    - Canonical edges: Number of canonical base pairs
    - Non-canonical edges: Number of non-canonical base pairs
    - Backbone edges: Number of phosphodiester bonds
    - Saturation [%]: Percentage of possible edges realized
    """
    rows = []
    
    for idx, data in enumerate(dataset):
        try:
            # Extract sequence ID and sequence
            seq_id = data.id if isinstance(data.id, str) else str(data.id)
            seq = data.seq if isinstance(data.seq, str) else str(data.seq)
            seq_length = len(seq)
            
            # Get edge types (string labels like 'phosphodiester', 'canonical', 'non-canonical')
            edge_types = data.edge_type if hasattr(data, 'edge_type') else []
            
            # Count different edge types
            canonical_edges = edge_types.count('canonical') // 2
            non_canonical_edges = edge_types.count('non-canonical') // 2
            backbone_edges = edge_types.count('phosphodiester') // 2
            
            # Get edge index to find paired positions
            edge_index = data.edge_index.numpy() if hasattr(data.edge_index, 'numpy') else data.edge_index
            if edge_index.shape[0] == 2:
                edge_index = edge_index.T
            
            # Count paired positions (exclude phosphodiester bonds)
            paired_positions = set()
            for i, (idx_a, idx_b) in enumerate(edge_index):
                if i < len(edge_types) and edge_types[i] != 'phosphodiester':
                    paired_positions.add(int(idx_a))
                    paired_positions.add(int(idx_b))
            
            # Count paired and unpaired nucleotides
            paired_nts = len(paired_positions)
            unpaired_nts = seq_length - paired_nts
            
            # Number of nodes
            num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else seq_length
            
            # Calculate saturation (percentage of possible edges)
            # Maximum possible edges for a complete graph: n*(n-1)/2
            max_edges = seq_length * (seq_length - 1) // 2
            actual_edges = len(edge_index) // 2
            saturation = (actual_edges / max_edges * 100) if max_edges > 0 else 0
            
            rows.append({
                'Seq ID': seq_id,
                'ID': seq_id,
                'Length [nt]': seq_length,
                'Unpaired nts': unpaired_nts,
                'Paired nts': paired_nts,
                'Nodes': num_nodes,
                'Canonical edges': canonical_edges,
                'Non-canonical edges': non_canonical_edges,
                'Backbone edges': backbone_edges,
                'Saturation [%]': round(saturation, 2),
            })
        
        except Exception as e:
            print(f"Warning: Error processing sequence {idx}: {e}")
            continue
    
    return pd.DataFrame(rows)


def create_latex_summary_table(df: pd.DataFrame) -> str:
    """
    Create a LaTeX table with the dataset summary statistics.
    
    Format:
    Seq ID & \multicolumn{2}{c|}{Structure data} & \multicolumn{5}{c}{Graph model}
    """
    latex_lines = []
    
    # Table begin
    latex_lines.append(r'\begin{tabular}{c|cc|ccccc}')
    latex_lines.append(r'\thickhline')
    
    # Header row with multicolumns
    latex_lines.append(r'Seq ID & \multicolumn{2}{c|}{Structure data} & \multicolumn{5}{c}{Graph model} \\')
    
    # Subheader row
    subheader = (r'     & Length [nt] & Unpaired nts & Paired nts & Nodes & '
                 r'Canonical edges & Non-canonical edges & Backbone edges & Saturation [\%]')
    latex_lines.append(subheader + r' \\')
    latex_lines.append(r'\thickhline')
    
    # Data rows
    for _, row in df.iterrows():
        row_data = [
            str(row['Seq ID']),
            str(row['Length [nt]']),
            str(row['Unpaired nts']),
            str(row['Paired nts']),
            str(row['Nodes']),
            str(row['Canonical edges']),
            str(row['Non-canonical edges']),
            str(row['Backbone edges']),
            f"{row['Saturation [%]']:.2f}",
        ]
        latex_lines.append(' & '.join(row_data) + r' \\')
    
    latex_lines.append(r'\thickhline')
    latex_lines.append(r'\end{tabular}')
    
    return '\n'.join(latex_lines)


@click.command()
@click.option('--dataset', type=str, required=True,
              help='Dataset to load: train, validation, all, or evaluation datasets (casp, large, rnapuzzles, smallmed, ts1, ts2)')
@click.option('--output-dir', type=click.Path(), default=None,
              help='Directory to save output tables (default: outputs/)')
@click.option('--format', type=click.Choice(['csv', 'latex', 'both']), default='both',
              help='Output format: csv, latex, or both')
def main(dataset: str, output_dir: str, format: str):
    """
    Generate dataset statistics table.
    
    This script loads a PyTorch Geometric dataset of RNA graphs and computes
    statistics for each sequence including structure and graph metrics.
    
    Examples:
        python scripts/generate_stats_table.py --dataset train
        python scripts/generate_stats_table.py --dataset casp
        python scripts/generate_stats_table.py --dataset validation --format latex
    """
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: '{dataset}'...")
    dataset_path = Path(BASE_DIR) / "data"
    data = load_dataset(dataset, root=str(dataset_path))
    
    print(f"Loaded {len(data)} sequences")
    print("Calculating statistics...")
    
    # Calculate statistics
    df = calculate_dataset_statistics(data)
    
    print(f"\nDataset Summary Statistics ({dataset}):")
    print(df.to_string())
    
    # Save outputs
    if format in ['csv', 'both']:
        csv_output = output_dir / f"dataset_stats_{dataset}.csv"
        df.to_csv(csv_output, index=False)
        print(f"\nCSV table saved to: {csv_output}")
    
    if format in ['latex', 'both']:
        latex_output = output_dir / f"dataset_stats_{dataset}.tex"
        latex_table = create_latex_summary_table(df)
        with open(latex_output, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {latex_output}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total sequences: {len(df)}")
    print(f"Average sequence length: {df['Length [nt]'].mean():.1f} nt")
    print(f"Average paired nucleotides: {df['Paired nts'].mean():.1f}")
    print(f"Average unpaired nucleotides: {df['Unpaired nts'].mean():.1f}")
    print(f"Average nodes per sequence: {df['Nodes'].mean():.1f}")
    print(f"Average canonical edges: {df['Canonical edges'].mean():.1f}")
    print(f"Average non-canonical edges: {df['Non-canonical edges'].mean():.1f}")
    print(f"Average backbone edges: {df['Backbone edges'].mean():.1f}")
    print(f"Average saturation: {df['Saturation [%]'].mean():.2f}%")


if __name__ == "__main__":
    main()
