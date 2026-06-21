"""
dataset.py

Defines PyTorch/PyTorch Geometric Dataset class for loading, caching, 
and batching molecular graphs from Tox21 dataset.
Includes dataset splitting and pipeline execution.
"""

import os
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

from src.data.rdkit_parser import validate_and_parse_smiles
from src.data.graph_builder import mol_to_graph_data

# The 12 target toxicity labels of Tox21
TOX21_LABELS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

def load_and_preprocess_tox21(
    csv_path: str, 
    desalt: bool = True
) -> Tuple[List[Data], pd.DataFrame]:
    """
    Loads raw tox21.csv, validates SMILES, builds graphs, and extracts labels.
    Missing labels (NaN) are filled with float('nan').
    
    Returns:
        A list of PyG Data objects and the clean DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tox21 CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    
    graphs = []
    valid_rows = []
    
    print(f"Loaded {len(df)} rows from {csv_path}. Processing smiles...")

    for idx, row in df.iterrows():
        smiles = row['smiles']
        mol = validate_and_parse_smiles(smiles, desalt=desalt)
        
        if mol is None:
            # Skip invalid SMILES
            continue
            
        # Get labels, preserve NaNs as NaN (we check for NaN during training loss)
        labels = []
        for col in TOX21_LABELS:
            val = row.get(col, np.nan)
            labels.append(float(val) if pd.notna(val) else float('nan'))
            
        y = torch.tensor(labels, dtype=torch.float)
        
        # Build graph
        graph_data = mol_to_graph_data(mol, y=y)
        # Store compound metadata inside the graph if needed
        graph_data.smiles = smiles
        if 'mol_id' in row:
            graph_data.mol_id = row['mol_id']
            
        graphs.append(graph_data)
        valid_rows.append(row)

    clean_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    print(f"Successfully processed {len(graphs)} / {len(df)} valid molecular graphs.")
    return graphs, clean_df


def split_dataset(
    graphs: List[Data], 
    train_ratio: float = 0.70, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Splits the molecular graph dataset into Train, Val, and Test subsets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    # Shuffle and split
    indices = np.arange(len(graphs))
    
    # First split off train
    train_idx, val_test_idx = train_test_split(
        indices, 
        train_size=train_ratio, 
        random_state=seed, 
        shuffle=True
    )
    
    # Second split val & test
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        val_test_idx, 
        train_size=relative_val_ratio, 
        random_state=seed, 
        shuffle=True
    )
    
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    print(f"Splits complete: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
    return train_graphs, val_graphs, test_graphs


def save_processed_pipeline(
    csv_path: str, 
    save_path: str, 
    desalt: bool = True, 
    seed: int = 42
) -> Dict[str, List[Data]]:
    """
    Runs the entire preprocessing pipeline: Load -> Validate/Clean -> Graph Builder -> Split -> Save.
    """
    graphs, _ = load_and_preprocess_tox21(csv_path, desalt=desalt)
    train_g, val_g, test_g = split_dataset(graphs, seed=seed)
    
    data_dict = {
        'train': train_g,
        'val': val_g,
        'test': test_g
    }
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data_dict, save_path)
    print(f"Saved processed graph datasets to {save_path}")
    return data_dict
