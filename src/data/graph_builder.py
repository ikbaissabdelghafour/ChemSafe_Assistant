"""
graph_builder.py

Converts molecular representations (parsed RDKit molecules) into graph data structures
(nodes, edges, atom/bond features) suitable for GNN input.
"""

import torch
from rdkit import Chem
from torch_geometric.data import Data

# Allowed atom types for one-hot encoding (covers majority of organic molecules in Tox21)
ALLOWED_ATOMS = [
    'H', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'P', 'S', 'Cl', 'K', 'Ca', 
    'Fe', 'Cu', 'Zn', 'Br', 'I'
]

# Hybridization list for one-hot encoding
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]

def get_atom_features(atom: Chem.Atom) -> torch.Tensor:
    """
    Extracts atom features and returns a torch FloatTensor.
    """
    symbol = atom.GetSymbol()
    # 1. Atom Symbol One-hot
    symbol_feat = [1.0 if symbol == allowed else 0.0 for allowed in ALLOWED_ATOMS]
    # Handle unknown elements (out of list)
    if sum(symbol_feat) == 0:
        symbol_feat.append(1.0)  # Unknown element token
    else:
        symbol_feat.append(0.0)

    # 2. Degree One-hot (0 to 5+)
    degree = atom.GetDegree()
    degree_feat = [1.0 if degree == d else 0.0 for d in range(6)]

    # 3. Hybridization One-hot
    hybrid = atom.GetHybridization()
    hybrid_feat = [1.0 if hybrid == h else 0.0 for h in HYBRIDIZATIONS]
    if sum(hybrid_feat) == 0:
        hybrid_feat.append(1.0)  # Unknown or none
    else:
        hybrid_feat.append(0.0)

    # 4. Formal Charge
    charge_feat = [float(atom.GetFormalCharge())]

    # 5. Aromaticity (binary)
    aromatic_feat = [1.0 if atom.GetIsAromatic() else 0.0]

    # 6. Hydrogen count
    h_feat = [float(atom.GetTotalNumHs())]

    all_features = symbol_feat + degree_feat + hybrid_feat + charge_feat + aromatic_feat + h_feat
    return torch.tensor(all_features, dtype=torch.float)


def get_bond_features(bond: Chem.Bond) -> torch.Tensor:
    """
    Extracts bond features and returns a torch FloatTensor.
    """
    bond_type = bond.GetBondType()
    
    # 1. Bond type One-hot
    type_feat = [
        1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0,
    ]
    
    # 2. Is Conjugated
    conjug_feat = [1.0 if bond.GetIsConjugated() else 0.0]
    
    # 3. Is in ring
    ring_feat = [1.0 if bond.IsInRing() else 0.0]
    
    all_features = type_feat + conjug_feat + ring_feat
    return torch.tensor(all_features, dtype=torch.float)


def mol_to_graph_data(mol: Chem.Mol, y: torch.Tensor = None) -> Data:
    """
    Converts an RDKit molecule to a PyTorch Geometric Data object.
    
    Args:
        mol: An RDKit Mol object.
        y: Optional tensor of target labels.
        
    Returns:
        A torch_geometric.data.Data object.
    """
    # Node features
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(get_atom_features(atom))
    x = torch.stack(node_features, dim=0)

    # Edges & Edge features
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        # RDKit bonds are undirected; we need to add edges in both directions for GNN
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_feat = get_bond_features(bond)
        
        # Forward edge
        edge_indices.append([start, end])
        edge_features.append(edge_feat)
        
        # Backward edge
        edge_indices.append([end, start])
        edge_features.append(edge_feat)

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_features, dim=0)
    else:
        # Handling isolated atoms/no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)  # 6 features in get_bond_features

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    if y is not None:
        data.y = y

    return data
