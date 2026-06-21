"""
rdkit_parser.py

Handles parsing SMILES strings using RDKit, verifying molecular validity, 
and computing basic molecular properties.
"""

from typing import Optional
from rdkit import Chem
from rdkit.Chem import SaltRemover

def validate_and_parse_smiles(smiles: str, desalt: bool = True) -> Optional[Chem.Mol]:
    """
    Validates a SMILES string and parses it into an RDKit Mol object.
    
    Args:
        smiles: The SMILES string of the compound.
        desalt: If True, removes salts/counterions and returns the largest fragment.
        
    Returns:
        Chem.Mol object if valid, else None.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Sanitize molecule
        Chem.SanitizeMol(mol)
        
        # Remove salts/mixtures (keep only largest organic fragment)
        if desalt:
            remover = SaltRemover.SaltRemover()
            mol = remover.StripMol(mol, dontRemoveEverything=True)
            
            # Re-sanitize after stripping
            Chem.SanitizeMol(mol)
            
        return mol
    except Exception:
        return None
