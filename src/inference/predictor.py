"""
predictor.py

Handles inference of the trained GNN model on a new, single SMILES string.
Loads the saved checkpoint, preprocesses the molecule into a graph,
runs the forward pass, and returns 12 toxicity probabilities.
"""

import torch
from torch_geometric.data import Data, Batch

from src.data.rdkit_parser import validate_and_parse_smiles
from src.data.graph_builder import mol_to_graph_data
from src.model.gnn import Tox21GNN
from src.utils.config import TOX21_LABELS


class ToxicityPredictor:
    """
    Loads a trained Tox21GNN checkpoint and predicts toxicity probabilities
    for new SMILES inputs.

    Args:
        checkpoint_path: Path to the saved model checkpoint (.pt file).
        num_node_features: Must match the feature dimension used during training.
        hidden_dim: Must match the hidden_dim used during training.
        device: 'cuda' or 'cpu'. Auto-detects if not provided.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_node_features: int,
        hidden_dim: int = 128,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path

        # Build model and load weights
        self.model = Tox21GNN(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_tasks=12,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded from {checkpoint_path}")
        print(f"   Checkpoint epoch: {checkpoint.get('epoch', '?')}")
        print(f"   Val AUC: {checkpoint.get('val_auc', '?')}")

    def predict(self, smiles: str) -> dict:
        """
        Predict toxicity probabilities for a single SMILES string.

        Args:
            smiles: A SMILES representation of the molecule.

        Returns:
            Dictionary with:
                - 'smiles': The input SMILES.
                - 'valid': Whether the SMILES was valid.
                - 'probabilities': Dict mapping each Tox21 task name to its probability.
                - 'raw_probs': List of 12 float probabilities.
        """
        # Step 1: Validate and parse SMILES
        mol = validate_and_parse_smiles(smiles, desalt=True)

        if mol is None:
            return {
                "smiles": smiles,
                "valid": False,
                "error": "Invalid SMILES string. Could not parse with RDKit.",
                "probabilities": None,
                "raw_probs": None,
            }

        # Step 2: Convert molecule to graph
        graph = mol_to_graph_data(mol)

        # Step 3: Batch the single graph (PyG requires batching even for 1 sample)
        batch = Batch.from_data_list([graph]).to(self.device)

        # Step 4: Run inference
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # Step 5: Build result dictionary
        prob_dict = {}
        for i, label in enumerate(TOX21_LABELS):
            prob_dict[label] = round(float(probs[i]), 4)

        return {
            "smiles": smiles,
            "valid": True,
            "probabilities": prob_dict,
            "raw_probs": [round(float(p), 4) for p in probs],
        }

    def predict_batch(self, smiles_list: list) -> list:
        """
        Predict toxicity for a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            List of prediction result dictionaries.
        """
        results = []
        for smiles in smiles_list:
            results.append(self.predict(smiles))
        return results
