"""
main.py

Main entry point of the ChemSafe Assistant.
Runs a command-line interface to orchestrate the pipeline:
SMILES -> GNN Prediction -> Risk Analysis -> (Gemini LLM - Coming Soon).
"""

import sys
import os

# Ensure the src folder can be imported
sys.path.insert(0, os.path.abspath('.'))

from src.utils.config import MODEL_SAVE_PATH
from src.inference.predictor import ToxicityPredictor
from src.inference.risk_engine import analyze_risk, format_risk_report
from src.data.rdkit_parser import validate_and_parse_smiles
from src.data.graph_builder import mol_to_graph_data

def get_num_node_features() -> int:
    """Helper to dynamically determine node feature size from the graph builder."""
    dummy_mol = validate_and_parse_smiles("C")
    dummy_graph = mol_to_graph_data(dummy_mol)
    return dummy_graph.x.shape[1]

def main():
    print("\n" + "="*55)
    print("      🧪 CHEMSAFE ASSISTANT INITIALIZATION")
    print("="*55)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"\n❌ ERROR: Model checkpoint not found at '{MODEL_SAVE_PATH}'.")
        print("Please train the model first by running the training pipeline.")
        sys.exit(1)

    print("\nLoading trained GNN model...")
    num_features = get_num_node_features()
    
    try:
        predictor = ToxicityPredictor(
            checkpoint_path=MODEL_SAVE_PATH,
            num_node_features=num_features
        )
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load the model. {e}")
        sys.exit(1)

    print("\n✅ Ready! Interactive Predictor Started.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            smiles_input = input("Enter a SMILES string (e.g. CCO for Ethanol): ").strip()
            
            if smiles_input.lower() == 'exit':
                break
                
            if not smiles_input:
                continue

            # Run real prediction
            prediction = predictor.predict(smiles_input)

            if not prediction["valid"]:
                print(f"❌ Error: {prediction.get('error', 'Invalid SMILES')}\n")
                continue

            # Run risk analysis
            report = analyze_risk(prediction)
            
            # Print report
            formatted_output = format_risk_report(report)
            print("\n" + formatted_output + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}\n")

if __name__ == "__main__":
    main()
