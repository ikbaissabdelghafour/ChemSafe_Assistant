import sys
import os

# Ensure the src folder can be imported
sys.path.insert(0, os.path.abspath('.'))

from src.inference.risk_engine import analyze_risk, format_risk_report

import random
from src.data.rdkit_parser import validate_and_parse_smiles

def run_test():
    print("🧪 ChemSafe Assistant - Interactive Tester")
    print("Type 'exit' to quit.\n")
    
    while True:
        # Prompt user for input
        smiles_input = input("Enter a SMILES string (e.g. CCO for Ethanol): ").strip()
        
        if smiles_input.lower() == 'exit':
            break
            
        if not smiles_input:
            continue
            
        # 1. Validate the SMILES using our real RDKit parser!
        mol = validate_and_parse_smiles(smiles_input)
        if mol is None:
            print("❌ Invalid SMILES string. RDKit could not parse it. Try again.\n")
            continue
            
        print(f"\n✅ Valid Molecule! Generating simulated GNN predictions for '{smiles_input}'...")
        
        # 2. Simulate GNN Prediction (because the real model isn't trained yet)
        # We generate random probabilities just to test the Risk Engine logic
        mock_probs = {
            "NR-AR": random.uniform(0.0, 1.0),
            "NR-AR-LBD": random.uniform(0.0, 1.0),
            "NR-AhR": random.uniform(0.0, 1.0),
            "NR-Aromatase": random.uniform(0.0, 1.0),
            "NR-ER": random.uniform(0.0, 1.0),
            "NR-ER-LBD": random.uniform(0.0, 1.0),
            "NR-PPAR-gamma": random.uniform(0.0, 1.0),
            "SR-ARE": random.uniform(0.0, 1.0),
            "SR-ATAD5": random.uniform(0.0, 1.0),
            "SR-HSE": random.uniform(0.0, 1.0),
            "SR-MMP": random.uniform(0.0, 1.0),
            "SR-p53": random.uniform(0.0, 1.0)
        }

        mock_gnn_prediction = {
            "valid": True,
            "smiles": smiles_input,
            "probabilities": mock_probs
        }

        # 3. Pass this data to the real Risk Engine
        report = analyze_risk(mock_gnn_prediction)
        
        # 4. Print the beautifully formatted text report
        formatted_output = format_risk_report(report)
        print("\n" + formatted_output + "\n")

if __name__ == "__main__":
    run_test()
