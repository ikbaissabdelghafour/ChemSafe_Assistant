# ChemSafe Assistant

ChemSafe Assistant is an AI system designed for chemical safety education.

## Pipeline Flow
`SMILES → RDKit → Graph → GNN (Tox21) → Risk Engine → Google Gemini → Explanation`

## Project Structure
- `colab/`: Google Colab notebooks for training and preprocessing.
- `data/`: Contains raw data and processed graphs.
- `src/`: Main source files for parsing, dataset building, GNN architecture, risk interpretation, and LLM orchestration.
- `saved_models/`: Stores trained neural networks.
