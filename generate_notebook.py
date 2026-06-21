import os
import json
import re

def generate_notebook(train_py_path, output_ipynb_path):
    with open(train_py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    cells = []
    
    # Header cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ChemSafe Assistant - Local Training\n",
            "\n",
            "This notebook trains the GNN locally."
        ]
    })

    # Split the file by print("\n--- X. Title ---")
    # This regex looks for: print("\n--- 1. Title ---")
    sections = re.split(r'print\("\\n---\s*\d+\.\s*(.*?)\s*---"\)', content)
    
    # sections[0] is the top imports and setup
    top_code = sections[0].strip()
    if top_code:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in top_code.split('\n')]
        })
        # Remove the last trailing newline
        if cells[-1]["source"] and cells[-1]["source"][-1].endswith("\n"):
            cells[-1]["source"][-1] = cells[-1]["source"][-1].rstrip("\n")

    # The rest are pairs of (Title, Code)
    for i in range(1, len(sections), 2):
        title = sections[i].strip()
        code = sections[i+1].strip()
        
        # Add Markdown cell for title
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {title}"]
        })
        
        # Add Code cell
        if code:
            lines = [line + "\n" for line in code.split('\n')]
            if lines and lines[-1].endswith("\n"):
                lines[-1] = lines[-1].rstrip("\n")
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines
            })

    # Create the notebook dictionary
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(output_ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

if __name__ == '__main__':
    train_py = os.path.join('c:\\Projects\\ChemSafe_Assistant', 'train.py')
    output_nb = os.path.join('c:\\Projects\\ChemSafe_Assistant', 'local_train.ipynb')
    generate_notebook(train_py, output_nb)
    print(f"Generated {output_nb} from {train_py}")
