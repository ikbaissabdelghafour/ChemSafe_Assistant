import json

def update_notebook():
    with open("local_train.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "y_pred_v = (y_score_v > 0.5).astype(int)" in src:
                src = src.replace("model.load_state_dict(checkpoint['model_state_dict'])\n", "model.load_state_dict(checkpoint['model_state_dict'])\n    best_thresholds = checkpoint.get('best_thresholds', [0.5] * 12)\n")
                src = src.replace("y_pred_v = (y_score_v > 0.5).astype(int)", "best_thresh = best_thresholds[i]\n        y_pred_v = (y_score_v > best_thresh).astype(int)")
                
                # Split back into lines
                lines = [line + "\n" for line in src.split("\n")]
                # Fix the last line
                lines[-1] = lines[-1].rstrip("\n")
                if not lines[-1]:
                    lines.pop()
                    
                cell["source"] = lines

    with open("local_train.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    update_notebook()
