"""
risk_engine.py

Translates raw GNN probabilities (12 endpoints) into risk levels (Low, Medium, High)
and aggregates them into a global toxicity risk score.

Risk Thresholds:
    - High Risk   : probability >= 0.70
    - Medium Risk  : 0.30 <= probability < 0.70
    - Low Risk     : probability < 0.30

Global Risk Strategy:
    - If ANY task is High Risk → Global = High
    - Else if ANY task is Medium Risk → Global = Medium
    - Else → Global = Low
    
    A weighted global score (0-100) is also computed for finer granularity.
"""

from typing import Dict, List, Optional
from src.utils.config import TOX21_LABELS

# ── Risk Thresholds ──────────────────────────────────────────────────────
HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.30

# ── Human-readable descriptions of each Tox21 assay ─────────────────────
ASSAY_DESCRIPTIONS = {
    "NR-AR":         "Androgen Receptor activity",
    "NR-AR-LBD":     "Androgen Receptor Ligand Binding Domain",
    "NR-AhR":        "Aryl Hydrocarbon Receptor activation",
    "NR-Aromatase":  "Aromatase enzyme inhibition",
    "NR-ER":         "Estrogen Receptor activity",
    "NR-ER-LBD":     "Estrogen Receptor Ligand Binding Domain",
    "NR-PPAR-gamma": "Peroxisome Proliferator-Activated Receptor Gamma",
    "SR-ARE":        "Antioxidant Response Element activation",
    "SR-ATAD5":      "DNA damage (genotoxicity) response",
    "SR-HSE":        "Heat Shock Response activation",
    "SR-MMP":        "Mitochondrial Membrane Potential disruption",
    "SR-p53":        "Tumor suppressor p53 pathway activation",
}

# ── Pathway criticality weights (higher = more dangerous endpoint) ───────
# Used for computing the weighted global score.
ASSAY_WEIGHTS = {
    "NR-AR":         1.0,
    "NR-AR-LBD":     1.0,
    "NR-AhR":        1.2,   # Carcinogen-related pathway
    "NR-Aromatase":  1.0,
    "NR-ER":         1.0,
    "NR-ER-LBD":     1.0,
    "NR-PPAR-gamma": 0.8,
    "SR-ARE":        1.0,
    "SR-ATAD5":      1.3,   # DNA damage — high criticality
    "SR-HSE":        0.9,
    "SR-MMP":        1.2,   # Mitochondrial toxicity — organ damage
    "SR-p53":        1.3,   # Tumor suppressor — cancer pathway
}


def classify_risk(probability: float) -> str:
    """Classify a single probability into a risk level."""
    if probability >= HIGH_THRESHOLD:
        return "High"
    elif probability >= MEDIUM_THRESHOLD:
        return "Medium"
    else:
        return "Low"


def get_risk_emoji(level: str) -> str:
    """Return an emoji indicator for the risk level."""
    return {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(level, "⚪")


def analyze_risk(prediction: dict) -> Optional[dict]:
    """
    Perform full risk analysis on a prediction result from ToxicityPredictor.

    Args:
        prediction: Dictionary returned by ToxicityPredictor.predict().
                    Must contain 'valid', 'probabilities', and 'smiles'.

    Returns:
        Risk report dictionary, or None if prediction was invalid.
    """
    if not prediction.get("valid", False):
        return None

    probabilities = prediction["probabilities"]
    smiles = prediction["smiles"]

    # ── Per-task risk classification ─────────────────────────────────
    task_risks = []
    for label in TOX21_LABELS:
        prob = probabilities[label]
        level = classify_risk(prob)
        task_risks.append({
            "task": label,
            "description": ASSAY_DESCRIPTIONS.get(label, ""),
            "probability": prob,
            "risk_level": level,
            "emoji": get_risk_emoji(level),
        })

    # ── Global risk level (worst-case override) ──────────────────────
    risk_levels = [t["risk_level"] for t in task_risks]

    if "High" in risk_levels:
        global_risk_level = "High"
    elif "Medium" in risk_levels:
        global_risk_level = "Medium"
    else:
        global_risk_level = "Low"

    # ── Weighted global score (0–100 scale) ──────────────────────────
    weighted_sum = 0.0
    weight_total = 0.0
    for label in TOX21_LABELS:
        w = ASSAY_WEIGHTS.get(label, 1.0)
        weighted_sum += probabilities[label] * w
        weight_total += w

    global_score = round((weighted_sum / weight_total) * 100, 1)

    # ── Identify high-risk pathways for LLM focus ────────────────────
    high_risk_tasks = [t for t in task_risks if t["risk_level"] == "High"]
    medium_risk_tasks = [t for t in task_risks if t["risk_level"] == "Medium"]

    # ── Build final report ───────────────────────────────────────────
    report = {
        "smiles": smiles,
        "global_risk_level": global_risk_level,
        "global_risk_emoji": get_risk_emoji(global_risk_level),
        "global_score": global_score,
        "task_risks": task_risks,
        "high_risk_count": len(high_risk_tasks),
        "medium_risk_count": len(medium_risk_tasks),
        "high_risk_tasks": high_risk_tasks,
        "medium_risk_tasks": medium_risk_tasks,
        "summary": _build_summary(smiles, global_risk_level, global_score, high_risk_tasks),
    }

    return report


def _build_summary(
    smiles: str,
    global_level: str,
    global_score: float,
    high_risk_tasks: List[dict],
) -> str:
    """Build a concise text summary of the risk analysis."""
    lines = []
    lines.append(f"Molecule: {smiles}")
    lines.append(f"Global Risk: {get_risk_emoji(global_level)} {global_level} (score: {global_score}/100)")

    if high_risk_tasks:
        lines.append(f"⚠️  {len(high_risk_tasks)} high-risk pathway(s) detected:")
        for t in high_risk_tasks:
            lines.append(f"   • {t['task']} ({t['description']}): {t['probability']:.1%}")
    else:
        lines.append("No high-risk pathways detected.")

    return "\n".join(lines)


def format_risk_report(report: dict) -> str:
    """
    Format the full risk report as a human-readable string for display.

    Args:
        report: Dictionary returned by analyze_risk().

    Returns:
        Formatted multi-line string.
    """
    if report is None:
        return "❌ Could not generate risk report (invalid molecule)."

    lines = []
    lines.append("=" * 55)
    lines.append("  CHEMSAFE TOXICITY RISK REPORT")
    lines.append("=" * 55)
    lines.append(f"  Molecule : {report['smiles']}")
    lines.append(f"  Global   : {report['global_risk_emoji']} {report['global_risk_level']} Risk")
    lines.append(f"  Score    : {report['global_score']} / 100")
    lines.append("-" * 55)

    for t in report["task_risks"]:
        prob_bar = "█" * int(t["probability"] * 20)
        prob_bar = prob_bar.ljust(20, "░")
        lines.append(
            f"  {t['emoji']} {t['task']:<18} {prob_bar} {t['probability']:.1%}  [{t['risk_level']}]"
        )

    lines.append("-" * 55)

    if report["high_risk_count"] > 0:
        lines.append(f"  ⚠️  {report['high_risk_count']} HIGH-RISK pathway(s) require attention!")
    if report["medium_risk_count"] > 0:
        lines.append(f"  ⚡ {report['medium_risk_count']} MEDIUM-RISK pathway(s) detected.")
    if report["high_risk_count"] == 0 and report["medium_risk_count"] == 0:
        lines.append("  ✅ All pathways are LOW risk.")

    lines.append("=" * 55)
    return "\n".join(lines)
