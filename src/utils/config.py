"""
config.py

Global configurations, hyperparameters, paths, and environment variable loaders.
Central source of truth for constants shared across all modules.
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; .env file won't be loaded

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "tox21.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "graphs.pt")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "saved_models", "gnn_tox21.pt")

# ── Tox21 Label Names (single source of truth) ──────────────────────────
TOX21_LABELS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]
NUM_TASKS = len(TOX21_LABELS)

# ── Model Hyperparameters ────────────────────────────────────────────────
HIDDEN_DIM = 128
DROPOUT = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 15

# ── Risk Thresholds ──────────────────────────────────────────────────────
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.30

# ── Gemini API ───────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
