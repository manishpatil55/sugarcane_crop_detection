"""Verify the persisted model artefact loads and matches the trained schema."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import joblib
import yaml

art = joblib.load("models/saved/best.pkl")

# Required keys
required = {"model", "feature_names", "optimal_threshold", "best_model_name", "n_features"}
missing = required - set(art.keys())
assert not missing, f"Artefact missing keys: {missing}"

print(f"best_model_name   : {art['best_model_name']}")
print(f"n_features        : {art['n_features']}")
print(f"optimal_threshold : {art['optimal_threshold']:.4f}")
print(f"tuned_params      : {art.get('tuned_params')}")
print(f"feature_names[:3] : {art['feature_names'][:3]}")
print(f"feature_names[-3:]: {art['feature_names'][-3:]}")

# Check config was updated
cfg = yaml.safe_load(open("config.yaml"))
print(f"\nconfig.inference.active_model         : {cfg['inference']['active_model']}")
print(f"config.inference.probability_threshold: {cfg['inference']['probability_threshold']}")
print(f"config.inference.model_path           : {cfg['inference']['model_path']}")

# Verify SugarcanePredictor loads the same artefact and exposes the same threshold
from inference.predictor import SugarcanePredictor
p = SugarcanePredictor(model_path="models/saved/best.pkl", config_path="config.yaml")
p.load_models()
assert p.model is not None
assert len(p.feature_names) == art["n_features"]
assert abs(p.optimal_threshold - art["optimal_threshold"]) < 1e-6
assert p.model_type == art["best_model_name"]
print(f"\nSugarcanePredictor loaded OK:")
print(f"  model_type        : {p.model_type}")
print(f"  feature_names len : {len(p.feature_names)}")
print(f"  optimal_threshold : {p.optimal_threshold:.4f}")

print("\nARTEFACT OK")
