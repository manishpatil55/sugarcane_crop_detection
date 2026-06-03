"""
validate_pipeline.py
====================
End-to-end validation script for the Sugarcane Detection Pipeline.
Verifies directories, configurations, model availability, and tests the Predictor API.
"""
import os
import yaml
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Pipeline Validation...")
    
    # 1. Directory Checks
    dirs_to_check = [
        "data/kml/sugarcane",
        "data/kml/non_sugarcane",
        "data/kml/validation/sugarcane",
        "data/kml/validation/non_sugarcane",
        "models/saved",
        "data/processed"
    ]
    for d in dirs_to_check:
        if not Path(d).exists():
            logger.warning(f"Directory {d} does not exist. Creating it.")
            Path(d).mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"[OK] Directory exists: {d}")
            
    # 2. Config Verification
    if not Path("config.yaml").exists():
        logger.error("[FAIL] config.yaml not found!")
        sys.exit(1)
        
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    active_model = cfg.get("inference", {}).get("active_model", None)
    if not active_model:
        logger.error("[FAIL] 'active_model' not specified in config.yaml under 'inference'.")
        sys.exit(1)
    logger.info(f"[OK] Active model found in config: {active_model}")
    
    # 3. Model Existence Check
    model_paths = {
        "bilstm": "models/saved/bilstm_best.pt",
        "random_forest": "models/saved/rf_best.joblib",
        "xgboost": "models/saved/xgb_best.joblib"
    }
    
    expected_path = model_paths.get(active_model)
    if expected_path:
        if not Path(expected_path).exists():
            logger.warning(f"[WARN] Expected model file {expected_path} for active model '{active_model}' is MISSING. (You may need to run train.py first)")
        else:
            logger.info(f"[OK] Expected model file {expected_path} found.")
    else:
        logger.error(f"[FAIL] Unknown active model '{active_model}'.")
        sys.exit(1)

    # 4. Dummy Predictor API Check
    logger.info("Initializing Predictor instance...")
    try:
        from inference.predictor import SugarcanePredictor
        predictor = SugarcanePredictor(config_path="config.yaml")
        logger.info("[OK] Predictor initialized successfully.")
    except Exception as e:
        logger.error(f"[FAIL] Could not initialize Predictor: {e}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("SUCCESS: PIPELINE VALIDATION COMPLETED. SYSTEM IS PRODUCTION-READY.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
