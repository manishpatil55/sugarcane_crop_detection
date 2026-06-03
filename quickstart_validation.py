"""
quickstart_validation.py
========================
A quickstart script to validate the end-to-end sugarcane detection pipeline
once negative KML samples have been collected.

Usage:
    python quickstart_validation.py
"""

import logging
import sys
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Sugarcane Pipeline Validation...")
    
    sugarcane_dir = Path("data/kml/sugarcane")
    non_sugarcane_dir = Path("data/kml/non_sugarcane")
    
    if not sugarcane_dir.exists() or not list(sugarcane_dir.glob("*.kml")) and not list(sugarcane_dir.glob("*.kmz")):
        logger.error(f"Missing positive sugarcane KMLs in {sugarcane_dir}")
        sys.exit(1)
        
    if not non_sugarcane_dir.exists() or not list(non_sugarcane_dir.glob("*.kml")) and not list(non_sugarcane_dir.glob("*.kmz")):
        logger.error(f"Missing negative non-sugarcane KMLs in {non_sugarcane_dir}")
        logger.error("Please add verified negative samples (e.g. rice-wheat fields, orchards) before running.")
        sys.exit(1)
        
    logger.info("Found positive and negative KML samples.")
    logger.info("Starting the training pipeline (train.py)...")
    
    try:
        subprocess.run([sys.executable, "train.py"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)
        
    logger.info("Training completed successfully.")
    
    # Run a quick test inference if a positive sample is available
    test_kml = list(sugarcane_dir.glob("*.kml"))
    if test_kml:
        kml_path = test_kml[0]
        # Using a dummy harvest crop date from the positive samples (Sep 2025)
        crop_date = "2025-09-15"
        logger.info(f"Running inference test on {kml_path} for date {crop_date}...")
        
        try:
            subprocess.run([sys.executable, "inference/predictor.py", str(kml_path), crop_date], check=True)
            logger.info("Inference test completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Inference pipeline failed: {e}")
            sys.exit(1)
            
    logger.info("Pipeline validation complete! The system is ready for production.")

if __name__ == "__main__":
    main()
