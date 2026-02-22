import os
import numpy as np
import logging
from typing import Optional, Tuple
from .features import extract_quality_features, get_feature_names

logger = logging.getLogger(__name__)

class ProteinQualityClassifier:
    """
    Random Forest classifier to predict if a protein structure is "High Quality"
    (biophysically plausible) or "Low Quality" (likely has steric clashes or
    geometry violations).

    This is a classical machine learning model (scikit-learn RandomForest), not
    a neural network or generative AI. It uses structural geometry features
    (Ramachandran angles, steric clashes, bond lengths, radius of gyration) as
    inputs.

    The model is trained on synth-pdb generated data:
    - Positive samples: Minimized, valid structures
    - Negative samples: Raw unminimized structures, decoys, and random perturbations
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_names = get_feature_names()
        
        if model_path:
            self.load_model(model_path)
        else:
            # Try to load default model included in package
            default_path = os.path.join(os.path.dirname(__file__), "models", "quality_filter_v1.joblib")
            if os.path.exists(default_path):
                self.load_model(default_path)
            else:
                logger.warning(f"No model found at {default_path}. Classifier is not initialized.")

    def load_model(self, path: str):
        """Loads a pre-trained scikit-learn model."""
        try:
            import joblib
        except ImportError:
            logger.error("joblib is not installed. Install synth-pdb[ai] to use the quality filter.")
            self.model = None
            return
            
        try:
            self.model = joblib.load(path)
            logger.info(f"Loaded quality classifier model from {path}")
        except Exception as e:
            logger.error(f"Failed to load quality classifier model: {e}")
            self.model = None

    def predict(self, pdb_content: str) -> Tuple[bool, float, dict]:
        """
        Predicts quality of a PDB structure.
        
        Returns:
            is_good (bool): True if probability > 0.5
            probability (float): Confidence score (0.0 - 1.0)
            features (dict): The extracted feature values used for prediction
        """
        if self.model is None:
            raise RuntimeError("Classifier model is not loaded.")
            
        # Extract features
        features_vec = extract_quality_features(pdb_content)
        
        # Predict
        # Reshape to (1, n_features) for scikit-learn
        prob = self.model.predict_proba(features_vec.reshape(1, -1))[0, 1]
        is_good = prob > 0.5
        
        # Create feature dict for debugging/logging
        feature_dict = dict(zip(self.feature_names, features_vec))
        
        return is_good, prob, feature_dict
