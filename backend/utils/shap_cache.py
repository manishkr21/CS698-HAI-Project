"""
SHAP caching module - handles caching of SHAP explainer and values
"""
import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path
import os
from typing import Dict, Any, Optional

from repositories.model_repository import load_artifact
from utils.config import FEATURE_NAMES

# Get the absolute path to the backend directory
BACKEND_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = BACKEND_DIR / "resources" / "shap_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'mitigated_models_ovr.pkl'
PREPROCESSOR_NAME = 'preprocessor_no_sens.pkl'

class SHAPPredictor:
    """Wrapper class for model prediction to make it picklable"""
    def __init__(self, model):
        self.model = model
    
    def __call__(self, data):
        if isinstance(self.model, dict) and 'models' in self.model:
            # Handle one-vs-rest models dictionary
            class_scores = np.zeros((len(data), len(self.model['models'])))
            for class_idx, model_info in self.model['models'].items():
                mitigator = model_info['model']
                if hasattr(mitigator, 'predict_proba'):
                    class_scores[:, class_idx] = mitigator.predict_proba(data)[:, 1]
                else:
                    class_scores[:, class_idx] = mitigator.predict(data)
            return class_scores
        else:
            # Handle single model case
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(data)
            return self.model.predict(data)

def initialize_shap_cache() -> Dict[str, Any]:
    """Initialize SHAP explainer and cache basic values"""
    try:
        print("Initializing SHAP cache...")
        # Load model and preprocessor
        model = load_artifact(MODEL_NAME)
        preprocessor = load_artifact(PREPROCESSOR_NAME)
        
        # Create background data
        background_data = pd.DataFrame([{
            col: 0.5 if col not in ['Age at enrollment', 'Application order'] else 20
            for col in FEATURE_NAMES if col in preprocessor.feature_names_in_
        } for _ in range(50)])
        
        # Preprocess background data
        background_processed = preprocessor.transform(background_data)
        
        # Create predictor instance with proper model
        predictor = SHAPPredictor(model)
        
        # Create KernelExplainer with proper model
        explainer = shap.KernelExplainer(
            predictor,
            background_processed,
            link="identity"  # Use identity link for direct probability outputs
        )
        
        # Get SHAP values for background data
        background_shap_values = explainer.shap_values(background_processed)
        
        # Cache everything
        cache = {
            'explainer': explainer,
            'background_data': background_data,
            'background_processed': background_processed,
            'background_shap_values': background_shap_values,
            'predictor': predictor,
            'feature_names': list(background_data.columns)
        }
        
        # Save cache
        cache_file = CACHE_DIR / 'shap_cache.pkl'
        print(f"Saving SHAP cache to {cache_file}")
        joblib.dump(cache, cache_file)
        
        return cache
    except Exception as e:
        print(f"Error initializing SHAP cache: {e}")
        raise

def get_shap_cache() -> Dict[str, Any]:
    """Get cached SHAP values, initialize if not exists"""
    cache_file = CACHE_DIR / 'shap_cache.pkl'
    
    try:
        if not cache_file.exists():
            return initialize_shap_cache()
        
        print(f"Loading SHAP cache from {cache_file}")
        cache = joblib.load(cache_file)
        print("SHAP cache loaded successfully")
        return cache
    except Exception as e:
        print(f"Error loading SHAP cache: {e}")
        print("Reinitializing cache...")
        # If loading fails, try to reinitialize
        if cache_file.exists():
            cache_file.unlink()
        return initialize_shap_cache()

def clear_shap_cache():
    """Clear the SHAP cache"""
    cache_file = CACHE_DIR / 'shap_cache.pkl'
    if cache_file.exists():
        try:
            cache_file.unlink()
            print("SHAP cache cleared successfully")
        except Exception as e:
            print(f"Error clearing SHAP cache: {e}")