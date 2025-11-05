"""
LIME caching module - handles caching of LIME global importance values
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os
from typing import Dict, Any, Optional

from repositories.model_repository import load_artifact
from utils.config import FEATURE_NAMES, CLASS_NAMES

# Get the absolute path to the backend directory
BACKEND_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = BACKEND_DIR / "resources" / "lime_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def initialize_lime_cache(model_type: str = 'best_model') -> Dict[str, Any]:
    """Initialize LIME cache with global feature importance values"""
    try:
        print(f"Initializing LIME cache for {model_type}...")
        # Load model and preprocessor
        if model_type == 'best_model':
            model = load_artifact('best_model.pkl')
            preprocessor = load_artifact('preprocessor.pkl')
        else:
            model = load_artifact('gb_model_unbiased_features.pkl')
            preprocessor = load_artifact('preprocessor_no_sens.pkl')
        
        # Create background data using feature ranges
        feature_ranges = {
            'Marital status': (0, 6),
            'Application mode': (1, 17),
            'Application order': (0, 9),
            'Course': (1, 17),
            'Daytime/evening attendance': (0, 1),
            'Previous qualification': (1, 17),
            'Nationality': (1, 21),
            'Mother\'s qualification': (1, 34),
            'Father\'s qualification': (1, 34),
            'Mother\'s occupation': (1, 46),
            'Father\'s occupation': (1, 46),
            'Displaced': (0, 1),
            'Educational special needs': (0, 1),
            'Debtor': (0, 1),
            'Tuition fees up to date': (0, 1),
            'Gender': (0, 1),
            'Scholarship holder': (0, 1),
            'Age at enrollment': (17, 70),
            'International': (0, 1),
            'Curricular units 1st sem (credited)': (0, 20),
            'Curricular units 1st sem (enrolled)': (0, 20),
            'Curricular units 1st sem (evaluations)': (0, 40),
            'Curricular units 1st sem (approved)': (0, 20),
            'Curricular units 1st sem (grade)': (0, 20),
            'Curricular units 2nd sem (credited)': (0, 20),
            'Curricular units 2nd sem (enrolled)': (0, 20),
            'Curricular units 2nd sem (evaluations)': (0, 40),
            'Curricular units 2nd sem (approved)': (0, 20),
            'Curricular units 2nd sem (grade)': (0, 20),
            'Unemployment rate': (8, 16),
            'Inflation rate': (-1, 5),
            'GDP': (-5, 5)
        }
        
        num_samples = 1000
        background_data = pd.DataFrame()
        np.random.seed(42)
        
        for feature in FEATURE_NAMES:
            if feature in feature_ranges:
                min_val, max_val = feature_ranges[feature]
                if feature in ['Age at enrollment', 'Unemployment rate', 'Inflation rate', 'GDP']:
                    values = np.random.uniform(min_val, max_val, num_samples)
                else:
                    values = np.random.randint(min_val, max_val + 1, num_samples)
            else:
                values = np.random.uniform(0, 1, num_samples)
            background_data[feature] = values
        
        # Preprocess background data
        background_processed = preprocessor.transform(background_data)
        
        # Store initial global importance values
        feature_importance = {}
        for feature in FEATURE_NAMES:
            # Initialize with small random values to prevent zero importance
            feature_importance[feature] = float(np.random.uniform(0.001, 0.01))
        
        # Cache everything
        cache = {
            'background_data': background_data,
            'background_processed': background_processed,
            'feature_importance': feature_importance,
            'model_type': model_type,
            'feature_names': FEATURE_NAMES,
            'num_samples_processed': 0
        }
        
        # Save cache
        cache_file = CACHE_DIR / f'lime_cache_{model_type}.pkl'
        print(f"Saving LIME cache to {cache_file}")
        joblib.dump(cache, cache_file)
        
        return cache
    
    except Exception as e:
        print(f"Error initializing LIME cache: {e}")
        raise

def get_lime_cache(model_type: str = 'best_model') -> Dict[str, Any]:
    """Get cached LIME values, initialize if not exists"""
    cache_file = CACHE_DIR / f'lime_cache_{model_type}.pkl'
    
    try:
        if not cache_file.exists():
            return initialize_lime_cache(model_type)
        
        print(f"Loading LIME cache from {cache_file}")
        cache = joblib.load(cache_file)  # Changed from dump to load
        print("LIME cache loaded successfully")
        return cache
    except Exception as e:
        print(f"Error loading LIME cache: {e}")
        print("Reinitializing cache...")
        # If loading fails, try to reinitialize
        if cache_file.exists():
            cache_file.unlink()
        return initialize_lime_cache(model_type)

def update_lime_cache(new_importances: Dict[str, float], model_type: str = 'best_model'):
    """Update the cached LIME importance values with new samples"""
    try:
        cache = get_lime_cache(model_type)
        
        # Update feature importance using exponential moving average
        alpha = 0.1  # Weight for new values
        current_importances = cache['feature_importance']
        
        for feature in FEATURE_NAMES:
            if feature in new_importances:
                current_value = current_importances.get(feature, 0.0)
                new_value = new_importances[feature]
                current_importances[feature] = alpha * new_value + (1 - alpha) * current_value
        
        # Update cache
        cache['feature_importance'] = current_importances
        cache['num_samples_processed'] += 1
        
        # Save updated cache
        cache_file = CACHE_DIR / f'lime_cache_{model_type}.pkl'
        joblib.dump(cache, cache_file)
        
    except Exception as e:
        print(f"Error updating LIME cache: {e}")
        raise

def clear_lime_cache(model_type: str = 'best_model'):
    """Clear the LIME cache for the specified model type"""
    cache_file = CACHE_DIR / f'lime_cache_{model_type}.pkl'
    if cache_file.exists():
        try:
            cache_file.unlink()
            print(f"LIME cache for {model_type} cleared successfully")
        except Exception as e:
            print(f"Error clearing LIME cache: {e}")