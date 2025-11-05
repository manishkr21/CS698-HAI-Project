"""
Router for model explanations using LIME
"""
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Dict, Optional
from pydantic import BaseModel
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime import lime_tabular
import joblib
from repositories.model_repository import load_artifact
from utils.config import FEATURE_NAMES, CLASS_NAMES, ARTIFACT_FOLDER
from utils.lime_cache import get_lime_cache, update_lime_cache
import json

router = APIRouter(prefix="/lime", tags=["LIME Explanations"])

# Create a mapping between class names and indices
CLASS_NAME_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
CLASS_IDX_TO_NAME = {idx: name for idx, name in enumerate(CLASS_NAMES)}

class LimeModelInput(BaseModel):
    features: List[float]
    model_type: str = 'best_model'
    num_features: int = 10
    num_samples: int = 5000

class GlobalLimeInput(BaseModel):
    model_type: str = 'best_model'
    num_features: int = 10
    num_samples: int = 5000
    sample_size: int = 100

def initialize_lime_explainer(model_type: str = 'best_model'):
    """Initialize LIME explainer with training data"""
    # Load model and preprocessor
    if model_type == 'best_model':
        model = load_artifact('best_model.pkl')
        preprocessor = load_artifact('preprocessor.pkl')
    else:
        model = load_artifact('gb_model_unbiased_features.pkl')
        preprocessor = load_artifact('preprocessor_no_sens.pkl')
    
    # Create background data using uniform distribution over feature ranges
    np.random.seed(42)
    num_samples = 1000
    
    # Define feature ranges based on domain knowledge
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
    
    # Generate background data using feature ranges
    background_data = pd.DataFrame()
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
    
    # Initialize LIME explainer with background data and specific settings
    explainer = lime_tabular.LimeTabularExplainer(
        background_processed,
        feature_names=FEATURE_NAMES,
        class_names=CLASS_NAMES,
        mode='classification',
        discretize_continuous=True,
        sample_around_instance=True,  # Enable sampling around instances
        kernel_width=3,  # Increase kernel width for more stable explanations
        verbose=False,
        random_state=42
    )
    
    return {
        'explainer': explainer,
        'model': model,
        'preprocessor': preprocessor,
        'background_data': background_processed
    }

# Initialize explainers
LIME_DATA = {
    'best_model': initialize_lime_explainer('best_model'),
    'unbiased': initialize_lime_explainer('unbiased')
}

@router.post("/local/explanation")
async def get_local_explanation_data(model_input: LimeModelInput):
    """Get local LIME explanation data without visualization"""
    lime_data = LIME_DATA[model_input.model_type]
    explainer = lime_data['explainer']
    model = lime_data['model']
    preprocessor = lime_data['preprocessor']
    
    # Set feature count based on model type
    feature_count = 36 if model_input.model_type == 'best_model' else 19
    
    # Create a DataFrame with all required columns initialized to 0
    all_features = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES)
    
    # Update values for provided features
    for i, value in enumerate(model_input.features):
        if i < len(FEATURE_NAMES):
            all_features.iloc[0, i] = value
    
    # Process instance
    X_processed = preprocessor.transform(all_features)
    
    # Get prediction and convert string label to index if needed
    pred_class_str = model.predict(X_processed)[0]
    pred_class = CLASS_NAME_TO_IDX[pred_class_str] if isinstance(pred_class_str, str) else int(pred_class_str)
    
    # Get probabilities
    pred_proba = model.predict_proba(X_processed)[0]

    # Generate explanation for all classes
    explanation = explainer.explain_instance(
        X_processed[0],
        model.predict_proba,
        num_features=feature_count,
        num_samples=model_input.num_samples,
        top_labels=len(CLASS_NAMES)
    )
    
    # Get available labels first
    available_labels = explanation.available_labels()
    
    # Use the first available label if predicted class is not in explanations
    label_to_use = pred_class if pred_class in available_labels else available_labels[0]
    
    # Get feature importance and explanations
    feature_importance = explanation.as_list(label=label_to_use)
    
    return {
        "prediction": {
            "class": pred_class,
            "class_name": CLASS_NAMES[pred_class],
            "original_prediction": pred_class_str,
            "probability": float(pred_proba[pred_class]),
            "explained_class": label_to_use,
            "explained_class_name": CLASS_NAMES[label_to_use]
        },
        "explanations": [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in feature_importance
        ]
    }

@router.post("/local/plot")
async def get_local_explanation_plot(model_input: LimeModelInput):
    """Get local LIME explanation visualization"""
    lime_data = LIME_DATA[model_input.model_type]
    explainer = lime_data['explainer']
    model = lime_data['model']
    preprocessor = lime_data['preprocessor']
    
    # Set feature count based on model type
    feature_count = 36 if model_input.model_type == 'base' else 19
    
    # Create a DataFrame with all required columns initialized to 0
    all_features = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES)
    
    # Update values for provided features
    for i, value in enumerate(model_input.features):
        if i < len(FEATURE_NAMES):
            all_features.iloc[0, i] = value
    
    # Process instance
    X_processed = preprocessor.transform(all_features)
    
    # Get prediction and convert string label to index if needed
    pred_class_str = model.predict(X_processed)[0]
    pred_class = CLASS_NAME_TO_IDX[pred_class_str] if isinstance(pred_class_str, str) else int(pred_class_str)

    # Generate explanation
    explanation = explainer.explain_instance(
        X_processed[0],
        model.predict_proba,
        num_features=feature_count,
        num_samples=model_input.num_samples,
        top_labels=len(CLASS_NAMES)
    )
    
    # Get available labels and use appropriate label
    available_labels = explanation.available_labels()
    label_to_use = pred_class if pred_class in available_labels else available_labels[0]
    
    # Create visualization
    explanation.as_pyplot_figure(label=label_to_use)
    plt.title(f"LIME Explanation")
    plt.tight_layout()
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")


import re

def extract_feature_name(text: str) -> str:
    """
    Extract only the feature name from the LIME output string.
    e.g.:
    'Curricular units 1st sem (evaluations) <= 0.10' → 'Curricular units 1st sem (evaluations)'
    '-0.58 < Gender <= 1.72' → 'Gender'
    '0.19 < Curricular units 2nd sem (evaluations) <= 2.17' → 'Curricular units 2nd sem (evaluations)'
    """
    # 1) Remove numeric ranges
    cleaned = re.sub(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", "", text)

    # 2) Remove operators and spaces
    cleaned = cleaned.replace("<=", "").replace(">=", "").replace("<", "").replace(">", "").strip()

    return cleaned


@router.post("/global/explanation")
async def get_global_lime_explanation(data: GlobalLimeInput):
    """
    Compute global LIME explanation by aggregating multiple local explanations.
    Returns top features across sampled instances.
    """

    lime_data = LIME_DATA[data.model_type]

    explainer = lime_data["explainer"]
    model = lime_data["model"]
    preprocessor = lime_data["preprocessor"]
    background_data = lime_data["background_data"]

    num_features = data.num_features
    num_samples = data.num_samples
    sample_size = data.sample_size

    # Randomly sample instances
    np.random.seed(42)
    idxs = np.random.choice(len(background_data), size=sample_size, replace=False)
    sampled_instances = background_data[idxs]

    # Aggregation dictionary
    feature_scores = {f: 0.0 for f in FEATURE_NAMES}

    print("Computing global LIME explanations...")

    for i in range(sample_size):
        x = sampled_instances[i]

        try:
            exp = explainer.explain_instance(
                x,
                model.predict_proba,
                num_features=num_features,
                num_samples=num_samples,
                top_labels=len(CLASS_NAMES),
            )

            pred_class = np.argmax(model.predict_proba([x])[0])
            local_exp = exp.as_list(label=pred_class)

            print(f"Sample {i}, Predicted class: {CLASS_NAMES[pred_class]}, Explanation: {local_exp}")

            # Aggregate
            for feat, score in local_exp:
                feature = extract_feature_name(feat)

                if feature in feature_scores:
                    feature_scores[feature] += score
                else:
                    # Ignore unknown transformed features
                    pass

        except Exception as e:
            print("LIME explanation failed at sample", i, e)
            continue

    # Convert to dataframe
    df_scores = pd.DataFrame(
        [{"feature": k, "importance": v} for k, v in feature_scores.items()]
    )

    # Sort descending absolute contribution
    df_scores["abs_importance"] = df_scores["importance"].abs()
    df_scores = df_scores.sort_values("abs_importance", ascending=False)

    # Take top-N
    top_features = df_scores.head(num_features).to_dict(orient="records")

    return {
        "global_explanations": top_features,
        "num_samples_used": sample_size,
        "description": "Higher absolute importance indicates stronger influence globally.",
    }


@router.post("/global/plot")
async def get_global_lime_plot(data: GlobalLimeInput):
    """
    Generate a global LIME explanation plot by aggregating feature importance.
    Returns a PNG bar chart.
    """

    # --- Reuse global explanation logic ---
    response = await get_global_lime_explanation(data)
    top_features = response["global_explanations"]

    # Extract data
    features = [item["feature"] for item in top_features]
    importance = [item["importance"] for item in top_features]

    # --- Plotting ---
    plt.figure(figsize=(12, 0.5 * 36))
    plt.barh(features, importance)
    plt.gca().invert_yaxis()
    plt.title("Global LIME Feature Importance")
    plt.xlabel("Aggregated Importance Score")
    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
