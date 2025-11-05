"""
Router for model explanations using SHAP with artifacts
"""
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import List, Dict, Optional, Union
from pydantic import BaseModel
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from repositories.model_repository import load_artifact
from utils.config import FEATURE_NAMES, CLASS_NAMES, ARTIFACT_FOLDER
import shap
import base64

# Pydantic models for request/response validation
class ModelInfo(BaseModel):
    id: str
    name: str
    description: str

class AvailableModelsResponse(BaseModel):
    models: List[ModelInfo]
    default: Optional[str]

class FeatureImportanceResponse(BaseModel):
    feature_importance: Dict[str, float]
    model_type: str
    top_features: List[str]

class FeatureNamesResponse(BaseModel):
    feature_names: List[str]
    model_type: str

class WaterfallModelInput(BaseModel):
    class_index: int
    features: List[float]
    model_type: str = 'best_model'
    max_display: int = 20

class ForcePlotModelInput(BaseModel):
    features: List[float]
    model_type: str = 'best_model'
    class_index: Optional[int] = None

class WaterfallResponse(BaseModel):
    """Response model for waterfall plot endpoint"""
    image: str
    shap_values: dict[str, Union[dict[str, float], float]]

router = APIRouter(prefix="/explain", tags=["Explanations"])

def load_shap_data(model_type: str = 'best_model'):
    """Load SHAP values and related data from artifacts"""
    try:
        if (model_type == 'best_model'):
            shap_data = load_artifact('shap_values_best_model.pkl')
        elif (model_type == 'unbiased'):
            shap_data = load_artifact('shap_values_gb_model_unbiased_features.pkl')
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        return shap_data
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"SHAP data not found for {model_type}: {str(e)}"
        )

def format_shap_values(shap_data):
    """Ensure SHAP values are in proper matrix format"""
    shap_values = shap_data['shap_values']
    if isinstance(shap_values, list):
        # Multi-class case - ensure each class's values are 2D
        formatted_values = [
            sv if len(sv.shape) > 1 else sv.reshape(sv.shape[0], 1)
            for sv in shap_values
        ]
    else:
        # Binary case - ensure values are 2D
        formatted_values = (
            shap_values if len(shap_values.shape) > 1 
            else shap_values.reshape(shap_values.shape[0], 1)
        )
    
    shap_data['shap_values'] = formatted_values
    return shap_data

# Initialize SHAP values with proper formatting
SHAP_VALUES = format_shap_values(load_shap_data())

@router.get("/available-models", response_model=AvailableModelsResponse)
async def get_available_explanation_models():
    """Get list of models with available SHAP explanations"""
    available_models = []
    
    # Check for best model SHAP values
    if (ARTIFACT_FOLDER / 'shap_values_best_model.pkl').exists():
        available_models.append(ModelInfo(
            id='best_model',
            name='Best Model (All Features)',
            description='SHAP explanations for the best performing model using all features'
        ))
    
    # Check for unbiased model SHAP values
    if (ARTIFACT_FOLDER / 'shap_values_gb_model_unbiased_features.pkl').exists():
        available_models.append(ModelInfo(
            id='unbiased',
            name='Unbiased Model',
            description='SHAP explanations for model trained on unbiased features only'
        ))
    
    return AvailableModelsResponse(
        models=available_models,
        default='best_model' if available_models else None
    )

@router.get("/global/feature-importance", response_model=FeatureImportanceResponse)
async def get_global_feature_importance(
    model_type: str = Query('best_model', description="Model type for SHAP explanations"),
    top_k: Optional[int] = Query(None, description="Number of top features to return")
):
    """Get global feature importance using SHAP values"""
    shap_values = SHAP_VALUES['shap_values']
    feature_names = SHAP_VALUES['feature_names']
    
    # For multi-class, average SHAP values across classes and samples
    if isinstance(shap_values, list):
        feature_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        feature_importance = np.abs(shap_values).mean(axis=0)
    
    # Create importance dictionary
    global_importance = {
        feature: float(importance)
        for feature, importance in zip(feature_names, feature_importance)
    }
    
    # Sort by importance
    sorted_importance = dict(sorted(
        global_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    # Filter top K if specified
    if top_k:
        sorted_importance = dict(list(sorted_importance.items())[:top_k])
    
    return FeatureImportanceResponse(
        feature_importance=sorted_importance,
        model_type=model_type,
        top_features=list(sorted_importance.keys())
    )

def plot_multiclass_shap_beeswarm(shap_values, background, feature_names=None, max_display=36):
    """
    shap_values: list of np.arrays, one per class → [(N,F), (N,F), (N,F)]
    background:  np.array (M,F)   BUT M may != N
    """

    # ✅ convert → (C, N, F)
    shap_values_arr = np.array(shap_values)   # list → array
    shap_mean = np.mean(shap_values_arr, axis=0)   # (N,F)

    print("Averaged SHAP shape:", shap_mean.shape)
    print("Background shape:", background.shape)

    N_shap = shap_mean.shape[0]
    N_bg   = background.shape[0]

    # ✅ Fix size mismatch by slicing
    if N_shap != N_bg:
        print(f"⚠ Resizing background: using first {N_shap} rows of background")
        background = background[:N_shap]

    # ✅ Re-check
    if shap_mean.shape != background.shape:
        raise ValueError(
            f"Final shape mismatch: SHAP {shap_mean.shape} vs background {background.shape}"
        )

    shap_ex = shap.Explanation(
        values=shap_mean,
        data=background,
        feature_names=feature_names,
    )

    shap.plots.beeswarm(shap_ex, max_display=max_display, show=False)

@router.get("/global/summary-plot", response_class=StreamingResponse)
async def get_global_summary_plot(
    model_type: str = Query('best_model', description="Model type for SHAP explanations"),
    plot_type: str = Query('bar', description="Type of plot: 'bar' or 'beeswarm'"),
    max_display: int = Query(20, description="Maximum number of features to display")
):
    """Generate global SHAP summary plot"""

    shap_data = load_shap_data(model_type)   # <- your function
    shap_values = shap_data['shap_values']
    background = shap_data['background_data']
    feature_names = shap_data['feature_names']

    print(f"Generating {plot_type} summary plot for model type: {model_type}")

    plt.figure(figsize=(12, 0.5 * len(feature_names)))

    # -----------------------------
    # ✅ BAR PLOT
    # -----------------------------
    if plot_type == "bar":
        if isinstance(shap_values, list):   # multiclass
            shap_values_arr = np.array(shap_values)      # (C,N,F)
            shap_mean = np.mean(np.abs(shap_values_arr), axis=0)   # (N,F)

            shap.summary_plot(
                shap_mean,
                background,
                feature_names=feature_names,
                plot_type='bar',
                max_display=max_display,
                show=False
            )
        else:   # single class
            shap.summary_plot(
                shap_values,
                background,
                feature_names=feature_names,
                plot_type='bar',
                max_display=max_display,
                show=False
            )

    # -----------------------------
    # ✅ BEESWARM (MULTICLASS SAFE)
    # -----------------------------
    elif plot_type == "beeswarm":
        if isinstance(shap_values, list):     # multiclass
            plot_multiclass_shap_beeswarm(
                shap_values=shap_values,
                background=background,
                feature_names=feature_names,
                max_display=max_display
            )
        else:     # single class
            shap_ex = shap.Explanation(
                values=shap_values,
                data=background,
                feature_names=feature_names
            )

            shap.plots.beeswarm(shap_ex, max_display=max_display, show=False)

    # -----------------------------
    # FIGURE OUTPUT
    # -----------------------------
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@router.post("/local/waterfall", response_model=WaterfallResponse)
async def get_local_waterfall_plot(model_input: WaterfallModelInput):
    """Generate local SHAP waterfall plot and values for a specific class"""
    shap_data = load_shap_data(model_input.model_type)
    shap_values = shap_data['shap_values']
    feature_names = shap_data['feature_names']
    
    X = pd.DataFrame([model_input.features], columns=feature_names)
    
    # Prepare the SHAP values response
    if isinstance(shap_values, list):
        # Multi-class case
        explainer = shap.KernelExplainer(lambda x: x, shap_data['background_data'])
        explainer.expected_value = [0] * len(shap_values)
        current_shap_values = shap_values[model_input.class_index][0]
        base_value = float(explainer.expected_value[model_input.class_index])
    else:
        explainer = shap.KernelExplainer(lambda x: x, shap_data['background_data'])
        current_shap_values = shap_values[0]
        base_value = float(explainer.expected_value)

    # Create feature value dictionary
    feature_values = {
        name: float(value) 
        for name, value in zip(feature_names, current_shap_values)
    }
    
    # Sort by absolute value for consistent ordering with plot
    feature_values = dict(sorted(
        feature_values.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )[:model_input.max_display])

    # Generate plot
    plt.figure(figsize=(12, len(feature_names) * 0.5))
    
    if isinstance(shap_values, list):
        shap.waterfall_plot(
            shap.Explanation(
                values=current_shap_values,
                base_values=explainer.expected_value[model_input.class_index],
                data=X.iloc[0],
                feature_names=feature_names
            ),
            show=False,
            max_display=model_input.max_display
        )
    else:
        shap.waterfall_plot(
            shap.Explanation(
                values=current_shap_values,
                base_values=explainer.expected_value,
                data=X.iloc[0],
                feature_names=feature_names
            ),
            show=False,
            max_display=model_input.max_display
        )
    
    title = f"Feature Contributions for {CLASS_NAMES[model_input.class_index]} Prediction"
    if model_input.model_type == 'unbiased':
        title += " (Unbiased Features)"
    plt.title(title)
    plt.tight_layout()
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Convert image buffer to base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Return properly structured response
    return WaterfallResponse(
        image=image_base64,
        shap_values={
            "feature_values": feature_values,
            "base_value": base_value
        }
    )

@router.post("/local/force-plot", response_class=StreamingResponse)
async def get_local_force_plot(model_input: ForcePlotModelInput):
    """Generate local SHAP force plot"""
    shap_data = load_shap_data(model_input.model_type)
    shap_values = shap_data['shap_values']
    feature_names = shap_data['feature_names']
    
    X = pd.DataFrame([model_input.features], columns=feature_names)
    
    plt.figure(figsize=(20, 3))
    if isinstance(shap_values, list):
        if model_input.class_index is None:
            # If no class specified, use the predicted class
            model_input.class_index = np.argmax([sv[0].sum() for sv in shap_values])
        
        explainer = shap.KernelExplainer(lambda x: x, shap_data['background_data'])
        explainer.expected_value = [0] * len(shap_values)
        
        shap.force_plot(
            explainer.expected_value[model_input.class_index],
            shap_values[model_input.class_index][0],
            X.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False       
        )
    else:
        explainer = shap.KernelExplainer(lambda x: x, shap_data['background_data'])
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            X.iloc[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
    
    title = f"SHAP Force Plot"
    if model_input.class_index is not None:
        title += f" for {CLASS_NAMES[model_input.class_index]}"
    if model_input.model_type == 'unbiased':
        title += " (Unbiased Features)"
    plt.title(title)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")

@router.get("/feature-names", response_model=FeatureNamesResponse)
async def get_feature_names(
    model_type: str = Query('best_model', description="Model type for feature names")
):
    """Get feature names for the specified model type"""
    shap_data = load_shap_data(model_type)
    return FeatureNamesResponse(
        feature_names=shap_data['feature_names'],
        model_type=model_type
    )
