"""
Model management router - handles model info, training, and metrics
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
from datetime import datetime

from repositories.model_repository import (
    get_available_models, check_model_exists, get_training_summary,
    train_models, train_fair_model, get_fair_model_summary
)
from utils.config import FEATURE_NAMES, CLASS_NAMES, SENSITIVE_FEATURE, TARGET_COL, FAIRLEARN_AVAILABLE

router = APIRouter(tags=["Models"])


@router.get("/")
def root():
    """API information"""
    return {
        "name": "Student Dropout Prediction API",
        "version": "2.0.0",
        "description": "ML API with One-vs-Rest bias mitigation for multi-class classification",
        "endpoints": {
            "GET /health": "Health check",
            "GET /models": "List available models",
            "GET /metrics": "Get training metrics",
            "GET /features": "Get feature information",
            "POST /predict": "Single prediction",
            "POST /predict/batch": "Batch predictions",
            "POST /predict/structured": "Prediction with named features",
            "POST /predict/with-bias-mitigation": "Prediction with bias mitigation option",
            "POST /predict/compare": "Compare baseline vs mitigated predictions",
            "POST /train": "Train models with uploaded data",
            "GET /fairness/summary": "Get fairness evaluation summary"
        },
        "fairlearn_available": FAIRLEARN_AVAILABLE,
        "mitigation_approach": "One-vs-Rest for multi-class problems"
    }


@router.get("/health")
def health_check():
    """Health check"""
    models_available = check_model_exists('best_model.pkl')
    mitigated_available = check_model_exists('mitigated_models_ovr.pkl')
    
    return {
        "status": "healthy",
        "models_loaded": models_available,
        "mitigated_model_loaded": mitigated_available,
        "fairlearn_enabled": FAIRLEARN_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/features")
def get_features():
    """Get feature information"""
    return {
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "class_names": CLASS_NAMES,
        "sensitive_feature": SENSITIVE_FEATURE
    }


@router.get("/models")
def list_models():
    """List available models"""
    models = get_available_models()
    
    return {
        "available_models": models,
        "best_model_available": check_model_exists('best_model.pkl'),
        "mitigated_model_available": check_model_exists('mitigated_models_ovr.pkl')
    }


@router.get("/metrics")
def get_metrics():
    """Get training metrics and model performance"""
    try:
        data = get_training_summary()
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No training metrics found. Train models first.")


@router.post("/train")
async def train_endpoint(file: UploadFile = File(...)):
    """Train models with uploaded CSV"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files supported")
    
    try:
        # Read CSV
        contents = await file.read()
        
        # Try to read with semicolon delimiter first
        df = pd.read_csv(pd.io.common.BytesIO(contents), sep=';')
        
        print(f"Loaded dataset with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Validate required columns
        required_cols = FEATURE_NAMES + [TARGET_COL]
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}. Expected columns: {required_cols}"
            )
        
        # Train models
        summary = train_models(df)
        
        return {
            "status": "success",
            "message": "Models trained successfully",
            "summary": summary
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@router.post("/train-fair")
async def train_fair_endpoint(file: UploadFile = File(...)):
    """
    Train fair models with comprehensive fairness metrics
    
    This endpoint trains models specifically optimized for fairness by:
    - Converting multi-class problem to binary (Dropout vs Others)
    - Evaluating fairness across multiple protected attributes (Gender, International, Scholarship)
    - Computing Statistical Parity, Equal Opportunity, and Disparate Impact metrics
    - Using class weighting to handle imbalanced data
    
    Parameters:
    -----------
    file : CSV file
        Must contain all features in FEATURE_NAMES plus Target column
        
    Returns:
    --------
    Training summary with fairness metrics for each model
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files supported")
    
    try:
        # Read CSV
        contents = await file.read()
        
        # Try different separators
        try:
            df = pd.read_csv(pd.io.common.BytesIO(contents), sep=';')
        except:
            df = pd.read_csv(pd.io.common.BytesIO(contents), sep=',')
        
        print(f"Loaded dataset for fair model training with shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Validate required columns
        required_cols = FEATURE_NAMES + [TARGET_COL]
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}. Expected columns: {required_cols}"
            )
        
        # Train fair model
        summary = train_fair_model(df)
        
        return {
            "status": "success",
            "message": "Fair models trained successfully with comprehensive fairness metrics",
            "model_type": "fair_with_cv",
            "protected_attributes": summary['protected_attributes'],
            "privileged_groups": summary['privileged_groups'],
            "best_model": summary['best_model'],
            "best_fairness_score": summary['best_fairness_score'],
            "fairness_evaluation_type": summary['fairness_evaluation_type'],
            "summary": summary
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Fair model training error: {str(e)}")


@router.get("/metrics/fair")
def get_fair_metrics():
    """
    Get fair model training metrics and fairness evaluation
    
    Returns comprehensive fairness metrics including:
    - Statistical Parity Difference: Difference in positive prediction rates
    - Equal Opportunity Difference: Difference in True Positive Rates
    - Disparate Impact: Ratio of positive rates between groups
    - Per-model fairness scores across Gender, International, and Scholarship attributes
    """
    try:
        data = get_fair_model_summary()
        return data
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail="No fair model metrics found. Train fair model first using POST /train-fair"
        )
