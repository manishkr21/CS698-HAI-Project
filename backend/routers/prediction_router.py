"""
Prediction router - handles all prediction-related endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
import pandas as pd
import numpy as np

from models.schemas import (
    PredictionInput, BatchPredictionInput, StudentData,
    PredictionOutput, BiasMitigation
)
from repositories.model_repository import (
    predict_baseline, predict_mitigated, load_artifact, check_model_exists, predict_fair
)
from utils.config import FEATURE_NAMES, CLASS_NAMES, SENSITIVE_FEATURE

router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post("/", response_model=dict)
def predict(input_data: PredictionInput):
    """Make single prediction using best model"""
    if not check_model_exists('best_model.pkl'):
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    
    try:
        result = predict_baseline(input_data.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/batch")
def predict_batch(input_data: BatchPredictionInput):
    """Make batch predictions"""
    if not check_model_exists('best_model.pkl'):
        raise HTTPException(status_code=503, detail="Model not loaded. Train first.")
    
    try:
        model = load_artifact('best_model.pkl')
        df = pd.DataFrame(input_data.samples, columns=FEATURE_NAMES)
        
        predictions = [int(p) for p in model.predict(df)]
        prediction_labels = [CLASS_NAMES[p] for p in predictions]
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)
            probabilities = [
                {CLASS_NAMES[i]: float(p) for i, p in enumerate(prob)}
                for prob in proba
            ]
        
        return {
            "predictions": predictions,
            "prediction_labels": prediction_labels,
            "probabilities": probabilities,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/structured")
def predict_structured(student: StudentData):
    """Predict using structured student data"""
    features = [
        student.marital_status, student.application_mode, student.application_order,
        student.course, student.daytime_evening, student.previous_qualification,
        student.previous_qualification_grade, student.nationality,
        student.mother_qualification, student.father_qualification,
        student.mother_occupation, student.father_occupation,
        student.admission_grade, student.displaced, student.educational_special_needs,
        student.debtor, student.tuition_fees_up_to_date, student.gender,
        student.scholarship_holder, student.age_at_enrollment, student.international,
        student.curricular_units_1st_sem_credited, student.curricular_units_1st_sem_enrolled,
        student.curricular_units_1st_sem_evaluations, student.curricular_units_1st_sem_approved,
        student.curricular_units_1st_sem_grade, student.curricular_units_1st_sem_without_evaluations,
        student.curricular_units_2nd_sem_credited, student.curricular_units_2nd_sem_enrolled,
        student.curricular_units_2nd_sem_evaluations, student.curricular_units_2nd_sem_approved,
        student.curricular_units_2nd_sem_grade, student.curricular_units_2nd_sem_without_evaluations,
        student.unemployment_rate, student.inflation_rate, student.gdp
    ]
    
    return predict(PredictionInput(features=features))


@router.post("/with-bias-mitigation", response_model=PredictionOutput)
def predict_with_bias_mitigation(
    input_data: PredictionInput,
    bias_mitigation: BiasMitigation = Query(BiasMitigation.BASELINE, description="Bias mitigation approach")
):
    """Make prediction with specified bias mitigation approach"""
    # try:
    if bias_mitigation == BiasMitigation.AFTER:
        if not check_model_exists('mitigated_models_ovr.pkl'):
            raise HTTPException(
                status_code=404, 
                detail="Mitigated model not available. Train with fairlearn enabled."
            )
        result = predict_mitigated(input_data.features)
        result['bias_mitigation'] = bias_mitigation.value
    else:
        result = predict_baseline(input_data.features)
        result['bias_mitigation'] = bias_mitigation.value
    
    return PredictionOutput(**result)
    # except FileNotFoundError as e:
    #     raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/compare")
def compare_predictions(input_data: PredictionInput):
    """Compare predictions between baseline and bias-mitigated models"""
    results = {}
    
    # Get baseline prediction
    try:
        baseline_result = predict_baseline(input_data.features)
        results['baseline'] = {
            'prediction': baseline_result['prediction'],
            'prediction_label': baseline_result['prediction_label'],
            'probabilities': baseline_result['probabilities']
        }
    except Exception as e:
        results['baseline'] = {'error': str(e)}
    
    # Get mitigated prediction
    try:
        if check_model_exists('mitigated_models_ovr.pkl'):
            mitigated_result = predict_mitigated(input_data.features)
            results['mitigated'] = {
                'prediction': mitigated_result['prediction'],
                'prediction_label': mitigated_result['prediction_label'],
                'probabilities': mitigated_result['probabilities']
            }
        else:
            results['mitigated'] = {'error': 'Mitigated model not available'}
    except Exception as e:
        results['mitigated'] = {'error': str(e)}
    
    return {
        'comparison': results,
        'timestamp': datetime.now().isoformat(),
        'predictions_match': results.get('baseline', {}).get('prediction') == results.get('mitigated', {}).get('prediction'),
        'note': 'Mitigated model uses One-vs-Rest approach for multi-class fairness'
    }


@router.post("/fair")
def predict_fair_model(input_data: PredictionInput):
    """
    Make prediction using fair model with comprehensive fairness metrics
    
    This endpoint uses a model trained specifically for fairness across multiple protected attributes:
    - Gender (Male vs Female)
    - International status (Domestic vs International)
    - Scholarship holder (With vs Without scholarship)
    
    The model is evaluated using:
    - Statistical Parity Difference
    - Equal Opportunity Difference
    - Disparate Impact
    
    Parameters:
    -----------
    input_data : PredictionInput
        Features must be in the same order as FEATURE_NAMES (36 features total)
        
    Returns:
    --------
    Prediction with class label, probabilities, and fairness information
    """
    if not check_model_exists('best_fair_model.pkl'):
        raise HTTPException(
            status_code=503, 
            detail="Fair model not loaded. Train fair model first using POST /train-fair"
        )
    
    try:
        result = predict_fair(input_data.features)
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Fair model artifacts missing: {str(e)}. Please train fair model first."
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Fair model prediction error: {str(e)}")
