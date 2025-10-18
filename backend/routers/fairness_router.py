"""
Fairness evaluation router - handles fairness metrics and evaluation
"""
from fastapi import APIRouter, HTTPException
from repositories.model_repository import get_training_summary
from utils.config import FAIRLEARN_AVAILABLE

router = APIRouter(prefix="/fairness", tags=["Fairness"])


@router.get("/summary")
def get_fairness_summary():
    """Get fairness evaluation summary from training"""
    try:
        summary = get_training_summary()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No training summary found. Train models first."
        )
    
    if 'mitigated' not in summary.get('results', {}):
        raise HTTPException(
            status_code=404,
            detail="No fairness evaluation available. Ensure fairlearn is installed and retrain models."
        )
    
    mitigated_data = summary['results']['mitigated']
    
    return {
        'mitigated_accuracy': mitigated_data.get('test_accuracy'),
        'baseline_accuracy': summary['results'][summary['best_model']]['test_accuracy'],
        'group_metrics': mitigated_data.get('group_metrics'),
        'classification_report': mitigated_data.get('classification_report'),
        'approach': mitigated_data.get('approach'),
        'fairlearn_available': FAIRLEARN_AVAILABLE,
        'timestamp': summary.get('timestamp')
    }


@router.get("/custom-metrics")
def get_custom_fairness_metrics():
    """Get custom fairness metrics (statistical parity, equal opportunity, disparate impact)"""
    try:
        summary = get_training_summary()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No training summary found. Train models first."
        )
    
    if 'mitigated' not in summary.get('results', {}):
        raise HTTPException(
            status_code=404,
            detail="No fairness evaluation available. Ensure fairlearn is installed and retrain models."
        )
    
    mitigated_data = summary['results']['mitigated']
    group_metrics = mitigated_data.get('group_metrics', {})
    custom_metrics = group_metrics.get('custom_fairness_metrics', {})
    
    if not custom_metrics:
        raise HTTPException(
            status_code=404,
            detail="Custom fairness metrics not found. Please retrain models to generate these metrics."
        )
    
    return {
        'custom_fairness_metrics': custom_metrics,
        'sensitive_feature': 'Gender',
        'classes': list(custom_metrics.keys()),
        'metrics_included': [
            'statistical_parity_difference',
            'equal_opportunity_difference', 
            'disparate_impact',
            'privileged_positive_rate',
            'unprivileged_positive_rate',
            'privileged_tpr',
            'unprivileged_tpr'
        ],
        'timestamp': summary.get('timestamp')
    }
