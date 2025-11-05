"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

class BiasMitigation(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    BASELINE = "baseline"

class PredictionInput(BaseModel):
    features: List[float] = Field(..., min_items=36, max_items=36)
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [1, 1, 1, 1, 1, 1, 100, 1, 1, 1, 1, 1, 120, 0, 0, 0, 1, 1, 0, 20, 0, 0, 6, 0, 6, 13.5, 0, 0, 6, 0, 6, 13.5, 0, 10.8, 1.4, 1.74]
            }
        }

class BatchPredictionInput(BaseModel):
    samples: List[List[float]]

class StudentData(BaseModel):
    marital_status: int
    application_mode: int
    application_order: int
    course: int
    daytime_evening: int
    previous_qualification: int
    previous_qualification_grade: float
    nationality: int
    mother_qualification: int
    father_qualification: int
    mother_occupation: int
    father_occupation: int
    admission_grade: float
    displaced: int
    educational_special_needs: int
    debtor: int
    tuition_fees_up_to_date: int
    gender: int
    scholarship_holder: int
    age_at_enrollment: int
    international: int
    curricular_units_1st_sem_credited: int
    curricular_units_1st_sem_enrolled: int
    curricular_units_1st_sem_evaluations: int
    curricular_units_1st_sem_approved: int
    curricular_units_1st_sem_grade: float
    curricular_units_1st_sem_without_evaluations: int
    curricular_units_2nd_sem_credited: int
    curricular_units_2nd_sem_enrolled: int
    curricular_units_2nd_sem_evaluations: int
    curricular_units_2nd_sem_approved: int
    curricular_units_2nd_sem_grade: float
    curricular_units_2nd_sem_without_evaluations: int
    unemployment_rate: float
    inflation_rate: float
    gdp: float

class PredictionOutput(BaseModel):
    prediction: int
    prediction_label: str
    probabilities: Optional[Dict[str, float]]
    model_used: str
    bias_mitigation: str
    timestamp: str
