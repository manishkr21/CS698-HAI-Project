"""
FastAPI Backend for Student Dropout Prediction
Modular architecture with routers and repositories
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path

from routers import model_router, prediction_router, fairness_router, explanation_router, lime_router
from repositories.model_repository import train_models

# Create FastAPI app
app = FastAPI(
    title="Student Dropout Prediction API",
    description="ML API with bias mitigation, fairness evaluation, and SHAP explanations",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(model_router.router)
app.include_router(prediction_router.router)
app.include_router(fairness_router.router)
app.include_router(explanation_router.router)
app.include_router(lime_router.router)  # Add LIME router

# if not exist create artifacts directory
artifacts_path = Path('artifacts')
artifacts_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    import uvicorn
    
    # Check if dataset exists for initial training
    dataset_path = Path('dataset.csv')
    if dataset_path.exists():
        print(f"Found {dataset_path}, training models...")
        try:
            # Try different separators
            try:
                df = pd.read_csv(dataset_path, sep=';')
            except:
                df = pd.read_csv(dataset_path, sep=',')
            
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            train_models(df)
        except Exception as e:
            print(f"Failed to train on startup: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No dataset.csv found. Upload data via API to train models.")
    
    print("\n" + "="*70)
    print("Starting FastAPI server...")
    print("API Documentation: http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
