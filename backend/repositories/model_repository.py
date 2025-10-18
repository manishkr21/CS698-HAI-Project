"""
Repository layer for model operations
Contains all business logic for training, prediction, and fairness evaluation
"""
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE

from utils.config import (
    ARTIFACT_FOLDER, RANDOM_STATE, TARGET_COL, SENSITIVE_FEATURE,
    CLASS_NAMES, FEATURE_NAMES, FAIRLEARN_AVAILABLE, UNBIASED_FEATURES
)

if FAIRLEARN_AVAILABLE:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    from fairlearn.metrics import MetricFrame, demographic_parity_difference


def save_artifact(obj: Any, filename: str) -> Path:
    """Save artifact to disk"""
    path = ARTIFACT_FOLDER / filename
    joblib.dump(obj, path)
    return path


def load_artifact(filename: str) -> Any:
    """Load artifact from disk"""
    path = ARTIFACT_FOLDER / filename
    if not path.exists():
        raise FileNotFoundError(f"Artifact {filename} not found")
    return joblib.load(path)


def check_model_exists(model_name: str) -> bool:
    """Check if a model artifact exists"""
    return (ARTIFACT_FOLDER / model_name).exists()


def get_available_models() -> List[str]:
    """Get list of available trained models"""
    models = []
    for model_file in ARTIFACT_FOLDER.glob("*_pipeline.pkl"):
        models.append(model_file.stem.replace('_pipeline', ''))
    return models


def cross_validate_model(model, X, y, X_test, y_test, cv):
    """
    Perform cross validation and return detailed metrics including test set performance
    """
    # Keep data as DataFrames (don't convert to numpy arrays)
    # The sklearn pipeline needs DataFrames to work with named columns
    
    # Initialize lists to store metrics
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    test_accuracy_scores = []  # Track test set accuracy per fold
    
    print(f"\nFold-wise performance:")
    print("-" * 50)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        # Split data for this fold - keep as DataFrames/Series
        if isinstance(X, pd.DataFrame):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            
        if isinstance(y, pd.Series):
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        else:
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train_fold, y_train_fold)
        
        # Get predictions
        val_pred = model.predict(X_val_fold)
        test_pred = model.predict(X_test)
        
        # Calculate validation metrics
        fold_accuracy = accuracy_score(y_val_fold, val_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val_fold, val_pred, average='weighted', zero_division=0
        )
        
        # Calculate test set accuracy
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # Store metrics
        accuracy_scores.append(fold_accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        test_accuracy_scores.append(test_accuracy)
        
        # Print fold results
        print(f"Fold {fold}:")
        print(f"  Validation Accuracy: {fold_accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    return {
        'accuracy': np.array(accuracy_scores),
        'precision': np.array(precision_scores),
        'recall': np.array(recall_scores),
        'f1': np.array(f1_scores),
        'test_accuracy': np.array(test_accuracy_scores)
    }


def train_models(df: pd.DataFrame) -> Dict:
    """Train baseline and mitigated models with SMOTE"""
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("STARTING MODEL TRAINING WITH SMOTE")
    print("="*70)
    
    # Prepare data
    X = df[FEATURE_NAMES].copy()
    y = df[TARGET_COL]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nOriginal Training Data Distribution:")
    print(f"Total training samples: {len(y_train)}")
    print(y_train.value_counts().to_dict())
    
    # Preprocessing
    numeric_features = [f for f in FEATURE_NAMES if f != SENSITIVE_FEATURE]
    preprocessor = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), numeric_features)],
        remainder='passthrough'
    )
    
    # Fit and transform training data
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Apply SMOTE to balance training data
    print(f"\nApplying SMOTE to balance training data...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"\nResampled Training Data Distribution (after SMOTE):")
    print(f"Total training samples: {len(y_train_resampled)}")
    print(pd.Series(y_train_resampled).value_counts().to_dict())
    
    # Convert back to DataFrame for compatibility
    X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=RANDOM_STATE
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE
        ),
        'Logistic Regression': LogisticRegression(
            solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE, multi_class='multinomial'
        )
    }
    
    results = {}
    best_accuracy = 0
    best_model_name = None
    best_model = None
    
    # Create k-fold cross validator
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Training {name} (with SMOTE-resampled data)")
        print(f"{'='*70}")
        
        # Initialize lists to store metrics
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        test_accuracy_scores = []
        
        print(f"\nFold-wise performance:")
        print("-" * 50)
        
        # Perform cross-validation on resampled data
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_resampled, y_train_resampled), 1):
            # Split resampled data
            X_train_fold = X_train_resampled[train_idx]
            X_val_fold = X_train_resampled[val_idx]
            y_train_fold = y_train_resampled[train_idx]
            y_val_fold = y_train_resampled[val_idx]
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Get predictions
            val_pred = model.predict(X_val_fold)
            test_pred = model.predict(X_test_scaled)
            
            # Calculate validation metrics
            fold_accuracy = accuracy_score(y_val_fold, val_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val_fold, val_pred, average='weighted', zero_division=0
            )
            
            # Calculate test set accuracy
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # Store metrics
            accuracy_scores.append(fold_accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            test_accuracy_scores.append(test_accuracy)
            
            # Print fold results
            print(f"Fold {fold}:")
            print(f"  Validation Accuracy: {fold_accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
        
        # Print summary statistics
        print(f"\n{name} Cross-Validation Summary (SMOTE-resampled data):")
        print(f"  Validation Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
        print(f"  Validation Precision: {np.mean(precision_scores):.4f} (+/- {np.std(precision_scores):.4f})")
        print(f"  Validation Recall: {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})")
        print(f"  Validation F1-Score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
        print(f"  Test Accuracy (avg across folds): {np.mean(test_accuracy_scores):.4f} (+/- {np.std(test_accuracy_scores):.4f})")
        
        # Train final model on full resampled training set
        model.fit(X_train_resampled, y_train_resampled)
        
        # Final test evaluation
        y_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, 
                                      target_names=CLASS_NAMES, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{name} Final Model (trained on full SMOTE-resampled data):")
        print(f"  Test Accuracy: {test_accuracy:.4f}")

        results[name] = {
            'cv_accuracy_mean': float(np.mean(accuracy_scores)),
            'cv_accuracy_std': float(np.std(accuracy_scores)),
            'cv_precision_mean': float(np.mean(precision_scores)),
            'cv_precision_std': float(np.std(precision_scores)),
            'cv_recall_mean': float(np.mean(recall_scores)),
            'cv_recall_std': float(np.std(recall_scores)),
            'cv_f1_mean': float(np.mean(f1_scores)),
            'cv_f1_std': float(np.std(f1_scores)),
            'cv_test_accuracy_mean': float(np.mean(test_accuracy_scores)),
            'cv_test_accuracy_std': float(np.std(test_accuracy_scores)),
            'test_accuracy': float(test_accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'smote_applied': True,
            'original_train_size': len(y_train),
            'resampled_train_size': len(y_train_resampled)
        }
        
        # Save model
        save_artifact(model, f'{name.lower().replace(" ", "_")}_model.pkl')
        
        # Track best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_name = name
            best_model = model
    
    # Save best model and preprocessor
    save_artifact(best_model, 'best_model.pkl')
    save_artifact(preprocessor, 'preprocessor.pkl')
    
    # Bias mitigation with SMOTE
    mitigated_results = train_mitigated_models_with_smote(
        X_train, X_test, y_train, y_test, best_model, best_accuracy, preprocessor
    )
    if mitigated_results:
        results['mitigated'] = mitigated_results
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models': list(models.keys()),
        'best_model': best_model_name,
        'results': results,
        'feature_names': FEATURE_NAMES,
        'class_names': CLASS_NAMES,
        'training_time_seconds': (datetime.now() - start_time).total_seconds(),
        'smote_applied': True,
        'original_train_samples': len(y_train),
        'resampled_train_samples': len(y_train_resampled),
        'test_samples': len(y_test),
        'class_distribution_before_smote': y_train.value_counts().to_dict(),
        'class_distribution_after_smote': pd.Series(y_train_resampled).value_counts().to_dict()
    }
    
    with open(ARTIFACT_FOLDER / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ TRAINING COMPLETE")
    print(f"Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    print(f"SMOTE Applied: Training samples increased from {len(y_train)} to {len(y_train_resampled)}")
    print(f"Summary saved to artifacts/training_summary.json")
    print(f"{'='*70}\n")
    
    return summary


def train_mitigated_models(X_train, X_test, y_train, y_test, best_pipeline, best_accuracy) -> Optional[Dict]:
    """Train bias-mitigated models using One-vs-Rest approach"""
    if not FAIRLEARN_AVAILABLE or SENSITIVE_FEATURE not in X_train.columns:
        return None
    
    print("\n" + "="*70)
    print("APPLYING BIAS MITIGATION (One-vs-Rest for Multi-Class)")
    print("="*70)
    
    try:
        # Extract sensitive feature
        sensitive_train = X_train[SENSITIVE_FEATURE]
        sensitive_test = X_test[SENSITIVE_FEATURE]
        
        X_train_no_sens = X_train.drop(columns=[SENSITIVE_FEATURE])
        X_test_no_sens = X_test.drop(columns=[SENSITIVE_FEATURE])
        
        # Create preprocessor without sensitive feature
        numeric_features = [f for f in FEATURE_NAMES if f != SENSITIVE_FEATURE]
        numeric_features_no_sens = [f for f in numeric_features if f in X_train_no_sens.columns]
        preprocessor_no_sens = ColumnTransformer(
            transformers=[('scaler', StandardScaler(), numeric_features_no_sens)],
            remainder='passthrough'
        )
        
        # Preprocess
        X_train_processed = preprocessor_no_sens.fit_transform(X_train_no_sens)
        X_test_processed = preprocessor_no_sens.transform(X_test_no_sens)
        
        print(f"Training data shape: {X_train_processed.shape}")
        print(f"Target classes: {sorted(y_train.unique())}")
        
        # Train One-vs-Rest mitigated models
        mitigated_models = {}
        unique_classes = sorted(y_train.unique())
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        for target_class in unique_classes:
            class_idx = class_to_idx[target_class]
            print(f"\nTraining mitigated model for class '{target_class}' (index {class_idx})...")
            
            # Convert to binary problem
            y_train_binary = (y_train == target_class).astype(int)
            
            print(f"  Binary distribution - Train: {y_train_binary.value_counts().to_dict()}")
            
            # Create base estimator
            base_estimator = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=RANDOM_STATE
            )
            
            # Apply mitigation
            mitigator = ExponentiatedGradient(
                estimator=base_estimator,
                constraints=DemographicParity(),
                max_iter=50,
                nu=1e-6
            )
            
            mitigator.fit(X_train_processed, y_train_binary, 
                        sensitive_features=sensitive_train)
            
            mitigated_models[class_idx] = {
                'model': mitigator,
                'class_name': target_class
            }
            
            # Evaluate binary classifier
            y_pred_binary = mitigator.predict(X_test_processed)
            y_test_binary = (y_test == target_class).astype(int)
            acc = accuracy_score(y_test_binary, y_pred_binary)
            print(f"  Binary accuracy for class '{target_class}': {acc:.4f}")
        
        # Save mitigated models
        save_artifact({
            'models': mitigated_models,
            'class_to_idx': class_to_idx,
            'unique_classes': unique_classes
        }, 'mitigated_models_ovr.pkl')
        save_artifact(preprocessor_no_sens, 'preprocessor_no_sens.pkl')
        
        # Evaluate on test set
        mitigated_accuracy, group_metrics = evaluate_mitigated_models(
            mitigated_models, unique_classes, X_test_processed, 
            y_test, sensitive_test, best_pipeline, X_test
        )
        
        mitigated_report = classification_report(
            y_test, 
            [unique_classes[np.argmax([mitigated_models[i]['model'].predict_proba(X_test_processed)[j, 1] 
                                       if hasattr(mitigated_models[i]['model'], 'predict_proba') 
                                       else mitigated_models[i]['model'].predict(X_test_processed)[j] 
                                       for i in range(len(unique_classes))])] 
             for j in range(len(X_test_processed))],
            output_dict=True,
            target_names=CLASS_NAMES,
            zero_division=0
        )
        
        print(f"\n{'='*70}")
        print(f"✓ BIAS MITIGATION COMPLETE")
        print(f"  Baseline accuracy: {best_accuracy:.4f}")
        print(f"  Mitigated accuracy: {mitigated_accuracy:.4f}")
        print(f"  Accuracy change: {(mitigated_accuracy - best_accuracy):.4f}")
        print(f"{'='*70}\n")
        
        return {
            'test_accuracy': mitigated_accuracy,
            'classification_report': mitigated_report,
            'group_metrics': group_metrics,
            'approach': 'One-vs-Rest with DemographicParity per class',
            'note': 'Multi-class bias mitigation using binary classifiers'
        }
        
    except Exception as e:
        print(f"\n❌ Mitigation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_mitigated_models(mitigated_models, unique_classes, X_test_processed, 
                              y_test, sensitive_test, baseline_model, X_test) -> Tuple[float, Dict]:
    """Evaluate mitigated models and compute fairness metrics"""
    # Get predictions
    class_scores = np.zeros((len(X_test_processed), len(mitigated_models)))
    
    for class_idx, model_info in mitigated_models.items():
        mitigator = model_info['model']
        if hasattr(mitigator, 'predict_proba'):
            class_scores[:, class_idx] = mitigator.predict_proba(X_test_processed)[:, 1]
        else:
            class_scores[:, class_idx] = mitigator.predict(X_test_processed)
    
    y_pred_mitigated_idx = np.argmax(class_scores, axis=1)
    y_pred_mitigated = np.array([unique_classes[idx] for idx in y_pred_mitigated_idx])
    mitigated_accuracy = float(accuracy_score(y_test, y_pred_mitigated))
    
    print(f"\n✓ Mitigated model (OvR) test accuracy: {mitigated_accuracy:.4f}")
    
    # Compute group metrics
    def accuracy_metric(y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
    # Baseline group metrics
    y_pred_baseline = baseline_model.predict(X_test)
    mf_baseline = MetricFrame(
        metrics=accuracy_metric,
        y_true=y_test,
        y_pred=y_pred_baseline,
        sensitive_features=sensitive_test
    )
    
    # Mitigated group metrics
    mf_mitigated = MetricFrame(
        metrics=accuracy_metric,
        y_true=y_test,
        y_pred=y_pred_mitigated,
        sensitive_features=sensitive_test
    )
    
    print(f"\nBaseline group metrics:")
    print(f"  {mf_baseline.by_group.to_dict()}")
    print(f"\nMitigated group metrics:")
    print(f"  {mf_mitigated.by_group.to_dict()}")
    
    # Calculate demographic parity
    dp_metrics = {}
    for target_class in sorted(y_test.unique()):
        y_pred_baseline_binary = (y_pred_baseline == target_class).astype(int)
        y_pred_mitigated_binary = (y_pred_mitigated == target_class).astype(int)
        
        dp_diff_baseline = demographic_parity_difference(
            y_pred_baseline_binary, y_pred_baseline_binary,
            sensitive_features=sensitive_test
        )
        dp_diff_mitigated = demographic_parity_difference(
            y_pred_mitigated_binary, y_pred_mitigated_binary,
            sensitive_features=sensitive_test
        )
        
        dp_metrics[target_class] = {
            'baseline': float(dp_diff_baseline),
            'mitigated': float(dp_diff_mitigated),
            'improvement': float(abs(dp_diff_baseline) - abs(dp_diff_mitigated))
        }
    
    group_metrics = {
        'baseline_by_group': {str(k): float(v) for k, v in mf_baseline.by_group.to_dict().items()},
        'mitigated_by_group': {str(k): float(v) for k, v in mf_mitigated.by_group.to_dict().items()},
        'baseline_overall': float(mf_baseline.overall),
        'mitigated_overall': float(mf_mitigated.overall),
        'fairness_metrics': {
            'demographic_parity_by_class': dp_metrics
        }
    }
    
    return mitigated_accuracy, group_metrics


def train_mitigated_models_with_smote(X_train, X_test, y_train, y_test, best_model, best_accuracy, preprocessor) -> Optional[Dict]:
    """Train bias-mitigated models with SMOTE using One-vs-Rest approach with only unbiased features"""
    if not FAIRLEARN_AVAILABLE or SENSITIVE_FEATURE not in X_train.columns:
        print("\n⚠️  Fairlearn not available or sensitive feature missing. Skipping bias mitigation.")
        return None
    
    print("\n" + "="*70)
    print("APPLYING BIAS MITIGATION WITH SMOTE (One-vs-Rest for Multi-Class)")
    print("Using ONLY UNBIASED FEATURES for fair model training")
    print("="*70)
    
    try:
        # Extract sensitive feature
        sensitive_train = X_train[SENSITIVE_FEATURE]
        sensitive_test = X_test[SENSITIVE_FEATURE]
        
        # Use only UNBIASED_FEATURES for training mitigated models
        print(f"\nFiltering features:")
        print(f"  Total features available: {len(X_train.columns)}")
        print(f"  Unbiased features to use: {len(UNBIASED_FEATURES)}")
        print(f"  Unbiased features: {UNBIASED_FEATURES}")
        
        # Select only unbiased features
        X_train_unbiased = X_train[UNBIASED_FEATURES].copy()
        X_test_unbiased = X_test[UNBIASED_FEATURES].copy()
        
        # Create preprocessor for unbiased features only
        preprocessor_unbiased = ColumnTransformer(
            transformers=[('scaler', StandardScaler(), UNBIASED_FEATURES)],
            remainder='passthrough'
        )
        
        # Preprocess
        X_train_processed = preprocessor_unbiased.fit_transform(X_train_unbiased)
        X_test_processed = preprocessor_unbiased.transform(X_test_unbiased)
        
        # Apply SMOTE to training data
        print(f"\nApplying SMOTE to balance training data before bias mitigation...")
        print(f"Original training samples: {len(y_train)}")
        print(f"Class distribution: {y_train.value_counts().to_dict()}")
        
        smote = SMOTE(random_state=RANDOM_STATE)
        
        # Create temporary dataframe with sensitive feature for resampling
        # Convert to DataFrame with string column names to avoid mixed type issues
        X_train_with_sens = pd.DataFrame(
            X_train_processed, 
            columns=[str(i) for i in range(X_train_processed.shape[1])]
        )
        X_train_with_sens['sensitive'] = sensitive_train.values
        
        # Apply SMOTE
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_with_sens, y_train)
        
        # Separate sensitive feature after resampling
        sensitive_train_resampled = X_train_resampled['sensitive'].values
        X_train_resampled = X_train_resampled.drop(columns=['sensitive']).values
        
        print(f"After SMOTE - Training samples: {len(y_train_resampled)}")
        print(f"Class distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")
        
        # Train One-vs-Rest mitigated models
        print("\nTraining One-vs-Rest Bias-Mitigated Models (with unbiased features only)...")
        mitigated_models = {}
        unique_classes = sorted(y_train_resampled.unique())
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        for target_class in unique_classes:
            class_idx = class_to_idx[target_class]
            print(f"\nTraining mitigated model for class '{target_class}' (index {class_idx})...")
            
            # Convert to binary problem
            y_train_binary = (y_train_resampled == target_class).astype(int)
            
            print(f"  Binary distribution: {pd.Series(y_train_binary).value_counts().to_dict()}")
            
            # Create base estimator
            base_estimator = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=RANDOM_STATE
            )
            
            # Apply mitigation
            mitigator = ExponentiatedGradient(
                estimator=base_estimator,
                constraints=DemographicParity(),
                max_iter=50,
                nu=1e-6
            )
            
            mitigator.fit(X_train_resampled, y_train_binary, 
                        sensitive_features=sensitive_train_resampled)
            
            mitigated_models[class_idx] = {
                'model': mitigator,
                'class_name': target_class
            }
            
            # Evaluate binary classifier
            y_pred_binary = mitigator.predict(X_test_processed)
            y_test_binary = (y_test == target_class).astype(int)
            acc = accuracy_score(y_test_binary, y_pred_binary)
            print(f"  Binary accuracy: {acc:.4f}")
        
        # Save mitigated models
        save_artifact({
            'models': mitigated_models,
            'class_to_idx': class_to_idx,
            'unique_classes': unique_classes,
            'features_used': UNBIASED_FEATURES
        }, 'mitigated_models_ovr.pkl')
        save_artifact(preprocessor_unbiased, 'preprocessor_no_sens.pkl')
        
        # Evaluate on test set
        print("\nEvaluating mitigated models on test set...")
        class_scores = np.zeros((len(X_test_processed), len(mitigated_models)))
        
        for class_idx, model_info in mitigated_models.items():
            mitigator = model_info['model']
            if hasattr(mitigator, 'predict_proba'):
                class_scores[:, class_idx] = mitigator.predict_proba(X_test_processed)[:, 1]
            else:
                class_scores[:, class_idx] = mitigator.predict(X_test_processed)
        
        y_pred_mitigated_idx = np.argmax(class_scores, axis=1)
        y_pred_mitigated = np.array([unique_classes[idx] for idx in y_pred_mitigated_idx])
        mitigated_accuracy = float(accuracy_score(y_test, y_pred_mitigated))
        
        mitigated_report = classification_report(
            y_test, 
            y_pred_mitigated,
            output_dict=True,
            target_names=CLASS_NAMES,
            zero_division=0
        )
        
        # Compute fairness metrics
        def accuracy_metric(y_true, y_pred):
            return accuracy_score(y_true, y_pred)
        
        # Baseline group metrics (using preprocessor to transform X_test)
        X_test_scaled = preprocessor.transform(X_test)
        y_pred_baseline = best_model.predict(X_test_scaled)
        
        mf_baseline = MetricFrame(
            metrics=accuracy_metric,
            y_true=y_test,
            y_pred=y_pred_baseline,
            sensitive_features=sensitive_test
        )
        
        # Mitigated group metrics
        mf_mitigated = MetricFrame(
            metrics=accuracy_metric,
            y_true=y_test,
            y_pred=y_pred_mitigated,
            sensitive_features=sensitive_test
        )
        
        group_metrics = {
            'baseline_by_group': {str(k): float(v) for k, v in mf_baseline.by_group.to_dict().items()},
            'mitigated_by_group': {str(k): float(v) for k, v in mf_mitigated.by_group.to_dict().items()},
            'baseline_overall': float(mf_baseline.overall),
            'mitigated_overall': float(mf_mitigated.overall)
        }
        
        # Calculate custom fairness metrics for each class using one-vs-rest
        print("\n" + "="*70)
        print("CALCULATING CUSTOM FAIRNESS METRICS")
        print("="*70)
        
        custom_fairness_metrics = {}
        for target_class in unique_classes:
            print(f"\nCalculating fairness metrics for class '{target_class}'...")
            
            # Convert to binary (one-vs-rest)
            y_test_binary = (y_test == target_class).astype(int)
            y_pred_baseline_binary = (y_pred_baseline == target_class).astype(int)
            y_pred_mitigated_binary = (y_pred_mitigated == target_class).astype(int)
            
            # Define protected attributes (Gender in this case)
            protected_attrs = {SENSITIVE_FEATURE: sensitive_test.values}
            
            # Define privileged groups (assuming Gender=1 is privileged, adjust as needed)
            # You may need to adjust this based on your data encoding
            privileged_groups = {SENSITIVE_FEATURE: 1}
            
            # Calculate fairness metrics for baseline
            baseline_fairness = calculate_fairness_metrics(
                y_test_binary, 
                y_pred_baseline_binary,
                protected_attrs,
                privileged_groups
            )
            
            # Calculate fairness metrics for mitigated
            mitigated_fairness = calculate_fairness_metrics(
                y_test_binary,
                y_pred_mitigated_binary,
                protected_attrs,
                privileged_groups
            )
            
            custom_fairness_metrics[target_class] = {
                'baseline': baseline_fairness,
                'mitigated': mitigated_fairness
            }
            
            # Log the metrics
            print(f"  Baseline - Statistical Parity Diff: {baseline_fairness[SENSITIVE_FEATURE]['statistical_parity_difference']:.4f}")
            print(f"  Mitigated - Statistical Parity Diff: {mitigated_fairness[SENSITIVE_FEATURE]['statistical_parity_difference']:.4f}")
            print(f"  Baseline - Equal Opportunity Diff: {baseline_fairness[SENSITIVE_FEATURE]['equal_opportunity_difference']:.4f}")
            print(f"  Mitigated - Equal Opportunity Diff: {mitigated_fairness[SENSITIVE_FEATURE]['equal_opportunity_difference']:.4f}")
            print(f"  Baseline - Disparate Impact: {baseline_fairness[SENSITIVE_FEATURE]['disparate_impact']:.4f}")
            print(f"  Mitigated - Disparate Impact: {mitigated_fairness[SENSITIVE_FEATURE]['disparate_impact']:.4f}")
        
        # Add custom fairness metrics to group_metrics
        group_metrics['custom_fairness_metrics'] = custom_fairness_metrics
        
        print(f"\n{'='*70}")
        print(f"✓ BIAS MITIGATION WITH SMOTE COMPLETE")
        print(f"  Features used: UNBIASED_FEATURES only ({len(UNBIASED_FEATURES)} features)")
        print(f"  Baseline accuracy: {best_accuracy:.4f}")
        print(f"  Mitigated accuracy: {mitigated_accuracy:.4f}")
        print(f"  Accuracy change: {(mitigated_accuracy - best_accuracy):.4f}")
        print(f"  Training samples (after SMOTE): {len(y_train_resampled)}")
        print(f"  Custom fairness metrics calculated for all classes")
        print(f"{'='*70}\n")
        
        return {
            'test_accuracy': mitigated_accuracy,
            'classification_report': mitigated_report,
            'group_metrics': group_metrics,
            'approach': 'One-vs-Rest with DemographicParity + SMOTE',
            'smote_applied': True,
            'original_train_size': len(y_train),
            'resampled_train_size': len(y_train_resampled),
            'features_used': UNBIASED_FEATURES,
            'num_features': len(UNBIASED_FEATURES),
            'note': 'Multi-class bias mitigation using binary classifiers with SMOTE and UNBIASED_FEATURES only'
        }
        
    except Exception as e:
        print(f"\n❌ Mitigation with SMote failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_baseline(features: List[float]) -> Dict:
    """Make prediction using baseline model"""
    model = load_artifact('best_model.pkl')
    df = pd.DataFrame([features], columns=FEATURE_NAMES)
    
    prediction = model.predict(df)[0]
    
    probabilities = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(df)[0]
        probabilities = {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)}
    
    return {
        "prediction": prediction,
        "prediction_label": prediction,
        "probabilities": probabilities,
        "model_used": "Best Model (Baseline)",
        "timestamp": datetime.now().isoformat()
    }


def predict_mitigated(features: List[float]) -> Dict:
    """Make prediction using mitigated model (uses only unbiased features)"""
    mitigated_models = load_artifact('mitigated_models_ovr.pkl')
    preprocessor = load_artifact('preprocessor_no_sens.pkl')
    
    # Create dataframe with all features
    df = pd.DataFrame([features], columns=FEATURE_NAMES)
    
    # Extract only unbiased features (same as used during training)
    df_unbiased = df[UNBIASED_FEATURES]
    X_processed = preprocessor.transform(df_unbiased)
    
    # Get scores from each binary classifier
    class_scores = np.zeros(len(mitigated_models['models']))
    for class_idx, model_info in mitigated_models['models'].items():
        mitigator = model_info['model']
        if hasattr(mitigator, 'predict_proba'):
            class_scores[class_idx] = mitigator.predict_proba(X_processed)[0, 1]
        else:
            class_scores[class_idx] = mitigator.predict(X_processed)[0]
    
    prediction_idx = int(np.argmax(class_scores))
    prediction_label = CLASS_NAMES[prediction_idx]
    probabilities = {CLASS_NAMES[i]: float(score) for i, score in enumerate(class_scores)}
    
    # Normalize probabilities
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {k: v/total for k, v in probabilities.items()}
    
    return {
        "prediction": prediction_label,
        "prediction_label": prediction_label,
        "probabilities": probabilities,
        "model_used": "Mitigated Model (One-vs-Rest with Unbiased Features)",
        "features_used": UNBIASED_FEATURES,
        "num_features": len(UNBIASED_FEATURES),
        "timestamp": datetime.now().isoformat()
    }


def get_training_summary() -> Dict:
    """Load training summary from disk"""
    summary_path = ARTIFACT_FOLDER / 'training_summary.json'
    
    if not summary_path.exists():
        raise FileNotFoundError("No training summary found")
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def calculate_fairness_metrics(y_true, y_pred, protected_attributes, privileged_groups):
    """
    Calculate fairness metrics across protected attributes
    
    Parameters:
    -----------
    y_true : array-like (binary: 0/1)
        True labels
    y_pred : array-like (binary: 0/1)
        Predicted labels
    protected_attributes : dict
        Dictionary with protected attribute arrays
    privileged_groups : dict 
        Dictionary defining privileged values for each attribute
    
    Returns:
    --------
    dict : Fairness metrics for each protected attribute
    """
    
    # Ensure arrays are flat and numeric
    y_true = np.asarray(y_true).ravel().astype(int)  # flatten and convert to int
    y_pred = np.asarray(y_pred).ravel().astype(int)
    
    metrics = {}   # Store fairness metrics
    for attr_name, attr_values in protected_attributes.items():  # iterate over protected attributes
        attr_values = np.asarray(attr_values).ravel()    # flatten
        
        # Masks
        privileged_mask = attr_values == privileged_groups[attr_name]  # privileged group mask
        unprivileged_mask = ~privileged_mask   # unprivileged group mask
        
        # Statistical Parity (predicted positive rates)
        pred_privileged = y_pred[privileged_mask].mean() if privileged_mask.any() else 0.0  # avoid division by zero
        pred_unprivileged = y_pred[unprivileged_mask].mean() if unprivileged_mask.any() else 0.0  # avoid division by zero
        stat_parity = pred_privileged - pred_unprivileged  # difference in positive rates
        
        # Equal Opportunity (TPR difference)
        pos_mask = y_true == 1  # positive class mask
        tpr_privileged = y_pred[privileged_mask & pos_mask].mean() if np.any(privileged_mask & pos_mask) else 0.0  # avoid division by zero
        tpr_unprivileged = y_pred[unprivileged_mask & pos_mask].mean() if np.any(unprivileged_mask & pos_mask) else 0.0  # avoid division by zero
        eq_opp = tpr_privileged - tpr_unprivileged  # difference in TPR
        
        # Disparate Impact
        di = (pred_unprivileged / pred_privileged) if pred_privileged > 0 else np.inf  # avoid division by zero
        
        metrics[attr_name] = {
            'statistical_parity_difference': float(stat_parity),  # convert to float for JSON serialization
            'equal_opportunity_difference': float(eq_opp), # convert to float for JSON serialization
            'disparate_impact': float(di), # convert to float for JSON serialization
            'privileged_positive_rate': float(pred_privileged),
            'unprivileged_positive_rate': float(pred_unprivileged),
            'privileged_tpr': float(tpr_privileged),
            'unprivileged_tpr': float(tpr_unprivileged)
        }
    
    return metrics


def cross_validate_model_with_fairness(model, X, y, X_test, y_test, protected_attributes, privileged_groups, cv):
    """
    Perform cross-validation and compute fairness metrics on test set
    
    Parameters:
    -----------
    model : sklearn estimator
        The model to train
    X : DataFrame
        Training features
    y : array-like
        Training labels (encoded for multi-class)
    X_test : DataFrame
        Test features
    y_test : array-like
        Test labels (binary for fairness evaluation)
    protected_attributes : dict
        Dictionary with protected attribute arrays for test set
    privileged_groups : dict
        Dictionary defining privileged values for each attribute
    cv : cross-validator
        Cross-validation strategy
        
    Returns:
    --------
    dict : Performance and fairness metrics
    """
    # Run normal CV metrics
    base_metrics = cross_validate_model(model, X, y, X_test, y_test, cv)
    
    # Train final model on full training data
    model.fit(X, y)
    y_pred = model.predict(X_test)
    
    # Compute fairness metrics
    fairness_metrics = calculate_fairness_metrics(
        y_true=y_test,
        y_pred=y_pred,
        protected_attributes=protected_attributes,
        privileged_groups=privileged_groups
    )
    
    return {**base_metrics, 'fairness': fairness_metrics}


def train_fair_model(df: pd.DataFrame) -> Dict:
    """
    Train a fair model with comprehensive fairness metrics across multiple protected attributes
    
    This function trains models specifically for fairness evaluation by:
    1. Converting multi-class to binary (Dropout vs Others)
    2. Evaluating fairness across Gender, International status, and Scholarship holder
    3. Computing Statistical Parity, Equal Opportunity, and Disparate Impact
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with features and target
        
    Returns:
    --------
    dict : Training summary with fairness metrics
    """
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("TRAINING FAIR MODEL WITH COMPREHENSIVE FAIRNESS METRICS")
    print("="*70)
    
    # Prepare data
    X = df[FEATURE_NAMES].copy()
    y = df[TARGET_COL]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nOriginal Training Data Distribution:")
    print(f"Total training samples: {len(y_train)}")
    print(y_train.value_counts().to_dict())
    
    # Encode labels for multi-class training
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Save label encoder
    save_artifact(label_encoder, 'label_encoder_fair.pkl')
    
    print(f"\nEncoded classes: {dict(enumerate(label_encoder.classes_))}")
    
    # Convert to binary for fairness evaluation: Dropout=1, else=0
    dropout_class = list(label_encoder.classes_).index("Dropout")
    y_train_binary = (y_train_encoded == dropout_class).astype(int)
    y_test_binary = (y_test_encoded == dropout_class).astype(int)
    
    print(f"\nBinary distribution for fairness evaluation (Dropout vs Others):")
    print(f"  Training: {pd.Series(y_train_binary).value_counts().to_dict()}")
    print(f"  Test: {pd.Series(y_test_binary).value_counts().to_dict()}")
    
    # Preprocessing
    numeric_features = [f for f in FEATURE_NAMES if f != SENSITIVE_FEATURE]
    preprocessor = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), numeric_features)],
        remainder='passthrough'
    )
    
    # Fit and transform
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Convert to DataFrame for easier access
    X_train_df = pd.DataFrame(X_train_scaled, columns=FEATURE_NAMES)
    X_test_df = pd.DataFrame(X_test_scaled, columns=FEATURE_NAMES)
    
    # Define protected attributes for fairness evaluation (test set)
    protected_attributes = {
        'Gender': X_test['Gender'].values,
        'International': X_test['International'].values,
        'Scholarship holder': X_test['Scholarship holder'].values
    }
    
    # Define privileged groups
    privileged_groups = {
        'Gender': 1,                    # Male (assuming 1=Male, 0=Female)
        'International': 0,             # Domestic students
        'Scholarship holder': 1         # Scholarship holders
    }
    
    print(f"\nProtected attributes for fairness evaluation:")
    print(f"  Gender: Male (1) vs Female (0)")
    print(f"  International: Domestic (0) vs International (1)")
    print(f"  Scholarship holder: With scholarship (1) vs Without (0)")
    
    # Train multiple models with fairness metrics
    models = {
        'Fair Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'Fair Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE
        ),
        'Fair Logistic Regression': LogisticRegression(
            solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE, 
            multi_class='multinomial', class_weight='balanced'
        )
    }
    
    results = {}
    best_fairness_score = float('inf')  # Lower is better for fairness
    best_model_name = None
    best_model = None
    
    # Create k-fold cross validator
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Training {name} with Fairness Evaluation")
        print(f"{'='*70}")
        
        # Cross-validate with fairness metrics
        metrics = cross_validate_model_with_fairness(
            model=model,
            X=X_train_df,
            y=y_train_encoded,          # Multi-class for performance
            X_test=X_test_df,
            y_test=y_test_binary,       # Binary for fairness
            protected_attributes=protected_attributes,
            privileged_groups=privileged_groups,
            cv=kfold
        )
        
        # Performance metrics
        print(f"\n{name} Performance Metrics:")
        print(f"  Validation Accuracy: {metrics['accuracy'].mean():.4f} (+/- {metrics['accuracy'].std():.4f})")
        print(f"  Test Accuracy: {metrics['test_accuracy'].mean():.4f} (+/- {metrics['test_accuracy'].std():.4f})")
        
        # Fairness metrics
        print(f"\n{name} Fairness Metrics (Dropout prediction):")
        for attr, fair_metrics in metrics['fairness'].items():
            print(f"\n  {attr} Group Fairness:")
            print(f"    Statistical Parity Difference: {fair_metrics['statistical_parity_difference']:.4f}")
            print(f"    Equal Opportunity Difference: {fair_metrics['equal_opportunity_difference']:.4f}")
            print(f"    Disparate Impact: {fair_metrics['disparate_impact']:.4f}")
            print(f"    Privileged Positive Rate: {fair_metrics['privileged_positive_rate']:.4f}")
            print(f"    Unprivileged Positive Rate: {fair_metrics['unprivileged_positive_rate']:.4f}")
        
        # Calculate overall fairness score (average absolute values of unfairness metrics)
        fairness_score = np.mean([
            abs(fair_metrics['statistical_parity_difference']) + 
            abs(fair_metrics['equal_opportunity_difference'])
            for fair_metrics in metrics['fairness'].values()
        ])
        
        print(f"\n  Overall Fairness Score: {fairness_score:.4f} (lower is better)")
        
        # Store results
        results[name] = {
            'cv_accuracy_mean': float(metrics['accuracy'].mean()),
            'cv_accuracy_std': float(metrics['accuracy'].std()),
            'test_accuracy_mean': float(metrics['test_accuracy'].mean()),
            'test_accuracy_std': float(metrics['test_accuracy'].std()),
            'fairness_metrics': metrics['fairness'],
            'fairness_score': float(fairness_score),
            'model_type': 'fair_cv',
            'protected_attributes': list(protected_attributes.keys()),
            'privileged_groups': privileged_groups
        }
        
        # Save model
        model.fit(X_train_df, y_train_encoded)
        save_artifact(model, f'{name.lower().replace(" ", "_")}_fair_model.pkl')
        
        # Track best fair model
        if fairness_score < best_fairness_score:
            best_fairness_score = fairness_score
            best_model_name = name
            best_model = model
    
    # Save best fair model and preprocessor
    save_artifact(best_model, 'best_fair_model.pkl')
    save_artifact(preprocessor, 'preprocessor_fair.pkl')
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'fair_with_cv',
        'models': list(models.keys()),
        'best_model': best_model_name,
        'best_fairness_score': float(best_fairness_score),
        'results': results,
        'feature_names': FEATURE_NAMES,
        'class_names': CLASS_NAMES,
        'label_encoding': {int(k): v for k, v in enumerate(label_encoder.classes_)},
        'dropout_class_index': int(dropout_class),
        'protected_attributes': list(protected_attributes.keys()),
        'privileged_groups': privileged_groups,
        'training_time_seconds': (datetime.now() - start_time).total_seconds(),
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'fairness_evaluation_type': 'Binary (Dropout vs Others)',
        'note': 'Models trained with class weighting and evaluated for fairness across multiple protected attributes'
    }
    
    # Save fair model summary
    with open(ARTIFACT_FOLDER / 'fair_model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ FAIR MODEL TRAINING COMPLETE")
    print(f"Best Fair Model: {best_model_name} (Fairness Score: {best_fairness_score:.4f})")
    print(f"Summary saved to artifacts/fair_model_summary.json")
    print(f"{'='*70}\n")
    
    return summary


def predict_fair(features: List[float]) -> Dict:
    """
    Make prediction using fair model
    
    Parameters:
    -----------
    features : List[float]
        Input features in the same order as FEATURE_NAMES
        
    Returns:
    --------
    dict : Prediction results with probabilities
    """
    model = load_artifact('best_fair_model.pkl')
    preprocessor = load_artifact('preprocessor_fair.pkl')
    label_encoder = load_artifact('label_encoder_fair.pkl')
    
    # Create dataframe
    df = pd.DataFrame([features], columns=FEATURE_NAMES)
    
    # Preprocess
    X_processed = preprocessor.transform(df)
    X_df = pd.DataFrame(X_processed, columns=FEATURE_NAMES)
    
    # Get prediction (encoded)
    prediction_encoded = model.predict(X_df)[0]
    
    # Decode prediction
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get probabilities
    probabilities = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_df)[0]
        probabilities = {
            label_encoder.inverse_transform([i])[0]: float(p) 
            for i, p in enumerate(proba)
        }
    
    return {
        "prediction": prediction_label,
        "prediction_label": prediction_label,
        "prediction_encoded": int(prediction_encoded),
        "probabilities": probabilities,
        "model_used": "Best Fair Model (with Comprehensive Fairness Metrics)",
        "protected_attributes_evaluated": ["Gender", "International", "Scholarship holder"],
        "timestamp": datetime.now().isoformat()
    }


def get_fair_model_summary() -> Dict:
    """Load fair model summary from disk"""
    summary_path = ARTIFACT_FOLDER / 'fair_model_summary.json'
    
    if not summary_path.exists():
        raise FileNotFoundError("No fair model summary found. Train fair model first.")
    
    with open(summary_path, 'r') as f:
        return json.load(f)
