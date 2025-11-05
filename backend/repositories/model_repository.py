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


def get_training_summary() -> Dict:
    """Get the training summary from the artifacts folder"""
    try:
        with open(ARTIFACT_FOLDER / 'training_summary.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "error": "No training summary found. Please train models first.",
            "status": "not_found"
        }
    except Exception as e:
        return {
            "error": f"Error loading training summary: {str(e)}",
            "status": "error"
        }


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


def calculate_fairness_metrics(y_true, y_pred, protected_attrs, privileged_groups):
    """Calculate fairness metrics for binary classification"""
    metrics = {}
    
    for attr, values in protected_attrs.items():
        # Convert to binary arrays
        privileged_value = privileged_groups[attr]
        unprivileged_mask = values != privileged_value
        privileged_mask = values == privileged_value
        
        # Calculate probabilities for each group
        priv_prob = y_pred[privileged_mask].mean()
        unpriv_prob = y_pred[unprivileged_mask].mean()
        
        # Statistical Parity Difference
        statistical_parity_diff = priv_prob - unpriv_prob
        
        # Equal Opportunity Difference (True Positive Rate difference)
        tpr_priv = (y_pred[privileged_mask] & y_true[privileged_mask]).sum() / y_true[privileged_mask].sum()
        tpr_unpriv = (y_pred[unprivileged_mask] & y_true[unprivileged_mask]).sum() / y_true[unprivileged_mask].sum()
        equal_opportunity_diff = tpr_priv - tpr_unpriv
        
        # Disparate Impact
        disparate_impact = unpriv_prob / priv_prob if priv_prob > 0 else float('inf')
        
        metrics[attr] = {
            'statistical_parity_difference': float(statistical_parity_diff),
            'equal_opportunity_difference': float(equal_opportunity_diff),
            'disparate_impact': float(disparate_impact)
        }
    
    return metrics


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
    
    # Initialize SHAP explainer for feature importance
    try:
        import shap
        print("\nInitializing SHAP explainer...")
        # Get preprocessed training data
        X_train_processed = preprocessor.transform(X_train)
        
        # Create background dataset (subset of training data)
        background_data = shap.sample(X_train_processed, 100)  # Use 100 background samples
        
        # Create KernelExplainer which works for any model type
        def model_predict(X):
            return best_model.predict_proba(X)
        
        explainer = shap.KernelExplainer(model_predict, background_data)
        print("✓ Created SHAP KernelExplainer")
        
        # Calculate SHAP values for a subset of test data
        print("Calculating SHAP values (this may take a few minutes)...")
        X_test_subset = X_test_scaled[:50]  # Use smaller subset for demonstration
        shap_values = explainer.shap_values(X_test_subset)
        
        # Ensure SHAP values are in proper matrix format
        if isinstance(shap_values, list):
            # Multi-class case - ensure each class's values are 2D
            shap_values = [
                sv.reshape(sv.shape[0], -1) if len(sv.shape) == 1 else sv
                for sv in shap_values
            ]
        else:
            # Binary case - ensure values are 2D
            shap_values = shap_values.reshape(shap_values.shape[0], -1) if len(shap_values.shape) == 1 else shap_values
        
        # Save SHAP artifacts
        shap_data = {
            'shap_values': shap_values,
            'feature_names': FEATURE_NAMES,
            'background_data': background_data,
            'X_test_subset': X_test_subset
        }
        save_artifact(shap_data, 'shap_values_best_model.pkl')
        print("✓ SHAP values calculated and saved successfully")
        
        # Print feature importance summary
        if isinstance(shap_values, list):  # For multiclass, shap_values is a list of arrays
            print("\nFeature Importance Summary (average absolute SHAP values across classes):")
            # Compute average absolute SHAP values across all classes
            avg_shap = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
            feature_importance = dict(zip(FEATURE_NAMES, avg_shap))
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            print("\nTop 5 most important features (SHAP):")
            for feature, importance in sorted_features[:5]:
                print(f"  {feature}: {importance:.4f}")
                
    except Exception as e:
        print(f"\n⚠️ Error creating SHAP explainer: {str(e)}")
        print("Continuing without SHAP values...")
    
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
        
        # save model with gradient boosting for comparison
        print("\nTraining Gradient Boosting model on unbiased features for comparison...")
        gb_model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE
        )
        gb_model.fit(X_train_resampled, y_train_resampled)
        save_artifact(gb_model, 'gb_model_unbiased_features.pkl')

        # Calculate SHAP values for the unbiased model
        try:
            import shap
            print("\nCalculating SHAP values for unbiased model...")
            
            # Use TreeExplainer for gradient boosting model
            explainer = shap.TreeExplainer(gb_model)
            
            # Calculate SHAP values for test set subset
            X_test_subset = X_test_processed[:100]  # Use first 100 samples
            shap_values = explainer.shap_values(X_test_subset)
            
            # Ensure proper shape for multi-class case
            if isinstance(shap_values, list):
                shap_values = [
                    sv if len(sv.shape) > 1 else sv.reshape(sv.shape[0], 1)
                    for sv in shap_values
                ]
            else:
                shap_values = (
                    shap_values if len(shap_values.shape) > 1 
                    else shap_values.reshape(shap_values.shape[0], 1)
                )
            
            # Save SHAP values and related data
            save_artifact({
                'shap_values': shap_values,
                'feature_names': UNBIASED_FEATURES,
                'background_data': X_test_subset,
                'model_type': 'gradient_boosting_unbiased'
            }, 'shap_values_gb_model_unbiased_features.pkl')
            
            print("✓ SHAP values calculated and saved for unbiased model")
            
        except Exception as e:
            print(f"\n⚠️ Error calculating SHAP values: {str(e)}")
            print("Continuing without SHAP values...")

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


def train_fair_model(df: pd.DataFrame) -> Dict:
    """Train fair models with comprehensive fairness metrics"""
    summary = train_models(df)  # Use existing training pipeline
    
    # Add fairness-specific metadata
    summary.update({
        'protected_attributes': [SENSITIVE_FEATURE],
        'privileged_groups': {SENSITIVE_FEATURE: 1},
        'best_model': summary.get('best_model', 'Unknown'),
        'best_fairness_score': summary.get('results', {}).get('mitigated', {}).get('test_accuracy', 0),
        'fairness_evaluation_type': 'One-vs-Rest with DemographicParity'
    })
    
    return summary

def get_fair_model_summary() -> Dict:
    """Get fair model training summary"""
    return get_training_summary()  # Reuse existing summary which includes fairness metrics


def predict_baseline(features: List[Any]) -> Dict:
    """Make prediction using best model"""
    try:
        # Load model and preprocessor
        model = load_artifact('best_model.pkl')
        preprocessor = load_artifact('preprocessor.pkl')
        
        # Convert features to DataFrame
        df = pd.DataFrame([features], columns=FEATURE_NAMES)
        
        # Preprocess features
        X = preprocessor.transform(df)
        
        # Get prediction
        prediction = model.predict(X)[0]
        # Handle both string and integer predictions
        if isinstance(prediction, str):
            prediction_label = prediction
            # Map the string label back to its index
            prediction = CLASS_NAMES.index(prediction)
        else:
            prediction_label = CLASS_NAMES[prediction]
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            probabilities = {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)}
        
        return {
            'prediction': prediction,
            'prediction_label': prediction_label,
            'probabilities': probabilities,
            'timestamp': datetime.now().isoformat(),
            'model_used': 'best_model',  # Add model_used field
            'bias_mitigation': 'baseline'
        }
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def predict_mitigated(features: List[Any]) -> Dict:
    """Make prediction using mitigated models"""
    try:
        # Load mitigated models and preprocessor
        mitigated_data = load_artifact('mitigated_models_ovr.pkl')
        preprocessor = load_artifact('preprocessor_no_sens.pkl')
        
        # Get model info
        mitigated_models = mitigated_data['models']
        class_to_idx = mitigated_data['class_to_idx']
        unique_classes = mitigated_data['unique_classes']
        features_used = mitigated_data['features_used']
        
        # Convert features to DataFrame and select only unbiased features
        df = pd.DataFrame([features], columns=FEATURE_NAMES)
        df_unbiased = df[features_used]
        
        # Preprocess features
        X = preprocessor.transform(df_unbiased)
        
        # Get predictions from each binary classifier
        class_scores = np.zeros(len(unique_classes))
        class_probas = {}
        
        for class_idx, model_info in mitigated_models.items():
            mitigator = model_info['model']
            if hasattr(mitigator, 'predict_proba'):
                class_scores[class_idx] = mitigator.predict_proba(X)[0][1]
            else:
                class_scores[class_idx] = float(mitigator.predict(X)[0])
            
            class_name = model_info['class_name']
            class_probas[class_name] = float(class_scores[class_idx])
        
        # Get final prediction
        prediction_idx = np.argmax(class_scores)
        prediction = unique_classes[prediction_idx]
        prediction_label = CLASS_NAMES[prediction]
        
        return {
            'prediction': int(prediction),
            'prediction_label': prediction_label,
            'probabilities': class_probas,
            'timestamp': datetime.now().isoformat(),
            'model_used': 'mitigated_ovr',  # Add model_used field
            'note': 'Prediction made using One-vs-Rest mitigated models',
            'bias_mitigation': 'mitigated'
        }
    except Exception as e:
        raise Exception(f"Mitigated prediction failed: {str(e)}")

def predict_fair(features: List[Any]) -> Dict:
    """Make prediction using fair model with comprehensive fairness metrics"""
    # For now, this uses the same logic as predict_mitigated
    return predict_mitigated(features)

