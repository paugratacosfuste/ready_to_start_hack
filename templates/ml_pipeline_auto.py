"""
================================================================================
HACKATHON AUTO-ML PIPELINE v2.0
================================================================================

USAGE: You only need to set TWO things:
  1. DATA_SOURCE — path to CSV or URL
  2. TARGET — name of the target column

EVERYTHING else is auto-detected:
  - Feature types (numerical vs categorical)
  - Columns to drop (IDs, high-cardinality)
  - Missing value strategy
  - Preprocessor selection
  - Scoring metric (based on class balance)
  - Model selection, tuning, evaluation, and plotting

OPTIONAL overrides:
  - COLUMNS_TO_DROP: manually specify columns to remove
  - SCORING_METRIC: override auto-detected metric
  - TASK_TYPE: 'auto', 'classification', or 'regression'
  - FAST_MODE: True for quick results, False for thorough search

================================================================================
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import json
import sys
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Sklearn - Preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# Sklearn - Model Selection
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Sklearn - Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Sklearn - Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Sklearn - Metrics
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# Optional: XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. pip install xgboost to enable it.")


# =============================================================================
# ✏️  CONFIGURATION — EDIT THESE TWO LINES
# =============================================================================

DATA_SOURCE = None   # ← "path/to/data.csv" or "https://url/to/data.csv"
TARGET = None        # ← "target_column_name"

# =============================================================================
# 🔧 OPTIONAL OVERRIDES (leave as-is for auto-detection)
# =============================================================================

COLUMNS_TO_DROP = []      # e.g., ['PassengerId', 'Name'] — empty = auto-detect
SCORING_METRIC = 'auto'   # 'auto', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'r2', 'neg_mse'
TASK_TYPE = 'auto'         # 'auto', 'classification', 'regression'
FAST_MODE = True           # True = skip GridSearch (faster), False = full tuning
SAVE_MODEL = True          # Save best model as .pkl
OUTPUT_DIR = 'ml_outputs'  # Where to save plots and model


# =============================================================================
# 🤖 FULLY AUTOMATED PIPELINE — DO NOT MODIFY BELOW
# =============================================================================

def detect_task_type(y):
    """Auto-detect if this is classification or regression."""
    if y.dtype == 'object' or y.dtype.name == 'category':
        return 'classification'
    n_unique = y.nunique()
    if n_unique <= 20 and n_unique / len(y) < 0.05:
        return 'classification'
    return 'regression'


def auto_detect_drops(df, target):
    """Auto-detect columns that should be dropped."""
    drops = []
    for col in df.columns:
        if col == target:
            continue
        # Drop likely ID columns
        if 'id' in col.lower() and df[col].nunique() > 0.8 * len(df):
            drops.append(col)
            continue
        # Drop columns with >60% missing
        if df[col].isna().mean() > 0.6:
            drops.append(col)
            continue
        # Drop very high cardinality text (likely names, descriptions)
        if df[col].dtype == 'object' and df[col].nunique() > 0.5 * len(df):
            drops.append(col)
            continue
    return drops


def auto_detect_features(df, target, columns_to_drop):
    """Auto-detect numerical and categorical features."""
    feature_cols = [c for c in df.columns if c != target and c not in columns_to_drop]
    
    numerical = []
    categorical = []
    
    for col in feature_cols:
        if df[col].dtype in ['object', 'category']:
            categorical.append(col)
        elif df[col].nunique() <= 10:
            categorical.append(col)  # Low-cardinality numeric → treat as categorical
        else:
            numerical.append(col)
    
    return numerical, categorical


def build_preprocessor(numerical_features, categorical_features):
    """Build sklearn preprocessor based on detected feature types."""
    transformers = []
    
    if numerical_features:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipeline, numerical_features))
    
    if categorical_features:
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipeline, categorical_features))
    
    if not transformers:
        return 'passthrough'
    
    return ColumnTransformer(transformers)


def auto_select_metric(y, task_type):
    """Auto-select best metric based on data characteristics."""
    if task_type == 'regression':
        return 'r2'
    
    class_ratio = y.value_counts(normalize=True).min()
    if class_ratio < 0.3:
        return 'f1'
    return 'accuracy'


def get_models(task_type, preprocessor):
    """Get model dict based on task type."""
    if task_type == 'classification':
        models = {
            'LogisticRegression': Pipeline([
                ('preprocessor', preprocessor),
                ('clf', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            'DecisionTree': Pipeline([
                ('preprocessor', preprocessor),
                ('clf', DecisionTreeClassifier(random_state=42))
            ]),
            'RandomForest': Pipeline([
                ('preprocessor', preprocessor),
                ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'GradientBoost': Pipeline([
                ('preprocessor', preprocessor),
                ('clf', GradientBoostingClassifier(random_state=42))
            ]),
        }
        if HAS_XGBOOST:
            models['XGBoost'] = Pipeline([
                ('preprocessor', preprocessor),
                ('clf', XGBClassifier(n_estimators=100, learning_rate=0.1,
                                       eval_metric='logloss', random_state=42,
                                       use_label_encoder=False))
            ])
    else:  # regression
        models = {
            'LinearRegression': Pipeline([
                ('preprocessor', preprocessor),
                ('clf', LinearRegression())
            ]),
            'Ridge': Pipeline([
                ('preprocessor', preprocessor),
                ('clf', Ridge(random_state=42))
            ]),
            'RandomForest': Pipeline([
                ('preprocessor', preprocessor),
                ('clf', RandomForestRegressor(n_estimators=100, random_state=42))
            ]),
            'GradientBoost': Pipeline([
                ('preprocessor', preprocessor),
                ('clf', GradientBoostingRegressor(random_state=42))
            ]),
        }
        if HAS_XGBOOST:
            models['XGBoost'] = Pipeline([
                ('preprocessor', preprocessor),
                ('clf', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
            ])
    
    return models


def get_param_grids(task_type):
    """Get hyperparameter grids for tuning."""
    if task_type == 'classification':
        return {
            'LogisticRegression': {
                'clf__C': [0.01, 0.1, 1, 10],
                'clf__penalty': ['l1', 'l2'],
                'clf__solver': ['liblinear']
            },
            'DecisionTree': {
                'clf__max_depth': [5, 10, 20, None],
                'clf__min_samples_split': [2, 5, 10],
            },
            'RandomForest': {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [10, 20, None],
                'clf__min_samples_split': [2, 5],
            },
            'GradientBoost': {
                'clf__n_estimators': [100, 200],
                'clf__learning_rate': [0.05, 0.1, 0.2],
                'clf__max_depth': [3, 5, 7],
            },
            'XGBoost': {
                'clf__n_estimators': [100, 200],
                'clf__learning_rate': [0.05, 0.1, 0.2],
                'clf__max_depth': [3, 5, 7],
            }
        }
    else:
        return {
            'LinearRegression': {},
            'Ridge': {'clf__alpha': [0.01, 0.1, 1, 10, 100]},
            'RandomForest': {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [10, 20, None],
            },
            'GradientBoost': {
                'clf__n_estimators': [100, 200],
                'clf__learning_rate': [0.05, 0.1],
                'clf__max_depth': [3, 5],
            },
            'XGBoost': {
                'clf__n_estimators': [100, 200],
                'clf__learning_rate': [0.05, 0.1],
                'clf__max_depth': [3, 5],
            }
        }


def run_pipeline(df, target, columns_to_drop, scoring_metric, task_type,
                 fast_mode, save_model, output_dir):
    """Run the complete automated ML pipeline."""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("🤖 HACKATHON AUTO-ML PIPELINE v2.0")
    print("=" * 70)
    
    # ── Step 1: Auto-detect columns to drop ──
    auto_drops = auto_detect_drops(df, target)
    all_drops = list(set(columns_to_drop + auto_drops))
    if all_drops:
        print(f"\n📋 Dropping columns: {all_drops}")
        df = df.drop(columns=all_drops, errors='ignore')
    
    # ── Step 2: Auto-detect task type ──
    y = df[target]
    if task_type == 'auto':
        task_type = detect_task_type(y)
    print(f"\n🎯 Task type: {task_type.upper()}")
    
    # Encode target if needed
    label_encoder = None
    if task_type == 'classification' and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), name=target)
        print(f"   Encoded target classes: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    # ── Step 3: Auto-detect features ──
    X = df.drop(columns=[target])
    numerical_features, categorical_features = auto_detect_features(df, target, all_drops)
    print(f"\n📊 Features detected:")
    print(f"   Numerical ({len(numerical_features)}): {numerical_features[:10]}{'...' if len(numerical_features) > 10 else ''}")
    print(f"   Categorical ({len(categorical_features)}): {categorical_features[:10]}{'...' if len(categorical_features) > 10 else ''}")
    
    # ── Step 4: Auto-select metric ──
    if scoring_metric == 'auto':
        scoring_metric = auto_select_metric(y, task_type)
    print(f"\n📈 Scoring metric: {scoring_metric.upper()}")
    
    if task_type == 'classification':
        class_dist = y.value_counts(normalize=True)
        print(f"   Class distribution: {dict(class_dist.round(3))}")
    
    # ── Step 5: Build preprocessor ──
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    
    # ── Step 6: Train-test split ──
    split_kwargs = {'test_size': 0.2, 'random_state': 42}
    if task_type == 'classification':
        split_kwargs['stratify'] = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)
    print(f"\n✂️  Split: {X_train.shape[0]} train / {X_test.shape[0]} test")
    
    # ── Step 7: Model comparison ──
    print("\n" + "=" * 70)
    print(f"🏋️ TRAINING MODELS (optimizing {scoring_metric})")
    print("=" * 70)
    
    models = get_models(task_type, preprocessor)
    results = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            if task_type == 'classification':
                y_pred = model.predict(X_test)
                if scoring_metric == 'roc_auc':
                    y_proba = model.predict_proba(X_test)[:, 1]
                    score = roc_auc_score(y_test, y_proba)
                elif scoring_metric == 'precision':
                    score = precision_score(y_test, y_pred, zero_division=0)
                elif scoring_metric == 'recall':
                    score = recall_score(y_test, y_pred, zero_division=0)
                elif scoring_metric == 'f1':
                    score = f1_score(y_test, y_pred, zero_division=0)
                else:
                    score = accuracy_score(y_test, y_pred)
            else:
                y_pred = model.predict(X_test)
                if scoring_metric == 'r2':
                    score = r2_score(y_test, y_pred)
                elif scoring_metric == 'neg_mse':
                    score = -mean_squared_error(y_test, y_pred)
                else:
                    score = r2_score(y_test, y_pred)
            
            results[name] = score
            print(f"  ✅ {name:25s} → {scoring_metric}: {score:.4f}")
        except Exception as e:
            print(f"  ❌ {name:25s} → Error: {e}")
    
    if not results:
        print("\n❌ All models failed. Check your data.")
        return None
    
    best_model_name = max(results, key=results.get)
    print(f"\n{'⭐' * 3} BEST: {best_model_name} ({scoring_metric}: {results[best_model_name]:.4f}) {'⭐' * 3}")
    
    # ── Step 8: Hyperparameter tuning (if not fast mode) ──
    best_model = models[best_model_name]
    
    if not fast_mode:
        print("\n" + "=" * 70)
        print(f"🔧 HYPERPARAMETER TUNING: {best_model_name}")
        print("=" * 70)
        
        param_grids = get_param_grids(task_type)
        param_grid = param_grids.get(best_model_name, {})
        
        if param_grid:
            grid_search = GridSearchCV(
                estimator=best_model,
                param_grid=param_grid,
                cv=5,
                scoring=scoring_metric,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"\n  Best params: {grid_search.best_params_}")
            print(f"  Best CV score: {grid_search.best_score_:.4f}")
        else:
            print(f"  No tuning parameters for {best_model_name}")
    
    # ── Step 9: Final evaluation ──
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION")
    print("=" * 70)
    
    y_pred_test = best_model.predict(X_test)
    
    if task_type == 'classification':
        y_proba_test = best_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, zero_division=0),
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1': f1_score(y_test, y_pred_test, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_test) if len(np.unique(y_test)) == 2 else 'N/A'
        }
        
        for k, v in metrics.items():
            indicator = " ← OPTIMIZED" if k == scoring_metric else ""
            if isinstance(v, float):
                print(f"  {k:12s}: {v:.4f}{indicator}")
            else:
                print(f"  {k:12s}: {v}{indicator}")
        
        print(f"\n{classification_report(y_test, y_pred_test)}")
        
    else:  # regression
        metrics = {
            'r2': r2_score(y_test, y_pred_test),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mse': mean_squared_error(y_test, y_pred_test)
        }
        for k, v in metrics.items():
            indicator = " ← OPTIMIZED" if k == scoring_metric else ""
            print(f"  {k:12s}: {v:.4f}{indicator}")
    
    # ── Step 10: Cross-validation ──
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring=scoring_metric)
    print(f"\n  5-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # ── Step 11: Plots ──
    print("\n" + "=" * 70)
    print("📈 GENERATING PLOTS")
    print("=" * 70)
    
    if task_type == 'classification':
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Auto-ML Results — {best_model_name} (optimized for {scoring_metric.upper()})',
                     fontsize=14, fontweight='bold', y=1.02)
        
        # Confusion Matrix
        ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=axes[0, 0], cmap='Blues')
        axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
        
        # ROC Curve
        if len(np.unique(y_test)) == 2:
            RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=axes[0, 1])
            axes[0, 1].plot([0, 1], [0, 1], 'k--')
            axes[0, 1].set_title('ROC Curve', fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'ROC: Binary only', ha='center')
        
        # CV Scores
        axes[1, 0].bar(range(1, 6), cv_scores, color='steelblue', edgecolor='black')
        axes[1, 0].axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
        axes[1, 0].set_title('Cross-Validation Scores', fontweight='bold')
        axes[1, 0].legend()
        
        # Model Comparison
        model_names = list(results.keys())
        scores = list(results.values())
        colors = ['green' if n == best_model_name else 'steelblue' for n in model_names]
        bars = axes[1, 1].barh(model_names, scores, color=colors, edgecolor='black')
        axes[1, 1].set_title(f'Model Comparison ({scoring_metric})', fontweight='bold')
        for bar, score in zip(bars, scores):
            axes[1, 1].text(score + 0.002, bar.get_y() + bar.get_height()/2,
                            f'{score:.4f}', va='center', fontsize=10)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Auto-ML Results — {best_model_name}', fontsize=14, fontweight='bold')
        
        # Actual vs Predicted
        axes[0].scatter(y_test, y_pred_test, alpha=0.5)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[0].set_xlabel('Actual')
        axes[0].set_ylabel('Predicted')
        axes[0].set_title('Actual vs Predicted', fontweight='bold')
        
        # Residuals
        residuals = y_test - y_pred_test
        axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='black')
        axes[1].set_title('Residual Distribution', fontweight='bold')
        
        # Model Comparison
        model_names = list(results.keys())
        scores = list(results.values())
        colors = ['green' if n == best_model_name else 'steelblue' for n in model_names]
        axes[2].barh(model_names, scores, color=colors, edgecolor='black')
        axes[2].set_title(f'Model Comparison ({scoring_metric})', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'results_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Plots saved to {plot_path}")
    plt.close()
    
    # ── Step 12: Feature Importance ──
    try:
        clf = best_model.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importances = np.abs(clf.coef_.flatten())
        else:
            importances = None
        
        if importances is not None:
            # Get feature names after preprocessing
            if hasattr(preprocessor, 'get_feature_names_out'):
                feat_names = preprocessor.get_feature_names_out()
            else:
                feat_names = [f'Feature_{i}' for i in range(len(importances))]
            
            feat_imp = pd.DataFrame({'feature': feat_names[:len(importances)], 'importance': importances})
            feat_imp = feat_imp.sort_values('importance', ascending=True).tail(15)
            
            plt.figure(figsize=(10, 6))
            plt.barh(feat_imp['feature'], feat_imp['importance'], color='steelblue')
            plt.title(f'Top Features — {best_model_name}', fontweight='bold')
            plt.tight_layout()
            fi_path = os.path.join(output_dir, f'feature_importance_{timestamp}.png')
            plt.savefig(fi_path, dpi=150, bbox_inches='tight')
            print(f"  ✅ Feature importance saved to {fi_path}")
            plt.close()
    except Exception as e:
        print(f"  ⚠️  Feature importance: {e}")
    
    # ── Step 13: Save model ──
    if save_model:
        model_path = os.path.join(output_dir, f'best_model_{timestamp}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"  ✅ Model saved to {model_path}")
        
        # Also save a metadata file
        meta = {
            'model_name': best_model_name,
            'task_type': task_type,
            'scoring_metric': scoring_metric,
            'best_score': results[best_model_name],
            'all_scores': results,
            'metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in metrics.items()},
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'features': {'numerical': numerical_features, 'categorical': categorical_features},
            'timestamp': timestamp
        }
        meta_path = os.path.join(output_dir, f'metadata_{timestamp}.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  ✅ Metadata saved to {meta_path}")
    
    # ── Summary ──
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"""
    Dataset:        {X.shape[0]} samples, {X.shape[1]} features
    Task:           {task_type}
    Best Model:     {best_model_name}
    {scoring_metric.upper():14s}: {results[best_model_name]:.4f}
    CV Score:       {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
    Output:         {output_dir}/
    """)
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'model_comparison': results,
        'metrics': metrics,
        'task_type': task_type,
        'scoring_metric': scoring_metric,
        'cv_scores': cv_scores,
        'predictions': y_pred_test,
        'label_encoder': label_encoder,
        'feature_names': {'numerical': numerical_features, 'categorical': categorical_features}
    }


# =============================================================================
# 🚀 RUN
# =============================================================================

if __name__ == "__main__":
    
    if DATA_SOURCE is None or TARGET is None:
        print("❌ Please set DATA_SOURCE and TARGET at the top of the file.")
        print("   Example:")
        print('     DATA_SOURCE = "data/transactions.csv"')
        print('     TARGET = "is_fraud"')
        sys.exit(1)
    
    # Load data
    print(f"📂 Loading data from: {DATA_SOURCE}")
    if DATA_SOURCE.startswith('http'):
        df = pd.read_csv(DATA_SOURCE)
    else:
        if DATA_SOURCE.endswith('.xlsx') or DATA_SOURCE.endswith('.xls'):
            df = pd.read_excel(DATA_SOURCE)
        else:
            df = pd.read_csv(DATA_SOURCE)
    
    print(f"   Shape: {df.shape}")
    print(f"   Target: {TARGET}")
    print(f"   Columns: {list(df.columns)}")
    
    # Run pipeline
    results = run_pipeline(
        df=df,
        target=TARGET,
        columns_to_drop=COLUMNS_TO_DROP,
        scoring_metric=SCORING_METRIC,
        task_type=TASK_TYPE,
        fast_mode=FAST_MODE,
        save_model=SAVE_MODEL,
        output_dir=OUTPUT_DIR
    )
    
    if results:
        print("\n🎉 Done! Access results:")
        print("   results['best_model']       → trained model")
        print("   results['metrics']           → performance metrics")
        print("   results['model_comparison']  → all model scores")
        print(f"   results['predictions']       → test set predictions")
