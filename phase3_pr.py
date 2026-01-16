"""
Phase 3 PR: Fast Model Training for PR (OPTIMIZED - Using Pre-calculated Percentages)

This is the PR-specific implementation for training models on PR (Points + Rebounds).

Key Optimizations:
1. Downloads comprehensive PR file with ALL threshold columns ONCE (not 33 times)
2. Reuses the DataFrame for all 33 models (thresholds 8-40)
3. Just filters columns per threshold (no repeated downloads)
4. Pre-calculated percentages make training ~10x faster
5. Expected time: 15-20 minutes for all 33 models

Author: NBA Props Prediction System
Date: 2025-12-31
"""

import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
from typing import Tuple, Dict

# Setup logger
logger = logging.getLogger(__name__)


def train_model_for_threshold(df_all: pd.DataFrame, threshold: int) -> Tuple[xgb.XGBClassifier, float, Dict]:
    """
    Train XGBoost model for specific PR threshold.

    Uses pre-loaded comprehensive DataFrame, just filters columns.

    Args:
        df_all: Comprehensive DataFrame with all PR percentage columns
        threshold: PR threshold for binary classification

    Returns:
        Tuple of (model, precision, metrics_dict)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training model for {threshold}+ PR")
    logger.info(f"{'='*80}")

    # Step 1: Create target variable
    y = (df_all['PR'] >= threshold).astype(int)
    logger.info(f"  Target distribution: {y.value_counts().to_dict()}")

    # Step 2: Select features for this threshold
    # Base features (same for all thresholds)
    base_features = [
        'last_5_avg', 'last_10_avg', 'last_20_avg',
        'season_avg', 'last_season_avg',
        'lineup_average', 'h2h_avg',
        'opp_strength'
    ]

    # Threshold-specific percentage columns
    threshold_pct_features = [
        f'last_5_pct_{threshold}',
        f'last_10_pct_{threshold}',
        f'last_20_pct_{threshold}',
        f'season_pct_{threshold}',
        f'last_season_pct_{threshold}',
        f'lineup_pct_{threshold}',
        f'h2h_pct_{threshold}'
    ]

    # Combine all features
    feature_cols = base_features + threshold_pct_features

    # Check which features exist in DataFrame
    available_features = [col for col in feature_cols if col in df_all.columns]
    missing_features = [col for col in feature_cols if col not in df_all.columns]

    if missing_features:
        logger.warning(f"  Missing features: {missing_features}")

    # Select features from comprehensive DataFrame
    X = df_all[available_features].copy()

    # Fill NaN values with 0
    X = X.fillna(0)

    logger.info(f"  Features shape: {X.shape}")
    logger.info(f"  Features: {available_features}")

    # Step 3: Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Step 4: Train XGBoost
    logger.info(f"  Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Step 5: Evaluate
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate precision (TP / (FP + TP))
    precision = tp / (fp + tp) if (fp + tp) > 0 else 0.0

    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    metrics = {
        'threshold': threshold,
        'precision': precision,
        'accuracy': accuracy,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'n_features': X.shape[1]
    }

    logger.info(f"  {threshold}+ PR: Precision = {precision:.4f}")
    logger.info(f"  Accuracy = {accuracy:.4f}")
    logger.info(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

    return model, precision, metrics


def run_phase_3_pr(s3_handler, threshold_start: int = 8, threshold_end: int = 40) -> Tuple[bool, Dict]:
    """
    Execute Phase 3 PR (OPTIMIZED): Model Training for PR using pre-calculated percentages.

    This is the PR-specific implementation that:
    1. Downloads comprehensive PR file ONCE
    2. Reuses DataFrame for all 33 models (thresholds 8-40)
    3. Trains models sequentially (one threshold at a time)
    4. Saves each model to S3 immediately in separate PR folder

    Prerequisites:
    - Phase 2.5 PR must be completed (pre-calculated percentage file must exist)

    Args:
        s3_handler: S3Handler instance
        threshold_start: Starting threshold (default: 8 for PR, representing o/u 7.5)
        threshold_end: Ending threshold (default: 40 for PR, so thresholds 8-40 inclusive representing o/u 7.5-39.5)

    Returns:
        Tuple of (success: bool, stats: dict)
    """
    logger.info("=" * 80)
    logger.info("PHASE 3 PR (OPTIMIZED): FAST MODEL TRAINING FOR PR")
    logger.info("=" * 80)
    logger.info(f"Stat Type: PR (Points + Rebounds)")
    logger.info(f"Threshold range: {threshold_start}-{threshold_end}")
    logger.info(f"Total models to train: {threshold_end - threshold_start + 1}")
    logger.info("=" * 80)

    try:
        from s3_utils import S3_PLAYER_BUCKET, S3_MODEL_BUCKET

        # ============================================================================
        # STEP 1: Load Comprehensive PR Data (ONCE)
        # ============================================================================
        logger.info("\nSTEP 1: Loading comprehensive PR data with all percentage columns...")

        data_key = f'processed_data_pr/processed_with_pr_pct_{threshold_start}-{threshold_end - 1}.csv'
        logger.info(f"  Downloading: s3://{S3_PLAYER_BUCKET}/{data_key}")

        df_all = s3_handler.download_dataframe(S3_PLAYER_BUCKET, data_key)

        if df_all is None:
            error_msg = (
                f"Failed to download pre-calculated PR data from S3.\n"
                f"  Expected file: s3://{S3_PLAYER_BUCKET}/{data_key}\n"
                f"  Please run Phase 2.5 PR first to generate this file."
            )
            logger.error(error_msg)
            return False, {'error': error_msg}

        logger.info(f"✓ Loaded comprehensive PR data:")
        logger.info(f"  Rows: {len(df_all):,}")
        logger.info(f"  Columns: {len(df_all.columns)}")
        logger.info(f"  Memory usage: {df_all.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

        # DEBUG: Check what percentage columns actually exist
        pct_columns = [col for col in df_all.columns if '_pct_' in col]
        logger.info(f"\n  DEBUG: Found {len(pct_columns)} percentage columns")

        # Check specifically for threshold start
        last_season_col = f'last_season_pct_{threshold_start}'
        lineup_col = f'lineup_pct_{threshold_start}'
        logger.info(f"  DEBUG: '{last_season_col}' exists: {last_season_col in df_all.columns}")
        logger.info(f"  DEBUG: '{lineup_col}' exists: {lineup_col in df_all.columns}")

        if last_season_col in df_all.columns:
            logger.info(f"  DEBUG: {last_season_col} - NaN count: {df_all[last_season_col].isna().sum()}, Non-zero count: {(df_all[last_season_col] != 0).sum()}")
        if lineup_col in df_all.columns:
            logger.info(f"  DEBUG: {lineup_col} - NaN count: {df_all[lineup_col].isna().sum()}, Non-zero count: {(df_all[lineup_col] != 0).sum()}")

        # Verify required columns exist
        required_base_cols = ['PR', 'last_5_avg', 'last_10_avg', 'last_20_avg', 'season_avg']
        missing_cols = [col for col in required_base_cols if col not in df_all.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            logger.error(error_msg)
            return False, {'error': error_msg}

        # ============================================================================
        # STEP 2: Train Models (One Threshold at a Time)
        # ============================================================================
        logger.info("\nSTEP 2: Training models for each PR threshold...")
        logger.info(f"  Using the SAME DataFrame for all {threshold_end - threshold_start + 1} models")
        logger.info(f"  This is the key optimization - no repeated downloads!")

        model_results = {}

        for threshold in range(threshold_start, threshold_end + 1):
            # Train model for this threshold
            model, precision, metrics = train_model_for_threshold(df_all, threshold)

            # Save model to S3 in PR folder
            logger.info(f"  Saving model to S3...")
            model_filename = f'xgb_pr_{threshold}plus_precision_{precision:.4f}.pkl'
            model_key = f'models_pr/{model_filename}'

            # Serialize model
            model_bytes = pickle.dumps(model)

            # Upload to S3
            s3_handler.s3_client.put_object(
                Bucket=S3_MODEL_BUCKET,
                Key=model_key,
                Body=model_bytes
            )

            logger.info(f"  ✓ Model saved: s3://{S3_MODEL_BUCKET}/{model_key}")

            # Store metrics
            model_results[threshold] = {
                'filename': model_filename,
                's3_key': model_key,
                **metrics
            }

        # ============================================================================
        # STEP 3: Generate Summary
        # ============================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3 PR (OPTIMIZED) - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Stat Type: PR (Points + Rebounds)")
        logger.info(f"Models trained: {len(model_results)}")
        logger.info(f"Threshold range: {threshold_start}-{threshold_end}")
        logger.info("")
        logger.info("Precision scores by threshold:")

        # Sort by threshold and display
        for threshold in sorted(model_results.keys()):
            metrics = model_results[threshold]
            precision = metrics['precision']
            accuracy = metrics['accuracy']
            logger.info(
                f"  {threshold:2d}+ PR: "
                f"Precision={precision:.4f}, "
                f"Accuracy={accuracy:.4f}, "
                f"Features={metrics['n_features']}"
            )

        # Calculate average precision
        avg_precision = np.mean([m['precision'] for m in model_results.values()])
        avg_accuracy = np.mean([m['accuracy'] for m in model_results.values()])

        logger.info("")
        logger.info(f"Average Precision: {avg_precision:.4f}")
        logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
        logger.info("=" * 80)
        logger.info("✓ PHASE 3 PR COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        stats = {
            'stat_type': 'PR',
            'total_models': len(model_results),
            'threshold_range': f"{threshold_start}-{threshold_end}",
            'avg_precision': avg_precision,
            'avg_accuracy': avg_accuracy,
            'model_results': model_results,
            'data_rows': len(df_all),
            'data_columns': len(df_all.columns)
        }

        return True, stats

    except Exception as e:
        logger.error(f"Phase 3 PR (Optimized) failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}


if __name__ == '__main__':
    # For local testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from s3_utils import S3Handler

    s3_handler = S3Handler()

    # Train PR models (thresholds 8-40, representing o/u 7.5-39.5)
    print("\nTraining PR models (thresholds 8-40)...")
    success, stats = run_phase_3_pr(s3_handler, threshold_start=8, threshold_end=40)

    if success:
        print("\n✓ Phase 3 PR (Optimized) completed successfully!")
        print(f"Stats: {stats}")
    else:
        print("\n✗ Phase 3 PR (Optimized) failed!")
        print(f"Error: {stats.get('error')}")
