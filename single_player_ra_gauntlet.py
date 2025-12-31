"""
Single Player RA Gauntlet Script

This script loads the RA dataset and models, then runs the prop gauntlet for a specific
player ID provided via command line, returning props they're likely to hit with 78%+ confidence.

Usage:
    python single_player_ra_gauntlet.py <player_id>

Example:
    python single_player_ra_gauntlet.py 2544  # For LeBron James

Author: NBA Props Prediction System
Date: 2025-12-31
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
from datetime import datetime
import pickle
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for RA
CONFIDENCE_THRESHOLD = 0.78
RA_CONFIG = {
    'threshold_start': 5,
    'threshold_end': 27,  # 5-26 inclusive
    'min_line': 4.5,
    'max_line': 25.5,
}


def load_ra_dataset(s3_handler):
    """
    Load the RA processed dataset from S3.

    Args:
        s3_handler: S3Handler instance

    Returns:
        DataFrame: RA processed dataset
    """
    from s3_utils import S3_PLAYER_BUCKET

    logger.info("Loading RA dataset from S3...")

    df = s3_handler.download_dataframe(
        S3_PLAYER_BUCKET,
        'processed_data_ra/processed_with_ra_pct_5-26.csv'
    )

    logger.info(f"✓ Loaded {len(df):,} rows from RA dataset")

    # Ensure GAME_DATE_PARSED is datetime
    if 'GAME_DATE_PARSED' in df.columns:
        df['GAME_DATE_PARSED'] = pd.to_datetime(df['GAME_DATE_PARSED'], errors='coerce')
    elif 'GAME_DATE' in df.columns:
        df['GAME_DATE_PARSED'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y', errors='coerce')
    else:
        raise ValueError("Neither GAME_DATE_PARSED nor GAME_DATE column found in data")

    # Calculate RA if not present
    if 'RA' not in df.columns:
        if 'REB' in df.columns and 'AST' in df.columns:
            df['RA'] = df['REB'] + df['AST']
        else:
            raise ValueError("Cannot calculate RA: REB or AST columns missing")

    return df


def load_ra_models(s3_handler):
    """
    Load all RA models from S3 (thresholds 5-26).

    Args:
        s3_handler: S3Handler instance

    Returns:
        dict: {5: model_5, 6: model_6, ..., 26: model_26}
    """
    from s3_utils import S3_MODEL_BUCKET

    logger.info("Loading RA models from S3...")

    models = {}
    threshold_start = RA_CONFIG['threshold_start']
    threshold_end = RA_CONFIG['threshold_end']

    for threshold in range(threshold_start, threshold_end):
        try:
            model_prefix = f'models_ra/xgb_ra_{threshold}plus_'

            # Get list of objects with this prefix
            s3_client = s3_handler.s3_client
            response = s3_client.list_objects_v2(
                Bucket=S3_MODEL_BUCKET,
                Prefix=model_prefix
            )

            if 'Contents' not in response or len(response['Contents']) == 0:
                logger.warning(f"  No model found for threshold {threshold}")
                continue

            # Get the first matching model file
            model_key = response['Contents'][0]['Key']

            # Download model
            obj = s3_client.get_object(Bucket=S3_MODEL_BUCKET, Key=model_key)
            model = pickle.loads(obj['Body'].read())

            models[threshold] = model

        except Exception as e:
            logger.warning(f"  Error loading model for threshold {threshold}: {e}")
            continue

    logger.info(f"✓ Loaded {len(models)} RA models (thresholds {threshold_start}-{threshold_end-1})")
    return models


def get_player_data(player_id, df):
    """
    Get all games for a specific player.

    Args:
        player_id: Player ID to filter for
        df: Full dataset

    Returns:
        DataFrame: All games for this player, sorted by date
    """
    player_games = df[df['Player_ID'] == player_id].copy()
    player_games = player_games.sort_values('GAME_DATE_PARSED', ascending=False)

    if len(player_games) == 0:
        logger.error(f"No games found for Player_ID {player_id}")
        return None

    # Get player info
    player_name = player_games.iloc[0].get('PLAYER', 'Unknown')
    team = player_games.iloc[0].get('TEAM', 'Unknown')

    logger.info(f"Found {len(player_games)} games for {player_name} (Player_ID: {player_id}, Team: {team})")

    return player_games


def reconstruct_features_for_player(player_games, full_df, thresholds, opponent='OPP'):
    """
    Reconstruct features for the most recent game (for future prediction).

    This is a simplified version that uses the most recent game as the baseline.

    Args:
        player_games: All games for this player (sorted by date, most recent first)
        full_df: Full dataset (for opponent strength calculation)
        thresholds: List of thresholds
        opponent: Opponent team abbreviation (default: 'OPP' if not specified)

    Returns:
        dict: Reconstructed features
    """
    if len(player_games) == 0:
        return None

    # Most recent game
    latest_game = player_games.iloc[0]
    player_id = latest_game['Player_ID']
    team = latest_game['TEAM']
    current_date = latest_game['GAME_DATE_PARSED']
    current_season = latest_game['SEASON_ID']
    lineup_id = latest_game.get('LINEUP_ID', 0)

    # Get games up to and including current game
    player_all_games = full_df[
        (full_df['Player_ID'] == player_id) &
        (full_df['GAME_DATE_PARSED'] <= current_date)
    ].sort_values('GAME_DATE_PARSED')

    player_team_games = player_all_games[player_all_games['TEAM'] == team]
    player_season_games = player_team_games[player_team_games['SEASON_ID'] == current_season]
    player_lineup_games = player_team_games[player_team_games['LINEUP_ID'] == lineup_id]

    # H2H games (last 3 seasons)
    target_seasons = [current_season, current_season - 1, current_season - 2]
    player_h2h_games = player_all_games[
        (player_all_games['OPPONENT'] == opponent) &
        (player_all_games['SEASON_ID'].isin(target_seasons))
    ]

    # Base features
    features = {
        'Player_ID': player_id,
        'LINEUP_ID': lineup_id,
    }

    # Rolling averages
    features['last_5_avg'] = player_team_games.tail(5)['RA'].mean() if len(player_team_games) > 0 else 0
    features['last_10_avg'] = player_team_games.tail(10)['RA'].mean() if len(player_team_games) > 0 else 0
    features['last_20_avg'] = player_team_games.tail(20)['RA'].mean() if len(player_team_games) > 0 else 0
    features['season_avg'] = player_season_games['RA'].mean() if len(player_season_games) > 0 else 0

    # Last season average
    last_season_id = current_season - 1
    player_last_season_games = player_all_games[player_all_games['SEASON_ID'] == last_season_id]
    features['last_season_avg'] = player_last_season_games['RA'].mean() if len(player_last_season_games) > 0 else 0

    features['lineup_average'] = player_lineup_games['RA'].mean() if len(player_lineup_games) > 0 else 0
    features['h2h_avg'] = player_h2h_games['RA'].mean() if len(player_h2h_games) > 0 else 0

    # Opponent strength (simplified - use 0 if no opponent data)
    features['opp_strength'] = 0.0

    # Threshold-specific percentage features
    for threshold in thresholds:
        last_5_games = player_team_games.tail(5)
        features[f'last_5_pct_{threshold}'] = (
            (last_5_games['RA'] >= threshold).mean() if len(last_5_games) > 0 else 0
        )

        last_10_games = player_team_games.tail(10)
        features[f'last_10_pct_{threshold}'] = (
            (last_10_games['RA'] >= threshold).mean() if len(last_10_games) > 0 else 0
        )

        last_20_games = player_team_games.tail(20)
        features[f'last_20_pct_{threshold}'] = (
            (last_20_games['RA'] >= threshold).mean() if len(last_20_games) > 0 else 0
        )

        features[f'season_pct_{threshold}'] = (
            (player_season_games['RA'] >= threshold).mean() if len(player_season_games) > 0 else 0
        )

        features[f'last_season_pct_{threshold}'] = latest_game.get(f'last_season_pct_{threshold}', 0)

        # Lineup percentage
        if len(player_lineup_games) > 1:
            features[f'lineup_pct_{threshold}'] = (
                (player_lineup_games['RA'] >= threshold).mean()
            )
        elif len(player_lineup_games) == 1:
            player_team_history = player_team_games[player_team_games['GAME_DATE_PARSED'] < current_date]
            if len(player_team_history) > 0:
                player_team_sorted = player_team_history.sort_values('GAME_DATE_PARSED')
                lineup_changed = player_team_sorted['LINEUP_ID'].ne(player_team_sorted['LINEUP_ID'].shift(1))
                first_lineup_games = player_team_sorted[lineup_changed]
                if len(first_lineup_games) > 0:
                    features[f'lineup_pct_{threshold}'] = (
                        (first_lineup_games['RA'] >= threshold).mean()
                    )
                else:
                    features[f'lineup_pct_{threshold}'] = 0.0
            else:
                features[f'lineup_pct_{threshold}'] = 0.0
        else:
            features[f'lineup_pct_{threshold}'] = 0.0

        features[f'h2h_pct_{threshold}'] = (
            (player_h2h_games['RA'] >= threshold).mean() if len(player_h2h_games) > 0 else 0
        )

    return features


def prepare_model_input(features_dict, model):
    """
    Prepare input features for model prediction.

    Args:
        features_dict: Dictionary of feature values
        model: XGBoost model

    Returns:
        DataFrame: Single row with features in correct order
    """
    expected_features = model.get_booster().feature_names
    feature_values = [features_dict.get(feat, 0) for feat in expected_features]
    X = pd.DataFrame([feature_values], columns=expected_features)
    return X


def run_over_gauntlet(reconstructed_features, models):
    """
    Run OVER gauntlet starting from lowest threshold.

    Args:
        reconstructed_features: Dict of reconstructed features
        models: Dict of loaded models

    Returns:
        dict or None: Prediction result if passed threshold
    """
    thresholds = sorted(models.keys())
    highest_passed = None
    final_probability = 0.0

    for threshold in thresholds:
        threshold_features = {
            'Player_ID': reconstructed_features['Player_ID'],
            'LINEUP_ID': reconstructed_features['LINEUP_ID'],
            'lineup_average': reconstructed_features['lineup_average'],
            'last_5_avg': reconstructed_features['last_5_avg'],
            'last_10_avg': reconstructed_features['last_10_avg'],
            'last_20_avg': reconstructed_features['last_20_avg'],
            'season_avg': reconstructed_features['season_avg'],
            'last_season_avg': reconstructed_features['last_season_avg'],
            'h2h_avg': reconstructed_features['h2h_avg'],
            'opp_strength': reconstructed_features['opp_strength'],
            f'last_5_pct_{threshold}': reconstructed_features[f'last_5_pct_{threshold}'],
            f'last_10_pct_{threshold}': reconstructed_features[f'last_10_pct_{threshold}'],
            f'last_20_pct_{threshold}': reconstructed_features[f'last_20_pct_{threshold}'],
            f'season_pct_{threshold}': reconstructed_features[f'season_pct_{threshold}'],
            f'last_season_pct_{threshold}': reconstructed_features[f'last_season_pct_{threshold}'],
            f'lineup_pct_{threshold}': reconstructed_features[f'lineup_pct_{threshold}'],
            f'h2h_pct_{threshold}': reconstructed_features[f'h2h_pct_{threshold}']
        }

        try:
            X = prepare_model_input(threshold_features, models[threshold])
            prob = models[threshold].predict_proba(X)[0][1]

            if prob >= CONFIDENCE_THRESHOLD:
                highest_passed = threshold
                final_probability = prob
            else:
                break
        except Exception as e:
            logger.warning(f"  Error predicting at threshold {threshold}: {e}")
            break

    if highest_passed is not None:
        line = highest_passed - 0.5
        if line >= RA_CONFIG['min_line'] and line <= RA_CONFIG['max_line']:
            return {
                'prop_type': 'OVER',
                'line': line,
                'threshold': highest_passed,
                'confidence': final_probability,
                'prop': f"Over {line} RA"
            }
    return None


def run_under_gauntlet(reconstructed_features, models):
    """
    Run UNDER gauntlet starting from high threshold.

    Args:
        reconstructed_features: Dict of reconstructed features
        models: Dict of loaded models

    Returns:
        dict or None: Prediction result if passed threshold
    """
    thresholds = sorted(models.keys())
    season_avg = reconstructed_features.get('season_avg', 0)
    max_threshold = max(thresholds) if thresholds else 26
    starting_threshold = min(max_threshold, max(10, int(season_avg) + 5))

    lowest_passed = None
    under_confidence = 0.0

    for threshold in reversed(thresholds):
        if threshold > starting_threshold:
            continue

        threshold_features = {
            'Player_ID': reconstructed_features['Player_ID'],
            'LINEUP_ID': reconstructed_features['LINEUP_ID'],
            'lineup_average': reconstructed_features['lineup_average'],
            'last_5_avg': reconstructed_features['last_5_avg'],
            'last_10_avg': reconstructed_features['last_10_avg'],
            'last_20_avg': reconstructed_features['last_20_avg'],
            'season_avg': reconstructed_features['season_avg'],
            'last_season_avg': reconstructed_features['last_season_avg'],
            'h2h_avg': reconstructed_features['h2h_avg'],
            'opp_strength': reconstructed_features['opp_strength'],
            f'last_5_pct_{threshold}': reconstructed_features[f'last_5_pct_{threshold}'],
            f'last_10_pct_{threshold}': reconstructed_features[f'last_10_pct_{threshold}'],
            f'last_20_pct_{threshold}': reconstructed_features[f'last_20_pct_{threshold}'],
            f'season_pct_{threshold}': reconstructed_features[f'season_pct_{threshold}'],
            f'last_season_pct_{threshold}': reconstructed_features[f'last_season_pct_{threshold}'],
            f'lineup_pct_{threshold}': reconstructed_features[f'lineup_pct_{threshold}'],
            f'h2h_pct_{threshold}': reconstructed_features[f'h2h_pct_{threshold}']
        }

        try:
            X = prepare_model_input(threshold_features, models[threshold])
            prob = models[threshold].predict_proba(X)[0][1]

            if prob <= (1 - CONFIDENCE_THRESHOLD):
                lowest_passed = threshold
                under_confidence = 1 - prob
            else:
                break
        except Exception as e:
            logger.warning(f"  Error predicting UNDER at threshold {threshold}: {e}")
            break

    if lowest_passed is not None:
        line = lowest_passed + 0.5
        if line >= RA_CONFIG['min_line'] and line <= RA_CONFIG['max_line']:
            return {
                'prop_type': 'UNDER',
                'line': line,
                'threshold': lowest_passed,
                'confidence': under_confidence,
                'prop': f"Under {line} RA"
            }
    return None


def display_prediction(prediction, player_name, reconstructed_features):
    """
    Display prediction result in terminal.

    Args:
        prediction: Prediction dict
        player_name: Player name
        reconstructed_features: Reconstructed features dict
    """
    if prediction is None:
        logger.info(f"\n❌ No props passed the 78% confidence threshold for {player_name}")
        return

    print("\n" + "=" * 80)
    print(f"🎯 PROP PREDICTION FOR {player_name.upper()}")
    print("=" * 80)
    print(f"Prop: {prediction['prop']}")
    print(f"Type: {prediction['prop_type']}")
    print(f"Line: {prediction['line']}")
    print(f"Confidence: {prediction['confidence']:.1%}")
    print()
    print("Performance Metrics:")
    print(f"  Last 5 Games:    {reconstructed_features.get(f'last_5_pct_{prediction["threshold"]}', 0):.1%}")
    print(f"  Last 10 Games:   {reconstructed_features.get(f'last_10_pct_{prediction["threshold"]}', 0):.1%}")
    print(f"  This Season:     {reconstructed_features.get(f'season_pct_{prediction["threshold"]}', 0):.1%}")
    print(f"  Last Season:     {reconstructed_features.get(f'last_season_pct_{prediction["threshold"]}', 0):.1%}")
    print()
    print("Averages:")
    print(f"  Last 5 Avg:      {reconstructed_features.get('last_5_avg', 0):.1f} RA")
    print(f"  Season Avg:      {reconstructed_features.get('season_avg', 0):.1f} RA")
    print(f"  Last Season Avg: {reconstructed_features.get('last_season_avg', 0):.1f} RA")
    print("=" * 80)


def main(player_id):
    """
    Main execution function.

    Args:
        player_id: Player ID to run gauntlet for
    """
    logger.info("=" * 80)
    logger.info("SINGLE PLAYER RA PROP GAUNTLET")
    logger.info("=" * 80)
    logger.info(f"Player ID: {player_id}")
    logger.info(f"Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%}")
    logger.info("")

    # Load environment variables
    load_dotenv()

    # Initialize S3 handler
    from s3_utils import S3Handler
    s3_handler = S3Handler()

    # Load RA dataset
    df = load_ra_dataset(s3_handler)

    # Load RA models
    models = load_ra_models(s3_handler)

    if len(models) == 0:
        logger.error("Failed to load RA models. Exiting.")
        return 1

    # Get player data
    player_games = get_player_data(player_id, df)
    if player_games is None:
        logger.error(f"No data found for Player_ID {player_id}. Exiting.")
        return 1

    player_name = player_games.iloc[0].get('PLAYER', 'Unknown')

    # Reconstruct features
    logger.info("\nReconstructing features for prediction...")
    thresholds = sorted(models.keys())
    reconstructed_features = reconstruct_features_for_player(
        player_games, df, thresholds
    )

    if reconstructed_features is None:
        logger.error("Failed to reconstruct features. Exiting.")
        return 1

    logger.info("✓ Features reconstructed")

    # Run both gauntlets
    logger.info("\nRunning OVER gauntlet...")
    over_pred = run_over_gauntlet(reconstructed_features, models)

    logger.info("Running UNDER gauntlet...")
    under_pred = run_under_gauntlet(reconstructed_features, models)

    # Display results
    predictions = []
    if over_pred:
        predictions.append(over_pred)
        logger.info(f"✓ OVER prediction: {over_pred['prop']} ({over_pred['confidence']:.1%})")

    if under_pred:
        predictions.append(under_pred)
        logger.info(f"✓ UNDER prediction: {under_pred['prop']} ({under_pred['confidence']:.1%})")

    # If both predictions, check separation
    if len(predictions) == 2:
        separation = abs(under_pred['line'] - over_pred['line'])
        if separation < 4.0:
            # Keep only higher confidence
            if over_pred['confidence'] >= under_pred['confidence']:
                logger.info(f"ℹ Keeping OVER only (separation={separation:.1f}, higher confidence)")
                predictions = [over_pred]
            else:
                logger.info(f"ℹ Keeping UNDER only (separation={separation:.1f}, higher confidence)")
                predictions = [under_pred]
        else:
            logger.info(f"✓ Keeping BOTH predictions (separation={separation:.1f})")

    # Display final predictions
    if len(predictions) == 0:
        display_prediction(None, player_name, reconstructed_features)
    else:
        for pred in predictions:
            display_prediction(pred, player_name, reconstructed_features)

    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python single_player_ra_gauntlet.py <player_id>")
        print("\nExample:")
        print("  python single_player_ra_gauntlet.py 2544")
        sys.exit(1)

    try:
        player_id = int(sys.argv[1])
    except ValueError:
        print(f"Error: Player ID must be an integer. Got: {sys.argv[1]}")
        sys.exit(1)

    sys.exit(main(player_id))
