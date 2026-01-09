"""
Phase 4: Daily Prediction Generation

This module generates daily NBA player prop predictions using a model gauntlet approach.
Players from today's games are evaluated through a series of ML models (10+ through 51+ PRA),
with predictions stopping when confidence drops below 78%.

IMPORTANT: Feature Reconstruction for Prediction
The processed data uses shift(1) to exclude the current game during training. However,
when predicting FUTURE games, the most recent game is now in the PAST. Therefore, we
reconstruct all features (rolling averages, percentages) to INCLUDE the current game's
PRA value. This ensures predictions are based on the most up-to-date performance data.

Author: Generated for NBA Props Prediction System
Date: 2025-12-11
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, date
import requests
from bs4 import BeautifulSoup
import json
import psycopg2
from psycopg2.extras import execute_values
from nba_api.stats.endpoints import leaguedashplayerstats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
ROTOWIRE_URL = 'https://www.rotowire.com/basketball/nba-lineups.php'
CONFIDENCE_THRESHOLD = 0.78  # 78% probability threshold for gauntlet


# ============================================================================
# 4.1: SCRAPE ROTOWIRE FOR TODAY'S GAMES
# ============================================================================

def normalize_player_name(name):
    """
    Normalize player name for consistent formatting.

    Args:
        name: Player name string

    Returns:
        str: Normalized name in slug format
    """
    name = name.replace(' Jr.', '').replace(' Sr.', '').replace(' III', '').replace(' II', '').replace(' IV', '')
    name = name.replace('.', '').replace("'", '')
    slug = name.lower().strip().replace(' ', '-')
    return slug


def scrape_rotowire_for_teams_playing_today():
    """
    Scrape RotoWire to get teams playing today.

    Returns:
        dict: {
            'teams_playing': set of team abbreviations,
            'team_matchups': dict mapping team to opponent and home/away,
            'game_date': today's date string
        }
    """
    logger.info("=" * 80)
    logger.info(f"Scraping RotoWire for today's games ({date.today()})...")
    logger.info("=" * 80)

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        response = requests.get(ROTOWIRE_URL, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        teams_playing = set()
        team_matchups = {}

        # Find all NBA lineup games
        lineup_games = soup.find_all('div', class_='lineup is-nba')
        logger.info(f"Found {len(lineup_games)} NBA games on RotoWire")

        for game_container in lineup_games:
            try:
                lineup_box = game_container.find('div', class_='lineup__box')
                if not lineup_box:
                    continue

                # Get team abbreviations
                lineup_top = lineup_box.find('div', class_='lineup__top')
                if not lineup_top:
                    continue

                team_abbr_elements = lineup_top.find_all('div', class_='lineup__abbr')
                if len(team_abbr_elements) < 2:
                    continue

                away_team_abbr = team_abbr_elements[0].get_text(strip=True)
                home_team_abbr = team_abbr_elements[1].get_text(strip=True)

                # Add to teams playing
                teams_playing.add(away_team_abbr)
                teams_playing.add(home_team_abbr)

                # Store matchup info
                team_matchups[away_team_abbr] = {
                    'opponent': home_team_abbr,
                    'is_home': False
                }
                team_matchups[home_team_abbr] = {
                    'opponent': away_team_abbr,
                    'is_home': True
                }

                logger.info(f"  {away_team_abbr} @ {home_team_abbr}")

            except Exception as e:
                logger.warning(f"Error parsing game container: {e}")
                continue

        logger.info(f"✓ Found {len(teams_playing)} teams playing today")

        return {
            'teams_playing': teams_playing,
            'team_matchups': team_matchups,
            'game_date': date.today().strftime('%b %d, %Y')
        }

    except Exception as e:
        logger.error(f"Error scraping RotoWire: {e}")
        return {
            'teams_playing': set(),
            'team_matchups': {},
            'game_date': date.today().strftime('%b %d, %Y')
        }


# ============================================================================
# 4.2: FILTER PLAYERS PLAYING TODAY
# ============================================================================

def fetch_league_leaders():
    """
    Fetch all players with 10+ PRA using LeagueDashPlayerStats.

    This approach includes ALL players regardless of games played percentage,
    unlike LeagueLeaders which filters for players with 70%+ games played.

    Returns:
        DataFrame: League leaders data with columns: PLAYER_ID, PLAYER, TEAM, PTS, REB, AST, PRA
    """
    logger.info("Fetching all players with 10+ PRA (including players with low game counts)...")

    try:
        ll_df = leaguedashplayerstats.LeagueDashPlayerStats(per_mode_detailed='PerGame').get_data_frames()[0]

        # Calculate PRA
        ll_df['PRA'] = ll_df['PTS'] + ll_df['REB'] + ll_df['AST']

        # Filter 10+ PRA
        ll_df = ll_df[ll_df['PRA'] >= 10.0]
        ll_df = ll_df.sort_values('PRA', ascending=False)

        logger.info(f"✓ Found {len(ll_df)} players with PRA >= 10")

        # Map columns to match expected format (LeagueDashPlayerStats uses PLAYER_NAME instead of PLAYER)
        return ll_df[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'PTS', 'REB', 'AST', 'PRA']].rename(
            columns={'PLAYER_NAME': 'PLAYER', 'TEAM_ABBREVIATION': 'TEAM'}
        )

    except Exception as e:
        logger.error(f"Error fetching league leaders: {e}")
        return pd.DataFrame()


def get_players_playing_today(s3_handler, teams_playing, league_leaders_df, stat_type='PRA'):
    """
    Filter processed data for players on teams playing today.

    Logic:
    1. Filter for current season (2024-25)
    2. For each player, get their most recent game in the current season
    3. Check if their most recent game was for a team playing today
    4. If yes, include them (they're still on that team, even if they missed recent games)
    5. If no, exclude them (they switched teams)

    Args:
        s3_handler: S3Handler instance
        teams_playing: Set of team abbreviations playing today
        league_leaders_df: League leaders
        stat_type: Stat type ('PRA', 'RA', 'PA', 'PR')

    Returns:
        Tuple of (filtered_df, full_df):
            - filtered_df: Filtered rows (most recent per player on teams playing today)
            - full_df: Full processed dataset (for feature reconstruction)
    """
    from s3_utils import S3_PLAYER_BUCKET

    # Data file paths based on stat type
    data_paths = {
        'PRA': 'processed_data/processed_with_all_pct_10-51.csv',
        'RA': 'processed_data_ra/processed_with_ra_pct_5-26.csv',
        'PA': 'processed_data_pa/processed_with_pa_pct_5-40.csv',
        'PR': 'processed_data_pr/processed_with_pr_pct_5-40.csv',
    }

    data_path = data_paths.get(stat_type.upper())
    if not data_path:
        raise ValueError(f"Invalid stat_type: {stat_type}")

    logger.info(f"Loading {stat_type} processed data and filtering for today's players...")

    # Download processed data with all percentages
    df = s3_handler.download_dataframe(S3_PLAYER_BUCKET, data_path)

    logger.info(f"Loaded {len(df):,} rows from processed data")

    # Ensure GAME_DATE_PARSED is datetime
    if 'GAME_DATE_PARSED' in df.columns:
        # Column exists but might be string type - convert to datetime
        df['GAME_DATE_PARSED'] = pd.to_datetime(df['GAME_DATE_PARSED'], errors='coerce')
    elif 'GAME_DATE' in df.columns:
        # Column doesn't exist - create from GAME_DATE
        df['GAME_DATE_PARSED'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y', errors='coerce')
    else:
        raise ValueError("Neither GAME_DATE_PARSED nor GAME_DATE column found in data")

    # Filter for current season (2025-26 season starts October 2025)
    current_season_start = pd.to_datetime('2025-10-01')
    df_current_season = df[df['GAME_DATE_PARSED'] >= current_season_start].copy()
    logger.info(f"Filtered to {len(df_current_season):,} rows for current season (2025-26)")

    # For each player, get their most recent game in the current season
    df_current_season_sorted = df_current_season.sort_values('GAME_DATE_PARSED', ascending=False)
    df_player_latest = df_current_season_sorted.groupby('Player_ID').first().reset_index()
    logger.info(f"Found most recent games for {len(df_player_latest)} players in current season")

    # Filter to only keep players whose most recent game was for a team playing today
    # This handles both: players who missed recent games (injury) AND excludes traded players
    df_latest = df_player_latest[df_player_latest['TEAM'].isin(teams_playing)].copy()
    logger.info(f"Selected {len(df_latest)} players whose most recent game was for teams playing today")

    # Merge with league leaders to get player names
    df_latest = df_latest.merge(
        league_leaders_df[['PLAYER_ID', 'PLAYER']],
        left_on='Player_ID',
        right_on='PLAYER_ID',
        how='inner'
    )

    logger.info(f"✓ {len(df_latest)} players from league leaders playing today")

    # Return both the filtered players and the full dataset (for feature reconstruction)
    return df_latest, df


# ============================================================================
# 4.3: LOAD MODELS FROM S3
# ============================================================================

def load_all_models(s3_handler, stat_type='PRA', threshold_start=None, threshold_end=None):
    """
    Load all trained models from S3 for a specific stat type.

    Args:
        s3_handler: S3Handler instance
        stat_type: Stat type ('PRA', 'RA', 'PA', 'PR')
        threshold_start: Starting threshold (default: depends on stat_type)
        threshold_end: Ending threshold (exclusive, default: depends on stat_type)

    Returns:
        dict: {threshold: model, ...}
    """
    from s3_utils import S3_MODEL_BUCKET
    import pickle

    # Default thresholds based on stat type
    stat_configs = {
        'PRA': {'start': 10, 'end': 52, 'folder': 'models', 'prefix': 'xgb_pra'},
        'RA': {'start': 5, 'end': 27, 'folder': 'models_ra', 'prefix': 'xgb_ra'},
        'PA': {'start': 8, 'end': 41, 'folder': 'models_pa', 'prefix': 'xgb_pa'},
        'PR': {'start': 8, 'end': 41, 'folder': 'models_pr', 'prefix': 'xgb_pr'},
    }

    config = stat_configs.get(stat_type.upper())
    if not config:
        raise ValueError(f"Invalid stat_type: {stat_type}. Must be one of {list(stat_configs.keys())}")

    if threshold_start is None:
        threshold_start = config['start']
    if threshold_end is None:
        threshold_end = config['end']

    logger.info(f"Loading {stat_type} models for thresholds {threshold_start}-{threshold_end-1}...")

    models = {}

    for threshold in range(threshold_start, threshold_end):
        try:
            # List all model files for this threshold
            model_prefix = f"{config['folder']}/{config['prefix']}_{threshold}plus_"

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

            if threshold % 10 == 0:
                logger.info(f"  Loaded models up to threshold {threshold}")

        except Exception as e:
            logger.warning(f"  Error loading model for threshold {threshold}: {e}")
            continue

    logger.info(f"✓ Loaded {len(models)} {stat_type} models from S3")
    return models


# ============================================================================
# 4.4: RECONSTRUCT FEATURES FOR PREDICTION
# ============================================================================

def get_opponent_strength(full_df, opponent_team, current_season):
    """
    Get the most recent opponent strength for a given opponent team.

    Args:
        full_df: Full processed dataset
        opponent_team: Opponent team abbreviation
        current_season: Current season ID

    Returns:
        float: Most recent opponent strength value for this opponent, or 0.0 if not found
    """
    # Filter for games against this opponent in the current season
    opp_games = full_df[
        (full_df['OPPONENT'] == opponent_team) &
        (full_df['SEASON_ID'] == current_season)
    ].copy()

    if len(opp_games) == 0:
        # Try previous season if no current season data
        opp_games = full_df[
            (full_df['OPPONENT'] == opponent_team) &
            (full_df['SEASON_ID'] == current_season - 1)
        ].copy()

    if len(opp_games) == 0:
        return 0.0

    # Sort by date and get the most recent game's opponent strength
    opp_games = opp_games.sort_values('GAME_DATE_PARSED', ascending=False)
    most_recent_opp_strength = opp_games.iloc[0].get('opp_strength', 0.0)

    # Handle NaN values
    if pd.isna(most_recent_opp_strength):
        return 0.0

    return most_recent_opp_strength


def normalize_team_abbreviation(team_abbr, full_df, player_name=""):
    """
    Normalize team abbreviation and validate it exists in data.
    Handles alternate abbreviations (BRK/BKN, CHO/CHA).

    Args:
        team_abbr: Original team abbreviation
        full_df: Full processed dataset
        player_name: Player name for logging context (optional)

    Returns:
        str: Normalized team abbreviation that exists in data
    """
    # Map of alternate abbreviations
    alternate_abbrs = {
        'BRK': 'BKN',
        'BKN': 'BRK',
        'CHO': 'CHA',
        'CHA': 'CHO'
    }

    # Check if abbreviation exists in data
    available_teams = full_df['OPPONENT'].unique()

    if team_abbr in available_teams:
        return team_abbr

    # Try alternate abbreviation
    alt_abbr = alternate_abbrs.get(team_abbr)
    if alt_abbr and alt_abbr in available_teams:
        logger.info(f"H2H: Using alternate abbreviation {alt_abbr} for {team_abbr}" +
                   (f" (player: {player_name})" if player_name else ""))
        return alt_abbr

    # If neither found, log warning and return original
    logger.warning(f"H2H: Team abbreviation '{team_abbr}' not found in data" +
                  (f" (player: {player_name})" if player_name else ""))
    return team_abbr


def reconstruct_features_for_prediction(player_row, full_df, thresholds, today_opponent):
    """
    Reconstruct features by INCLUDING the current game's PRA for future prediction.

    The processed data uses shift(1) to exclude the current game during training.
    For prediction, we need to include the current game's PRA in all rolling windows
    and percentages, as it's now in the past relative to the game we're predicting.

    Args:
        player_row: Most recent game row with existing features
        full_df: Full processed dataset with all player histories
        thresholds: List of PRA thresholds (10-51)
        today_opponent: Today's actual opponent from team matchups (not from player_row)

    Returns:
        dict: Reconstructed features including current game's PRA
    """
    player_id = player_row['Player_ID']
    team = player_row['TEAM']
    current_date = player_row['GAME_DATE_PARSED']
    current_season = player_row['SEASON_ID']
    lineup_id = player_row.get('LINEUP_ID', 0)
    opponent = today_opponent  # Use today's opponent, not the one from most recent game

    # Get all games for this player up to and including current game
    player_all_games = full_df[
        (full_df['Player_ID'] == player_id) &
        (full_df['GAME_DATE_PARSED'] <= current_date)
    ].sort_values('GAME_DATE_PARSED')

    # Get player-team games (for team-specific stats)
    player_team_games = player_all_games[player_all_games['TEAM'] == team]

    # Get current season games
    player_season_games = player_team_games[player_team_games['SEASON_ID'] == current_season]

    # Get lineup games
    player_lineup_games = player_team_games[player_team_games['LINEUP_ID'] == lineup_id]

    # Get H2H games (last 3 seasons) - normalize opponent abbreviation first
    target_seasons = [current_season, current_season - 1, current_season - 2]
    normalized_opponent = normalize_team_abbreviation(opponent, full_df, player_row.get('PLAYER', ''))
    player_h2h_games = player_all_games[
        (player_all_games['OPPONENT'] == normalized_opponent) &
        (player_all_games['SEASON_ID'].isin(target_seasons))
    ]
    if len(player_h2h_games) == 0:
        logger.info(f"PRA H2H: No games found for {player_row.get('PLAYER', '')} vs {normalized_opponent}")

    # ========================================================================
    # BASE FEATURES (same for all thresholds)
    # ========================================================================

    features = {
        'Player_ID': player_id,
        'LINEUP_ID': lineup_id,
    }

    # Rolling averages (including current game)
    features['last_5_avg'] = player_team_games.tail(5)['PRA'].mean() if len(player_team_games) > 0 else 0
    features['last_10_avg'] = player_team_games.tail(10)['PRA'].mean() if len(player_team_games) > 0 else 0
    features['last_20_avg'] = player_team_games.tail(20)['PRA'].mean() if len(player_team_games) > 0 else 0

    # Season average (including current game)
    features['season_avg'] = player_season_games['PRA'].mean() if len(player_season_games) > 0 else 0

    # Last season average (unchanged - copy from player_row)
    features['last_season_avg'] = player_row.get('last_season_avg', 0)

    # Lineup average (including current game)
    features['lineup_average'] = player_lineup_games['PRA'].mean() if len(player_lineup_games) > 0 else 0

    # H2H average (including current game)
    features['h2h_avg'] = player_h2h_games['PRA'].mean() if len(player_h2h_games) > 0 else 0

    # Opponent strength - get from opponent's most recent games
    features['opp_strength'] = get_opponent_strength(full_df, opponent, current_season)

    # ========================================================================
    # THRESHOLD-SPECIFIC PERCENTAGE FEATURES
    # ========================================================================

    for threshold in thresholds:
        # Rolling percentages (including current game)
        last_5_games = player_team_games.tail(5)
        features[f'last_5_pct_{threshold}'] = (
            (last_5_games['PRA'] >= threshold).mean() if len(last_5_games) > 0 else 0
        )

        last_10_games = player_team_games.tail(10)
        features[f'last_10_pct_{threshold}'] = (
            (last_10_games['PRA'] >= threshold).mean() if len(last_10_games) > 0 else 0
        )

        last_20_games = player_team_games.tail(20)
        features[f'last_20_pct_{threshold}'] = (
            (last_20_games['PRA'] >= threshold).mean() if len(last_20_games) > 0 else 0
        )

        # Season percentage (including current game)
        features[f'season_pct_{threshold}'] = (
            (player_season_games['PRA'] >= threshold).mean() if len(player_season_games) > 0 else 0
        )

        # Last season percentage (unchanged - copy from player_row)
        features[f'last_season_pct_{threshold}'] = player_row.get(f'last_season_pct_{threshold}', 0)

        # Lineup percentage (including current game)
        if len(player_lineup_games) > 1:
            # Has previous games with this lineup - use actual percentage
            features[f'lineup_pct_{threshold}'] = (
                (player_lineup_games['PRA'] >= threshold).mean()
            )
        elif len(player_lineup_games) == 1:
            # First game with this lineup - use "new lineup percentage"
            # Find all previous "first games with new lineups" for this player-team

            # Exclude the current game
            player_team_history = player_team_games[player_team_games['GAME_DATE_PARSED'] < current_date]

            if len(player_team_history) > 0:
                # Sort by date to identify lineup changes
                player_team_sorted = player_team_history.sort_values('GAME_DATE_PARSED')

                # Find games where lineup changed from previous game (first game with new lineup)
                lineup_changed = player_team_sorted['LINEUP_ID'].ne(player_team_sorted['LINEUP_ID'].shift(1))
                first_lineup_games = player_team_sorted[lineup_changed]

                if len(first_lineup_games) > 0:
                    # Calculate percentage from previous first-lineup-games
                    features[f'lineup_pct_{threshold}'] = (
                        (first_lineup_games['PRA'] >= threshold).mean()
                    )
                else:
                    # No previous first-lineup-games, use 0
                    features[f'lineup_pct_{threshold}'] = 0.0
            else:
                # No previous games for this player-team, use 0
                features[f'lineup_pct_{threshold}'] = 0.0
        else:
            # Should not happen (no games with this lineup including current game)
            features[f'lineup_pct_{threshold}'] = 0.0

        # H2H percentage (including current game)
        features[f'h2h_pct_{threshold}'] = (
            (player_h2h_games['PRA'] >= threshold).mean() if len(player_h2h_games) > 0 else 0
        )

    return features


# ============================================================================
# 4.5: RUN MODEL GAUNTLET
# ============================================================================

def prepare_model_input(features_dict, model):
    """
    Prepare input features for model prediction.

    Args:
        features_dict: Dictionary of feature values
        model: XGBoost model (to get feature names)

    Returns:
        DataFrame: Single row with features in correct order
    """
    # Get expected feature names from model
    expected_features = model.get_booster().feature_names

    # Create DataFrame with features in correct order
    feature_values = [features_dict.get(feat, 0) for feat in expected_features]
    X = pd.DataFrame([feature_values], columns=expected_features)

    return X


def run_pra_over_gauntlet(reconstructed_features, models):
    """
    Run OVER gauntlet for PRA starting from lowest threshold.

    Args:
        reconstructed_features: Dict of reconstructed PRA features
        models: Dict of loaded PRA models

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
            logger.warning(f"  Error predicting at PRA threshold {threshold}: {e}")
            break

    if highest_passed is not None:
        line = highest_passed - 0.5
        # PRA line range: 9.5 to 50.5
        if line > 9:
            return {
                'prop_type': 'OVER',
                'line': line,
                'threshold': highest_passed,
                'confidence': final_probability
            }
    return None


def run_pra_under_gauntlet(reconstructed_features, models):
    """
    Run UNDER gauntlet for PRA starting from high threshold.

    Args:
        reconstructed_features: Dict of reconstructed PRA features
        models: Dict of loaded PRA models

    Returns:
        dict or None: Prediction result if passed threshold
    """
    thresholds = sorted(models.keys())
    season_avg = reconstructed_features.get('season_avg', 0)
    max_threshold = max(thresholds) if thresholds else 51
    starting_threshold = min(max_threshold, max(15, int(season_avg) + 5))

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
            logger.warning(f"  Error predicting UNDER at PRA threshold {threshold}: {e}")
            break

    if lowest_passed is not None:
        line = lowest_passed + 0.5
        # PRA line range: 9.5 to 50.5
        if line > 9:
            return {
                'prop_type': 'UNDER',
                'line': line,
                'threshold': lowest_passed,
                'confidence': under_confidence
            }
    return None


def run_gauntlet_for_player(player_row, models, team_matchups, full_df):
    """
    Run PRA gauntlet (OVER and UNDER) for a single player using reconstructed features.

    Args:
        player_row: DataFrame row with player data and all percentage columns
        models: Dictionary of loaded models
        team_matchups: Matchup info from RotoWire
        full_df: Full processed dataset for feature reconstruction

    Returns:
        list: List of prediction dicts (can be 0, 1, or 2 predictions)
    """
    player_id = player_row['Player_ID']
    player_name = player_row['PLAYER']
    team = player_row['TEAM']

    # Get matchup info
    matchup_info = team_matchups.get(team)
    if not matchup_info:
        logger.warning(f"  No matchup info for {player_name} ({team})")
        return []

    opponent = matchup_info['opponent']
    is_home = matchup_info['is_home']
    matchup_str = f"{team} {'vs' if is_home else '@'} {opponent}"

    # Reconstruct features including current game's PRA for all thresholds
    thresholds = sorted(models.keys())
    reconstructed_features = reconstruct_features_for_prediction(
        player_row, full_df, thresholds, opponent  # Pass today's opponent
    )

    # Run both gauntlets
    over_pred = run_pra_over_gauntlet(reconstructed_features, models)
    under_pred = run_pra_under_gauntlet(reconstructed_features, models)

    # Build prediction list
    predictions = []

    if over_pred:
        predictions.append({
            'NAME': player_name,
            'MATCHUP': matchup_str,
            'GAME_DATE': datetime.today().strftime('%b %d, %Y'),
            'PROP': f"Over {over_pred['line']} PRA",
            'LINE': over_pred['line'],
            'CONFIDENCE_SCORE': over_pred['confidence'],
            'PLAYER_ID': player_id,
            'OPPONENT': opponent,
            'THRESHOLD': over_pred['threshold'],
            'player_row': player_row,
            'reconstructed_features': reconstructed_features
        })
        logger.info(f"  ✓ {player_name}: OVER {over_pred['line']} PRA (prob={over_pred['confidence']:.3f})")

    if under_pred:
        predictions.append({
            'NAME': player_name,
            'MATCHUP': matchup_str,
            'GAME_DATE': datetime.today().strftime('%b %d, %Y'),
            'PROP': f"Under {under_pred['line']} PRA",
            'LINE': under_pred['line'],
            'CONFIDENCE_SCORE': under_pred['confidence'],
            'PLAYER_ID': player_id,
            'OPPONENT': opponent,
            'THRESHOLD': under_pred['threshold'],
            'player_row': player_row,
            'reconstructed_features': reconstructed_features
        })
        logger.info(f"  ✓ {player_name}: UNDER {under_pred['line']} PRA (prob={under_pred['confidence']:.3f})")

    # Check separation if both predictions exist
    if len(predictions) == 2:
        separation = abs(under_pred['line'] - over_pred['line'])
        if separation < 4.0:
            # Keep only higher confidence
            if over_pred['confidence'] >= under_pred['confidence']:
                logger.info(f"  ℹ Keeping OVER only (separation={separation:.1f}, higher confidence)")
                predictions = [predictions[0]]  # Keep OVER
            else:
                logger.info(f"  ℹ Keeping UNDER only (separation={separation:.1f}, higher confidence)")
                predictions = [predictions[1]]  # Keep UNDER
        else:
            logger.info(f"  ✓ Keeping BOTH predictions (separation={separation:.1f})")

    if len(predictions) == 0:
        logger.info(f"  ✗ {player_name}: No PRA predictions passed threshold")

    return predictions


# ============================================================================
# 4.6: CALCULATE DISPLAY METRICS
# ============================================================================

def calculate_display_metrics(reconstructed_features, threshold):
    """
    Calculate performance rates for different time periods using reconstructed features.

    Args:
        reconstructed_features: Reconstructed features dict (including current game)
        threshold: PRA threshold

    Returns:
        dict: Performance rates including lineup_pct and opp_strength
    """
    return {
        'LAST_5': reconstructed_features.get(f'last_5_pct_{threshold}', 0),
        'LAST_10': reconstructed_features.get(f'last_10_pct_{threshold}', 0),
        'THIS_SEASON': reconstructed_features.get(f'season_pct_{threshold}', 0),
        'LAST_SEASON': reconstructed_features.get(f'last_season_pct_{threshold}', 0),
        'H2H': reconstructed_features.get(f'h2h_pct_{threshold}', 0),
        'LINEUP_PCT': reconstructed_features.get(f'lineup_pct_{threshold}', 0),
        'OPP_STRENGTH': reconstructed_features.get('opp_strength', 0)
    }


# ============================================================================
# RA GAUNTLET FUNCTIONS (Rebounds + Assists)
# ============================================================================

def reconstruct_features_for_ra_prediction(player_row, full_df, thresholds, opponent):
    """
    Reconstruct RA-specific features for prediction.

    Args:
        player_row: DataFrame row with player data
        full_df: Full RA processed dataset
        thresholds: List of RA thresholds to reconstruct (5-26)
        opponent: Opponent team abbreviation

    Returns:
        dict: Reconstructed features for RA prediction
    """
    player_id = player_row['Player_ID']
    lineup_id = player_row['LINEUP_ID']
    team = player_row['TEAM']
    current_season = 22025  # 2024-25 season

    # Filter player's historical games
    player_all_games = full_df[full_df['Player_ID'] == player_id].copy()

    if len(player_all_games) == 0:
        logger.warning(f"  No historical data for player {player_id}")
        return None

    # Sort by game date
    player_all_games = player_all_games.sort_values('GAME_DATE_PARSED')

    # Get current season games (before today)
    player_season_games = player_all_games[player_all_games['SEASON_ID'] == current_season].copy()
    player_team_games = player_season_games[player_season_games['TEAM'] == team].copy()

    # Get lineup games
    player_lineup_games = player_team_games[player_team_games['LINEUP_ID'] == lineup_id].copy()

    # Get H2H games (last 3 seasons) - normalize opponent abbreviation first
    normalized_opponent = normalize_team_abbreviation(opponent, full_df, player_row.get('PLAYER', ''))
    target_seasons = [current_season, current_season - 1, current_season - 2]
    player_h2h_games = player_all_games[
        (player_all_games['OPPONENT'] == normalized_opponent) &
        (player_all_games['SEASON_ID'].isin(target_seasons))
    ].copy()
    if len(player_h2h_games) == 0:
        logger.info(f"RA H2H: No games found for {player_row.get('PLAYER', '')} vs {normalized_opponent}")

    # Base features
    features = {
        'Player_ID': player_id,
        'LINEUP_ID': lineup_id,
    }

    # Rolling averages
    features['last_5_avg'] = player_team_games.tail(5)['RA'].mean() if len(player_team_games) >= 5 else player_team_games['RA'].mean()
    features['last_10_avg'] = player_team_games.tail(10)['RA'].mean() if len(player_team_games) >= 10 else player_team_games['RA'].mean()
    features['last_20_avg'] = player_team_games.tail(20)['RA'].mean() if len(player_team_games) >= 20 else player_team_games['RA'].mean()
    features['season_avg'] = player_season_games['RA'].mean() if len(player_season_games) > 0 else 0

    # Last season average
    last_season_id = current_season - 1
    player_last_season_games = player_all_games[player_all_games['SEASON_ID'] == last_season_id]
    features['last_season_avg'] = player_last_season_games['RA'].mean() if len(player_last_season_games) > 0 else 0

    features['lineup_average'] = player_lineup_games['RA'].mean() if len(player_lineup_games) > 0 else 0
    features['h2h_avg'] = player_h2h_games['RA'].mean() if len(player_h2h_games) > 0 else 0

    # Opponent strength - get from opponent's most recent games
    features['opp_strength'] = get_opponent_strength(full_df, opponent, current_season)

    # Threshold-specific percentage features
    for threshold in thresholds:
        # Last 5 games
        last_5_games = player_team_games.tail(5)
        features[f'last_5_pct_{threshold}'] = (
            (last_5_games['RA'] >= threshold).mean() if len(last_5_games) > 0 else 0
        )

        # Last 10 games
        last_10_games = player_team_games.tail(10)
        features[f'last_10_pct_{threshold}'] = (
            (last_10_games['RA'] >= threshold).mean() if len(last_10_games) > 0 else 0
        )

        # Last 20 games
        last_20_games = player_team_games.tail(20)
        features[f'last_20_pct_{threshold}'] = (
            (last_20_games['RA'] >= threshold).mean() if len(last_20_games) > 0 else 0
        )

        # Season percentage
        features[f'season_pct_{threshold}'] = (
            (player_season_games['RA'] >= threshold).mean() if len(player_season_games) > 0 else 0
        )

        # Last season percentage - get from latest game if available
        if len(player_team_games) > 0:
            latest_game = player_team_games.iloc[-1]
            features[f'last_season_pct_{threshold}'] = latest_game.get(f'last_season_pct_{threshold}', 0)
        else:
            features[f'last_season_pct_{threshold}'] = 0

        # Lineup percentage (with fallback logic)
        if len(player_lineup_games) > 1:
            features[f'lineup_pct_{threshold}'] = (
                (player_lineup_games['RA'] >= threshold).mean()
            )
        elif len(player_lineup_games) == 1:
            # Fallback to first game with different lineup
            player_team_history = player_team_games.copy()
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

        # H2H percentage
        features[f'h2h_pct_{threshold}'] = (
            (player_h2h_games['RA'] >= threshold).mean() if len(player_h2h_games) > 0 else 0
        )

    return features


def run_ra_over_gauntlet(reconstructed_features, models):
    """
    Run OVER gauntlet for RA starting from lowest threshold.

    Args:
        reconstructed_features: Dict of reconstructed RA features
        models: Dict of loaded RA models

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
            logger.warning(f"  Error predicting at RA threshold {threshold}: {e}")
            break

    if highest_passed is not None:
        line = highest_passed - 0.5
        # RA line range: 4.5 to 25.5
        if line >= 4.5 and line <= 25.5:
            return {
                'prop_type': 'OVER',
                'line': line,
                'threshold': highest_passed,
                'confidence': final_probability
            }
    return None


def run_ra_under_gauntlet(reconstructed_features, models):
    """
    Run UNDER gauntlet for RA starting from high threshold.

    Args:
        reconstructed_features: Dict of reconstructed RA features
        models: Dict of loaded RA models

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
            logger.warning(f"  Error predicting UNDER at RA threshold {threshold}: {e}")
            break

    if lowest_passed is not None:
        line = lowest_passed + 0.5
        # RA line range: 4.5 to 25.5
        if line >= 4.5 and line <= 25.5:
            return {
                'prop_type': 'UNDER',
                'line': line,
                'threshold': lowest_passed,
                'confidence': under_confidence
            }
    return None


def run_ra_gauntlet_for_player(player_row, models, team_matchups, full_df):
    """
    Run RA gauntlet (OVER and UNDER) for a single player.

    Args:
        player_row: DataFrame row with player data
        models: Dictionary of loaded RA models
        team_matchups: Matchup info from RotoWire
        full_df: Full RA processed dataset

    Returns:
        list: List of prediction dicts (can be 0, 1, or 2 predictions)
    """
    player_id = player_row['Player_ID']
    player_name = player_row['PLAYER']
    team = player_row['TEAM']

    # Get matchup info
    matchup_info = team_matchups.get(team)
    if not matchup_info:
        logger.warning(f"  No matchup info for {player_name} ({team})")
        return []

    opponent = matchup_info['opponent']
    is_home = matchup_info['is_home']
    matchup_str = f"{team} {'vs' if is_home else '@'} {opponent}"

    # Reconstruct features for RA
    thresholds = sorted(models.keys())
    reconstructed_features = reconstruct_features_for_ra_prediction(
        player_row, full_df, thresholds, opponent
    )

    if reconstructed_features is None:
        logger.warning(f"  Failed to reconstruct RA features for {player_name}")
        return []

    # Run both gauntlets
    over_pred = run_ra_over_gauntlet(reconstructed_features, models)
    under_pred = run_ra_under_gauntlet(reconstructed_features, models)

    # Build prediction list
    predictions = []

    if over_pred:
        predictions.append({
            'NAME': player_name,
            'MATCHUP': matchup_str,
            'GAME_DATE': datetime.today().strftime('%b %d, %Y'),
            'PROP': f"Over {over_pred['line']} RA",
            'LINE': over_pred['line'],
            'CONFIDENCE_SCORE': over_pred['confidence'],
            'PLAYER_ID': player_id,
            'OPPONENT': opponent,
            'THRESHOLD': over_pred['threshold'],
            'player_row': player_row,
            'reconstructed_features': reconstructed_features
        })
        logger.info(f"  ✓ {player_name}: OVER {over_pred['line']} RA (prob={over_pred['confidence']:.3f})")

    if under_pred:
        predictions.append({
            'NAME': player_name,
            'MATCHUP': matchup_str,
            'GAME_DATE': datetime.today().strftime('%b %d, %Y'),
            'PROP': f"Under {under_pred['line']} RA",
            'LINE': under_pred['line'],
            'CONFIDENCE_SCORE': under_pred['confidence'],
            'PLAYER_ID': player_id,
            'OPPONENT': opponent,
            'THRESHOLD': under_pred['threshold'],
            'player_row': player_row,
            'reconstructed_features': reconstructed_features
        })
        logger.info(f"  ✓ {player_name}: UNDER {under_pred['line']} RA (prob={under_pred['confidence']:.3f})")

    # Check separation if both predictions exist
    if len(predictions) == 2:
        separation = abs(under_pred['line'] - over_pred['line'])
        if separation < 4.0:
            # Keep only higher confidence
            if over_pred['confidence'] >= under_pred['confidence']:
                logger.info(f"  ℹ Keeping OVER only (separation={separation:.1f}, higher confidence)")
                predictions = [predictions[0]]  # Keep OVER
            else:
                logger.info(f"  ℹ Keeping UNDER only (separation={separation:.1f}, higher confidence)")
                predictions = [predictions[1]]  # Keep UNDER
        else:
            logger.info(f"  ✓ Keeping BOTH predictions (separation={separation:.1f})")

    if len(predictions) == 0:
        logger.info(f"  ✗ {player_name}: No RA predictions passed threshold")

    return predictions


# ============================================================================
# PA GAUNTLET FUNCTIONS (Points + Assists)
# ============================================================================

def reconstruct_features_for_pa_prediction(player_row, full_df, thresholds, opponent):
    """
    Reconstruct PA-specific features for prediction.

    Args:
        player_row: DataFrame row with player data
        full_df: Full PA processed dataset
        thresholds: List of PA thresholds to reconstruct (5-39)
        opponent: Opponent team abbreviation

    Returns:
        dict: Reconstructed features for PA prediction
    """
    player_id = player_row['Player_ID']
    lineup_id = player_row['LINEUP_ID']
    team = player_row['TEAM']
    current_season = 22025  # 2024-25 season

    # Filter player's historical games
    player_all_games = full_df[full_df['Player_ID'] == player_id].copy()

    if len(player_all_games) == 0:
        logger.warning(f"  No historical data for player {player_id}")
        return None

    # Sort by game date
    player_all_games = player_all_games.sort_values('GAME_DATE_PARSED')

    # Get current season games (before today)
    player_season_games = player_all_games[player_all_games['SEASON_ID'] == current_season].copy()
    player_team_games = player_season_games[player_season_games['TEAM'] == team].copy()

    # Get lineup games
    player_lineup_games = player_team_games[player_team_games['LINEUP_ID'] == lineup_id].copy()

    # Get H2H games (last 3 seasons) - normalize opponent abbreviation first
    normalized_opponent = normalize_team_abbreviation(opponent, full_df, player_row.get('PLAYER', ''))
    target_seasons = [current_season, current_season - 1, current_season - 2]
    player_h2h_games = player_all_games[
        (player_all_games['OPPONENT'] == normalized_opponent) &
        (player_all_games['SEASON_ID'].isin(target_seasons))
    ].copy()
    if len(player_h2h_games) == 0:
        logger.info(f"PA H2H: No games found for {player_row.get('PLAYER', '')} vs {normalized_opponent}")

    # Base features
    features = {
        'Player_ID': player_id,
        'LINEUP_ID': lineup_id,
    }

    # Rolling averages
    features['last_5_avg'] = player_team_games.tail(5)['PA'].mean() if len(player_team_games) >= 5 else player_team_games['PA'].mean()
    features['last_10_avg'] = player_team_games.tail(10)['PA'].mean() if len(player_team_games) >= 10 else player_team_games['PA'].mean()
    features['last_20_avg'] = player_team_games.tail(20)['PA'].mean() if len(player_team_games) >= 20 else player_team_games['PA'].mean()
    features['season_avg'] = player_season_games['PA'].mean() if len(player_season_games) > 0 else 0

    # Last season average
    last_season_id = current_season - 1
    player_last_season_games = player_all_games[player_all_games['SEASON_ID'] == last_season_id]
    features['last_season_avg'] = player_last_season_games['PA'].mean() if len(player_last_season_games) > 0 else 0

    features['lineup_average'] = player_lineup_games['PA'].mean() if len(player_lineup_games) > 0 else 0
    features['h2h_avg'] = player_h2h_games['PA'].mean() if len(player_h2h_games) > 0 else 0

    # Opponent strength - get from opponent's most recent games
    features['opp_strength'] = get_opponent_strength(full_df, opponent, current_season)

    # Threshold-specific percentage features
    for threshold in thresholds:
        # Last 5 games
        last_5_games = player_team_games.tail(5)
        features[f'last_5_pct_{threshold}'] = (
            (last_5_games['PA'] >= threshold).mean() if len(last_5_games) > 0 else 0
        )

        # Last 10 games
        last_10_games = player_team_games.tail(10)
        features[f'last_10_pct_{threshold}'] = (
            (last_10_games['PA'] >= threshold).mean() if len(last_10_games) > 0 else 0
        )

        # Last 20 games
        last_20_games = player_team_games.tail(20)
        features[f'last_20_pct_{threshold}'] = (
            (last_20_games['PA'] >= threshold).mean() if len(last_20_games) > 0 else 0
        )

        # Season percentage
        features[f'season_pct_{threshold}'] = (
            (player_season_games['PA'] >= threshold).mean() if len(player_season_games) > 0 else 0
        )

        # Last season percentage - get from latest game if available
        if len(player_team_games) > 0:
            latest_game = player_team_games.iloc[-1]
            features[f'last_season_pct_{threshold}'] = latest_game.get(f'last_season_pct_{threshold}', 0)
        else:
            features[f'last_season_pct_{threshold}'] = 0

        # Lineup percentage (with fallback logic)
        if len(player_lineup_games) > 1:
            features[f'lineup_pct_{threshold}'] = (
                (player_lineup_games['PA'] >= threshold).mean()
            )
        elif len(player_lineup_games) == 1:
            # Fallback to first game with different lineup
            player_team_history = player_team_games.copy()
            if len(player_team_history) > 0:
                player_team_sorted = player_team_history.sort_values('GAME_DATE_PARSED')
                lineup_changed = player_team_sorted['LINEUP_ID'].ne(player_team_sorted['LINEUP_ID'].shift(1))
                first_lineup_games = player_team_sorted[lineup_changed]
                if len(first_lineup_games) > 0:
                    features[f'lineup_pct_{threshold}'] = (
                        (first_lineup_games['PA'] >= threshold).mean()
                    )
                else:
                    features[f'lineup_pct_{threshold}'] = 0.0
            else:
                features[f'lineup_pct_{threshold}'] = 0.0
        else:
            features[f'lineup_pct_{threshold}'] = 0.0

        # H2H percentage
        features[f'h2h_pct_{threshold}'] = (
            (player_h2h_games['PA'] >= threshold).mean() if len(player_h2h_games) > 0 else 0
        )

    return features


def run_pa_over_gauntlet(reconstructed_features, models):
    """
    Run OVER gauntlet for PA starting from lowest threshold.

    Args:
        reconstructed_features: Dict of reconstructed PA features
        models: Dict of loaded PA models

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
            logger.warning(f"  Error predicting at PA threshold {threshold}: {e}")
            break

    if highest_passed is not None:
        line = highest_passed - 0.5
        # PA line range: 4.5 to 38.5
        if line >= 4.5 and line <= 38.5:
            return {
                'prop_type': 'OVER',
                'line': line,
                'threshold': highest_passed,
                'confidence': final_probability
            }
    return None


def run_pa_under_gauntlet(reconstructed_features, models):
    """
    Run UNDER gauntlet for PA starting from high threshold.

    Args:
        reconstructed_features: Dict of reconstructed PA features
        models: Dict of loaded PA models

    Returns:
        dict or None: Prediction result if passed threshold
    """
    thresholds = sorted(models.keys())
    season_avg = reconstructed_features.get('season_avg', 0)
    # Handle NaN values
    if pd.isna(season_avg):
        season_avg = 0
    max_threshold = max(thresholds) if thresholds else 39
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
            logger.warning(f"  Error predicting UNDER at PA threshold {threshold}: {e}")
            break

    if lowest_passed is not None:
        line = lowest_passed + 0.5
        # PA line range: 4.5 to 38.5
        if line >= 4.5 and line <= 38.5:
            return {
                'prop_type': 'UNDER',
                'line': line,
                'threshold': lowest_passed,
                'confidence': under_confidence
            }
    return None


def run_pa_gauntlet_for_player(player_row, models, team_matchups, full_df):
    """
    Run PA gauntlet (OVER and UNDER) for a single player.

    Args:
        player_row: DataFrame row with player data
        models: Dictionary of loaded PA models
        team_matchups: Matchup info from RotoWire
        full_df: Full PA processed dataset

    Returns:
        list: List of prediction dicts (can be 0, 1, or 2 predictions)
    """
    player_id = player_row['Player_ID']
    player_name = player_row['PLAYER']
    team = player_row['TEAM']

    # Get matchup info
    matchup_info = team_matchups.get(team)
    if not matchup_info:
        logger.warning(f"  No matchup info for {player_name} ({team})")
        return []

    opponent = matchup_info['opponent']
    is_home = matchup_info['is_home']
    matchup_str = f"{team} {'vs' if is_home else '@'} {opponent}"

    # Reconstruct features for PA
    thresholds = sorted(models.keys())
    reconstructed_features = reconstruct_features_for_pa_prediction(
        player_row, full_df, thresholds, opponent
    )

    if reconstructed_features is None:
        logger.warning(f"  Failed to reconstruct PA features for {player_name}")
        return []

    # Run both gauntlets
    over_pred = run_pa_over_gauntlet(reconstructed_features, models)
    under_pred = run_pa_under_gauntlet(reconstructed_features, models)

    # Build prediction list
    predictions = []

    if over_pred:
        predictions.append({
            'NAME': player_name,
            'MATCHUP': matchup_str,
            'GAME_DATE': datetime.today().strftime('%b %d, %Y'),
            'PROP': f"Over {over_pred['line']} PA",
            'LINE': over_pred['line'],
            'CONFIDENCE_SCORE': over_pred['confidence'],
            'PLAYER_ID': player_id,
            'OPPONENT': opponent,
            'THRESHOLD': over_pred['threshold'],
            'player_row': player_row,
            'reconstructed_features': reconstructed_features
        })
        logger.info(f"  ✓ {player_name}: OVER {over_pred['line']} PA (prob={over_pred['confidence']:.3f})")

    if under_pred:
        predictions.append({
            'NAME': player_name,
            'MATCHUP': matchup_str,
            'GAME_DATE': datetime.today().strftime('%b %d, %Y'),
            'PROP': f"Under {under_pred['line']} PA",
            'LINE': under_pred['line'],
            'CONFIDENCE_SCORE': under_pred['confidence'],
            'PLAYER_ID': player_id,
            'OPPONENT': opponent,
            'THRESHOLD': under_pred['threshold'],
            'player_row': player_row,
            'reconstructed_features': reconstructed_features
        })
        logger.info(f"  ✓ {player_name}: UNDER {under_pred['line']} PA (prob={under_pred['confidence']:.3f})")

    # Check separation if both predictions exist
    if len(predictions) == 2:
        separation = abs(under_pred['line'] - over_pred['line'])
        if separation < 4.0:
            # Keep only higher confidence
            if over_pred['confidence'] >= under_pred['confidence']:
                logger.info(f"  ℹ Keeping OVER only (separation={separation:.1f}, higher confidence)")
                predictions = [predictions[0]]  # Keep OVER
            else:
                logger.info(f"  ℹ Keeping UNDER only (separation={separation:.1f}, higher confidence)")
                predictions = [predictions[1]]  # Keep UNDER
        else:
            logger.info(f"  ✓ Keeping BOTH predictions (separation={separation:.1f})")

    if len(predictions) == 0:
        logger.info(f"  ✗ {player_name}: No PA predictions passed threshold")

    return predictions


# ============================================================================
# PR GAUNTLET FUNCTIONS (Points + Rebounds)
# ============================================================================

def reconstruct_features_for_pr_prediction(player_row, full_df, thresholds, opponent):
    """
    Reconstruct PR-specific features for prediction.

    Args:
        player_row: DataFrame row with player data
        full_df: Full PR processed dataset
        thresholds: List of PR thresholds to reconstruct (5-39)
        opponent: Opponent team abbreviation

    Returns:
        dict: Reconstructed features for PR prediction
    """
    player_id = player_row['Player_ID']
    lineup_id = player_row['LINEUP_ID']
    team = player_row['TEAM']
    current_season = 22025  # 2024-25 season

    # Filter player's historical games
    player_all_games = full_df[full_df['Player_ID'] == player_id].copy()

    if len(player_all_games) == 0:
        logger.warning(f"  No historical data for player {player_id}")
        return None

    # Sort by game date
    player_all_games = player_all_games.sort_values('GAME_DATE_PARSED')

    # Get current season games (before today)
    player_season_games = player_all_games[player_all_games['SEASON_ID'] == current_season].copy()
    player_team_games = player_season_games[player_season_games['TEAM'] == team].copy()

    # Get lineup games
    player_lineup_games = player_team_games[player_team_games['LINEUP_ID'] == lineup_id].copy()

    # Get H2H games (last 3 seasons) - normalize opponent abbreviation first
    normalized_opponent = normalize_team_abbreviation(opponent, full_df, player_row.get('PLAYER', ''))
    target_seasons = [current_season, current_season - 1, current_season - 2]
    player_h2h_games = player_all_games[
        (player_all_games['OPPONENT'] == normalized_opponent) &
        (player_all_games['SEASON_ID'].isin(target_seasons))
    ].copy()
    if len(player_h2h_games) == 0:
        logger.info(f"PR H2H: No games found for {player_row.get('PLAYER', '')} vs {normalized_opponent}")

    # Base features
    features = {
        'Player_ID': player_id,
        'LINEUP_ID': lineup_id,
    }

    # Rolling averages
    features['last_5_avg'] = player_team_games.tail(5)['PR'].mean() if len(player_team_games) >= 5 else player_team_games['PR'].mean()
    features['last_10_avg'] = player_team_games.tail(10)['PR'].mean() if len(player_team_games) >= 10 else player_team_games['PR'].mean()
    features['last_20_avg'] = player_team_games.tail(20)['PR'].mean() if len(player_team_games) >= 20 else player_team_games['PR'].mean()
    features['season_avg'] = player_season_games['PR'].mean() if len(player_season_games) > 0 else 0

    # Last season average
    last_season_id = current_season - 1
    player_last_season_games = player_all_games[player_all_games['SEASON_ID'] == last_season_id]
    features['last_season_avg'] = player_last_season_games['PR'].mean() if len(player_last_season_games) > 0 else 0

    features['lineup_average'] = player_lineup_games['PR'].mean() if len(player_lineup_games) > 0 else 0
    features['h2h_avg'] = player_h2h_games['PR'].mean() if len(player_h2h_games) > 0 else 0

    # Opponent strength - get from opponent's most recent games
    features['opp_strength'] = get_opponent_strength(full_df, opponent, current_season)

    # Threshold-specific percentage features
    for threshold in thresholds:
        # Last 5 games
        last_5_games = player_team_games.tail(5)
        features[f'last_5_pct_{threshold}'] = (
            (last_5_games['PR'] >= threshold).mean() if len(last_5_games) > 0 else 0
        )

        # Last 10 games
        last_10_games = player_team_games.tail(10)
        features[f'last_10_pct_{threshold}'] = (
            (last_10_games['PR'] >= threshold).mean() if len(last_10_games) > 0 else 0
        )

        # Last 20 games
        last_20_games = player_team_games.tail(20)
        features[f'last_20_pct_{threshold}'] = (
            (last_20_games['PR'] >= threshold).mean() if len(last_20_games) > 0 else 0
        )

        # Season percentage
        features[f'season_pct_{threshold}'] = (
            (player_season_games['PR'] >= threshold).mean() if len(player_season_games) > 0 else 0
        )

        # Last season percentage - get from latest game if available
        if len(player_team_games) > 0:
            latest_game = player_team_games.iloc[-1]
            features[f'last_season_pct_{threshold}'] = latest_game.get(f'last_season_pct_{threshold}', 0)
        else:
            features[f'last_season_pct_{threshold}'] = 0

        # Lineup percentage (with fallback logic)
        if len(player_lineup_games) > 1:
            features[f'lineup_pct_{threshold}'] = (
                (player_lineup_games['PR'] >= threshold).mean()
            )
        elif len(player_lineup_games) == 1:
            # Fallback to first game with different lineup
            player_team_history = player_team_games.copy()
            if len(player_team_history) > 0:
                player_team_sorted = player_team_history.sort_values('GAME_DATE_PARSED')
                lineup_changed = player_team_sorted['LINEUP_ID'].ne(player_team_sorted['LINEUP_ID'].shift(1))
                first_lineup_games = player_team_sorted[lineup_changed]
                if len(first_lineup_games) > 0:
                    features[f'lineup_pct_{threshold}'] = (
                        (first_lineup_games['PR'] >= threshold).mean()
                    )
                else:
                    features[f'lineup_pct_{threshold}'] = 0.0
            else:
                features[f'lineup_pct_{threshold}'] = 0.0
        else:
            features[f'lineup_pct_{threshold}'] = 0.0

        # H2H percentage
        features[f'h2h_pct_{threshold}'] = (
            (player_h2h_games['PR'] >= threshold).mean() if len(player_h2h_games) > 0 else 0
        )

    return features


def run_pr_over_gauntlet(reconstructed_features, models):
    """
    Run OVER gauntlet for PR starting from lowest threshold.

    Args:
        reconstructed_features: Dict of reconstructed PR features
        models: Dict of loaded PR models

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
            logger.warning(f"  Error predicting at PR threshold {threshold}: {e}")
            break

    if highest_passed is not None:
        line = highest_passed - 0.5
        # PR line range: 4.5 to 38.5
        if line >= 4.5 and line <= 38.5:
            return {
                'prop_type': 'OVER',
                'line': line,
                'threshold': highest_passed,
                'confidence': final_probability
            }
    return None


def run_pr_under_gauntlet(reconstructed_features, models):
    """
    Run UNDER gauntlet for PR starting from high threshold.

    Args:
        reconstructed_features: Dict of reconstructed PR features
        models: Dict of loaded PR models

    Returns:
        dict or None: Prediction result if passed threshold
    """
    thresholds = sorted(models.keys())
    season_avg = reconstructed_features.get('season_avg', 0)
    # Handle NaN values
    if pd.isna(season_avg):
        season_avg = 0
    max_threshold = max(thresholds) if thresholds else 39
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
            logger.warning(f"  Error predicting UNDER at PR threshold {threshold}: {e}")
            break

    if lowest_passed is not None:
        line = lowest_passed + 0.5
        # PR line range: 4.5 to 38.5
        if line >= 4.5 and line <= 38.5:
            return {
                'prop_type': 'UNDER',
                'line': line,
                'threshold': lowest_passed,
                'confidence': under_confidence
            }
    return None


def run_pr_gauntlet_for_player(player_row, models, team_matchups, full_df):
    """
    Run PR gauntlet (OVER and UNDER) for a single player.

    Args:
        player_row: DataFrame row with player data
        models: Dictionary of loaded PR models
        team_matchups: Matchup info from RotoWire
        full_df: Full PR processed dataset

    Returns:
        list: List of prediction dicts (can be 0, 1, or 2 predictions)
    """
    player_id = player_row['Player_ID']
    player_name = player_row['PLAYER']
    team = player_row['TEAM']

    # Get matchup info
    matchup_info = team_matchups.get(team)
    if not matchup_info:
        logger.warning(f"  No matchup info for {player_name} ({team})")
        return []

    opponent = matchup_info['opponent']
    is_home = matchup_info['is_home']
    matchup_str = f"{team} {'vs' if is_home else '@'} {opponent}"

    # Reconstruct features for PR
    thresholds = sorted(models.keys())
    reconstructed_features = reconstruct_features_for_pr_prediction(
        player_row, full_df, thresholds, opponent
    )

    if reconstructed_features is None:
        logger.warning(f"  Failed to reconstruct PR features for {player_name}")
        return []

    # Run both gauntlets
    over_pred = run_pr_over_gauntlet(reconstructed_features, models)
    under_pred = run_pr_under_gauntlet(reconstructed_features, models)

    # Build prediction list
    predictions = []

    if over_pred:
        predictions.append({
            'NAME': player_name,
            'MATCHUP': matchup_str,
            'GAME_DATE': datetime.today().strftime('%b %d, %Y'),
            'PROP': f"Over {over_pred['line']} PR",
            'LINE': over_pred['line'],
            'CONFIDENCE_SCORE': over_pred['confidence'],
            'PLAYER_ID': player_id,
            'OPPONENT': opponent,
            'THRESHOLD': over_pred['threshold'],
            'player_row': player_row,
            'reconstructed_features': reconstructed_features
        })
        logger.info(f"  ✓ {player_name}: OVER {over_pred['line']} PR (prob={over_pred['confidence']:.3f})")

    if under_pred:
        predictions.append({
            'NAME': player_name,
            'MATCHUP': matchup_str,
            'GAME_DATE': datetime.today().strftime('%b %d, %Y'),
            'PROP': f"Under {under_pred['line']} PR",
            'LINE': under_pred['line'],
            'CONFIDENCE_SCORE': under_pred['confidence'],
            'PLAYER_ID': player_id,
            'OPPONENT': opponent,
            'THRESHOLD': under_pred['threshold'],
            'player_row': player_row,
            'reconstructed_features': reconstructed_features
        })
        logger.info(f"  ✓ {player_name}: UNDER {under_pred['line']} PR (prob={under_pred['confidence']:.3f})")

    # Check separation if both predictions exist
    if len(predictions) == 2:
        separation = abs(under_pred['line'] - over_pred['line'])
        if separation < 4.0:
            # Keep only higher confidence
            if over_pred['confidence'] >= under_pred['confidence']:
                logger.info(f"  ℹ Keeping OVER only (separation={separation:.1f}, higher confidence)")
                predictions = [predictions[0]]  # Keep OVER
            else:
                logger.info(f"  ℹ Keeping UNDER only (separation={separation:.1f}, higher confidence)")
                predictions = [predictions[1]]  # Keep UNDER
        else:
            logger.info(f"  ✓ Keeping BOTH predictions (separation={separation:.1f})")

    if len(predictions) == 0:
        logger.info(f"  ✗ {player_name}: No PR predictions passed threshold")

    return predictions


# ============================================================================
# 4.7: FETCH OPPONENT RANKINGS
# ============================================================================

def fetch_opponent_rankings_batch(opponents):
    """
    Fetch opponent defensive rankings from Basketball Reference.

    Args:
        opponents: Set of opponent team abbreviations

    Returns:
        dict: {
            'LAL': {'OPP_PTS_RANK': 5, 'OPP_REB_RANK': 12, 'OPP_AST_RANK': 8},
            ...
        }
    """
    logger.info(f"Fetching opponent rankings for {len(opponents)} teams...")

    try:
        # Fetch from Basketball Reference
        url_opp = "https://www.basketball-reference.com/leagues/NBA_2026.html"
        tables = pd.read_html(url_opp)
        opp_stats = tables[5]  # Opponent stats table

        # Team abbreviation to name mapping (includes alternate abbreviations)
        team_name_map = {
            'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics',
            'BRK': 'Brooklyn Nets', 'BKN': 'Brooklyn Nets',  # BRK or BKN
            'CHI': 'Chicago Bulls',
            'CHO': 'Charlotte Hornets', 'CHA': 'Charlotte Hornets',  # CHO or CHA
            'CLE': 'Cleveland Cavaliers',
            'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
            'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
            'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
            'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
            'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
            'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
            'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
            'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
        }

        rankings = {}

        for opp_abbr in opponents:
            try:
                team_name = team_name_map.get(opp_abbr)
                if not team_name:
                    logger.warning(f"  Unknown team abbreviation: {opp_abbr}")
                    rankings[opp_abbr] = {
                        'OPP_PTS_RANK': 0,
                        'OPP_REB_RANK': 0,
                        'OPP_AST_RANK': 0
                    }
                    continue

                # Filter out "League Average" row before ranking
                opp_stats_filtered = opp_stats[~opp_stats['Team'].str.contains('League Average', case=False, na=False)].copy()

                # Calculate rankings (lower is better for defense)
                # Add 1 to convert from 0-based index to 1-based ranking
                opp_pts_rk = opp_stats_filtered.sort_values('PTS', ascending=True, ignore_index=True)
                pts_rnk = opp_pts_rk.index[opp_pts_rk['Team'] == team_name].tolist()
                pts_rnk = (pts_rnk[0] + 1) if len(pts_rnk) > 0 else 0

                opp_trb_rk = opp_stats_filtered.sort_values('TRB', ascending=True, ignore_index=True)
                trb_rnk = opp_trb_rk.index[opp_trb_rk['Team'] == team_name].tolist()
                trb_rnk = (trb_rnk[0] + 1) if len(trb_rnk) > 0 else 0

                opp_ast_rk = opp_stats_filtered.sort_values('AST', ascending=True, ignore_index=True)
                ast_rnk = opp_ast_rk.index[opp_ast_rk['Team'] == team_name].tolist()
                ast_rnk = (ast_rnk[0] + 1) if len(ast_rnk) > 0 else 0

                rankings[opp_abbr] = {
                    'OPP_PTS_RANK': pts_rnk,
                    'OPP_REB_RANK': trb_rnk,
                    'OPP_AST_RANK': ast_rnk
                }

            except Exception as e:
                logger.warning(f"  Error fetching rankings for {opp_abbr}: {e}")
                rankings[opp_abbr] = {
                    'OPP_PTS_RANK': 0,
                    'OPP_REB_RANK': 0,
                    'OPP_AST_RANK': 0
                }

        logger.info(f"✓ Fetched opponent rankings")
        return rankings

    except Exception as e:
        logger.error(f"Error fetching opponent rankings: {e}")
        # Return empty rankings for all opponents
        return {opp: {'OPP_PTS_RANK': 0, 'OPP_REB_RANK': 0, 'OPP_AST_RANK': 0} for opp in opponents}


# ============================================================================
# 4.8: CONSTRUCT FINAL DATAFRAME
# ============================================================================

def construct_predictions_dataframe(predictions, opponent_rankings):
    """
    Construct final predictions DataFrame using reconstructed features.

    Args:
        predictions: List of prediction dicts from gauntlet
        opponent_rankings: Dict of opponent rankings

    Returns:
        DataFrame: Final predictions in Neon format
    """
    logger.info("Constructing final predictions DataFrame...")

    props_list = []

    for pred in predictions:
        # Get display metrics using the threshold and reconstructed features
        threshold = pred['THRESHOLD']
        reconstructed_features = pred['reconstructed_features']
        display_metrics = calculate_display_metrics(reconstructed_features, threshold)

        # Get opponent rankings
        opp_ranks = opponent_rankings.get(pred['OPPONENT'], {
            'OPP_PTS_RANK': 0,
            'OPP_REB_RANK': 0,
            'OPP_AST_RANK': 0
        })

        props_list.append({
            'NAME': pred['NAME'],
            'MATCHUP': pred['MATCHUP'],
            'GAME_DATE': pred['GAME_DATE'],
            'PROP': pred['PROP'],
            'LINE': pred['LINE'],
            'CONFIDENCE_SCORE': pred['CONFIDENCE_SCORE'],
            'LAST_5': display_metrics['LAST_5'],
            'LAST_10': display_metrics['LAST_10'],
            'THIS_SEASON': display_metrics['THIS_SEASON'],
            'LAST_SEASON': display_metrics['LAST_SEASON'],
            'H2H': display_metrics['H2H'],
            'LINEUP_PCT': display_metrics['LINEUP_PCT'],
            'OPP_STRENGTH': display_metrics['OPP_STRENGTH'],
            'OPP_PTS_RANK': opp_ranks['OPP_PTS_RANK'],
            'OPP_REB_RANK': opp_ranks['OPP_REB_RANK'],
            'OPP_AST_RANK': opp_ranks['OPP_AST_RANK'],
            'STAT_TYPE': pred.get('stat_type', 'PRA')
        })

    df = pd.DataFrame(props_list)

    if len(df) == 0:
        logger.warning("No predictions to construct DataFrame")
        return df

    # Sort by confidence score (highest first)
    df = df.sort_values('CONFIDENCE_SCORE', ascending=False)

    # Filter based on stat type (PRA: LINE > 9, RA: LINE >= 4.5, PA/PR: LINE >= 4.5)
    if len(df) > 0:
        df = df[
            ((df['STAT_TYPE'] == 'PRA') & (df['LINE'] > 9)) |
            ((df['STAT_TYPE'] == 'RA') & (df['LINE'] >= 4.5)) |
            ((df['STAT_TYPE'] == 'PA') & (df['LINE'] >= 4.5)) |
            ((df['STAT_TYPE'] == 'PR') & (df['LINE'] >= 4.5))
        ]

    # Fill NaN with 0
    df = df.fillna(0)

    logger.info(f"✓ Constructed DataFrame with {len(df)} predictions")
    return df


# ============================================================================
# 4.8: INJURY STATUS FILTERING
# ============================================================================

def get_injured_players():
    """
    Scrape ESPN injuries page to get list of injured players and their return dates.

    Returns:
        dict: Dictionary mapping player names to return date strings
              Example: {'Josh Minott': 'Jan 10', 'CJ McCollum': 'Jan 9'}
    """
    try:
        logger.info("Fetching injury data from ESPN...")
        url = 'https://www.espn.com/nba/injuries'
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        injured_players = {}

        # Find all injury tables (one per team)
        tables = soup.find_all('table', class_='Table')

        for table in tables:
            tbody = table.find('tbody')
            if not tbody:
                continue

            rows = tbody.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 4:
                    # ESPN table columns: Name, Position, Status, Comment, Date
                    name_cell = cells[0]
                    date_cell = cells[-1]  # Last column is usually the date

                    # Extract player name
                    player_name = name_cell.get_text(strip=True)
                    # Extract return date
                    return_date = date_cell.get_text(strip=True)

                    if player_name and return_date:
                        injured_players[player_name] = return_date

        logger.info(f"  Found {len(injured_players)} injured players")
        return injured_players

    except Exception as e:
        logger.warning(f"  Failed to fetch injury data from ESPN: {e}")
        logger.warning("  Continuing without injury filtering")
        return {}


def should_exclude_player(return_date_str, today):
    """
    Determine if a player should be excluded based on their return date.

    Args:
        return_date_str: Return date string from ESPN (e.g., "Jan 10", "Expected to be out until...")
        today: Today's date as datetime.date object

    Returns:
        bool: True if player should be excluded (return date is in future), False otherwise
    """
    if not return_date_str or return_date_str.strip() == '':
        # No return date means player is active or day-to-day
        return False

    # If it's a long description, it's likely a future return
    if len(return_date_str) > 15 or 'expected' in return_date_str.lower() or 'out' in return_date_str.lower():
        return True

    try:
        # Try to parse date like "Jan 10"
        # Assume current year
        current_year = today.year
        date_parts = return_date_str.split()

        if len(date_parts) >= 2:
            month_str = date_parts[0]
            day_str = date_parts[1].replace(',', '')

            # Parse month and day
            from datetime import datetime
            date_str = f"{month_str} {day_str} {current_year}"
            return_date = datetime.strptime(date_str, "%b %d %Y").date()

            # If return date is in the future (after today), exclude player
            if return_date > today:
                return True

    except Exception:
        # If we can't parse, assume it's a description and exclude
        return True

    return False


def filter_injured_players(props_df, today):
    """
    Filter out players who are injured and not returning today.

    Args:
        props_df: DataFrame with predictions
        today: Today's date as datetime.date object

    Returns:
        DataFrame: Filtered predictions with injured players removed
    """
    initial_count = len(props_df)

    # Get injured players
    injured_players = get_injured_players()

    if not injured_players:
        logger.info("No injury data available, skipping injury filtering")
        return props_df

    # Filter out injured players
    excluded_players = []

    for player_name, return_date_str in injured_players.items():
        if should_exclude_player(return_date_str, today):
            excluded_players.append(player_name)

    if excluded_players:
        logger.info(f"Excluding {len(excluded_players)} injured players:")
        for player in excluded_players:
            return_date = injured_players.get(player, 'Unknown')
            logger.info(f"  - {player} (return: {return_date})")

        # Filter DataFrame - remove rows where NAME matches any excluded player
        props_df = props_df[~props_df['NAME'].isin(excluded_players)].copy()

        filtered_count = initial_count - len(props_df)
        logger.info(f"Removed {filtered_count} predictions for injured players")
    else:
        logger.info("No players need to be excluded based on injury status")

    return props_df


# ============================================================================
# 4.9: UPLOAD TO NEON DATABASE
# ============================================================================

def upload_predictions_to_neon(props_df):
    """
    Upload predictions to Neon database.

    Args:
        props_df: DataFrame with predictions
    """
    logger.info("Uploading predictions to Neon database...")

    if len(props_df) == 0:
        logger.warning("No predictions to upload")
        return

    # Connect to Neon
    conn = psycopg2.connect(os.getenv('NEON_DATABASE_URL'))
    cursor = conn.cursor()

    try:
        # Truncate existing table
        cursor.execute("TRUNCATE TABLE nba_props RESTART IDENTITY")
        logger.info("  Truncated existing nba_props table")

        # Rename columns to match DB schema (lowercase)
        df_renamed = props_df.rename(columns={
            'NAME': 'name',
            'MATCHUP': 'matchup',
            'GAME_DATE': 'game_date',
            'PROP': 'prop',
            'LINE': 'line',
            'CONFIDENCE_SCORE': 'confidence_score',
            'LAST_5': 'last_5',
            'LAST_10': 'last_10',
            'THIS_SEASON': 'this_season',
            'LAST_SEASON': 'last_season',
            'H2H': 'h2h',
            'LINEUP_PCT': 'lineup_pct',
            'OPP_STRENGTH': 'opp_strength',
            'OPP_PTS_RANK': 'opp_pts_rank',
            'OPP_REB_RANK': 'opp_reb_rank',
            'OPP_AST_RANK': 'opp_ast_rank',
            'STAT_TYPE': 'stat_type'
        })

        # Replace NaN with None for SQL
        df_renamed = df_renamed.where(pd.notna(df_renamed), None)

        # Define columns in order
        columns = ['name', 'matchup', 'game_date', 'prop', 'line', 'confidence_score',
                   'last_5', 'last_10', 'this_season', 'last_season', 'h2h',
                   'lineup_pct', 'opp_strength', 'opp_pts_rank', 'opp_reb_rank', 'opp_ast_rank', 'stat_type']

        # Prepare values
        values = [tuple(row) for row in df_renamed[columns].values]

        # Insert
        insert_query = f"""
            INSERT INTO nba_props ({', '.join(columns)})
            VALUES %s
        """
        execute_values(cursor, insert_query, values)

        # Commit
        conn.commit()
        logger.info(f"✓ Uploaded {len(props_df)} predictions to Neon")

    except Exception as e:
        conn.rollback()
        logger.error(f"Error uploading to Neon: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


# ============================================================================
# STAT-TYPE SPECIFIC PREDICTION RUNNER
# ============================================================================

def run_stat_type_predictions(stat_type, s3_handler, teams_playing, team_matchups, league_leaders_df):
    """
    Run predictions for a specific stat type (PRA, RA, PA, or PR).

    This function handles:
    1. Loading stat-specific processed data
    2. Loading stat-specific models
    3. Running gauntlet for each player
    4. Returning predictions with stat_type included

    Args:
        stat_type: Stat type ('PRA', 'RA', 'PA', 'PR')
        s3_handler: S3Handler instance
        teams_playing: Set of teams playing today
        team_matchups: Dict of team matchup info
        league_leaders_df: DataFrame of league leaders

    Returns:
        list: Predictions for this stat type
    """
    logger.info("=" * 80)
    logger.info(f"RUNNING {stat_type} PREDICTIONS")
    logger.info("=" * 80)

    try:
        # Get players playing today for this stat type
        players_today_df, full_df = get_players_playing_today(
            s3_handler, teams_playing, league_leaders_df, stat_type=stat_type
        )

        if len(players_today_df) == 0:
            logger.warning(f"No players with {stat_type} data playing today")
            return []

        logger.info(f"Players with {stat_type} data playing today: {len(players_today_df)}")

        # Load models for this stat type
        models = load_all_models(s3_handler, stat_type=stat_type)

        if len(models) == 0:
            logger.error(f"Failed to load {stat_type} models from S3")
            return []

        # Run gauntlet for each player
        logger.info(f"Running {stat_type} gauntlet for each player...")
        predictions = []

        for idx, player_row in players_today_df.iterrows():
            if stat_type == 'PRA':
                # Use PRA gauntlet logic (can return 0, 1, or 2 predictions)
                pra_preds = run_gauntlet_for_player(player_row, models, team_matchups, full_df)
                for pred in pra_preds:
                    pred['stat_type'] = stat_type
                    predictions.append(pred)
            elif stat_type == 'RA':
                # Use RA-specific gauntlet logic (can return 0, 1, or 2 predictions)
                ra_preds = run_ra_gauntlet_for_player(player_row, models, team_matchups, full_df)
                for pred in ra_preds:
                    pred['stat_type'] = stat_type
                    predictions.append(pred)
            elif stat_type == 'PA':
                # Use PA-specific gauntlet logic (can return 0, 1, or 2 predictions)
                pa_preds = run_pa_gauntlet_for_player(player_row, models, team_matchups, full_df)
                for pred in pa_preds:
                    pred['stat_type'] = stat_type
                    predictions.append(pred)
            elif stat_type == 'PR':
                # Use PR-specific gauntlet logic (can return 0, 1, or 2 predictions)
                pr_preds = run_pr_gauntlet_for_player(player_row, models, team_matchups, full_df)
                for pred in pr_preds:
                    pred['stat_type'] = stat_type
                    predictions.append(pred)

        logger.info(f"{stat_type} predictions generated: {len(predictions)}")
        return predictions

    except Exception as e:
        logger.error(f"Error running {stat_type} predictions: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================================
# UNIFIED PHASE 4 - ALL STAT TYPES
# ============================================================================

def run_phase_4_unified(s3_handler):
    """
    Execute Phase 4: Daily Prediction Generation for ALL stat types.

    This is the unified version that runs predictions for:
    - PRA (Points + Rebounds + Assists)
    - RA (Rebounds + Assists)
    - PA (Points + Assists) - when ready
    - PR (Points + Rebounds) - when ready

    Args:
        s3_handler: S3Handler instance

    Returns:
        Tuple of (success, stats dict)
    """
    logger.info("=" * 80)
    logger.info("PHASE 4: UNIFIED DAILY PREDICTION GENERATION (ALL STAT TYPES)")
    logger.info("=" * 80)

    try:
        # ========================================================================
        # SHARED SETUP (ONCE FOR ALL STAT TYPES)
        # ========================================================================

        # Get today's date
        today = date.today()

        # Scrape RotoWire for today's games
        teams_data = scrape_rotowire_for_teams_playing_today()
        teams_playing = teams_data['teams_playing']
        team_matchups = teams_data['team_matchups']

        if len(teams_playing) == 0:
            logger.warning("No teams playing today - exiting")
            return False, {
                'error': 'No teams playing today',
                'teams_playing': 0,
                'predictions_by_stat': {}
            }

        logger.info(f"Teams playing today: {len(teams_playing)}")

        # Fetch league leaders (ONCE)
        league_leaders_df = fetch_league_leaders()
        if len(league_leaders_df) == 0:
            logger.error("Failed to fetch league leaders")
            return False, {'error': 'Failed to fetch league leaders'}

        logger.info(f"League leaders fetched: {len(league_leaders_df)}")

        # ========================================================================
        # RUN PREDICTIONS FOR EACH STAT TYPE
        # ========================================================================

        all_predictions = []
        predictions_by_stat = {}

        # Define which stat types to run (PRA, RA, PA, PR)
        stat_types_to_run = ['PRA', 'RA', 'PA', 'PR']

        for stat_type in stat_types_to_run:
            stat_predictions = run_stat_type_predictions(
                stat_type, s3_handler, teams_playing, team_matchups, league_leaders_df
            )
            all_predictions.extend(stat_predictions)
            predictions_by_stat[stat_type] = len(stat_predictions)

        logger.info(f"\nTotal predictions across all stat types: {len(all_predictions)}")

        if len(all_predictions) == 0:
            logger.warning("No predictions passed the gauntlet threshold for any stat type")
            return True, {
                'teams_playing': len(teams_playing),
                'predictions_by_stat': predictions_by_stat,
                'total_predictions': 0
            }

        # ========================================================================
        # FINALIZE AND UPLOAD
        # ========================================================================

        # Fetch opponent rankings
        unique_opponents = set(pred['OPPONENT'] for pred in all_predictions)
        opponent_rankings = fetch_opponent_rankings_batch(unique_opponents)

        # Construct final DataFrame
        props_df = construct_predictions_dataframe(all_predictions, opponent_rankings)

        # Filter out injured players
        props_df = filter_injured_players(props_df, today)

        # Upload to Neon (includes stat_type column)
        upload_predictions_to_neon(props_df)

        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ PHASE 4 UNIFIED COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Teams playing: {len(teams_playing)}")
        for stat_type, count in predictions_by_stat.items():
            logger.info(f"{stat_type} predictions: {count}")
        logger.info(f"Total predictions uploaded: {len(props_df)}")
        logger.info("=" * 80)

        return True, {
            'teams_playing': len(teams_playing),
            'predictions_by_stat': predictions_by_stat,
            'total_predictions': len(props_df)
        }

    except Exception as e:
        logger.error(f"Phase 4 Unified failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}


# ============================================================================
# MAIN PHASE 4 EXECUTION (LEGACY - PRA ONLY)
# ============================================================================
# NOTE: This function is DEPRECATED. Use run_phase_4_unified() instead.
# Kept for backwards compatibility only.

def run_phase_4(s3_handler):
    """
    Execute Phase 4: Daily Prediction Generation (PRA ONLY - DEPRECATED).

    DEPRECATED: Use run_phase_4_unified() instead for multi-stat support.

    Args:
        s3_handler: S3Handler instance

    Returns:
        Tuple of (success, stats dict)
    """
    logger.info("=" * 80)
    logger.info("PHASE 4: DAILY PREDICTION GENERATION")
    logger.info("=" * 80)

    try:
        # Get today's date
        today = date.today()

        # 4.1: Scrape RotoWire
        teams_data = scrape_rotowire_for_teams_playing_today()
        teams_playing = teams_data['teams_playing']
        team_matchups = teams_data['team_matchups']

        if len(teams_playing) == 0:
            logger.warning("No teams playing today - exiting")
            return False, {
                'error': 'No teams playing today',
                'teams_playing': 0,
                'players_evaluated': 0,
                'predictions_generated': 0,
                'predictions_uploaded': 0
            }

        logger.info(f"Teams playing today: {len(teams_playing)}")

        # 4.2: Get players playing today
        league_leaders_df = fetch_league_leaders()
        if len(league_leaders_df) == 0:
            logger.error("Failed to fetch league leaders")
            return False, {'error': 'Failed to fetch league leaders'}

        players_today_df, full_df = get_players_playing_today(
            s3_handler, teams_playing, league_leaders_df
        )

        if len(players_today_df) == 0:
            logger.warning("No players from league leaders playing today")
            return False, {'error': 'No players playing today'}

        logger.info(f"Players playing today: {len(players_today_df)}")

        # 4.3: Load models
        models = load_all_models(s3_handler)

        if len(models) == 0:
            logger.error("Failed to load models from S3")
            return False, {'error': 'Failed to load models'}

        # 4.4: Run gauntlet for each player (with feature reconstruction)
        logger.info("Running model gauntlet for each player (reconstructing features)...")
        predictions = []

        for idx, player_row in players_today_df.iterrows():
            pra_preds = run_gauntlet_for_player(player_row, models, team_matchups, full_df)
            predictions.extend(pra_preds)

        logger.info(f"Predictions generated: {len(predictions)}")

        if len(predictions) == 0:
            logger.warning("No predictions passed the gauntlet threshold")
            return True, {
                'teams_playing': len(teams_playing),
                'players_evaluated': len(players_today_df),
                'predictions_generated': 0,
                'predictions_uploaded': 0
            }

        # 4.7: Fetch opponent rankings
        unique_opponents = set(pred['OPPONENT'] for pred in predictions)
        opponent_rankings = fetch_opponent_rankings_batch(unique_opponents)

        # 4.8: Construct final DataFrame
        props_df = construct_predictions_dataframe(predictions, opponent_rankings)

        # Filter out injured players
        props_df = filter_injured_players(props_df, today)

        # 4.9: Upload to Neon
        upload_predictions_to_neon(props_df)

        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ PHASE 4 COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Teams playing: {len(teams_playing)}")
        logger.info(f"Players evaluated: {len(players_today_df)}")
        logger.info(f"Predictions generated: {len(predictions)}")
        logger.info(f"Predictions uploaded: {len(props_df)}")
        logger.info("=" * 80)

        # Display top 5 predictions
        if len(props_df) > 0:
            logger.info("\nTop 5 Predictions by Confidence Score:")
            logger.info("-" * 80)
            top_5 = props_df.head(5)[['NAME', 'MATCHUP', 'PROP', 'CONFIDENCE_SCORE']]
            for idx, row in top_5.iterrows():
                logger.info(f"  {row['NAME']}: {row['PROP']} ({row['MATCHUP']}) - {row['CONFIDENCE_SCORE']:.3f}")
            props_df.to_csv('props.csv')

        return True, {
            'teams_playing': len(teams_playing),
            'players_evaluated': len(players_today_df),
            'predictions_generated': len(predictions),
            'predictions_uploaded': len(props_df)
        }

    except Exception as e:
        logger.error(f"Phase 4 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False, {'error': str(e)}


if __name__ == '__main__':
    """Execute Phase 4 when script is run directly."""
    from s3_utils import S3Handler
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Initialize S3 handler
    s3_handler = S3Handler()

    # Run Phase 4 (Unified - supports PRA and RA)
    success, stats = run_phase_4_unified(s3_handler)

    # Exit with appropriate status code
    import sys
    sys.exit(0 if success else 1)
