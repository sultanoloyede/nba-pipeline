"""
Phase 4: Daily Prediction Generation

This module generates daily NBA player prop predictions using a model gauntlet approach.
Players from today's games are evaluated through a series of ML models for multiple stat types:
- PRA (Points + Rebounds + Assists): 10+ through 51+
- PA (Points + Assists): 8+ through 41+
- PR (Points + Rebounds): 8+ through 41+
- RA (Rebounds + Assists): 5+ through 26+

Predictions stop when confidence drops below 78%.

IMPORTANT: Feature Reconstruction for Prediction
The processed data uses shift(1) to exclude the current game during training. However,
when predicting FUTURE games, the most recent game is now in the PAST. Therefore, we
reconstruct all features (rolling averages, percentages) to INCLUDE the current game's
stat value. This ensures predictions are based on the most up-to-date performance data.

Author: Generated for NBA Props Prediction System
Date: 2025-12-29
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

# Stat-type specific configurations
STAT_TYPE_CONFIG = {
    'PRA': {
        'threshold_start': 10,
        'threshold_end': 52,  # 10-51 inclusive
        'min_line': 9.5,
        'max_line': 50.5,
        'league_filter': 10.0  # Filter for 10+ PRA players
    },
    'PA': {
        'threshold_start': 8,
        'threshold_end': 42,  # 8-41 inclusive
        'min_line': 7.5,
        'max_line': 40.5,
        'league_filter': 8.0
    },
    'PR': {
        'threshold_start': 8,
        'threshold_end': 42,  # 8-41 inclusive
        'min_line': 7.5,
        'max_line': 40.5,
        'league_filter': 8.0
    },
    'RA': {
        'threshold_start': 5,
        'threshold_end': 27,  # 5-26 inclusive
        'min_line': 4.5,
        'max_line': 25.5,
        'league_filter': 5.0
    }
}


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
    Also calculates PA, PR, and RA for all players.

    This approach includes ALL players regardless of games played percentage,
    unlike LeagueLeaders which filters for players with 70%+ games played.

    Returns:
        DataFrame: League leaders data with columns: PLAYER_ID, PLAYER, TEAM, PTS, REB, AST, PRA, PA, PR, RA
    """
    logger.info("Fetching all players with 10+ PRA (including players with low game counts)...")

    try:
        ll_df = leaguedashplayerstats.LeagueDashPlayerStats(per_mode_detailed='PerGame').get_data_frames()[0]

        # Calculate all stat types
        ll_df['PRA'] = ll_df['PTS'] + ll_df['REB'] + ll_df['AST']
        ll_df['PA'] = ll_df['PTS'] + ll_df['AST']
        ll_df['PR'] = ll_df['PTS'] + ll_df['REB']
        ll_df['RA'] = ll_df['REB'] + ll_df['AST']

        # Filter 10+ PRA (keep existing logic for player filtering)
        ll_df = ll_df[ll_df['PRA'] >= 10.0]
        ll_df = ll_df.sort_values('PRA', ascending=False)

        logger.info(f"✓ Found {len(ll_df)} players with PRA >= 10")

        # Map columns to match expected format (LeagueDashPlayerStats uses PLAYER_NAME instead of PLAYER)
        return ll_df[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'PTS', 'REB', 'AST', 'PRA', 'PA', 'PR', 'RA']].rename(
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
        league_leaders_df: League leaders with 10+ PRA
        stat_type: Stat type to load data for ('PRA', 'PA', 'PR', 'RA')

    Returns:
        Tuple of (filtered_df, full_df):
            - filtered_df: Filtered rows (most recent per player on teams playing today)
            - full_df: Full processed dataset (for feature reconstruction)
    """
    from s3_utils import S3_PLAYER_BUCKET

    logger.info(f"Loading {stat_type} processed data and filtering for today's players...")

    # Get configuration for this stat type
    config = STAT_TYPE_CONFIG[stat_type]
    threshold_start = config['threshold_start']
    threshold_end = config['threshold_end'] - 1  # Make it inclusive

    # Download processed data with all percentages for this stat type
    df = s3_handler.download_dataframe(
        S3_PLAYER_BUCKET,
        f'processed_data/processed_with_{stat_type.lower()}_pct_{threshold_start}-{threshold_end}.csv'
    )

    logger.info(f"Loaded {len(df):,} rows from processed data")

    # Ensure all stat columns exist in the dataframe
    if 'PTS' in df.columns and 'REB' in df.columns and 'AST' in df.columns:
        # Calculate all stat types if they don't exist
        if 'PRA' not in df.columns:
            df['PRA'] = df['PTS'] + df['REB'] + df['AST']
        if 'PA' not in df.columns:
            df['PA'] = df['PTS'] + df['AST']
        if 'PR' not in df.columns:
            df['PR'] = df['PTS'] + df['REB']
        if 'RA' not in df.columns:
            df['RA'] = df['REB'] + df['AST']

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

def load_all_models(s3_handler, stat_type='PRA', threshold_start=10, threshold_end=52):
    """
    Load all trained models from S3 for specified stat type.

    Args:
        s3_handler: S3Handler instance
        stat_type: Stat type to load models for ('PRA', 'PA', 'PR', 'RA')
        threshold_start: Starting threshold (default: 10)
        threshold_end: Ending threshold (exclusive, default: 52 for 10-51)

    Returns:
        dict: {10: model_10, 11: model_11, ..., 51: model_51}
    """
    from s3_utils import S3_MODEL_BUCKET
    import pickle

    logger.info(f"Loading {stat_type} models for thresholds {threshold_start}-{threshold_end-1}...")

    models = {}

    for threshold in range(threshold_start, threshold_end):
        try:
            # List all model files for this threshold
            model_prefix = f'models/xgb_{stat_type.lower()}_{threshold}plus_'

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

    logger.info(f"✓ Loaded {len(models)} models from S3")
    return models


# ============================================================================
# 4.4: RECONSTRUCT FEATURES FOR PREDICTION
# ============================================================================

def reconstruct_features_for_prediction(player_row, full_df, thresholds, today_opponent, stat_type='PRA'):
    """
    Reconstruct features by INCLUDING the current game's stat value for future prediction.

    The processed data uses shift(1) to exclude the current game during training.
    For prediction, we need to include the current game's stat value in all rolling windows
    and percentages, as it's now in the past relative to the game we're predicting.

    Args:
        player_row: Most recent game row with existing features
        full_df: Full processed dataset with all player histories
        thresholds: List of thresholds for the stat type
        today_opponent: Today's actual opponent from team matchups (not from player_row)
        stat_type: Stat type to reconstruct features for ('PRA', 'PA', 'PR', 'RA')

    Returns:
        dict: Reconstructed features including current game's stat value
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

    # Get H2H games (last 3 seasons)
    target_seasons = [current_season, current_season - 1, current_season - 2]
    player_h2h_games = player_all_games[
        (player_all_games['OPPONENT'] == opponent) &
        (player_all_games['SEASON_ID'].isin(target_seasons))
    ]

    # ========================================================================
    # BASE FEATURES (same for all thresholds)
    # ========================================================================

    features = {
        'Player_ID': player_id,
        'LINEUP_ID': lineup_id,
    }

    # Rolling averages (including current game)
    features['last_5_avg'] = player_team_games.tail(5)[stat_type].mean() if len(player_team_games) > 0 else 0
    features['last_10_avg'] = player_team_games.tail(10)[stat_type].mean() if len(player_team_games) > 0 else 0
    features['last_20_avg'] = player_team_games.tail(20)[stat_type].mean() if len(player_team_games) > 0 else 0

    # Season average (including current game)
    features['season_avg'] = player_season_games[stat_type].mean() if len(player_season_games) > 0 else 0

    # Last season average (unchanged - copy from player_row)
    features['last_season_avg'] = player_row.get('last_season_avg', 0)

    # Lineup average (including current game)
    features['lineup_average'] = player_lineup_games[stat_type].mean() if len(player_lineup_games) > 0 else 0

    # H2H average (including current game)
    features['h2h_avg'] = player_h2h_games[stat_type].mean() if len(player_h2h_games) > 0 else 0

    # Opponent strength (we'll use the value from player_row as it requires complex calculation)
    features['opp_strength'] = player_row.get('opp_strength', 0)

    # ========================================================================
    # THRESHOLD-SPECIFIC PERCENTAGE FEATURES
    # ========================================================================

    for threshold in thresholds:
        # Rolling percentages (including current game)
        last_5_games = player_team_games.tail(5)
        features[f'last_5_pct_{threshold}'] = (
            (last_5_games[stat_type] >= threshold).mean() if len(last_5_games) > 0 else 0
        )

        last_10_games = player_team_games.tail(10)
        features[f'last_10_pct_{threshold}'] = (
            (last_10_games[stat_type] >= threshold).mean() if len(last_10_games) > 0 else 0
        )

        last_20_games = player_team_games.tail(20)
        features[f'last_20_pct_{threshold}'] = (
            (last_20_games[stat_type] >= threshold).mean() if len(last_20_games) > 0 else 0
        )

        # Season percentage (including current game)
        features[f'season_pct_{threshold}'] = (
            (player_season_games[stat_type] >= threshold).mean() if len(player_season_games) > 0 else 0
        )

        # Last season percentage (unchanged - copy from player_row)
        features[f'last_season_pct_{threshold}'] = player_row.get(f'last_season_pct_{threshold}', 0)

        # Lineup percentage (including current game)
        if len(player_lineup_games) > 1:
            # Has previous games with this lineup - use actual percentage
            features[f'lineup_pct_{threshold}'] = (
                (player_lineup_games[stat_type] >= threshold).mean()
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
                        (first_lineup_games[stat_type] >= threshold).mean()
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
            (player_h2h_games[stat_type] >= threshold).mean() if len(player_h2h_games) > 0 else 0
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


def run_over_gauntlet_for_player(player_row, models, team_matchups, full_df, stat_type='PRA', min_line=9.5, max_line=50.5):
    """
    Run OVER gauntlet for a single player using reconstructed features.

    Starts at lowest threshold and increments until probability drops below 78%.

    Args:
        player_row: DataFrame row with player data and all percentage columns
        models: Dictionary of loaded models
        team_matchups: Matchup info from RotoWire
        full_df: Full processed dataset for feature reconstruction
        stat_type: Stat type being predicted ('PRA', 'PA', 'PR', 'RA')
        min_line: Minimum line threshold for this stat type
        max_line: Maximum line threshold for this stat type

    Returns:
        dict or None: Prediction result if passed threshold, else None
    """
    player_id = player_row['Player_ID']
    player_name = player_row['PLAYER']
    team = player_row['TEAM']

    # Get matchup info
    matchup_info = team_matchups.get(team)
    if not matchup_info:
        logger.warning(f"  No matchup info for {player_name} ({team})")
        return None

    opponent = matchup_info['opponent']
    is_home = matchup_info['is_home']
    matchup_str = f"{team} {'vs' if is_home else '@'} {opponent}"

    # Reconstruct features including current game's stat value for all thresholds
    thresholds = sorted(models.keys())
    reconstructed_features = reconstruct_features_for_prediction(
        player_row, full_df, thresholds, opponent, stat_type  # Pass today's opponent and stat type
    )

    # Run gauntlet (start at 10, go until failure)
    highest_passed = None
    final_probability = 0.0

    for threshold in thresholds:
        # Get features for this threshold (base + threshold-specific percentages)
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

        # Prepare input for model
        try:
            X = prepare_model_input(threshold_features, models[threshold])

            # Get prediction probability
            prob = models[threshold].predict_proba(X)[0][1]

            # Check if passed threshold
            if prob >= CONFIDENCE_THRESHOLD:
                highest_passed = threshold
                final_probability = prob
            else:
                # Failed - stop gauntlet
                break

        except Exception as e:
            logger.warning(f"  Error predicting for {player_name} at threshold {threshold}: {e}")
            break

    # If passed at least one threshold, create prediction
    # Check line is within valid range
    if highest_passed is not None:
        line = highest_passed - 0.5
        if line >= min_line and line <= max_line:
            logger.info(f"  ✓ {player_name} OVER: Passed up to {highest_passed}+ {stat_type} (prob={final_probability:.3f})")
            return {
                'NAME': player_name,
                'MATCHUP': matchup_str,
                'GAME_DATE': datetime.today().strftime('%b %d, %Y'),
                'PROP_TYPE': 'OVER',
                'PROP': f"Over {highest_passed - 0.5} {stat_type}",
                'LINE': highest_passed - 0.5,
                'STAT_TYPE': stat_type,  # NEW field
                'CONFIDENCE_SCORE': final_probability,
                'PLAYER_ID': player_id,
                'OPPONENT': opponent,
                'THRESHOLD': highest_passed,
                'player_row': player_row,  # Keep for display metrics
                'reconstructed_features': reconstructed_features  # Keep reconstructed features
            }
    return None


def run_under_gauntlet_for_player(player_row, models, team_matchups, full_df, stat_type='PRA', min_line=9.5, max_line=50.5):
    """
    Run UNDER gauntlet for a single player using reconstructed features.

    Starts at high threshold (based on season avg) and decrements until probability goes above 22%.
    Returns predictions where probability <= 22% (meaning 78%+ chance of staying under).

    Args:
        player_row: DataFrame row with player data and all percentage columns
        models: Dictionary of loaded models
        team_matchups: Matchup info from RotoWire
        full_df: Full processed dataset for feature reconstruction
        stat_type: Stat type being predicted ('PRA', 'PA', 'PR', 'RA')
        min_line: Minimum line threshold for this stat type
        max_line: Maximum line threshold for this stat type

    Returns:
        dict or None: Prediction result if passed threshold, else None
    """
    player_id = player_row['Player_ID']
    player_name = player_row['PLAYER']
    team = player_row['TEAM']

    # Get matchup info
    matchup_info = team_matchups.get(team)
    if not matchup_info:
        return None

    opponent = matchup_info['opponent']
    is_home = matchup_info['is_home']
    matchup_str = f"{team} {'vs' if is_home else '@'} {opponent}"

    # Reconstruct features including current game's stat value for all thresholds
    thresholds = sorted(models.keys())
    reconstructed_features = reconstruct_features_for_prediction(
        player_row, full_df, thresholds, opponent, stat_type
    )

    # Determine starting threshold for under gauntlet (dynamic based on player's season avg)
    season_avg = reconstructed_features.get('season_avg', 0)
    max_threshold = max(thresholds) if thresholds else 51
    starting_threshold = min(max_threshold, max(15, int(season_avg) + 5))  # Start a bit above their average

    # Run under gauntlet (start high, go down until failure)
    lowest_passed = None
    under_confidence = 0.0

    for threshold in reversed(thresholds):
        # Skip thresholds above our starting point
        if threshold > starting_threshold:
            continue

        # Get features for this threshold
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

        # Prepare input for model
        try:
            X = prepare_model_input(threshold_features, models[threshold])

            # Get prediction probability
            prob = models[threshold].predict_proba(X)[0][1]

            # Check if passed under threshold (LOW probability of hitting = good under bet)
            if prob <= (1 - CONFIDENCE_THRESHOLD):  # 22% or less
                lowest_passed = threshold
                under_confidence = 1 - prob  # Convert to "under confidence" (78%+)
            else:
                # Probability too high - stop gauntlet
                break

        except Exception as e:
            logger.warning(f"  Error predicting UNDER for {player_name} at threshold {threshold}: {e}")
            break

    # If passed at least one threshold, create under prediction
    # Check line is within valid range
    if lowest_passed is not None:
        line = lowest_passed + 0.5
        if line >= min_line and line <= max_line:
            logger.info(f"  ✓ {player_name} UNDER: Passed down to {lowest_passed}+ {stat_type} (under_prob={under_confidence:.3f})")
            return {
                'NAME': player_name,
                'MATCHUP': matchup_str,
                'GAME_DATE': datetime.today().strftime('%b %d, %Y'),
                'PROP_TYPE': 'UNDER',
                'PROP': f"Under {lowest_passed + 0.5} {stat_type}",
                'LINE': lowest_passed + 0.5,
                'STAT_TYPE': stat_type,  # NEW field
                'CONFIDENCE_SCORE': under_confidence,  # This is the confidence of going UNDER (78%+)
                'PLAYER_ID': player_id,
                'OPPONENT': opponent,
                'THRESHOLD': lowest_passed,
                'player_row': player_row,
                'reconstructed_features': reconstructed_features
            }
    return None


def run_both_gauntlets_for_player(player_row, models, team_matchups, full_df, stat_type='PRA', min_line=9.5, max_line=50.5):
    """
    Run both OVER and UNDER gauntlets for a player.

    Returns predictions that meet the criteria:
    - Both predictions if they have 4+ point separation
    - Single prediction if only one passes
    - No predictions if both fail

    Args:
        player_row: DataFrame row with player data and all percentage columns
        models: Dictionary of loaded models
        team_matchups: Matchup info from RotoWire
        full_df: Full processed dataset for feature reconstruction
        stat_type: Stat type being predicted ('PRA', 'PA', 'PR', 'RA')
        min_line: Minimum line threshold for this stat type
        max_line: Maximum line threshold for this stat type

    Returns:
        list: List of prediction dicts (0, 1, or 2 predictions)
    """
    predictions = []

    # Run over gauntlet
    over_pred = run_over_gauntlet_for_player(player_row, models, team_matchups, full_df, stat_type, min_line, max_line)
    if over_pred:
        predictions.append(over_pred)

    # Run under gauntlet
    under_pred = run_under_gauntlet_for_player(player_row, models, team_matchups, full_df, stat_type, min_line, max_line)
    if under_pred:
        predictions.append(under_pred)

    # If we have both, check for 4+ point separation
    if len(predictions) == 2:
        over_line = over_pred['LINE']
        under_line = under_pred['LINE']
        separation = abs(under_line - over_line)

        if separation < 4.0:
            # Not enough separation - keep only the higher confidence prediction
            if over_pred['CONFIDENCE_SCORE'] >= under_pred['CONFIDENCE_SCORE']:
                logger.info(f"    Keeping OVER (higher confidence, separation={separation:.1f})")
                return [over_pred]
            else:
                logger.info(f"    Keeping UNDER (higher confidence, separation={separation:.1f})")
                return [under_pred]
        else:
            logger.info(f"    Keeping BOTH (separation={separation:.1f})")

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
        dict: Performance rates
    """
    return {
        'LAST_5': reconstructed_features.get(f'last_5_pct_{threshold}', 0),
        'LAST_10': reconstructed_features.get(f'last_10_pct_{threshold}', 0),
        'THIS_SEASON': reconstructed_features.get(f'season_pct_{threshold}', 0),
        'LAST_SEASON': reconstructed_features.get(f'last_season_pct_{threshold}', 0),
        'H2H': reconstructed_features.get(f'h2h_pct_{threshold}', 0)
    }


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

        # Filter out League Average row to prevent 31st rank issue
        opp_stats = opp_stats[opp_stats['Team'] != 'League Average'].reset_index(drop=True)
        logger.info(f"Loaded opponent stats for {len(opp_stats)} teams (League Average row filtered out)")

        # Team abbreviation to name mapping
        # Note: RotoWire uses CHA and BKN, but some sources use CHO and BRK
        team_name_map = {
            'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics',
            'BRK': 'Brooklyn Nets', 'BKN': 'Brooklyn Nets',  # Both BRK and BKN for Brooklyn
            'CHI': 'Chicago Bulls',
            'CHO': 'Charlotte Hornets', 'CHA': 'Charlotte Hornets',  # Both CHO and CHA for Charlotte
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

                # Calculate rankings (lower is better for defense)
                # Note: Add 1 to convert from 0-indexed (0-29) to 1-indexed (1-30)
                opp_pts_rk = opp_stats.sort_values('PTS', ascending=True, ignore_index=True)
                pts_rnk = opp_pts_rk.index[opp_pts_rk['Team'] == team_name].tolist()
                pts_rnk = pts_rnk[0] + 1 if len(pts_rnk) > 0 else 0

                opp_trb_rk = opp_stats.sort_values('TRB', ascending=True, ignore_index=True)
                trb_rnk = opp_trb_rk.index[opp_trb_rk['Team'] == team_name].tolist()
                trb_rnk = trb_rnk[0] + 1 if len(trb_rnk) > 0 else 0

                opp_ast_rk = opp_stats.sort_values('AST', ascending=True, ignore_index=True)
                ast_rnk = opp_ast_rk.index[opp_ast_rk['Team'] == team_name].tolist()
                ast_rnk = ast_rnk[0] + 1 if len(ast_rnk) > 0 else 0

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

        # For UNDER predictions, use threshold + 1 to get correct percentage
        # Example: "Under 20.5" means PRA <= 20, which is 1 - P(PRA >= 21)
        if pred['PROP_TYPE'] == 'UNDER':
            # Calculate metrics for threshold + 1, then invert
            display_metrics = calculate_display_metrics(reconstructed_features, threshold + 1)
            display_metrics = {
                key: 1 - value
                for key, value in display_metrics.items()
            }
        else:
            # For OVER predictions, use threshold as-is
            display_metrics = calculate_display_metrics(reconstructed_features, threshold)

        # Get opponent rankings
        opp_ranks = opponent_rankings.get(pred['OPPONENT'], {
            'OPP_PTS_RANK': 0,
            'OPP_REB_RANK': 0,
            'OPP_AST_RANK': 0
        })

        # Get lineup_pct for this threshold
        lineup_pct = reconstructed_features.get(f'lineup_pct_{threshold}', 0)

        props_list.append({
            'NAME': pred['NAME'],
            'MATCHUP': pred['MATCHUP'],
            'GAME_DATE': pred['GAME_DATE'],
            'PROP_TYPE': pred['PROP_TYPE'],
            'PROP': pred['PROP'],
            'LINE': pred['LINE'],
            'STAT_TYPE': pred['STAT_TYPE'],  # NEW field
            'CONFIDENCE_SCORE': pred['CONFIDENCE_SCORE'],
            'LAST_5': display_metrics['LAST_5'],
            'LAST_10': display_metrics['LAST_10'],
            'THIS_SEASON': display_metrics['THIS_SEASON'],
            'LAST_SEASON': display_metrics['LAST_SEASON'],
            'H2H': display_metrics['H2H'],
            'LINEUP_PCT': lineup_pct,
            'OPP_PTS_RANK': opp_ranks['OPP_PTS_RANK'],
            'OPP_REB_RANK': opp_ranks['OPP_REB_RANK'],
            'OPP_AST_RANK': opp_ranks['OPP_AST_RANK']
        })

    df = pd.DataFrame(props_list)

    if len(df) == 0:
        logger.warning("No predictions to construct DataFrame")
        return df

    # Sort by confidence score (highest first)
    df = df.sort_values('CONFIDENCE_SCORE', ascending=False)

    # Line filtering is already applied in gauntlet functions per stat type

    # Fill NaN with 0
    df = df.fillna(0)

    logger.info(f"✓ Constructed DataFrame with {len(df)} predictions")
    return df


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
            'PROP_TYPE': 'prop_type',
            'PROP': 'prop',
            'LINE': 'line',
            'STAT_TYPE': 'stat_type',  # NEW
            'CONFIDENCE_SCORE': 'confidence_score',
            'LAST_5': 'last_5',
            'LAST_10': 'last_10',
            'THIS_SEASON': 'this_season',
            'LAST_SEASON': 'last_season',
            'H2H': 'h2h',
            'LINEUP_PCT': 'lineup_pct',
            'OPP_PTS_RANK': 'opp_pts_rank',
            'OPP_REB_RANK': 'opp_reb_rank',
            'OPP_AST_RANK': 'opp_ast_rank'
        })

        # Replace NaN with None for SQL
        df_renamed = df_renamed.where(pd.notna(df_renamed), None)

        # Define columns in order
        columns = ['name', 'matchup', 'game_date', 'prop_type', 'prop', 'line', 'stat_type',  # Added stat_type
                   'confidence_score', 'last_5', 'last_10', 'this_season', 'last_season', 'h2h', 'lineup_pct',
                   'opp_pts_rank', 'opp_reb_rank', 'opp_ast_rank']

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
# STAT-TYPE SPECIFIC EXECUTION
# ============================================================================

def run_phase_4_for_stat_type(s3_handler, teams_playing, team_matchups, league_leaders_df,
                               stat_type='PRA', threshold_start=10, threshold_end=52,
                               min_line=9.5, max_line=50.5):
    """
    Run Phase 4 for a specific stat type.

    Args:
        s3_handler: S3Handler instance
        teams_playing: Set of team abbreviations playing today
        team_matchups: Matchup info from RotoWire
        league_leaders_df: League leaders DataFrame
        stat_type: Stat type to generate predictions for
        threshold_start: Starting threshold for this stat type
        threshold_end: Ending threshold
        min_line: Minimum line for filtering predictions
        max_line: Maximum line for filtering predictions

    Returns:
        List of predictions for this stat type
    """
    logger.info(f"Loading {stat_type} data and models...")

    # Load models for this stat type
    models = load_all_models(s3_handler, stat_type, threshold_start, threshold_end)

    if len(models) == 0:
        logger.warning(f"No models found for {stat_type}")
        return []

    # Get players playing today with processed data for this stat type
    players_today_df, full_df = get_players_playing_today(
        s3_handler, teams_playing, league_leaders_df, stat_type
    )

    if len(players_today_df) == 0:
        logger.warning(f"No players from league leaders playing today for {stat_type}")
        return []

    logger.info(f"Running {stat_type} gauntlets for {len(players_today_df)} players...")

    # Run gauntlets for each player
    predictions = []
    for _, player_row in players_today_df.iterrows():
        player_preds = run_both_gauntlets_for_player(
            player_row, models, team_matchups, full_df,
            stat_type=stat_type,
            min_line=min_line,
            max_line=max_line
        )
        predictions.extend(player_preds)

    logger.info(f"Generated {len(predictions)} {stat_type} predictions")
    return predictions


# ============================================================================
# MAIN PHASE 4 EXECUTION
# ============================================================================

def run_phase_4(s3_handler):
    """
    Execute Phase 4: Daily Prediction Generation for ALL stat types.

    Args:
        s3_handler: S3Handler instance

    Returns:
        Tuple of (success, stats dict)
    """
    logger.info("=" * 80)
    logger.info("PHASE 4: DAILY PREDICTION GENERATION (ALL STAT TYPES)")
    logger.info("=" * 80)

    try:
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

        # 4.2: Fetch league leaders (calculates all stat types)
        league_leaders_df = fetch_league_leaders()
        if len(league_leaders_df) == 0:
            logger.error("Failed to fetch league leaders")
            return False, {'error': 'Failed to fetch league leaders'}

        # 4.3: Generate predictions for ALL stat types
        all_predictions = []
        stat_counts = {}

        for stat_type, config in STAT_TYPE_CONFIG.items():
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"Generating {stat_type} Predictions")
            logger.info("=" * 80)

            stat_predictions = run_phase_4_for_stat_type(
                s3_handler,
                teams_playing,
                team_matchups,
                league_leaders_df,
                stat_type=stat_type,
                threshold_start=config['threshold_start'],
                threshold_end=config['threshold_end'],
                min_line=config['min_line'],
                max_line=config['max_line']
            )

            all_predictions.extend(stat_predictions)
            stat_counts[stat_type] = len(stat_predictions)
            logger.info(f"✓ Generated {len(stat_predictions)} {stat_type} predictions")

        logger.info(f"\nTotal predictions generated: {len(all_predictions)}")
        for stat_type, count in stat_counts.items():
            logger.info(f"  {stat_type}: {count} predictions")

        if len(all_predictions) == 0:
            logger.warning("No predictions passed the gauntlet threshold")
            return True, {
                'teams_playing': len(teams_playing),
                'predictions_generated': 0,
                'predictions_uploaded': 0,
                'stat_counts': stat_counts
            }

        # 4.7: Fetch opponent rankings
        unique_opponents = set(pred['OPPONENT'] for pred in all_predictions)
        opponent_rankings = fetch_opponent_rankings_batch(unique_opponents)

        # 4.8: Construct final DataFrame
        props_df = construct_predictions_dataframe(all_predictions, opponent_rankings)

        # 4.9: Upload to Neon
        upload_predictions_to_neon(props_df)

        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ PHASE 4 COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Teams playing: {len(teams_playing)}")
        logger.info(f"Predictions generated: {len(all_predictions)}")
        logger.info(f"Predictions uploaded: {len(props_df)}")
        for stat_type, count in stat_counts.items():
            logger.info(f"  {stat_type}: {count} predictions")
        logger.info("=" * 80)

        # Display top predictions by stat type
        if len(props_df) > 0:
            logger.info("\nTop Predictions by Stat Type and Confidence Score:")
            logger.info("-" * 80)

            # Show top predictions for each stat type
            for stat_type in STAT_TYPE_CONFIG.keys():
                stat_df = props_df[props_df['STAT_TYPE'] == stat_type]
                if len(stat_df) > 0:
                    logger.info(f"\nTop {stat_type} Predictions:")
                    top_preds = stat_df.head(3)
                    for _, row in top_preds.iterrows():
                        logger.info(f"  {row['NAME']}: {row['PROP']} ({row['MATCHUP']}) - {row['CONFIDENCE_SCORE']:.3f}")

            props_df.to_csv('props.csv')

        return True, {
            'teams_playing': len(teams_playing),
            'predictions_generated': len(all_predictions),
            'predictions_uploaded': len(props_df),
            'stat_counts': stat_counts
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

    # Run Phase 4
    success, stats = run_phase_4(s3_handler)

    # Exit with appropriate status code
    import sys
    sys.exit(0 if success else 1)
