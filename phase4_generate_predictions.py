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


def get_players_playing_today(s3_handler, teams_playing, league_leaders_df):
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

    Returns:
        Tuple of (filtered_df, full_df):
            - filtered_df: Filtered rows (most recent per player on teams playing today)
            - full_df: Full processed dataset (for feature reconstruction)
    """
    from s3_utils import S3_PLAYER_BUCKET

    logger.info("Loading processed data and filtering for today's players...")

    # Download processed data with all percentages
    df = s3_handler.download_dataframe(
        S3_PLAYER_BUCKET,
        'processed_data/processed_with_all_pct_10-51.csv'
    )

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

def load_all_models(s3_handler, threshold_start=10, threshold_end=52):
    """
    Load all trained models from S3.

    Args:
        s3_handler: S3Handler instance
        threshold_start: Starting threshold (default: 10)
        threshold_end: Ending threshold (exclusive, default: 52 for 10-51)

    Returns:
        dict: {10: model_10, 11: model_11, ..., 51: model_51}
    """
    from s3_utils import S3_MODEL_BUCKET
    import pickle

    logger.info(f"Loading models for thresholds {threshold_start}-{threshold_end-1}...")

    models = {}

    for threshold in range(threshold_start, threshold_end):
        try:
            # List all model files for this threshold
            model_prefix = f'models/xgb_pra_{threshold}plus_'

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

    # Opponent strength (we'll use the value from player_row as it requires complex calculation)
    features['opp_strength'] = player_row.get('opp_strength', 0)

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


def run_gauntlet_for_player(player_row, models, team_matchups, full_df):
    """
    Run model gauntlet for a single player using reconstructed features.

    Args:
        player_row: DataFrame row with player data and all percentage columns
        models: Dictionary of loaded models
        team_matchups: Matchup info from RotoWire
        full_df: Full processed dataset for feature reconstruction

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

    # Reconstruct features including current game's PRA for all thresholds
    thresholds = sorted(models.keys())
    reconstructed_features = reconstruct_features_for_prediction(
        player_row, full_df, thresholds, opponent  # Pass today's opponent
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
    if highest_passed is not None:
        logger.info(f"  ✓ {player_name}: Passed up to {highest_passed}+ PRA (prob={final_probability:.3f})")
        return {
            'NAME': player_name,
            'MATCHUP': matchup_str,
            'GAME_DATE': datetime.today().strftime('%b %d, %Y'),
            'PROP': f"Over {highest_passed - 0.5} PRA",
            'LINE': highest_passed - 0.5,
            'CONFIDENCE_SCORE': final_probability,
            'PLAYER_ID': player_id,
            'OPPONENT': opponent,
            'THRESHOLD': highest_passed,
            'player_row': player_row,  # Keep for display metrics
            'reconstructed_features': reconstructed_features  # Keep reconstructed features
        }
    else:
        logger.info(f"  ✗ {player_name}: Failed gauntlet (no predictions)")
        return None


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

        # Team abbreviation to name mapping
        team_name_map = {
            'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BRK': 'Brooklyn Nets',
            'CHI': 'Chicago Bulls', 'CHO': 'Charlotte Hornets', 'CLE': 'Cleveland Cavaliers',
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
                opp_pts_rk = opp_stats.sort_values('PTS', ascending=True, ignore_index=True)
                pts_rnk = opp_pts_rk.index[opp_pts_rk['Team'] == team_name].tolist()
                pts_rnk = pts_rnk[0] if len(pts_rnk) > 0 else 0

                opp_trb_rk = opp_stats.sort_values('TRB', ascending=True, ignore_index=True)
                trb_rnk = opp_trb_rk.index[opp_trb_rk['Team'] == team_name].tolist()
                trb_rnk = trb_rnk[0] if len(trb_rnk) > 0 else 0

                opp_ast_rk = opp_stats.sort_values('AST', ascending=True, ignore_index=True)
                ast_rnk = opp_ast_rk.index[opp_ast_rk['Team'] == team_name].tolist()
                ast_rnk = ast_rnk[0] if len(ast_rnk) > 0 else 0

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

    # Filter LINE > 9 (same as daily_props_generator.py)
    df = df[df['LINE'] > 9]

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
            'PROP': 'prop',
            'LINE': 'line',
            'CONFIDENCE_SCORE': 'confidence_score',
            'LAST_5': 'last_5',
            'LAST_10': 'last_10',
            'THIS_SEASON': 'this_season',
            'LAST_SEASON': 'last_season',
            'H2H': 'h2h',
            'OPP_PTS_RANK': 'opp_pts_rank',
            'OPP_REB_RANK': 'opp_reb_rank',
            'OPP_AST_RANK': 'opp_ast_rank'
        })

        # Replace NaN with None for SQL
        df_renamed = df_renamed.where(pd.notna(df_renamed), None)

        # Define columns in order
        columns = ['name', 'matchup', 'game_date', 'prop', 'line', 'confidence_score',
                   'last_5', 'last_10', 'this_season', 'last_season', 'h2h',
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
# MAIN PHASE 4 EXECUTION
# ============================================================================

def run_phase_4(s3_handler):
    """
    Execute Phase 4: Daily Prediction Generation.

    Args:
        s3_handler: S3Handler instance

    Returns:
        Tuple of (success, stats dict)
    """
    logger.info("=" * 80)
    logger.info("PHASE 4: DAILY PREDICTION GENERATION")
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
            pred = run_gauntlet_for_player(player_row, models, team_matchups, full_df)
            if pred is not None:
                predictions.append(pred)

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

    # Run Phase 4
    success, stats = run_phase_4(s3_handler)

    # Exit with appropriate status code
    import sys
    sys.exit(0 if success else 1)
