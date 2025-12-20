"""
Phase 1: Data Fetching & Synchronization (OPTIMIZED)

Key Optimizations:
1. Lineup cache as dictionary for O(1) lookups (vs DataFrame filtering)
2. Batch S3 uploads (every 10 players)
3. Enhanced progress logging to prevent Modal heartbeat timeouts
4. Incremental processing (only fetch new games)

Author: Optimized for NBA Props Prediction System
Date: 2025-12-11
"""

import pandas as pd
import json
import time
import logging
from typing import Tuple, Dict, Optional, List, Set
from datetime import datetime
from nba_api.stats.endpoints import leaguedashplayerstats, playergamelog, boxscoretraditionalv3
from requests.exceptions import Timeout, ConnectionError
from http.client import RemoteDisconnected

# Import utility modules
from s3_utils import (
    S3Handler,
    S3_PLAYER_BUCKET,
    S3_LINEUP_BUCKET,
    download_metadata_from_s3,
    upload_metadata_to_s3,
    download_lineup_cache_from_s3,
    upload_lineup_cache_to_s3
)
from metadata_utils import create_default_metadata, update_fetch_time

# Setup logger
logger = logging.getLogger(__name__)

# Configuration
SEASONS = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
API_SLEEP_TIME = 0.6
LINEUP_API_SLEEP_TIME = 6.0
MAX_RETRIES = 3
BATCH_SIZE = 10  # Upload to S3 every N players


# ============================================================================
# HELPER FUNCTIONS - NAME PARSING
# ============================================================================

def parse_player_name(player_name: str) -> Tuple[str, str]:
    """
    Parse player name into firstname and lastname for filename.

    Examples:
        - "LeBron James" -> ("lebron", "james")
        - "Karl-Anthony Towns" -> ("karl-anthony", "towns")

    Args:
        player_name: Full player name from NBA API

    Returns:
        Tuple of (firstname, lastname) in lowercase with hyphens
    """
    name_clean = player_name.strip().replace('.', '')
    name_parts = name_clean.split()

    if len(name_parts) == 0:
        return "unknown", "player"
    elif len(name_parts) == 1:
        return name_parts[0].lower(), ""
    elif len(name_parts) == 2:
        return name_parts[0].lower(), name_parts[1].lower()
    else:
        firstname = name_parts[0].lower()
        lastname = "-".join(name_parts[1:]).lower()
        return firstname, lastname


def build_player_filename(player_name: str, player_id: int) -> str:
    """
    Build S3 filename from player name and ID.

    Format: {firstname}_{lastname}_{playerid}.csv

    Args:
        player_name: Full player name
        player_id: NBA player ID

    Returns:
        Filename string
    """
    firstname, lastname = parse_player_name(player_name)

    if lastname:
        return f"{firstname}_{lastname}_{player_id}.csv"
    else:
        return f"{firstname}_{player_id}.csv"


# ============================================================================
# HELPER FUNCTIONS - API RETRY LOGIC
# ============================================================================

def retry_api_call(func, max_retries: int = MAX_RETRIES):
    """
    Retry an API call with exponential backoff.

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts

    Returns:
        Result of the function call

    Raises:
        Exception if all retries fail
    """
    for attempt in range(max_retries):
        try:
            result = func()
            return result
        except (Timeout, ConnectionError, RemoteDisconnected, Exception) as e:
            error_msg = str(e)

            is_retryable = (
                'timed out' in error_msg.lower() or
                'connection' in error_msg.lower() or
                'remote end closed' in error_msg.lower() or
                isinstance(e, (Timeout, ConnectionError, RemoteDisconnected))
            )

            if not is_retryable or attempt == max_retries - 1:
                raise

            logger.warning(f"    Attempt {attempt + 1} failed: {error_msg}")
            logger.warning(f"    Retrying in {5 * (2 ** attempt)}s...")
            time.sleep(5 * (2 ** attempt))

    raise Exception("Max retries exceeded")


# ============================================================================
# LINEUP CACHE FUNCTIONS (OPTIMIZED WITH DICT)
# ============================================================================

def create_lineup_cache_dict(cache_df: pd.DataFrame) -> Dict[Tuple[int, str], Tuple[List, int]]:
    """
    Convert lineup cache DataFrame to dictionary for O(1) lookups.

    OPTIMIZATION: Dict lookup is O(1) vs DataFrame filtering O(n)

    Args:
        cache_df: DataFrame with cached lineup data

    Returns:
        Dict mapping (game_id, team_abbr) -> (lineup_list, lineup_id)
    """
    if cache_df is None or len(cache_df) == 0:
        return {}

    cache_dict = {}

    for _, row in cache_df.iterrows():
        game_id = int(row['GAME_ID'])
        team_abbr = row['TEAM_ABBREVIATION']
        lineup = json.loads(row['LINEUP'])
        lineup_id = int(row['LINEUP_ID'])

        cache_dict[(game_id, team_abbr)] = (lineup, lineup_id)

    return cache_dict


def get_lineup_from_cache_dict(cache_dict: Dict, game_id: int,
                               team_abbreviation: str) -> Tuple[Optional[List], Optional[int]]:
    """
    Retrieve lineup information from cache dictionary (O(1) lookup).

    Args:
        cache_dict: Dictionary with cached lineup data
        game_id: Game ID to look up
        team_abbreviation: Team abbreviation

    Returns:
        tuple: (lineup_list, lineup_id) if found, (None, None) if not found
    """
    key = (int(game_id), team_abbreviation)
    return cache_dict.get(key, (None, None))


def add_to_lineup_cache_dict(cache_dict: Dict, game_id: int,
                             team_abbreviation: str, lineup: List,
                             lineup_id: int) -> Dict:
    """
    Add a new entry to the lineup cache dictionary.

    Args:
        cache_dict: Dictionary with cached lineup data
        game_id: Game ID
        team_abbreviation: Team abbreviation
        lineup: List of player slugs
        lineup_id: Lineup ID (sum of player IDs)

    Returns:
        Updated cache dictionary
    """
    key = (int(game_id), team_abbreviation)
    cache_dict[key] = (lineup, int(lineup_id))
    return cache_dict


def convert_cache_dict_to_df(cache_dict: Dict) -> pd.DataFrame:
    """
    Convert lineup cache dictionary back to DataFrame for S3 storage.

    Args:
        cache_dict: Dictionary with cached lineup data

    Returns:
        DataFrame with columns: GAME_ID, TEAM_ABBREVIATION, LINEUP, LINEUP_ID
    """
    if len(cache_dict) == 0:
        return pd.DataFrame(columns=['GAME_ID', 'TEAM_ABBREVIATION', 'LINEUP', 'LINEUP_ID'])

    rows = []
    for (game_id, team_abbr), (lineup, lineup_id) in cache_dict.items():
        rows.append({
            'GAME_ID': game_id,
            'TEAM_ABBREVIATION': team_abbr,
            'LINEUP': json.dumps(lineup),
            'LINEUP_ID': lineup_id
        })

    return pd.DataFrame(rows)


def extract_team_from_matchup(matchup: str) -> str:
    """
    Extract team abbreviation from MATCHUP string.

    Examples:
        - "LAL vs. BOS" -> "LAL"
        - "GSW @ MIA" -> "GSW"

    Args:
        matchup: MATCHUP column value

    Returns:
        Team abbreviation
    """
    return matchup.split()[0]


# ============================================================================
# STEP 1: FETCH LEAGUE LEADERS
# ============================================================================

def fetch_league_leaders() -> pd.DataFrame:
    """
    Fetch all players with 10+ PRA using LeagueDashPlayerStats.

    This approach includes ALL players regardless of games played percentage,
    unlike LeagueLeaders which filters for players with 70%+ games played.

    Returns:
        DataFrame with columns: PLAYER_ID, PLAYER, TEAM, PTS, REB, AST, PRA
    """
    logger.info("Fetching all players with 10+ PRA (including players with low game counts)...")

    try:
        def fetch():
            return leaguedashplayerstats.LeagueDashPlayerStats(per_mode_detailed='PerGame').get_data_frames()[0]

        ll_df = retry_api_call(fetch)

        # Calculate PRA
        ll_df['PRA'] = ll_df['PTS'] + ll_df['REB'] + ll_df['AST']

        # Filter 10+ PRA
        ll_df = ll_df[ll_df['PRA'] >= 10.0]
        ll_df = ll_df.sort_values('PRA', ascending=False)

        logger.info(f"✓ Found {len(ll_df)} players with 10+ PRA")

        print(ll_df)

        # Map columns to match expected format (LeagueDashPlayerStats uses PLAYER_NAME instead of PLAYER)
        return ll_df[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'PTS', 'REB', 'AST', 'PRA']].rename(
            columns={'PLAYER_NAME': 'PLAYER', 'TEAM_ABBREVIATION': 'TEAM'}
        )

    except Exception as e:
        logger.error(f"Failed to fetch league leaders: {e}")
        raise


# ============================================================================
# STEP 2: FETCH PLAYER GAME LOGS (INCREMENTAL)
# ============================================================================

def get_latest_game_date(gamelog_df: pd.DataFrame) -> Optional[datetime]:
    """
    Extract the latest game date from a game log DataFrame.

    Args:
        gamelog_df: DataFrame with game log data

    Returns:
        datetime or None: Latest game date
    """
    if gamelog_df is None or len(gamelog_df) == 0:
        return None

    if 'GAME_DATE' not in gamelog_df.columns:
        return None

    try:
        gamelog_df['GAME_DATE_PARSED'] = pd.to_datetime(gamelog_df['GAME_DATE'], format='mixed')
        latest = gamelog_df['GAME_DATE_PARSED'].max()
        return latest
    except Exception as e:
        logger.warning(f"  Error parsing game dates: {e}")
        return None


def fetch_player_gamelogs(player_id: int, player_name: str,
                         seasons: List[str], existing_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Fetch game logs for a player across multiple seasons.

    Implements incremental updates:
    - If existing_df provided and has data, only fetch current season
    - Otherwise, fetch all seasons

    Args:
        player_id: NBA player ID
        player_name: Player's name
        seasons: List of season strings (e.g., ['2019-20', '2020-21'])
        existing_df: Existing game log DataFrame for incremental updates

    Returns:
        DataFrame: Combined game logs (new + existing)
    """
    all_gamelogs = []

    # Determine which seasons to fetch
    if existing_df is not None and len(existing_df) > 0:
        # Incremental mode: only fetch current season
        seasons_to_fetch = [seasons[-1]]
        logger.info(f"  Incremental update: checking {seasons_to_fetch[0]} only")
    else:
        # Full fetch mode: fetch all seasons
        seasons_to_fetch = seasons
        logger.info(f"  Full fetch: fetching {len(seasons_to_fetch)} seasons")

    for season in seasons_to_fetch:
        try:
            logger.info(f"    Fetching {season}...")

            def fetch_gamelog():
                return playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    season_type_all_star='Regular Season'
                ).get_data_frames()[0]

            gamelog_df = retry_api_call(fetch_gamelog)

            if len(gamelog_df) > 0:
                gamelog_df['SEASON'] = season
                all_gamelogs.append(gamelog_df)
                logger.info(f"    {season}: {len(gamelog_df)} games")
            else:
                logger.info(f"    {season}: No games")

            time.sleep(API_SLEEP_TIME)

        except Exception as e:
            logger.error(f"    {season} error: {e}")
            continue

    # Combine new data
    if len(all_gamelogs) > 0:
        new_df = pd.concat(all_gamelogs, ignore_index=True)

        # If incremental, merge with existing data
        if existing_df is not None and len(existing_df) > 0:
            combined_df = pd.concat([new_df, existing_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['Game_ID'], keep='first')

            new_games_count = len(combined_df) - len(existing_df)
            logger.info(f"  Added {new_games_count} new games")

            return combined_df
        else:
            logger.info(f"  Fetched {len(new_df)} total games")
            return new_df
    else:
        if existing_df is not None:
            logger.info(f"  No new games found")
            return existing_df
        else:
            logger.warning(f"  No data found for {player_name}")
            return pd.DataFrame()


# ============================================================================
# STEP 3: ADD LINEUP INFORMATION (OPTIMIZED)
# ============================================================================

def get_games_missing_lineup(gamelog_df: pd.DataFrame) -> Set[int]:
    """
    Extract set of game IDs that are missing lineup information.

    Args:
        gamelog_df: DataFrame with game log data

    Returns:
        set: Game IDs that need lineup information
    """
    if len(gamelog_df) == 0:
        return set()

    if 'LINEUP_ID' not in gamelog_df.columns:
        return set(gamelog_df['Game_ID'].tolist())

    missing = gamelog_df[
        (gamelog_df['LINEUP_ID'].isna()) | (gamelog_df['LINEUP_ID'] == 0.0)
    ]['Game_ID'].tolist()

    return set(missing)


def add_lineup_information_optimized(gamelog_df: pd.DataFrame, player_name: str,
                                    lineup_cache_dict: Dict,
                                    games_to_process: Optional[Set[int]] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Add starting lineup information using optimized cache dictionary (O(1) lookups).

    OPTIMIZATION: Uses dict for O(1) cache lookups instead of DataFrame filtering

    Args:
        gamelog_df: DataFrame with game log data
        player_name: Player's name for progress tracking
        lineup_cache_dict: Dictionary with cached lineup data
        games_to_process: Set of game IDs to process (None = all missing)

    Returns:
        tuple: (updated gamelog DataFrame, updated lineup cache dictionary)
    """
    if len(gamelog_df) == 0:
        return gamelog_df, lineup_cache_dict

    if games_to_process is None:
        games_to_process = get_games_missing_lineup(gamelog_df)

    if len(games_to_process) == 0:
        logger.info(f"  All games already have lineup information")
        return gamelog_df, lineup_cache_dict

    logger.info(f"  Processing lineup info for {len(games_to_process)} games")

    # Initialize columns if they don't exist
    if 'LINEUP' not in gamelog_df.columns:
        gamelog_df['LINEUP'] = None
    if 'LINEUP_ID' not in gamelog_df.columns:
        gamelog_df['LINEUP_ID'] = None

    games_processed = 0
    api_calls_made = 0
    cache_hits = 0

    for i in range(len(gamelog_df)):
        game_id = gamelog_df['Game_ID'].iloc[i]

        if game_id not in games_to_process:
            continue

        try:
            team_abbr = extract_team_from_matchup(gamelog_df['MATCHUP'].iloc[i])
            game_id_normalized = int(game_id)

            # Check cache first (O(1) lookup with dict!)
            lineup, lineup_id = get_lineup_from_cache_dict(lineup_cache_dict, game_id_normalized, team_abbr)

            if lineup is not None and lineup_id is not None:
                cache_hits += 1
            else:
                # Fetch from API
                game_id_padded = str(game_id).zfill(10)

                def fetch_boxscore():
                    bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id_padded)
                    return bs.get_data_frames()[0]

                boxscore_df = retry_api_call(fetch_boxscore)
                api_calls_made += 1

                # Extract starters
                starters = boxscore_df[boxscore_df['position'].str.len() > 0]

                # Cache lineups for BOTH teams
                teams_in_game = starters['teamTricode'].unique()
                for team_tricode in teams_in_game:
                    team_starters = starters[starters['teamTricode'] == team_tricode]
                    team_lineup_id = team_starters['personId'].sum()
                    team_lineup = team_starters['playerSlug'].values.tolist()

                    lineup_cache_dict = add_to_lineup_cache_dict(
                        lineup_cache_dict,
                        game_id_normalized,
                        team_tricode,
                        team_lineup,
                        team_lineup_id
                    )

                # Get lineup for player's team
                lineup, lineup_id = get_lineup_from_cache_dict(lineup_cache_dict, game_id_normalized, team_abbr)

                # IMPORTANT: 6-second sleep after boxscore API call
                time.sleep(LINEUP_API_SLEEP_TIME)

            # Store in gamelog
            gamelog_df.at[i, 'LINEUP'] = json.dumps(lineup)
            gamelog_df.at[i, 'LINEUP_ID'] = lineup_id

            games_processed += 1

            # Progress logging every 10 games
            if games_processed % 10 == 0:
                logger.info(f"    Progress: {games_processed}/{len(games_to_process)} (API: {api_calls_made}, Cache: {cache_hits})")

        except Exception as e:
            logger.error(f"    Error on game {game_id}: {e}")
            gamelog_df.at[i, 'LINEUP'] = None
            gamelog_df.at[i, 'LINEUP_ID'] = None
            continue

    logger.info(f"  Lineup processing complete: API calls={api_calls_made}, Cache hits={cache_hits}")

    return gamelog_df, lineup_cache_dict


# ============================================================================
# COLUMN ORDER STANDARDIZATION
# ============================================================================

def standardize_column_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame columns match the expected format exactly.

    Expected column order (32 columns):
    1-27: NBA API standard columns
    28: SEASON
    29: PLAYER_NAME
    30: PLAYER_ID (duplicate)
    31: LINEUP
    32: LINEUP_ID

    Args:
        df: DataFrame to standardize

    Returns:
        DataFrame with columns in correct order
    """
    expected_columns = [
        'SEASON_ID', 'Player_ID', 'Game_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN',
        'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS',
        'VIDEO_AVAILABLE', 'SEASON', 'PLAYER_NAME', 'PLAYER_ID', 'LINEUP', 'LINEUP_ID'
    ]

    # Get columns that exist in the dataframe
    available_columns = [col for col in expected_columns if col in df.columns]

    # Reorder to match expected format
    return df[available_columns]


# ============================================================================
# BATCH S3 UPLOAD
# ============================================================================

def batch_upload_to_s3(s3_handler: S3Handler, bucket: str, batch: List[Dict]) -> int:
    """
    Upload multiple player files to S3 in a batch.

    OPTIMIZATION: Reduces S3 operation overhead by batching uploads

    Args:
        s3_handler: S3Handler instance
        bucket: S3 bucket name
        batch: List of dicts with keys: 'filename', 'data', 'player_name'

    Returns:
        Number of successful uploads
    """
    successes = 0

    for item in batch:
        try:
            success = s3_handler.upload_dataframe(item['data'], bucket, item['filename'])
            if success:
                logger.info(f"  ✓ Uploaded: {item['filename']}")
                successes += 1
            else:
                logger.error(f"  Failed to upload: {item['filename']}")
        except Exception as e:
            logger.error(f"  Error uploading {item['filename']}: {e}")

    return successes


# ============================================================================
# MAIN PHASE 1 FUNCTION (OPTIMIZED)
# ============================================================================

def run_phase_1_optimized(s3_handler: Optional[S3Handler] = None) -> Tuple[bool, Dict]:
    """
    Execute Phase 1: Data Fetching & Synchronization (OPTIMIZED).

    Key Optimizations:
    1. Lineup cache as dictionary (O(1) lookups)
    2. Batch S3 uploads (every 10 players)
    3. Enhanced progress logging

    Args:
        s3_handler: Optional S3Handler instance (creates new if None)

    Returns:
        Tuple of (success status, statistics dict)
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: DATA FETCHING & SYNCHRONIZATION (OPTIMIZED)")
    logger.info("=" * 80)

    stats = {
        'players_updated': 0,
        'players_skipped': 0,
        'total_players': 0,
        'new_games_added': 0,
        'cache_hit_rate': 0.0
    }

    try:
        # Step 1.1: Initialize S3 handler
        if s3_handler is None:
            logger.info("Initializing S3 handler...")
            s3_handler = S3Handler()

        # Step 1.2: Load metadata
        logger.info("Loading pipeline metadata from S3...")
        metadata = download_metadata_from_s3(s3_handler)
        if metadata is None:
            logger.info("No metadata found, creating new metadata")
            metadata = create_default_metadata()

        # Step 1.3: Load lineup cache and convert to dict for O(1) lookups
        logger.info("Loading lineup cache from S3...")
        lineup_cache_df = download_lineup_cache_from_s3(s3_handler)
        if lineup_cache_df is None:
            logger.warning("No lineup cache in S3, creating new one")
            lineup_cache_dict = {}
        else:
            logger.info(f"✓ Loaded lineup cache with {len(lineup_cache_df)} entries")
            logger.info("Converting lineup cache to dictionary for O(1) lookups...")
            lineup_cache_dict = create_lineup_cache_dict(lineup_cache_df)
            logger.info(f"✓ Cache dictionary created: {len(lineup_cache_dict)} entries")

        # Step 1.4: Fetch league leaders
        players_df = fetch_league_leaders()

        # CRITICAL FIX: Deduplicate by PLAYER_ID to prevent duplicate processing
        # League leaders can return same player multiple times if traded between teams
        initial_count = len(players_df)
        players_df = players_df.drop_duplicates(subset=['PLAYER_ID'], keep='first')
        dedupe_count = initial_count - len(players_df)

        if dedupe_count > 0:
            logger.info(f"✓ Removed {dedupe_count} duplicate player entries (same player, different teams)")

        stats['total_players'] = len(players_df)

        # Step 1.5: Process each player with batch uploading
        logger.info(f"\nProcessing {len(players_df)} unique players...")
        logger.info("-" * 80)

        upload_batch = []

        for idx, row in players_df.iterrows():
            player_id = row['PLAYER_ID']
            player_name = row['PLAYER']

            # Progress logging every 10 players
            if (idx + 1) % 10 == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"PROGRESS: {idx + 1}/{len(players_df)} players processed")
                logger.info(f"{'='*80}\n")

            logger.info(f"\n[{idx + 1}/{len(players_df)}] {player_name} (ID: {player_id})")
            logger.info("-" * 40)

            try:
                # Build filename
                filename = build_player_filename(player_name, player_id)
                logger.info(f"  Filename: {filename}")

                # Download existing data from S3
                existing_df = s3_handler.download_dataframe(S3_PLAYER_BUCKET, filename)

                if existing_df is not None:
                    logger.info(f"  Found existing data: {len(existing_df)} games")
                else:
                    logger.info(f"  No existing data, full fetch required")

                # Fetch game logs (incremental if existing data found)
                gamelog_df = fetch_player_gamelogs(
                    player_id,
                    player_name,
                    SEASONS,
                    existing_df=existing_df
                )

                if len(gamelog_df) == 0:
                    logger.warning(f"  No data fetched, skipping")
                    stats['players_skipped'] += 1
                    continue

                # Add player identification
                gamelog_df['PLAYER_NAME'] = player_name
                gamelog_df['PLAYER_ID'] = player_id

                # Check for missing lineup data
                games_missing_lineup = get_games_missing_lineup(gamelog_df)

                if len(games_missing_lineup) > 0:
                    logger.info(f"  Found {len(games_missing_lineup)} games missing lineup data")
                    gamelog_df, lineup_cache_dict = add_lineup_information_optimized(
                        gamelog_df,
                        player_name,
                        lineup_cache_dict,
                        games_to_process=games_missing_lineup
                    )

                # Standardize column order to match expected format
                gamelog_df = standardize_column_order(gamelog_df)
                logger.info(f"  Final columns: {len(gamelog_df.columns)}")

                # Add to batch
                upload_batch.append({
                    'filename': filename,
                    'data': gamelog_df,
                    'player_name': player_name
                })

                stats['players_updated'] += 1

                # Upload batch when it reaches BATCH_SIZE
                if len(upload_batch) >= BATCH_SIZE:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"BATCH UPLOAD: Uploading {len(upload_batch)} players to S3")
                    logger.info(f"{'='*80}")
                    batch_upload_to_s3(s3_handler, S3_PLAYER_BUCKET, upload_batch)
                    upload_batch = []

                    # Save lineup cache checkpoint
                    logger.info("Saving lineup cache checkpoint...")
                    cache_df = convert_cache_dict_to_df(lineup_cache_dict)
                    upload_lineup_cache_to_s3(cache_df, s3_handler)
                    logger.info(f"✓ Lineup cache checkpoint saved ({len(lineup_cache_dict)} entries)")

            except Exception as e:
                logger.error(f"  ✗ Error processing {player_name} (ID: {player_id}): {e}")
                # Log detailed traceback for debugging
                import traceback
                logger.error(f"  Traceback: {traceback.format_exc()}")
                stats['players_skipped'] += 1
                # Track which players failed for summary
                if 'failed_players' not in stats:
                    stats['failed_players'] = []
                stats['failed_players'].append({
                    'player_id': player_id,
                    'player_name': player_name,
                    'error': str(e)
                })
                continue

        # Upload remaining batch
        if len(upload_batch) > 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"FINAL BATCH: Uploading {len(upload_batch)} remaining players")
            logger.info(f"{'='*80}")
            batch_upload_to_s3(s3_handler, S3_PLAYER_BUCKET, upload_batch)

        # Step 1.6: Final lineup cache upload
        logger.info("\nUploading final lineup cache to S3...")
        cache_df = convert_cache_dict_to_df(lineup_cache_dict)
        upload_lineup_cache_to_s3(cache_df, s3_handler)
        logger.info(f"✓ Lineup cache uploaded: {len(lineup_cache_dict)} entries")

        # Step 1.7: Update metadata
        logger.info("\nUpdating metadata...")
        metadata = update_fetch_time(metadata)
        upload_metadata_to_s3(metadata, s3_handler)
        logger.info("✓ Metadata updated")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1 SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total players: {stats['total_players']}")
        logger.info(f"Players updated: {stats['players_updated']}")
        logger.info(f"Players skipped: {stats['players_skipped']}")
        logger.info(f"Lineup cache entries: {len(lineup_cache_dict)}")

        # Report failed players if any
        if 'failed_players' in stats and len(stats['failed_players']) > 0:
            logger.warning(f"\n⚠️  FAILED PLAYERS ({len(stats['failed_players'])}):")
            for failed in stats['failed_players']:
                logger.warning(f"  - {failed['player_name']} (ID: {failed['player_id']}): {failed['error']}")

        logger.info("=" * 80)
        logger.info("✓ PHASE 1 COMPLETE")
        logger.info("=" * 80)

        return True, stats

    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False, stats


# ============================================================================
# MAIN ENTRY POINT (FOR TESTING)
# ============================================================================

def main():
    """Main entry point for standalone Phase 1 execution."""
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Run Phase 1 (optimized version)
    success, stats = run_phase_1_optimized()

    if success:
        logger.info("\n✓ Phase 1 completed successfully!")
    else:
        logger.error("\n✗ Phase 1 failed!")
        exit(1)


if __name__ == "__main__":
    main()
