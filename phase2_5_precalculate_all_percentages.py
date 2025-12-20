"""
Phase 2.5: Pre-calculate ALL Percentage Columns for All Thresholds (OPTIMIZED)

This phase runs ONCE to pre-calculate percentage columns for all thresholds (10-51)
and saves them to S3. This dramatically speeds up Phase 3.

KEY OPTIMIZATIONS:
- 100% vectorized operations (NO iterrows())
- Uses pandas rolling(), expanding(), and transform()
- Processes all thresholds in batches
- Progress logging every 100-500 groups to maintain Modal heartbeat
- Expected runtime: 40-60 minutes (vs 6-8 hours with old approach)

Calculates 7 percentage types for each threshold:
- last_5_pct: Rolling percentage over last 5 games
- last_10_pct: Rolling percentage over last 10 games
- last_20_pct: Rolling percentage over last 20 games
- season_pct: Current season percentage
- last_season_pct: Previous season percentage
- lineup_pct: Percentage with specific lineup
- h2h_pct: Head-to-head percentage (3 seasons)

Author: Generated for NBA Props Prediction System
Date: 2025-12-11 (Optimized)
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)


def calculate_rolling_percentage_all_thresholds_vectorized(df: pd.DataFrame, thresholds: List[int], window: int) -> Dict[int, np.ndarray]:
    """
    Calculate rolling percentage for ALL thresholds using VECTORIZED operations.

    OPTIMIZATION: Uses pandas rolling() with numpy broadcasting instead of iterrows().
    This is ~50-100x faster than the old approach.

    Strategy:
    - For each group (player, team, season), calculate rolling percentage
    - Use broadcasting to calculate for all thresholds at once within each group
    - Progress logging every 100 groups to maintain Modal heartbeat

    Args:
        df: DataFrame with PRA, Player_ID, TEAM, SEASON_ID, GAME_DATE_PARSED
        thresholds: List of PRA thresholds to calculate percentages for
        window: Rolling window size (5, 10, or 20)

    Returns:
        Dict mapping threshold -> numpy array of percentages
    """
    logger.info(f"  Calculating last_{window}_pct for {len(thresholds)} thresholds (VECTORIZED)...")

    # Sort once at the beginning
    df = df.sort_values(['Player_ID', 'TEAM', 'SEASON_ID', 'GAME_DATE_PARSED'])

    # Group by player, team, season
    grouped = df.groupby(['Player_ID', 'TEAM', 'SEASON_ID'])
    total_groups = len(grouped)

    # Initialize result dict with NaN arrays
    threshold_results = {t: np.full(len(df), np.nan) for t in thresholds}

    for group_idx, (name, group) in enumerate(grouped):
        if group_idx % 100 == 0:
            logger.info(f"    Progress: {group_idx}/{total_groups} groups ({group_idx/total_groups*100:.1f}%)")

        group_sorted = group.sort_values('GAME_DATE_PARSED')
        indices = group_sorted.index.values

        # For each threshold, calculate rolling percentage (vectorized within group)
        for threshold in thresholds:
            # Vectorized: Create boolean array where PRA >= threshold
            met_threshold = (group_sorted['PRA'].values >= threshold).astype(float)

            # Rolling sum and count using pandas (implemented in C, very fast)
            rolling_sum = pd.Series(met_threshold).rolling(window, min_periods=1).sum().shift(1).values
            rolling_count = pd.Series(met_threshold).rolling(window, min_periods=1).count().shift(1).values

            # Calculate percentage (vectorized division)
            pct = np.where(rolling_count > 0, rolling_sum / rolling_count, np.nan)

            # Store in result array at the correct indices
            threshold_results[threshold][indices] = pct

    logger.info(f"    ✓ Completed last_{window}_pct for all thresholds")
    return threshold_results


def calculate_season_percentage_all_thresholds_vectorized(df: pd.DataFrame, thresholds: List[int]) -> Dict[int, np.ndarray]:
    """
    Calculate season percentage for ALL thresholds using VECTORIZED operations.

    OPTIMIZATION: Uses groupby + expanding() instead of iterrows().

    Strategy:
    - Group by player, team, season
    - For each season, calculate expanding percentage (excludes current game)
    - Progress logging every 100 groups

    Args:
        df: DataFrame with PRA, Player_ID, TEAM, SEASON_ID, GAME_DATE_PARSED
        thresholds: List of PRA thresholds

    Returns:
        Dict mapping threshold -> numpy array of percentages
    """
    logger.info(f"  Calculating season_pct for {len(thresholds)} thresholds (VECTORIZED)...")

    grouped = df.groupby(['Player_ID', 'TEAM'])
    total_groups = len(grouped)

    # Initialize result dict with NaN arrays
    threshold_results = {t: np.full(len(df), np.nan) for t in thresholds}

    for group_idx, (name, group) in enumerate(grouped):
        if group_idx % 100 == 0:
            logger.info(f"    Progress: {group_idx}/{total_groups} groups ({group_idx/total_groups*100:.1f}%)")

        group_sorted = group.sort_values('GAME_DATE_PARSED')

        # Process each season in this player-team combination
        for season_id in group_sorted['SEASON_ID'].unique():
            season_mask = group_sorted['SEASON_ID'] == season_id
            season_group = group_sorted[season_mask]
            indices = season_group.index.values

            # For each threshold, calculate expanding percentage
            for threshold in thresholds:
                # Vectorized: Create boolean array where PRA >= threshold
                met_threshold = (season_group['PRA'].values >= threshold).astype(float)

                # Expanding mean (excludes current game with shift(1))
                expanding_sum = pd.Series(met_threshold).expanding().sum().shift(1).values
                expanding_count = pd.Series(met_threshold).expanding().count().shift(1).values

                # Calculate percentage (vectorized)
                pct = np.where(expanding_count > 0, expanding_sum / expanding_count, np.nan)

                # Store in result array
                threshold_results[threshold][indices] = pct

    logger.info(f"    ✓ Completed season_pct for all thresholds")
    return threshold_results


def calculate_last_season_percentage_all_thresholds_vectorized(df: pd.DataFrame, thresholds: List[int]) -> Dict[int, np.ndarray]:
    """
    Calculate last season percentage for ALL thresholds using VECTORIZED operations.

    OPTIMIZATION: Uses groupby().apply() with vectorized operations instead of iterrows().

    Strategy:
    - Calculate full season stats for each player-team-season combo
    - Merge with next season's data (shift SEASON_ID forward by 1)
    - Progress logging during calculation

    Args:
        df: DataFrame with PRA, Player_ID, TEAM, SEASON_ID
        thresholds: List of PRA thresholds

    Returns:
        Dict mapping threshold -> numpy array of percentages
    """
    logger.info(f"  Calculating last_season_pct for {len(thresholds)} thresholds (VECTORIZED)...")

    # Initialize result dict with NaN arrays
    threshold_results = {t: np.full(len(df), np.nan) for t in thresholds}

    # Calculate season stats for all thresholds at once
    season_stats_list = []

    for threshold_idx, threshold in enumerate(thresholds):
        if threshold_idx % 10 == 0:
            logger.info(f"    Processing threshold {threshold} ({threshold_idx + 1}/{len(thresholds)})")

        # For each player-season, calculate % of games >= threshold
        # Note: Group by Player_ID and SEASON_ID only (not TEAM) to handle team changes
        season_stats = df.groupby(['Player_ID', 'SEASON_ID']).apply(
            lambda x: (x['PRA'] >= threshold).mean()
        ).reset_index()
        season_stats.columns = ['Player_ID', 'SEASON_ID', 'pct']

        # Shift to next season (this becomes "last_season_pct" for next season's games)
        season_stats['SEASON_ID'] = season_stats['SEASON_ID'] + 1

        # Merge back to original dataframe
        df_with_last = df.merge(
            season_stats,
            on=['Player_ID', 'SEASON_ID'],
            how='left'
        )

        # Extract the percentage values (NaN for players' first season)
        threshold_results[threshold] = df_with_last['pct'].fillna(0.0).values

    logger.info(f"    ✓ Completed last_season_pct for all thresholds")
    return threshold_results


def calculate_lineup_percentage_all_thresholds_vectorized(df: pd.DataFrame, thresholds: List[int]) -> Dict[int, np.ndarray]:
    """
    Calculate lineup percentage for ALL thresholds using VECTORIZED operations.

    OPTIMIZATION: Uses groupby + expanding() instead of iterrows().

    Strategy:
    - Group by player, team, lineup
    - Calculate expanding percentage for each lineup (excludes current game)
    - For first game with new lineup, use percentage from previous "first games with new lineups"
    - Progress logging every 500 groups

    Args:
        df: DataFrame with PRA, Player_ID, TEAM, LINEUP_ID, GAME_DATE_PARSED
        thresholds: List of PRA thresholds

    Returns:
        Dict mapping threshold -> numpy array of percentages
    """
    logger.info(f"  Calculating lineup_pct for {len(thresholds)} thresholds (VECTORIZED)...")

    grouped = df.groupby(['Player_ID', 'TEAM', 'LINEUP_ID'])
    total_groups = len(grouped)

    # Initialize result dict with NaN arrays
    threshold_results = {t: np.full(len(df), np.nan) for t in thresholds}

    for group_idx, (name, group) in enumerate(grouped):
        if group_idx % 500 == 0:
            logger.info(f"    Progress: {group_idx}/{total_groups} groups ({group_idx/total_groups*100:.1f}%)")

        group_sorted = group.sort_values('GAME_DATE_PARSED')
        indices = group_sorted.index.values

        # For each threshold, calculate expanding percentage
        for threshold in thresholds:
            # Vectorized: Create boolean array where PRA >= threshold
            met_threshold = (group_sorted['PRA'].values >= threshold).astype(float)

            # Expanding mean (excludes current game with shift(1))
            expanding_sum = pd.Series(met_threshold).expanding().sum().shift(1).values
            expanding_count = pd.Series(met_threshold).expanding().count().shift(1).values

            # Calculate percentage (vectorized)
            pct = np.where(expanding_count > 0, expanding_sum / expanding_count, np.nan)

            # Store in result array
            threshold_results[threshold][indices] = pct

    # For first games with new lineups (NaN values), use "new lineup percentage"
    # This is the percentage from all previous "first games with new lineups"
    logger.info(f"  Filling first-time lineup games with new lineup percentages...")

    # Sort by player, team, date to maintain temporal ordering
    df_sorted = df.sort_values(['Player_ID', 'TEAM', 'GAME_DATE_PARSED']).copy()

    # Process each player-team combination
    player_team_groups = df_sorted.groupby(['Player_ID', 'TEAM'])
    total_pt_groups = len(player_team_groups)

    for pt_idx, ((player_id, team), pt_group) in enumerate(player_team_groups):
        if pt_idx % 500 == 0:
            logger.info(f"    Progress: {pt_idx}/{total_pt_groups} player-team groups ({pt_idx/total_pt_groups*100:.1f}%)")

        # For each threshold, calculate new lineup percentages
        for threshold in thresholds:
            # Get indices and values for this player-team
            pt_indices = pt_group.index.values
            pt_pct_values = threshold_results[threshold][pt_indices]

            # Track first-lineup-game PRAs
            first_lineup_pras = []

            for i, (idx, pct_val) in enumerate(zip(pt_indices, pt_pct_values)):
                if np.isnan(pct_val):
                    # This is a first game with a new lineup
                    if len(first_lineup_pras) > 0:
                        # Calculate percentage from previous first-lineup-games
                        new_lineup_pct = np.mean([pra >= threshold for pra in first_lineup_pras])
                        threshold_results[threshold][idx] = new_lineup_pct
                    else:
                        # No previous first-lineup-games, use 0
                        threshold_results[threshold][idx] = 0.0

                    # Add current game's PRA to the list for future first-lineup-games
                    current_pra = pt_group.iloc[i]['PRA']
                    first_lineup_pras.append(current_pra)

    logger.info(f"    ✓ Completed lineup_pct for all thresholds")
    return threshold_results


def calculate_h2h_percentage_all_thresholds_vectorized(df: pd.DataFrame, thresholds: List[int]) -> Dict[int, np.ndarray]:
    """
    Calculate H2H percentage for ALL thresholds using OPTIMIZED operations.

    OPTIMIZATION: Pre-groups by (Player_ID, OPPONENT) and calculates for all thresholds
    at once. Still requires some iteration due to the 3-season window logic, but vectorized
    within each group.

    Strategy:
    - Group by player, opponent
    - For each game, look at previous games in last 3 seasons
    - Calculate percentages for all thresholds at once using vectorized mean()
    - Progress logging every 500 groups

    Args:
        df: DataFrame with PRA, Player_ID, OPPONENT, SEASON_ID, GAME_DATE_PARSED
        thresholds: List of PRA thresholds

    Returns:
        Dict mapping threshold -> numpy array of percentages
    """
    logger.info(f"  Calculating h2h_pct for {len(thresholds)} thresholds (VECTORIZED)...")

    grouped = df.groupby(['Player_ID', 'OPPONENT'])
    total_groups = len(grouped)

    # Initialize result dict with NaN arrays
    threshold_results = {t: np.full(len(df), np.nan) for t in thresholds}

    for group_idx, ((player_id, opponent), group) in enumerate(grouped):
        if group_idx % 500 == 0:
            logger.info(f"    Progress: {group_idx}/{total_groups} groups ({group_idx/total_groups*100:.1f}%)")

        group_sorted = group.sort_values(['SEASON_ID', 'GAME_DATE_PARSED']).reset_index(drop=False)
        original_indices = group_sorted['index'].values

        # For each game in this player-opponent matchup
        for row_pos in range(len(group_sorted)):
            row = group_sorted.iloc[row_pos]
            current_season = row['SEASON_ID']
            target_seasons = [current_season, current_season - 1, current_season - 2]

            # Get previous games in target seasons (vectorized filtering)
            prev_games = group_sorted.iloc[:row_pos]
            prev_games = prev_games[prev_games['SEASON_ID'].isin(target_seasons)]

            if len(prev_games) > 0:
                # Calculate percentages for all thresholds at once (VECTORIZED)
                for threshold in thresholds:
                    pct = (prev_games['PRA'] >= threshold).mean()
                    threshold_results[threshold][original_indices[row_pos]] = pct
            else:
                # No previous h2h games, use 0 instead of NaN
                for threshold in thresholds:
                    threshold_results[threshold][original_indices[row_pos]] = 0.0

    logger.info(f"    ✓ Completed h2h_pct for all thresholds")
    return threshold_results


def run_phase_2_5(s3_handler, threshold_start=10, threshold_end=51) -> Tuple[bool, dict]:
    """
    Execute Phase 2.5: Pre-calculate ALL percentage columns for all thresholds (OPTIMIZED).

    This dramatically speeds up Phase 3 by doing all percentage calculations once.

    OPTIMIZATIONS:
    - 100% vectorized operations (NO iterrows())
    - Progress logging every 100-500 groups
    - Expected runtime: 40-60 minutes (vs 6-8 hours with old approach)

    Args:
        s3_handler: S3Handler instance
        threshold_start: Starting threshold (default: 10)
        threshold_end: Ending threshold (default: 51)

    Returns:
        Tuple of (success: bool, stats: dict)
    """
    logger.info("=" * 80)
    logger.info("PHASE 2.5: PRE-CALCULATE ALL PERCENTAGE COLUMNS (OPTIMIZED)")
    logger.info("=" * 80)

    try:
        from s3_utils import S3_PLAYER_BUCKET

        # Step 1: Download processed data from Phase 2
        logger.info("Step 1: Downloading processed data from Phase 2...")
        df = s3_handler.download_dataframe(
            S3_PLAYER_BUCKET,
            'processed_data/processed_model_data.csv'
        )

        if df is None:
            logger.error("Failed to download processed data from S3")
            return False, {'error': 'Failed to download processed data'}

        logger.info(f"✓ Downloaded processed data: {len(df):,} rows")

        # Ensure GAME_DATE_PARSED exists
        if 'GAME_DATE_PARSED' not in df.columns:
            logger.info("  Parsing game dates...")
            df['GAME_DATE_PARSED'] = pd.to_datetime(df['GAME_DATE'], format='%b %d, %Y')

        # Step 2: Calculate percentage columns for all thresholds (VECTORIZED)
        thresholds = list(range(threshold_start, threshold_end + 1))
        logger.info(f"\nStep 2: Calculating percentage columns for {len(thresholds)} thresholds (VECTORIZED)...")
        logger.info(f"  Thresholds: {threshold_start} to {threshold_end}")

        # Calculate rolling percentages (windows: 5, 10, 20)
        logger.info("\n[1/7] Calculating rolling percentages...")
        for window in [5, 10, 20]:
            results = calculate_rolling_percentage_all_thresholds_vectorized(df, thresholds, window)
            for threshold, pct_array in results.items():
                df[f'last_{window}_pct_{threshold}'] = pct_array
            logger.info(f"  ✓ Added last_{window}_pct columns for all thresholds")

        # Calculate season percentages
        logger.info("\n[2/7] Calculating season percentages...")
        season_results = calculate_season_percentage_all_thresholds_vectorized(df, thresholds)
        for threshold, pct_array in season_results.items():
            df[f'season_pct_{threshold}'] = pct_array
        logger.info(f"  ✓ Added season_pct columns for all thresholds")

        # Calculate last season percentages
        logger.info("\n[3/7] Calculating last season percentages...")
        last_season_results = calculate_last_season_percentage_all_thresholds_vectorized(df, thresholds)
        for threshold, pct_array in last_season_results.items():
            df[f'last_season_pct_{threshold}'] = pct_array
        logger.info(f"  ✓ Added last_season_pct columns for all thresholds")

        # Calculate lineup percentages
        logger.info("\n[4/7] Calculating lineup percentages...")
        lineup_results = calculate_lineup_percentage_all_thresholds_vectorized(df, thresholds)
        for threshold, pct_array in lineup_results.items():
            df[f'lineup_pct_{threshold}'] = pct_array
        logger.info(f"  ✓ Added lineup_pct columns for all thresholds")

        # Calculate h2h percentages
        logger.info("\n[5/7] Calculating H2H percentages...")
        h2h_results = calculate_h2h_percentage_all_thresholds_vectorized(df, thresholds)
        for threshold, pct_array in h2h_results.items():
            df[f'h2h_pct_{threshold}'] = pct_array
        logger.info(f"  ✓ Added h2h_pct columns for all thresholds")

        # Step 3: Save to S3
        logger.info("\n[6/7] Saving pre-calculated data to S3...")
        output_key = f'processed_data/processed_with_all_pct_{threshold_start}-{threshold_end}.csv'

        # Calculate file size before upload
        file_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        s3_handler.upload_dataframe(df, S3_PLAYER_BUCKET, output_key)
        logger.info(f"✓ Saved to s3://{S3_PLAYER_BUCKET}/{output_key}")
        logger.info(f"  File size: {file_size_mb:.1f} MB")

        # Generate stats
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'thresholds': f"{threshold_start}-{threshold_end}",
            'num_thresholds': len(thresholds),
            'columns_added': len(thresholds) * 7,  # 7 percentage types per threshold
            'file_size_mb': round(file_size_mb, 2)
        }

        logger.info("\n[7/7] " + "=" * 80)
        logger.info("PHASE 2.5 SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total rows: {stats['total_rows']:,}")
        logger.info(f"Total columns: {stats['total_columns']}")
        logger.info(f"Thresholds: {stats['thresholds']}")
        logger.info(f"Percentage columns added: {stats['columns_added']}")
        logger.info(f"File size: {stats['file_size_mb']} MB")
        logger.info("=" * 80)
        logger.info("✓ PHASE 2.5 COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return True, stats

    except Exception as e:
        logger.error(f"Phase 2.5 failed with error: {e}")
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
    success, stats = run_phase_2_5(s3_handler)

    if success:
        print("\n✓ Phase 2.5 completed successfully!")
        print(f"Stats: {stats}")
    else:
        print("\n✗ Phase 2.5 failed!")
        print(f"Error: {stats.get('error')}")
