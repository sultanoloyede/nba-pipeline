"""
Metadata Management Utilities Module

This module provides utilities for managing pipeline metadata including
last fetch times, training schedules, and run statistics.

Author: Generated for NBA Props Prediction System
Date: 2025-12-10
"""

from datetime import datetime
from typing import Dict, Optional
import logging

# Setup logger
logger = logging.getLogger(__name__)


def create_default_metadata() -> Dict:
    """
    Create default metadata structure.

    Returns:
        Dictionary with default metadata
    """
    return {
        "last_fetch": None,
        "last_training": None,
        "training_precision_scores": {},
        "last_run_stats": {
            "players_processed": 0,
            "props_generated": 0,
            "api_calls": 0,
            "cache_hits": 0
        },
        "version": "1.0.0",
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def should_train_models(metadata: Dict, training_interval_days: int = 7) -> bool:
    """
    Determine if models should be retrained based on last training date.

    Args:
        metadata: Pipeline metadata dictionary
        training_interval_days: Number of days between training runs

    Returns:
        bool: True if training is needed, False otherwise
    """
    if not metadata.get('last_training'):
        logger.info("No previous training found - training required")
        return True

    try:
        last_training = datetime.strptime(metadata['last_training'], '%Y-%m-%d %H:%M:%S')
        days_since_training = (datetime.now() - last_training).days

        if days_since_training >= training_interval_days:
            logger.info(f"Training required: {days_since_training} days since last training")
            return True
        else:
            logger.info(f"Training not required: Only {days_since_training} days since last training")
            return False

    except Exception as e:
        logger.error(f"Error checking training schedule: {e}")
        return True  # Train if there's an error


def update_fetch_time(metadata: Dict) -> Dict:
    """
    Update the last fetch timestamp in metadata.

    Args:
        metadata: Pipeline metadata dictionary

    Returns:
        Updated metadata dictionary
    """
    metadata['last_fetch'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return metadata


def update_training_time(metadata: Dict, precision_scores: Optional[Dict[int, float]] = None) -> Dict:
    """
    Update the last training timestamp and precision scores in metadata.

    Args:
        metadata: Pipeline metadata dictionary
        precision_scores: Optional dictionary mapping threshold to precision score

    Returns:
        Updated metadata dictionary
    """
    metadata['last_training'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if precision_scores:
        metadata['training_precision_scores'] = {
            str(threshold): score
            for threshold, score in precision_scores.items()
        }

    return metadata


def update_run_stats(metadata: Dict, players_processed: int, props_generated: int,
                     api_calls: int = 0, cache_hits: int = 0) -> Dict:
    """
    Update the last run statistics in metadata.

    Args:
        metadata: Pipeline metadata dictionary
        players_processed: Number of players processed
        props_generated: Number of props generated
        api_calls: Number of API calls made
        cache_hits: Number of cache hits

    Returns:
        Updated metadata dictionary
    """
    metadata['last_run_stats'] = {
        'players_processed': players_processed,
        'props_generated': props_generated,
        'api_calls': api_calls,
        'cache_hits': cache_hits,
        'cache_hit_rate': cache_hits / (api_calls + cache_hits) if (api_calls + cache_hits) > 0 else 0.0,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return metadata


def get_training_summary(metadata: Dict) -> str:
    """
    Generate a human-readable summary of training status.

    Args:
        metadata: Pipeline metadata dictionary

    Returns:
        Formatted string summary
    """
    if not metadata.get('last_training'):
        return "No training history found"

    summary = []
    summary.append(f"Last Training: {metadata['last_training']}")

    if metadata.get('training_precision_scores'):
        summary.append("\nPrecision Scores:")
        for threshold, score in sorted(metadata['training_precision_scores'].items(),
                                       key=lambda x: int(x[0])):
            summary.append(f"  {threshold}+: {score:.4f}")

    return '\n'.join(summary)


def get_run_summary(metadata: Dict) -> str:
    """
    Generate a human-readable summary of last run statistics.

    Args:
        metadata: Pipeline metadata dictionary

    Returns:
        Formatted string summary
    """
    if not metadata.get('last_run_stats'):
        return "No run history found"

    stats = metadata['last_run_stats']
    summary = []

    summary.append(f"Last Run: {metadata.get('last_fetch', 'Unknown')}")
    summary.append(f"Players Processed: {stats.get('players_processed', 0)}")
    summary.append(f"Props Generated: {stats.get('props_generated', 0)}")
    summary.append(f"API Calls: {stats.get('api_calls', 0)}")
    summary.append(f"Cache Hits: {stats.get('cache_hits', 0)}")

    if 'cache_hit_rate' in stats:
        summary.append(f"Cache Hit Rate: {stats['cache_hit_rate']:.1%}")

    return '\n'.join(summary)


def validate_metadata(metadata: Dict) -> bool:
    """
    Validate metadata structure.

    Args:
        metadata: Pipeline metadata dictionary

    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = ['last_fetch', 'last_training', 'training_precision_scores', 'last_run_stats']

    for key in required_keys:
        if key not in metadata:
            logger.warning(f"Missing required key in metadata: {key}")
            return False

    return True


def merge_metadata(old_metadata: Dict, new_metadata: Dict) -> Dict:
    """
    Merge new metadata with old, preserving history.

    Args:
        old_metadata: Existing metadata
        new_metadata: New metadata to merge

    Returns:
        Merged metadata dictionary
    """
    merged = old_metadata.copy()

    # Update fields from new metadata
    for key, value in new_metadata.items():
        if value is not None:
            merged[key] = value

    # Preserve history
    if 'history' not in merged:
        merged['history'] = []

    # Add snapshot to history
    merged['history'].append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'last_fetch': merged.get('last_fetch'),
        'last_training': merged.get('last_training'),
        'stats': merged.get('last_run_stats')
    })

    # Keep only last 30 history entries
    merged['history'] = merged['history'][-30:]

    return merged
