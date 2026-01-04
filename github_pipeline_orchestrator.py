#!/usr/bin/env python3
"""
GitHub Actions Pipeline Orchestrator for NBA Props Prediction

This script orchestrates the execution of all pipeline phases for GitHub Actions.
It replaces Modal deployment with a simple sequential runner.

Usage:
    python github_pipeline_orchestrator.py --phases "1,2,3,4"
    python github_pipeline_orchestrator.py --phases "all"
    python github_pipeline_orchestrator.py --phases "1,2,4"  # Skip model training
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import traceback

# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f'github_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the NBA Props pipeline phases"""

    def __init__(self, stat_type='PRA'):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.stat_type = stat_type.upper()

        # PRA Pipeline (default)
        pra_phases = {
            '1': {
                'name': 'Data Fetching',
                'script': 'phase1_fetch_data_optimized.py',
                'description': 'Fetches NBA player stats and game logs from NBA API',
                'estimated_time': '30-60 minutes'
            },
            '2': {
                'name': 'Data Processing (PRA)',
                'script': 'phase2_process_data.py',
                'description': 'Processes raw data and engineers PRA features',
                'estimated_time': '15-30 minutes'
            },
            '2.5': {
                'name': 'Pre-calculate Percentages (PRA)',
                'script': 'phase2_5_precalculate_all_percentages.py',
                'description': 'Pre-calculates all PRA threshold percentages (10-51)',
                'estimated_time': '30-40 minutes'
            },
            '3': {
                'name': 'Model Training (PRA)',
                'script': 'phase3_optimized.py',
                'description': 'Trains 42 XGBoost models for PRA thresholds',
                'estimated_time': '30-35 minutes'
            },
            '4': {
                'name': 'Generate Predictions (PRA)',
                'script': 'phase4_generate_predictions.py',
                'description': 'Generates daily PRA predictions and uploads to database',
                'estimated_time': '5-10 minutes'
            }
        }

        # RA Pipeline
        ra_phases = {
            '1': {
                'name': 'Data Fetching',
                'script': 'phase1_fetch_data_optimized.py',
                'description': 'Fetches NBA player stats and game logs from NBA API',
                'estimated_time': '30-60 minutes'
            },
            '2': {
                'name': 'Data Processing (RA)',
                'script': 'phase2_ra.py',
                'description': 'Processes raw data and engineers RA features',
                'estimated_time': '15-30 minutes'
            },
            '2.5': {
                'name': 'Pre-calculate Percentages (RA)',
                'script': 'phase2_5_ra.py',
                'description': 'Pre-calculates all RA threshold percentages (5-26)',
                'estimated_time': '30-40 minutes'
            },
            '3': {
                'name': 'Model Training (RA)',
                'script': 'phase3_ra.py',
                'description': 'Trains 22 XGBoost models for RA thresholds (5-26)',
                'estimated_time': '15-20 minutes'
            },
            '4': {
                'name': 'Generate Predictions (RA)',
                'script': 'phase4_ra.py',
                'description': 'Placeholder - use single_player_ra_gauntlet.py for testing',
                'estimated_time': '1 minute'
            }
        }

        # PA Pipeline
        pa_phases = {
            '1': {
                'name': 'Data Fetching',
                'script': 'phase1_fetch_data_optimized.py',
                'description': 'Fetches NBA player stats and game logs from NBA API',
                'estimated_time': '30-60 minutes'
            },
            '2': {
                'name': 'Data Processing (PA)',
                'script': 'phase2_pa.py',
                'description': 'Processes raw data and engineers PA features',
                'estimated_time': '15-30 minutes'
            },
            '2.5': {
                'name': 'Pre-calculate Percentages (PA)',
                'script': 'phase2_5_pa.py',
                'description': 'Pre-calculates all PA threshold percentages (5-39)',
                'estimated_time': '30-40 minutes'
            },
            '3': {
                'name': 'Model Training (PA)',
                'script': 'phase3_pa.py',
                'description': 'Trains 35 XGBoost models for PA thresholds (5-39)',
                'estimated_time': '20-25 minutes'
            },
            '4': {
                'name': 'Generate Predictions (PA)',
                'script': 'phase4_generate_predictions.py',
                'description': 'Generates daily PA predictions and uploads to database',
                'estimated_time': '5-10 minutes'
            }
        }

        # PR Pipeline
        pr_phases = {
            '1': {
                'name': 'Data Fetching',
                'script': 'phase1_fetch_data_optimized.py',
                'description': 'Fetches NBA player stats and game logs from NBA API',
                'estimated_time': '30-60 minutes'
            },
            '2': {
                'name': 'Data Processing (PR)',
                'script': 'phase2_pr.py',
                'description': 'Processes raw data and engineers PR features',
                'estimated_time': '15-30 minutes'
            },
            '2.5': {
                'name': 'Pre-calculate Percentages (PR)',
                'script': 'phase2_5_pr.py',
                'description': 'Pre-calculates all PR threshold percentages (5-39)',
                'estimated_time': '30-40 minutes'
            },
            '3': {
                'name': 'Model Training (PR)',
                'script': 'phase3_pr.py',
                'description': 'Trains 35 XGBoost models for PR thresholds (5-39)',
                'estimated_time': '20-25 minutes'
            },
            '4': {
                'name': 'Generate Predictions (PR)',
                'script': 'phase4_generate_predictions.py',
                'description': 'Generates daily PR predictions and uploads to database',
                'estimated_time': '5-10 minutes'
            }
        }

        # Select phases based on stat_type
        phase_map = {
            'PRA': pra_phases,
            'RA': ra_phases,
            'PA': pa_phases,
            'PR': pr_phases
        }

        self.phases = phase_map.get(self.stat_type, pra_phases)
        logger.info(f"Initialized orchestrator for {self.stat_type} pipeline")


    def validate_environment(self):
        """Validate required environment variables"""
        logger.info("Validating environment variables...")

        required_vars = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY',
            'AWS_REGION',
            'S3_PLAYER_BUCKET',
            'S3_LINEUP_BUCKET',
            'S3_MODEL_BUCKET',
            'NEON_DATABASE_URL'
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False

        logger.info("All required environment variables found")
        return True

    def run_phase(self, phase_num: str) -> bool:
        """
        Run a specific pipeline phase

        Args:
            phase_num: Phase number to run (e.g., '1', '2', '2.5', '3', '4')

        Returns:
            bool: True if phase succeeded, False otherwise
        """
        if phase_num not in self.phases:
            logger.error(f"Invalid phase number: {phase_num}")
            return False

        phase = self.phases[phase_num]
        script_path = os.path.join(self.script_dir, phase['script'])

        if not os.path.exists(script_path):
            logger.error(f"Phase {phase_num} script not found: {script_path}")
            return False

        logger.info("=" * 80)
        logger.info(f"Starting Phase {phase_num}: {phase['name']}")
        logger.info(f"Description: {phase['description']}")
        logger.info(f"Estimated time: {phase['estimated_time']}")
        logger.info("=" * 80)

        start_time = datetime.now()

        try:
            # Import and run the phase script as a module
            # This allows us to capture any errors properly
            logger.info(f"Executing: {phase['script']}")

            # Change to script directory to ensure relative imports work
            original_dir = os.getcwd()
            os.chdir(self.script_dir)

            # Execute the script
            with open(script_path, 'r') as f:
                script_code = f.read()

            # Create a namespace for the script execution
            script_globals = {
                '__name__': '__main__',
                '__file__': script_path,
            }

            exec(script_code, script_globals)

            # Restore original directory
            os.chdir(original_dir)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60

            logger.info(f"Phase {phase_num} completed successfully in {duration:.2f} minutes")
            return True

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60

            logger.error(f"Phase {phase_num} failed after {duration:.2f} minutes")
            logger.error(f"Error: {str(e)}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False

    def run_pipeline(self, phases_to_run: list) -> bool:
        """
        Run specified pipeline phases in sequence

        Args:
            phases_to_run: List of phase numbers to run (e.g., ['1', '2', '4'])

        Returns:
            bool: True if all phases succeeded, False otherwise
        """
        logger.info("Starting NBA Props Pipeline")
        logger.info(f"Stat Type: {self.stat_type}")
        logger.info(f"Phases to run: {', '.join(phases_to_run)}")

        # Validate environment first
        if not self.validate_environment():
            logger.error("Environment validation failed. Aborting pipeline.")
            return False

        pipeline_start = datetime.now()
        failed_phases = []

        # Run each phase sequentially
        for phase_num in phases_to_run:
            success = self.run_phase(phase_num)

            if not success:
                failed_phases.append(phase_num)
                logger.error(f"Phase {phase_num} failed. Stopping pipeline.")
                break

        pipeline_end = datetime.now()
        total_duration = (pipeline_end - pipeline_start).total_seconds() / 60

        # Summary
        logger.info("=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Duration: {total_duration:.2f} minutes")
        logger.info(f"Phases Requested: {', '.join(phases_to_run)}")

        if failed_phases:
            logger.error(f"Failed Phases: {', '.join(failed_phases)}")
            logger.error("Pipeline completed with errors")
            return False
        else:
            logger.info("All phases completed successfully!")
            return True


def parse_phases(phases_str: str) -> list:
    """
    Parse phases string into list of phase numbers

    Args:
        phases_str: Comma-separated phase numbers or 'all'

    Returns:
        list: List of phase numbers
    """
    if phases_str.lower() == 'all':
        return ['1', '2', '2.5', '3', '4']

    phases = [p.strip() for p in phases_str.split(',')]

    # Validate phases
    valid_phases = ['1', '2', '2.5', '3', '4']
    invalid = [p for p in phases if p not in valid_phases]

    if invalid:
        logger.error(f"Invalid phases specified: {', '.join(invalid)}")
        logger.error(f"Valid phases are: {', '.join(valid_phases)}")
        sys.exit(1)

    return phases


def parse_stat_types(stat_types_str: str) -> list:
    """
    Parse stat types string into list of stat types

    Args:
        stat_types_str: Comma-separated stat types or 'all'

    Returns:
        list: List of stat types
    """
    if stat_types_str.lower() == 'all':
        return ['PRA', 'RA', 'PA', 'PR']

    stat_types = [s.strip().upper() for s in stat_types_str.split(',')]

    # Validate stat types
    valid_stat_types = ['PRA', 'RA', 'PA', 'PR']
    invalid = [s for s in stat_types if s not in valid_stat_types]

    if invalid:
        logger.error(f"Invalid stat types specified: {', '.join(invalid)}")
        logger.error(f"Valid stat types are: {', '.join(valid_stat_types)}")
        sys.exit(1)

    return stat_types


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='NBA Props Pipeline Orchestrator for GitHub Actions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all phases for all stat types
  python github_pipeline_orchestrator.py --phases "all" --stat-types "all"

  # Run specific stat types (phases 2, 2.5, 3)
  python github_pipeline_orchestrator.py --phases "2,2.5,3" --stat-types "PRA,RA"

  # Run only PA predictions
  python github_pipeline_orchestrator.py --phases "2,2.5,3" --stat-types "PA"

  # Run daily pipeline for all stat types (no model training)
  python github_pipeline_orchestrator.py --phases "2,2.5" --stat-types "all"

  # Run weekly pipeline for all stat types (with model training)
  python github_pipeline_orchestrator.py --phases "2,2.5,3" --stat-types "all"
        """
    )

    parser.add_argument(
        '--phases',
        type=str,
        required=True,
        help='Comma-separated phase numbers to run (e.g., "2,2.5,3") or "all"'
    )

    parser.add_argument(
        '--stat-types',
        type=str,
        default='all',
        help='Comma-separated stat types to process (e.g., "PRA,RA") or "all" (default: all)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='auto',
        choices=['auto', 'full', 'incremental'],
        help='Phase 2.5 mode: auto (default), full (fresh run), or incremental'
    )

    args = parser.parse_args()

    # Parse phases and stat types
    phases_to_run = parse_phases(args.phases)
    stat_types_to_run = parse_stat_types(args.stat_types)

    logger.info("=" * 80)
    logger.info("NBA PROPS PIPELINE - MULTI-STAT TYPE RUN")
    logger.info("=" * 80)
    logger.info(f"Stat types to process: {', '.join(stat_types_to_run)}")
    logger.info(f"Phases to run for each: {', '.join(phases_to_run)}")
    logger.info("=" * 80)

    overall_start = datetime.now()
    failed_stat_types = []

    # Run pipeline for each stat type in succession
    for stat_type in stat_types_to_run:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STARTING {stat_type} PIPELINE")
        logger.info("=" * 80)

        # Create orchestrator for this stat type and run pipeline
        orchestrator = PipelineOrchestrator(stat_type=stat_type)
        success = orchestrator.run_pipeline(phases_to_run)

        if not success:
            failed_stat_types.append(stat_type)
            logger.error(f"{stat_type} pipeline failed. Continuing to next stat type...")
        else:
            logger.info(f"{stat_type} pipeline completed successfully!")

    overall_end = datetime.now()
    total_duration = (overall_end - overall_start).total_seconds() / 60

    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("OVERALL PIPELINE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Duration: {total_duration:.2f} minutes")
    logger.info(f"Stat Types Processed: {', '.join(stat_types_to_run)}")

    if failed_stat_types:
        logger.error(f"Failed Stat Types: {', '.join(failed_stat_types)}")
        logger.error("Some pipelines completed with errors")
        sys.exit(1)
    else:
        logger.info("All stat type pipelines completed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
