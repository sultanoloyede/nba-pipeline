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

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.phases = {
            '1': {
                'name': 'Data Fetching',
                'script': 'phase1_fetch_data_optimized.py',
                'description': 'Fetches NBA player stats and game logs from NBA API',
                'estimated_time': '30-60 minutes'
            },
            '2': {
                'name': 'Data Processing',
                'script': 'phase2_process_data.py',
                'description': 'Processes raw data and engineers features',
                'estimated_time': '15-30 minutes'
            },
            '2.5': {
                'name': 'Pre-calculate Percentages',
                'script': 'phase2_5_precalculate_all_percentages.py',
                'description': 'Pre-calculates all threshold percentages (10-51)',
                'estimated_time': '30-40 minutes'
            },
            '3': {
                'name': 'Model Training',
                'script': 'phase3_optimized.py',
                'description': 'Trains 42 XGBoost models for different PRA thresholds',
                'estimated_time': '30-35 minutes'
            },
            '4': {
                'name': 'Generate Predictions',
                'script': 'phase4_generate_predictions.py',
                'description': 'Generates daily predictions and uploads to database',
                'estimated_time': '5-10 minutes'
            }
        }

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
        logger.info(f"Phases to run: {', '.join(phases_to_run)}")

        # Validate environment first
        if not self.validate_environment():
            logger.error("Environment validation failed. Aborting pipeline.")
            return False

        pipeline_start = datetime.now()
        failed_phases = []

        # Special handling: If phase 3 is requested, ensure phase 2.5 runs first
        if '3' in phases_to_run and '2.5' not in phases_to_run:
            logger.info("Phase 3 requires Phase 2.5. Adding Phase 2.5 to pipeline.")
            # Insert 2.5 before 3
            idx = phases_to_run.index('3')
            phases_to_run.insert(idx, '2.5')

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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='NBA Props Pipeline Orchestrator for GitHub Actions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all phases
  python github_pipeline_orchestrator.py --phases "all"

  # Run daily pipeline (no model training)
  python github_pipeline_orchestrator.py --phases "1,2,4"

  # Run weekly pipeline (with model training)
  python github_pipeline_orchestrator.py --phases "1,2,3,4"

  # Run only prediction phase
  python github_pipeline_orchestrator.py --phases "4"
        """
    )

    parser.add_argument(
        '--phases',
        type=str,
        required=True,
        help='Comma-separated phase numbers to run (e.g., "1,2,4") or "all"'
    )

    args = parser.parse_args()

    # Parse phases
    phases_to_run = parse_phases(args.phases)

    # Create orchestrator and run pipeline
    orchestrator = PipelineOrchestrator()
    success = orchestrator.run_pipeline(phases_to_run)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
