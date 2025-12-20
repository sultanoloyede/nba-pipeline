"""
AWS S3 Utilities Module

This module provides utilities for interacting with AWS S3 buckets for the NBA Props Pipeline.
Handles uploading, downloading, and managing player data, lineup caches, models, and metadata.

Author: Generated for NBA Props Prediction System
Date: 2025-12-10
"""

import boto3
import pandas as pd
import pickle
import json
import logging
import os
from io import BytesIO, StringIO
from typing import Optional, List, Dict, Any
from botocore.exceptions import ClientError

# Setup logger
logger = logging.getLogger(__name__)

# Configuration - will be loaded from environment variables
S3_PLAYER_BUCKET = os.getenv('S3_PLAYER_BUCKET', 'deviation-io-player-bucket')
S3_LINEUP_BUCKET = os.getenv('S3_LINEUP_BUCKET', 'deviation-io-lineup-bucket')
S3_MODEL_BUCKET = os.getenv('S3_MODEL_BUCKET', 'deviation-io-model-bucket')


class S3Handler:
    """Handler for AWS S3 operations."""

    def __init__(self, aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 region_name: str = 'us-east-1'):
        """
        Initialize S3 handler.

        Args:
            aws_access_key_id: AWS access key (defaults to env variable)
            aws_secret_access_key: AWS secret key (defaults to env variable)
            region_name: AWS region name
        """
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        else:
            # Use default credentials (from env variables or IAM role)
            self.s3_client = boto3.client('s3', region_name=region_name)

        logger.info("S3 Handler initialized")

    def upload_dataframe(self, df: pd.DataFrame, bucket: str, key: str) -> bool:
        """
        Upload a DataFrame to S3 as CSV.

        Args:
            df: DataFrame to upload
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)

            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=csv_buffer.getvalue().encode('utf-8')
            )

            logger.info(f"✓ Uploaded DataFrame to s3://{bucket}/{key}")
            return True

        except ClientError as e:
            logger.error(f"Error uploading DataFrame to S3: {e}")
            return False

    def download_dataframe(self, bucket: str, key: str) -> Optional[pd.DataFrame]:
        """
        Download a DataFrame from S3 CSV.

        Args:
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            DataFrame if successful, None otherwise
        """
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(BytesIO(response['Body'].read()))
            logger.info(f"✓ Downloaded DataFrame from s3://{bucket}/{key}")
            return df

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"Key not found: s3://{bucket}/{key}")
            else:
                logger.error(f"Error downloading DataFrame from S3: {e}")
            return None

    def upload_model(self, model: Any, bucket: str, key: str) -> bool:
        """
        Upload a pickled model to S3.

        Args:
            model: Model object to pickle and upload
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            model_bytes = pickle.dumps(model)

            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=model_bytes
            )

            logger.info(f"✓ Uploaded model to s3://{bucket}/{key}")
            return True

        except Exception as e:
            logger.error(f"Error uploading model to S3: {e}")
            return False

    def download_model(self, bucket: str, key: str) -> Optional[Any]:
        """
        Download a pickled model from S3.

        Args:
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            Model object if successful, None otherwise
        """
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            model = pickle.loads(response['Body'].read())
            logger.info(f"✓ Downloaded model from s3://{bucket}/{key}")
            return model

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"Key not found: s3://{bucket}/{key}")
            else:
                logger.error(f"Error downloading model from S3: {e}")
            return None

    def upload_json(self, data: Dict, bucket: str, key: str) -> bool:
        """
        Upload a JSON object to S3.

        Args:
            data: Dictionary to upload as JSON
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            json_str = json.dumps(data, indent=2)

            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json_str.encode('utf-8')
            )

            logger.info(f"✓ Uploaded JSON to s3://{bucket}/{key}")
            return True

        except Exception as e:
            logger.error(f"Error uploading JSON to S3: {e}")
            return False

    def download_json(self, bucket: str, key: str) -> Optional[Dict]:
        """
        Download a JSON object from S3.

        Args:
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            Dictionary if successful, None otherwise
        """
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"✓ Downloaded JSON from s3://{bucket}/{key}")
            return data

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"Key not found: s3://{bucket}/{key}")
            else:
                logger.error(f"Error downloading JSON from S3: {e}")
            return None

    def list_objects(self, bucket: str, prefix: str = '') -> List[str]:
        """
        List objects in S3 bucket with given prefix.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix to filter objects

        Returns:
            List of object keys
        """
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

            if 'Contents' not in response:
                return []

            keys = [obj['Key'] for obj in response['Contents']]
            logger.info(f"✓ Listed {len(keys)} objects from s3://{bucket}/{prefix}")
            return keys

        except ClientError as e:
            logger.error(f"Error listing objects from S3: {e}")
            return []

    def object_exists(self, bucket: str, key: str) -> bool:
        """
        Check if an object exists in S3.

        Args:
            bucket: S3 bucket name
            key: S3 object key

        Returns:
            bool: True if exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False


# Convenience functions for specific use cases

def upload_player_to_s3(df: pd.DataFrame, filename: str, s3_handler: Optional[S3Handler] = None) -> bool:
    """
    Upload player game log to S3 player bucket.

    Args:
        df: Player game log DataFrame
        filename: Filename (e.g., 'karl-anthony_towns_1626157.csv')
        s3_handler: Optional S3Handler instance

    Returns:
        bool: Success status
    """
    if s3_handler is None:
        s3_handler = S3Handler()

    key = filename
    return s3_handler.upload_dataframe(df, S3_PLAYER_BUCKET, key)


def download_player_from_s3(filename: str, s3_handler: Optional[S3Handler] = None) -> Optional[pd.DataFrame]:
    """
    Download player game log from S3 player bucket.

    Args:
        filename: Filename (e.g., 'karl-anthony_towns_1626157.csv')
        s3_handler: Optional S3Handler instance

    Returns:
        DataFrame if successful, None otherwise
    """
    if s3_handler is None:
        s3_handler = S3Handler()

    key = filename
    return s3_handler.download_dataframe(S3_PLAYER_BUCKET, key)


def upload_lineup_cache_to_s3(df: pd.DataFrame, s3_handler: Optional[S3Handler] = None) -> bool:
    """
    Upload lineup cache to S3 lineup bucket.

    Args:
        df: Lineup cache DataFrame
        s3_handler: Optional S3Handler instance

    Returns:
        bool: Success status
    """
    if s3_handler is None:
        s3_handler = S3Handler()

    return s3_handler.upload_dataframe(df, S3_LINEUP_BUCKET, 'lineup_cache.csv')


def download_lineup_cache_from_s3(s3_handler: Optional[S3Handler] = None) -> Optional[pd.DataFrame]:
    """
    Download lineup cache from S3 lineup bucket.

    Args:
        s3_handler: Optional S3Handler instance

    Returns:
        DataFrame if successful, None otherwise
    """
    if s3_handler is None:
        s3_handler = S3Handler()

    return s3_handler.download_dataframe(S3_LINEUP_BUCKET, 'lineup_cache.csv')


def upload_model_to_s3(model: Any, threshold: int, precision: float,
                       s3_handler: Optional[S3Handler] = None) -> bool:
    """
    Upload trained model to S3 model bucket.

    Args:
        model: Trained model object
        threshold: PRA threshold
        precision: Model precision score
        s3_handler: Optional S3Handler instance

    Returns:
        bool: Success status
    """
    if s3_handler is None:
        s3_handler = S3Handler()

    filename = f'xgb_pra_{threshold}plus_precision_{precision:.4f}.pkl'
    key = f'models/{filename}'
    return s3_handler.upload_model(model, S3_MODEL_BUCKET, key)


def download_models_from_s3(s3_handler: Optional[S3Handler] = None) -> Dict[int, Any]:
    """
    Download all models from S3 model bucket.

    Args:
        s3_handler: Optional S3Handler instance

    Returns:
        Dictionary mapping threshold to model object
    """
    if s3_handler is None:
        s3_handler = S3Handler()

    models = {}

    # List all model files
    model_keys = s3_handler.list_objects(S3_MODEL_BUCKET, 'models/')

    for key in model_keys:
        if key.endswith('.pkl') and 'xgb_pra_' in key:
            try:
                # Extract threshold from filename
                # Format: xgb_pra_{threshold}plus_precision_{precision}.pkl
                threshold_str = key.split('_')[2].replace('plus', '')
                threshold = int(threshold_str)

                # Download model
                model = s3_handler.download_model(S3_MODEL_BUCKET, key)
                if model:
                    models[threshold] = model

            except Exception as e:
                logger.warning(f"Error processing model file {key}: {e}")
                continue

    logger.info(f"✓ Downloaded {len(models)} models from S3")
    return models


def upload_metadata_to_s3(metadata: Dict, s3_handler: Optional[S3Handler] = None) -> bool:
    """
    Upload pipeline metadata to S3 lineup bucket.

    Args:
        metadata: Metadata dictionary
        s3_handler: Optional S3Handler instance

    Returns:
        bool: Success status
    """
    if s3_handler is None:
        s3_handler = S3Handler()

    return s3_handler.upload_json(metadata, S3_LINEUP_BUCKET, 'metadata/pipeline_metadata.json')


def download_metadata_from_s3(s3_handler: Optional[S3Handler] = None) -> Optional[Dict]:
    """
    Download pipeline metadata from S3 lineup bucket.

    Args:
        s3_handler: Optional S3Handler instance

    Returns:
        Metadata dictionary if successful, None otherwise
    """
    if s3_handler is None:
        s3_handler = S3Handler()

    return s3_handler.download_json(S3_LINEUP_BUCKET, 'metadata/pipeline_metadata.json')


def list_player_files_in_s3(s3_handler: Optional[S3Handler] = None) -> List[str]:
    """
    List all player files in S3 player bucket.

    Args:
        s3_handler: Optional S3Handler instance

    Returns:
        List of player file keys
    """
    if s3_handler is None:
        s3_handler = S3Handler()

    return s3_handler.list_objects(S3_PLAYER_BUCKET, '')
