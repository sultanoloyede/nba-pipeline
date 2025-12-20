"""
Database Utilities Module

This module provides utilities for interacting with Neon PostgreSQL database
for storing and retrieving NBA props predictions.

Author: Generated for NBA Props Prediction System
Date: 2025-12-10
"""

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import pandas as pd
import logging
import os
from typing import Optional, List, Dict, Any

# Setup logger
logger = logging.getLogger(__name__)


class DatabaseHandler:
    """Handler for Neon PostgreSQL database operations."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database handler.

        Args:
            database_url: PostgreSQL connection string (defaults to env variable)
        """
        self.database_url = database_url or os.getenv('NEON_DATABASE_URL')

        if not self.database_url:
            raise ValueError("Database URL not provided and NEON_DATABASE_URL environment variable not set")

        self.conn = None
        logger.info("Database Handler initialized")

    def connect(self) -> bool:
        """
        Establish database connection.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.conn = psycopg2.connect(self.database_url)
            logger.info("✓ Connected to Neon database")
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False

    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def create_props_table(self) -> bool:
        """
        Create the nba_props table if it doesn't exist.

        Returns:
            bool: True if successful, False otherwise
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS nba_props (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            matchup VARCHAR(50) NOT NULL,
            game_date VARCHAR(50) NOT NULL,
            prop VARCHAR(100) NOT NULL,
            line FLOAT NOT NULL,
            probability FLOAT NOT NULL,
            last_5 FLOAT,
            last_10 FLOAT,
            last_20 FLOAT,
            this_season FLOAT,
            last_season FLOAT,
            h2h FLOAT,
            lineup_pct FLOAT,
            lineup_id BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        try:
            if not self.conn:
                self.connect()

            cursor = self.conn.cursor()
            cursor.execute(create_table_query)
            self.conn.commit()
            cursor.close()

            logger.info("✓ Props table created/verified")
            return True

        except Exception as e:
            logger.error(f"Error creating props table: {e}")
            if self.conn:
                self.conn.rollback()
            return False

    def upload_props(self, props_df: pd.DataFrame, truncate: bool = True) -> bool:
        """
        Upload props predictions to database.

        Args:
            props_df: DataFrame with props data
            truncate: If True, truncate table before inserting

        Returns:
            bool: True if successful, False otherwise
        """
        if len(props_df) == 0:
            logger.warning("No props to upload")
            return False

        try:
            if not self.conn:
                self.connect()

            cursor = self.conn.cursor()

            # Truncate table if requested
            if truncate:
                cursor.execute("TRUNCATE TABLE nba_props RESTART IDENTITY")
                logger.info("Table truncated")

            # Prepare column mapping
            column_mapping = {
                'NAME': 'name',
                'MATCHUP': 'matchup',
                'GAME_DATE': 'game_date',
                'PROP': 'prop',
                'LINE': 'line',
                'PROBABILITY': 'probability',
                'LAST_5': 'last_5',
                'LAST_10': 'last_10',
                'LAST_20': 'last_20',
                'THIS_SEASON': 'this_season',
                'LAST_SEASON': 'last_season',
                'H2H': 'h2h',
                'LINEUP_PCT': 'lineup_pct',
                'LINEUP_ID': 'lineup_id'
            }

            # Rename columns
            df_renamed = props_df.rename(columns=column_mapping)

            # Get columns in order
            columns = [
                'name', 'matchup', 'game_date', 'prop', 'line', 'probability',
                'last_5', 'last_10', 'last_20', 'this_season', 'last_season',
                'h2h', 'lineup_pct', 'lineup_id'
            ]

            # Ensure columns exist
            for col in columns:
                if col not in df_renamed.columns:
                    df_renamed[col] = None

            # Convert NaN to None
            df_renamed = df_renamed.where(pd.notna(df_renamed), None)

            # Prepare insert query
            insert_query = sql.SQL("""
                INSERT INTO nba_props ({})
                VALUES %s
            """).format(
                sql.SQL(', ').join(map(sql.Identifier, columns))
            )

            # Prepare values
            values = [tuple(row) for row in df_renamed[columns].values]

            # Execute batch insert
            execute_values(cursor, insert_query, values)

            self.conn.commit()
            cursor.close()

            logger.info(f"✓ Uploaded {len(props_df)} props to database")
            return True

        except Exception as e:
            logger.error(f"Error uploading props to database: {e}")
            if self.conn:
                self.conn.rollback()
            return False

    def get_props_for_date(self, game_date: str) -> Optional[pd.DataFrame]:
        """
        Retrieve props for a specific game date.

        Args:
            game_date: Game date string (e.g., 'Dec 10, 2025')

        Returns:
            DataFrame with props if successful, None otherwise
        """
        query = """
        SELECT * FROM nba_props
        WHERE game_date = %s
        ORDER BY probability DESC
        """

        try:
            if not self.conn:
                self.connect()

            df = pd.read_sql_query(query, self.conn, params=(game_date,))
            logger.info(f"✓ Retrieved {len(df)} props for {game_date}")
            return df

        except Exception as e:
            logger.error(f"Error retrieving props from database: {e}")
            return None

    def get_all_props(self, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Retrieve all props from database.

        Args:
            limit: Maximum number of rows to retrieve

        Returns:
            DataFrame with props if successful, None otherwise
        """
        query = "SELECT * FROM nba_props ORDER BY probability DESC"

        if limit:
            query += f" LIMIT {limit}"

        try:
            if not self.conn:
                self.connect()

            df = pd.read_sql_query(query, self.conn)
            logger.info(f"✓ Retrieved {len(df)} props from database")
            return df

        except Exception as e:
            logger.error(f"Error retrieving props from database: {e}")
            return None

    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[List[tuple]]:
        """
        Execute a custom SQL query.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            List of result tuples if successful, None otherwise
        """
        try:
            if not self.conn:
                self.connect()

            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            results = cursor.fetchall()
            cursor.close()

            logger.info(f"✓ Query executed, {len(results)} rows returned")
            return results

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            if self.conn:
                self.conn.rollback()
            return None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Convenience functions

def upload_props_to_neon(props_df: pd.DataFrame, database_url: Optional[str] = None,
                         truncate: bool = True) -> bool:
    """
    Upload props to Neon database (convenience function).

    Args:
        props_df: DataFrame with props data
        database_url: Optional database URL
        truncate: If True, truncate table before inserting

    Returns:
        bool: Success status
    """
    db = DatabaseHandler(database_url)

    try:
        db.connect()
        db.create_props_table()
        success = db.upload_props(props_df, truncate=truncate)
        db.disconnect()
        return success

    except Exception as e:
        logger.error(f"Error in upload_props_to_neon: {e}")
        db.disconnect()
        return False


def get_latest_props(database_url: Optional[str] = None, limit: int = 50) -> Optional[pd.DataFrame]:
    """
    Get latest props from Neon database (convenience function).

    Args:
        database_url: Optional database URL
        limit: Maximum number of rows to retrieve

    Returns:
        DataFrame with props if successful, None otherwise
    """
    db = DatabaseHandler(database_url)

    try:
        db.connect()
        df = db.get_all_props(limit=limit)
        db.disconnect()
        return df

    except Exception as e:
        logger.error(f"Error in get_latest_props: {e}")
        db.disconnect()
        return None


def create_database_tables(database_url: Optional[str] = None) -> bool:
    """
    Create all required database tables (convenience function).

    Args:
        database_url: Optional database URL

    Returns:
        bool: Success status
    """
    db = DatabaseHandler(database_url)

    try:
        db.connect()
        success = db.create_props_table()
        db.disconnect()
        return success

    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        db.disconnect()
        return False
