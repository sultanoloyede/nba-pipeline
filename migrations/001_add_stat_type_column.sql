-- Migration: Add stat_type column to nba_props table
-- Date: 2025-12-29
-- Description: Adds support for multiple stat types (PRA, PA, PR, RA)

-- Add stat_type column to nba_props table
ALTER TABLE nba_props
ADD COLUMN IF NOT EXISTS stat_type VARCHAR(10) DEFAULT 'PRA';

-- Add index on stat_type for faster queries
CREATE INDEX IF NOT EXISTS idx_nba_props_stat_type
ON nba_props(stat_type);

-- Add composite index on stat_type and prop_type for filtering
CREATE INDEX IF NOT EXISTS idx_nba_props_stat_prop_type
ON nba_props(stat_type, prop_type);

-- Add composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_nba_props_stat_confidence
ON nba_props(stat_type, confidence_score DESC);

-- Verify the column was added
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_name = 'nba_props' AND column_name = 'stat_type';

-- Show sample data (should show PRA as default for existing rows)
SELECT name, stat_type, prop, line, confidence_score
FROM nba_props
LIMIT 5;

COMMENT ON COLUMN nba_props.stat_type IS 'Stat type: PRA (Points+Rebounds+Assists), PA (Points+Assists), PR (Points+Rebounds), RA (Rebounds+Assists)';
