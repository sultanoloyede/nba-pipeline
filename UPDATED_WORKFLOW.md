# Updated Workflow - All Stat Types Processed by Default

## Summary of Changes

**Phase 2.5** and **Phase 3** now process **ALL stat types (PRA, PA, PR, RA) by default** with a single command!

---

## Simplified Workflow

### Complete Pipeline (All Stat Types)

```bash
# 1. Run Phase 2.5 - Recalculates features for ALL stat types
python phase2_5_precalculate_all_percentages.py

# 2. Run Phase 3 - Trains models for ALL stat types
python phase3_optimized.py

# 3. Run Phase 4 - Generates predictions for ALL stat types
python phase4_generate_predictions.py
```

**That's it!** Three simple commands process all stat types.

---

## Command Options

### Process All Stat Types (Default)

```bash
# Phase 2.5 - Process all stat types
python phase2_5_precalculate_all_percentages.py

# Phase 3 - Train all stat types
python phase3_optimized.py
```

### Process Single Stat Type (Optional)

```bash
# Phase 2.5 - Process only RA
python phase2_5_precalculate_all_percentages.py --stat-type RA

# Phase 3 - Train only RA models
python phase3_optimized.py --stat-type RA
```

### Processing Modes

**Phase 2.5** supports different modes:

```bash
# Auto mode (default) - Uses incremental if metadata exists, else full
python phase2_5_precalculate_all_percentages.py --mode auto

# Full mode - Reprocesses all data from scratch
python phase2_5_precalculate_all_percentages.py --mode full

# Incremental mode - Only processes new games
python phase2_5_precalculate_all_percentages.py --mode incremental
```

---

## What Each Phase Does

### Phase 2.5 (Data Processing)

**For EACH stat type (PRA, PA, PR, RA):**
1. Loads base data from Phase 2
2. **Recalculates ALL base features** using the correct stat type:
   - Rolling averages (last_5_avg, last_10_avg, last_20_avg)
   - Season averages (season_avg, last_season_avg)
   - Lineup average
   - H2H average
   - Opponent strength
3. Calculates percentage features for all thresholds
4. Saves to stat-specific file: `processed_with_{stat_type}_pct_{thresholds}.csv`

**Output:**
- `processed_with_pra_pct_10-51.csv` (42 PRA models worth of features)
- `processed_with_pa_pct_8-41.csv` (34 PA models worth of features)
- `processed_with_pr_pct_8-41.csv` (34 PR models worth of features)
- `processed_with_ra_pct_5-26.csv` (22 RA models worth of features)

### Phase 3 (Model Training)

**For EACH stat type (PRA, PA, PR, RA):**
1. Loads stat-specific processed data
2. Trains models for each threshold
3. Saves models to S3

**Output:**
- 42 PRA models (xgb_pra_10plus through xgb_pra_51plus)
- 34 PA models (xgb_pa_8plus through xgb_pa_41plus)
- 34 PR models (xgb_pr_8plus through xgb_pr_41plus)
- 22 RA models (xgb_ra_5plus through xgb_ra_26plus)
- **Total: 132 models**

### Phase 4 (Prediction Generation)

**For EACH stat type:**
1. Loads stat-specific processed data and models
2. Generates predictions using gauntlet approach
3. Combines all predictions into single output

**Output:**
- Single predictions file with all stat types
- Uploaded to Neon database

---

## Progress Tracking

Both Phase 2.5 and Phase 3 now show progress for each stat type and provide a final summary:

```
PHASE 2.5 FINAL SUMMARY
================================================================================
PRA: ✓ SUCCESS
  Rows: 125,432
  Mode: full
  Time: 342.5s

PA: ✓ SUCCESS
  Rows: 125,432
  Mode: full
  Time: 298.3s

PR: ✓ SUCCESS
  Rows: 125,432
  Mode: full
  Time: 301.7s

RA: ✓ SUCCESS
  Rows: 125,432
  Mode: full
  Time: 289.1s
================================================================================
```

```
PHASE 3 FINAL SUMMARY
================================================================================

PRA: ✓ SUCCESS
  Models Trained: 42
  Avg Precision: 0.8245
  Avg Accuracy: 0.7892

PA: ✓ SUCCESS
  Models Trained: 34
  Avg Precision: 0.8156
  Avg Accuracy: 0.7834

PR: ✓ SUCCESS
  Models Trained: 34
  Avg Precision: 0.8201
  Avg Accuracy: 0.7856

RA: ✓ SUCCESS
  Models Trained: 22
  Avg Precision: 0.8089
  Avg Accuracy: 0.7798

TOTAL MODELS TRAINED: 132
================================================================================
```

---

## Initial Setup (First Time Only)

```bash
# 1. Run Phase 2 (creates base data - only needed once or when adding new games)
python phase2_process_data.py

# 2. Run Phase 2.5 in full mode for all stat types
python phase2_5_precalculate_all_percentages.py --mode full

# 3. Train all models
python phase3_optimized.py

# 4. Generate predictions
python phase4_generate_predictions.py
```

---

## Daily Updates (Incremental Mode)

```bash
# 1. Update base data with new games
python phase2_process_data.py

# 2. Process new games for all stat types (uses incremental mode automatically)
python phase2_5_precalculate_all_percentages.py

# 3. Retrain models with updated data
python phase3_optimized.py

# 4. Generate fresh predictions
python phase4_generate_predictions.py
```

---

## Benefits of Default "All" Behavior

### 1. **Simplicity**
- No need to remember which stat types to process
- Single command handles everything
- Less room for error

### 2. **Consistency**
- All stat types always in sync
- No chance of forgetting to update one stat type
- Uniform processing across all models

### 3. **Automation-Friendly**
- Easy to set up cron jobs or scheduled tasks
- Simple bash scripts for complete pipeline
- Fewer commands to manage

### 4. **Transparency**
- Clear progress tracking for each stat type
- Summary shows which stat types succeeded/failed
- Easy to debug issues

---

## Backward Compatibility

You can still process individual stat types if needed:

```bash
# Just RA
python phase2_5_precalculate_all_percentages.py --stat-type RA
python phase3_optimized.py --stat-type RA

# Just PRA
python phase2_5_precalculate_all_percentages.py --stat-type PRA
python phase3_optimized.py --stat-type PRA
```

This is useful for:
- Testing changes to a specific stat type
- Debugging issues with one stat type
- Quick iterations during development

---

## Exit Codes

Both phases return appropriate exit codes for automation:

- `0`: All stat types processed successfully
- `1`: One or more stat types failed

This makes it easy to chain commands and handle errors:

```bash
# Stop pipeline if any step fails
python phase2_5_precalculate_all_percentages.py && \
python phase3_optimized.py && \
python phase4_generate_predictions.py
```

---

## Example: Complete Pipeline Script

Create `run_pipeline.sh`:

```bash
#!/bin/bash

set -e  # Exit on error

echo "Starting NBA Props Pipeline..."

echo "Phase 2: Processing base data..."
python phase2_process_data.py

echo "Phase 2.5: Calculating features for all stat types..."
python phase2_5_precalculate_all_percentages.py

echo "Phase 3: Training models for all stat types..."
python phase3_optimized.py

echo "Phase 4: Generating predictions..."
python phase4_generate_predictions.py

echo "✓ Pipeline completed successfully!"
```

Run it:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## Summary

**Before:**
```bash
# Had to run 4 separate commands for Phase 2.5
python phase2_5_precalculate_all_percentages.py --stat-type PRA
python phase2_5_precalculate_all_percentages.py --stat-type PA
python phase2_5_precalculate_all_percentages.py --stat-type PR
python phase2_5_precalculate_all_percentages.py --stat-type RA

# Had to run 4 separate commands for Phase 3
python phase3_optimized.py --stat-type PRA
python phase3_optimized.py --stat-type PA
python phase3_optimized.py --stat-type PR
python phase3_optimized.py --stat-type RA
```

**After:**
```bash
# Single command for Phase 2.5
python phase2_5_precalculate_all_percentages.py

# Single command for Phase 3
python phase3_optimized.py
```

**8 commands → 2 commands** 🎉
