# GitHub Deployment Guide - Stat Type Independence Fix

Complete step-by-step instructions to deploy the stat-type independence fix to GitHub and run it with full mode.

---

## Summary of Changes

### Code Changes:
1. ✅ **Phase 2.5**: Recalculates base features for each stat type
2. ✅ **Phase 3**: Trains all stat types by default
3. ✅ **Phase 4**: Recalculates last_season_avg and opp_strength
4. ✅ **Orchestrator**: Updated to use simplified multi-stat approach with mode support
5. ✅ **Workflow**: Added mode parameter for manual triggers

---

## Step 1: Commit and Push All Changes

```bash
# Navigate to scripts directory
cd /Users/bolajioloyede/Documents/FQ_Predictor/nba_betting_env/nba_props/deviation.io/domain/scripts

# Check status
git status

# Stage all modified files
git add phase2_5_precalculate_all_percentages.py
git add phase3_optimized.py
git add phase4_generate_predictions.py
git add github_pipeline_orchestrator.py

# Stage the workflow file
git add ../.github/workflows/nba-pipeline.yml

# Stage documentation
git add STAT_TYPE_INDEPENDENCE_FIX.md
git add UPDATED_WORKFLOW.md
git add GITHUB_DEPLOYMENT_GUIDE.md

# Create commit
git commit -m "Fix: Implement stat-type specific features for PA, PR, RA independence

Major Changes:
- Phase 2.5: Recalculate ALL base features (last_5_avg, season_avg, opp_strength, etc.) for each stat type
- Phase 3: Train all stat types (PRA, PA, PR, RA) with single command
- Phase 4: Recalculate last_season_avg and opp_strength using correct stat type
- Orchestrator: Simplified to use new default behavior + mode parameter support
- Workflow: Added mode parameter for Phase 2.5 (auto/full/incremental)

Impact:
- Each stat type now has fully independent feature calculation
- Fixes conservative prediction bias in PA, PR, RA models
- Reduces pipeline commands from 8 to 2 for Phases 2.5 & 3
- Total models: 132 (42 PRA + 34 PA + 34 PR + 22 RA)

Breaking Change:
- First run MUST use --mode full to regenerate processed data with new features
- Existing metadata and processed files will be outdated"

# Push to GitHub
git push origin main
```

---

## Step 2: Verify Push on GitHub

1. Go to your repository on GitHub
2. Check that the commit appears in the main branch
3. Verify the following files were updated:
   - `scripts/phase2_5_precalculate_all_percentages.py`
   - `scripts/phase3_optimized.py`
   - `scripts/phase4_generate_predictions.py`
   - `scripts/github_pipeline_orchestrator.py`
   - `.github/workflows/nba-pipeline.yml`

---

## Step 3: Manually Trigger GitHub Action with Full Mode

### Option A: Via GitHub Web Interface (Recommended)

1. **Go to Actions tab**
   - Navigate to: `https://github.com/YOUR_USERNAME/YOUR_REPO/actions`

2. **Select the NBA Props Pipeline workflow**
   - Click on "NBA Props Pipeline" in the left sidebar

3. **Click "Run workflow" button**
   - You'll see a dropdown button that says "Run workflow"

4. **Configure the run**
   - **Branch**: main (should be selected by default)
   - **Phases to run**: `all` (or `2,2.5,3,4` if you want to skip Phase 1)
   - **Phase 2.5 mode**: Select `full` from the dropdown ⚠️ **IMPORTANT**

5. **Click the green "Run workflow" button**

6. **Monitor the run**
   - The workflow will appear in the list
   - Click on it to see real-time logs
   - Expected duration: 4-6 hours for full pipeline

### Option B: Via GitHub CLI (Advanced)

If you have GitHub CLI installed:

```bash
# Install gh if you don't have it
# brew install gh (on macOS)
# or follow: https://cli.github.com/

# Authenticate
gh auth login

# Trigger workflow with full mode
gh workflow run nba-pipeline.yml \
  --field phases=all \
  --field mode=full

# Monitor the run
gh run watch
```

---

## Step 4: Monitor the Workflow

### What to Watch For:

1. **Phase 2 (Data Processing)**
   - Should complete in 15-30 minutes
   - Creates base data with all stat columns

2. **Phase 2.5 (Pre-calculate Percentages) - FULL MODE**
   - Will show: "Mode: full"
   - Processes ALL stat types (PRA, PA, PR, RA)
   - Expected duration: 120-150 minutes
   - Creates 4 new processed files with stat-specific features

3. **Phase 3 (Model Training)**
   - Trains ALL stat types
   - Expected duration: 120-150 minutes
   - Creates 132 models total

4. **Phase 4 (Predictions)**
   - Generates predictions for all stat types
   - Expected duration: 5-10 minutes

### Success Indicators:

Look for these messages in the logs:

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
...
```

```
PHASE 3 FINAL SUMMARY
================================================================================
PRA: ✓ SUCCESS
  Models Trained: 42
  Avg Precision: 0.8245

PA: ✓ SUCCESS
  Models Trained: 34
  Avg Precision: 0.8156
...

TOTAL MODELS TRAINED: 132
```

---

## Step 5: Verify New Metadata Created

After the successful run, new metadata files will be created in S3:

```
s3://your-bucket/processed_data/pra_metadata.json
s3://your-bucket/processed_data/pa_metadata.json
s3://your-bucket/processed_data/pr_metadata.json
s3://your-bucket/processed_data/ra_metadata.json
```

Each will contain:
```json
{
  "last_processed_date": "2025-12-31",
  "total_games": 125432,
  "stat_type": "PA",
  "threshold_start": 8,
  "threshold_end": 41,
  "mode": "full",
  "processed_at": "2025-12-31T15:30:45"
}
```

---

## Step 6: Future Runs (Daily/Weekly)

After the first successful full run, subsequent runs will use **auto mode** by default:

### Scheduled Runs (Automatic):

**Daily (Weekdays) - 10:00 AM UTC**
- Phases: 1, 2, 2.5, 4 (no model training)
- Mode: auto (uses incremental if possible)
- Duration: ~30-60 minutes

**Weekly (Sunday) - 9:00 AM UTC**
- Phases: 1, 2, 2.5, 3, 4 (with model training)
- Mode: auto (uses incremental if possible)
- Duration: ~4-6 hours

### Manual Runs:

For daily updates:
```
Phases: all (or 2,2.5,4)
Mode: auto
```

For retraining models:
```
Phases: all
Mode: auto
```

Only use `mode: full` again if you:
- Need to completely regenerate all data
- Suspect data corruption
- Make structural changes to features

---

## Troubleshooting

### Problem: Workflow Fails on Phase 2.5

**Check:**
1. Phase 2 completed successfully?
2. S3 bucket has `processed_data/processed_model_data.csv`?
3. Environment variables are set correctly?

**Solution:**
- Re-run with phases `2,2.5,3,4` and mode `full`

### Problem: Phase 3 Fails

**Check:**
1. Phase 2.5 completed for all stat types?
2. Processed files exist in S3:
   - `processed_with_pra_pct_10-51.csv`
   - `processed_with_pa_pct_8-41.csv`
   - `processed_with_pr_pct_8-41.csv`
   - `processed_with_ra_pct_5-26.csv`

**Solution:**
- Re-run Phase 2.5 with mode `full`
- Then run Phase 3

### Problem: Need to Cancel a Long-Running Workflow

1. Go to Actions tab
2. Click on the running workflow
3. Click "Cancel workflow" button (top right)

### Problem: Want to Check Logs After Failure

1. Go to the failed workflow run
2. Scroll down to "Artifacts" section
3. Download `pipeline-logs-XXXXX`
4. Extract and review log files

---

## Quick Reference Commands

### Push Changes:
```bash
git add -A
git commit -m "Your message"
git push origin main
```

### Check Remote Status:
```bash
git status
git log --oneline -5
```

### View Workflow Runs:
```bash
gh run list --limit 5
gh run view --web  # Opens latest run in browser
```

---

## Expected Timeline

| Phase | Duration | Key Output |
|-------|----------|------------|
| Phase 1 | 30-60 min | Raw player data |
| Phase 2 | 15-30 min | processed_model_data.csv |
| Phase 2.5 (full) | 120-150 min | 4 stat-specific processed files |
| Phase 3 | 120-150 min | 132 models |
| Phase 4 | 5-10 min | Daily predictions |
| **Total** | **4-6 hours** | Complete pipeline |

---

## Success Checklist

After the first full run, verify:

- [ ] Workflow completed without errors
- [ ] All 4 stat types processed in Phase 2.5
- [ ] 132 models trained in Phase 3
- [ ] Predictions generated in Phase 4
- [ ] New metadata files created in S3
- [ ] Predictions uploaded to Neon database

Once verified, future runs can use auto mode!

---

## Summary

1. ✅ Commit and push all changes
2. ✅ Go to GitHub Actions
3. ✅ Manually trigger workflow with `mode: full`
4. ✅ Monitor the 4-6 hour run
5. ✅ Verify success and new metadata
6. ✅ Future runs use `mode: auto` automatically

**The first full run is critical** - it regenerates all processed data with stat-specific features and retrains all models with the correct feature alignment!
