# Human Intervention Checklist
## Multi-Stat Type Implementation (PRA, PA, PR, RA)

This checklist outlines all manual steps required after the automated code refactoring is complete.

---

## 📋 Pre-Deployment Checklist

### ✅ 1. Database Migration (REQUIRED)

**Action**: Add `stat_type` column to your Neon database

**Steps**:
1. Connect to your Neon database:
   ```bash
   psql $NEON_DATABASE_URL
   ```

2. Run the migration script:
   ```bash
   psql $NEON_DATABASE_URL < migrations/001_add_stat_type_column.sql
   ```

   **OR** manually execute:
   ```sql
   ALTER TABLE nba_props
   ADD COLUMN IF NOT EXISTS stat_type VARCHAR(10) DEFAULT 'PRA';

   CREATE INDEX IF NOT EXISTS idx_nba_props_stat_type
   ON nba_props(stat_type);

   CREATE INDEX IF NOT EXISTS idx_nba_props_stat_prop_type
   ON nba_props(stat_type, prop_type);

   CREATE INDEX IF NOT EXISTS idx_nba_props_stat_confidence
   ON nba_props(stat_type, confidence_score DESC);
   ```

3. Verify the column was added:
   ```sql
   SELECT column_name, data_type
   FROM information_schema.columns
   WHERE table_name = 'nba_props' AND column_name = 'stat_type';
   ```

**Expected Output**: Should show `stat_type` column with type `character varying(10)`

---

### ✅ 2. Local Testing (RECOMMENDED)

**Action**: Test the updated pipeline locally before pushing to GitHub

#### Test Phase 2 (Add PA, PR, RA columns)
```bash
cd /Users/bolajioloyede/Documents/FQ_Predictor/nba_betting_env/nba_props/deviation.io/domain/scripts
python phase2_process_data.py
```

**Expected**: Should complete and log "Added PA, PR, RA columns"

---

#### Test Phase 2.5 (Multi-Stat + Incremental Mode)

**Test PRA (default):**
```bash
python phase2_5_precalculate_all_percentages.py --stat-type PRA --mode full
```

**Test PA:**
```bash
python phase2_5_precalculate_all_percentages.py --stat-type PA --mode full
```

**Test Incremental Mode** (after first full run):
```bash
python phase2_5_precalculate_all_percentages.py --stat-type PRA --mode incremental
```

**Expected**:
- Full mode: ~40-60 minutes per stat type
- Incremental mode: ~1-5 minutes (only for testing)
- Files created in S3:
  - `processed_data/processed_with_pra_pct_10-51.csv`
  - `processed_data/processed_with_pa_pct_8-41.csv`
  - `processed_data/processed_with_pr_pct_8-41.csv`
  - `processed_data/processed_with_ra_pct_5-26.csv`
  - `processed_data/PRA_metadata.json`
  - `processed_data/PA_metadata.json`
  - etc.

---

#### Test Phase 3 (Multi-Stat Model Training)

**Test PRA models:**
```bash
python phase3_optimized.py --stat-type PRA
```

**Test PA models:**
```bash
python phase3_optimized.py --stat-type PA
```

**Expected**:
- ~30-35 minutes per stat type
- Models created in S3:
  - `models/xgb_pra_10plus_*.pkl` through `models/xgb_pra_51plus_*.pkl` (42 models)
  - `models/xgb_pa_8plus_*.pkl` through `models/xgb_pa_41plus_*.pkl` (34 models)
  - `models/xgb_pr_8plus_*.pkl` through `models/xgb_pr_41plus_*.pkl` (34 models)
  - `models/xgb_ra_5plus_*.pkl` through `models/xgb_ra_26plus_*.pkl` (22 models)
  - **Total: 132 models**

---

#### Test Phase 4 (Multi-Stat Predictions)

```bash
python phase4_generate_predictions.py
```

**Expected**:
- Generates predictions for ALL stat types (PRA, PA, PR, RA)
- Creates `props.csv` with predictions from all stat types
- Logs show:
  ```
  Generating PRA Predictions
  ✓ Generated X PRA predictions

  Generating PA Predictions
  ✓ Generated X PA predictions

  Generating PR Predictions
  ✓ Generated X PR predictions

  Generating RA Predictions
  ✓ Generated X RA predictions
  ```
- Database should have `stat_type` column populated

---

#### Test Orchestrator (All Phases)

```bash
python github_pipeline_orchestrator.py --phases "2,2.5,3,4"
```

**Expected**:
- Phase 2: Completes successfully
- Phase 2.5: Runs for ALL 4 stat types sequentially
- Phase 3: Runs for ALL 4 stat types sequentially
- Phase 4: Generates predictions for all stat types
- Total time: ~4-6 hours for full run

---

### ✅ 3. Initial Full Pipeline Run (REQUIRED)

**Action**: Run the complete pipeline to generate all models and processed data

**Recommended Approach**: Use the orchestrator

```bash
# Run Phases 2, 2.5, and 3 to create all models and processed data
python github_pipeline_orchestrator.py --phases "2,2.5,3"
```

**Expected Duration**:
- Phase 2: ~15-30 minutes
- Phase 2.5: ~2-3 hours (4 stat types × 40-60 min each)
- Phase 3: ~2-3 hours (4 stat types × 30-35 min each)
- **Total: ~5-7 hours**

**What This Does**:
1. Processes data and adds PA, PR, RA columns
2. Calculates percentages for all stat types (PRA, PA, PR, RA)
3. Trains 132 models (42 PRA + 34 PA + 34 PR + 22 RA)
4. Uploads everything to S3

---

### ✅ 4. Verify S3 Contents (REQUIRED)

**Action**: Check that all files were created in S3

#### Processed Data Files (4 files):
```bash
aws s3 ls s3://$S3_PLAYER_BUCKET/processed_data/ --recursive | grep "processed_with_.*_pct_"
```

**Expected Output**:
```
processed_data/processed_with_pra_pct_10-51.csv
processed_data/processed_with_pa_pct_8-41.csv
processed_data/processed_with_pr_pct_8-41.csv
processed_data/processed_with_ra_pct_5-26.csv
```

#### Metadata Files (4 files):
```bash
aws s3 ls s3://$S3_PLAYER_BUCKET/processed_data/ --recursive | grep "metadata.json"
```

**Expected Output**:
```
processed_data/PRA_metadata.json
processed_data/PA_metadata.json
processed_data/PR_metadata.json
processed_data/RA_metadata.json
```

#### Model Files (132 models):
```bash
# Count models per stat type
aws s3 ls s3://$S3_MODEL_BUCKET/models/ --recursive | grep "xgb_pra_" | wc -l  # Should be 42
aws s3 ls s3://$S3_MODEL_BUCKET/models/ --recursive | grep "xgb_pa_" | wc -l   # Should be 34
aws s3 ls s3://$S3_MODEL_BUCKET/models/ --recursive | grep "xgb_pr_" | wc -l   # Should be 34
aws s3 ls s3://$S3_MODEL_BUCKET/models/ --recursive | grep "xgb_ra_" | wc -l   # Should be 22
```

**Total Model Count**: 132 models

---

### ✅ 5. Verify Database Schema (REQUIRED)

**Action**: Verify `stat_type` column exists and is populated

```sql
-- Check column exists
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'nba_props' AND column_name = 'stat_type';

-- Check data is populated
SELECT stat_type, COUNT(*) as count
FROM nba_props
GROUP BY stat_type
ORDER BY stat_type;
```

**Expected Output**:
```
stat_type | count
----------+-------
PA        | X
PR        | X
PRA       | X
RA        | X
```

---

### ✅ 6. Git Commit and Push (REQUIRED)

**Action**: Commit all changes and push to GitHub

```bash
cd /Users/bolajioloyede/Documents/FQ_Predictor/nba_betting_env/nba_props/deviation.io/domain/scripts

# Check what was changed
git status

# Add all modified files
git add phase2_process_data.py
git add phase2_5_precalculate_all_percentages.py
git add phase3_optimized.py
git add phase4_generate_predictions.py
git add github_pipeline_orchestrator.py
git add migrations/001_add_stat_type_column.sql
git add HUMAN_INTERVENTION_CHECKLIST.md

# Commit with descriptive message
git commit -m "feat: Add multi-stat type support (PA, PR, RA) + incremental mode

- Phase 2: Add PA, PR, RA calculations
- Phase 2.5: Add incremental mode + multi-stat support
- Phase 3: Train models for all stat types (132 total)
- Phase 4: Generate predictions for all stat types
- Orchestrator: Handle multi-stat workflows
- Database: Add stat_type column with indexes

🤖 Generated with Opus + Claude Code"

# Push to GitHub
git push origin main
```

**Important**: After pushing, GitHub Actions will use the updated code for scheduled runs.

---

### ✅ 7. Update GitHub Actions Workflow (OPTIONAL)

**Action**: Review and optionally update the GitHub Actions workflow

**File**: `.github/workflows/nba-pipeline.yml`

**Current Behavior**:
- Daily: Runs phases `2, 2.5` (data processing)
- Weekly: Runs phases `2, 2.5, 3` (includes model training)

**Updated Behavior** (automatically applied):
- Daily: Runs Phase 2 once, Phase 2.5 for all 4 stat types
- Weekly: Runs Phase 2 once, Phase 2.5 for all 4 stat types, Phase 3 for all 4 stat types

**Optional Changes**:
- Add Phase 4 to daily workflow to generate predictions
- Adjust timing/frequency
- Add notification hooks

---

## 🎯 Post-Deployment Monitoring

### ✅ 8. Monitor First GitHub Actions Run

**Action**: Watch the first scheduled GitHub Actions run

1. Go to: `https://github.com/YOUR_USERNAME/YOUR_REPO/actions`
2. Wait for next scheduled run (daily at 10:00 AM UTC or weekly Sunday 9:00 AM UTC)
3. Monitor the workflow logs

**Watch For**:
- Phase 2.5 runs for all 4 stat types
- Phase 3 runs for all 4 stat types (weekly only)
- No errors in stat type processing
- Models uploaded successfully

---

### ✅ 9. Verify Incremental Mode Works

**Action**: After 1-2 days, verify incremental mode is being used

**Check Logs**:
```
Phase 2.5 should log:
"Mode: incremental (metadata found)"
"Processing X new games since YYYY-MM-DD"
"Incremental update completed in ~3 minutes"
```

**If Full Mode is Running Every Time**:
- Check metadata files exist in S3
- Check metadata has correct `last_processed_date`
- Check mode is set to `'auto'` (default)

---

### ✅ 10. Validate Prediction Quality

**Action**: Review predictions across all stat types

1. Query database for sample predictions:
   ```sql
   SELECT name, stat_type, prop, line, confidence_score
   FROM nba_props
   WHERE confidence_score > 0.80
   ORDER BY stat_type, confidence_score DESC
   LIMIT 20;
   ```

2. Check prediction distribution:
   ```sql
   SELECT
       stat_type,
       prop_type,
       COUNT(*) as count,
       AVG(confidence_score) as avg_confidence,
       MIN(line) as min_line,
       MAX(line) as max_line
   FROM nba_props
   GROUP BY stat_type, prop_type
   ORDER BY stat_type, prop_type;
   ```

**Expected**:
- PA predictions: Lines between 7.5 and 40.5
- PR predictions: Lines between 7.5 and 40.5
- RA predictions: Lines between 4.5 and 25.5
- PRA predictions: Lines > 9 (existing)

---

## 📊 Performance Metrics

### Expected Processing Times:

| Phase | Stat Types | Time (Initial) | Time (Incremental) |
|-------|------------|----------------|-------------------|
| Phase 2 | All | 15-30 min | N/A |
| Phase 2.5 | PRA | 40-60 min | 1-5 min |
| Phase 2.5 | PA | 40-60 min | 1-5 min |
| Phase 2.5 | PR | 40-60 min | 1-5 min |
| Phase 2.5 | RA | 30-40 min | 1-5 min |
| Phase 3 | PRA | 30-35 min | N/A |
| Phase 3 | PA | 25-30 min | N/A |
| Phase 3 | PR | 25-30 min | N/A |
| Phase 3 | RA | 15-20 min | N/A |
| Phase 4 | All | 10-15 min | N/A |

**Total Initial Run**: 5-7 hours
**Daily Incremental**: 15-30 minutes (95%+ time savings)

---

## 🆘 Troubleshooting

### Issue: Models not found in Phase 4

**Symptom**: `"No model found for threshold X"`

**Solution**:
1. Check S3 for models: `aws s3 ls s3://$S3_MODEL_BUCKET/models/ | grep xgb_{stat_type}_`
2. Run Phase 3 for that stat type: `python phase3_optimized.py --stat-type {STAT_TYPE}`

---

### Issue: Processed data not found in Phase 4

**Symptom**: `"Failed to download processed data from S3"`

**Solution**:
1. Check S3 for processed files: `aws s3 ls s3://$S3_PLAYER_BUCKET/processed_data/`
2. Run Phase 2.5 for that stat type: `python phase2_5_precalculate_all_percentages.py --stat-type {STAT_TYPE} --mode full`

---

### Issue: Database column missing

**Symptom**: `"column 'stat_type' does not exist"`

**Solution**:
1. Run migration: `psql $NEON_DATABASE_URL < migrations/001_add_stat_type_column.sql`
2. Verify: `SELECT column_name FROM information_schema.columns WHERE table_name = 'nba_props' AND column_name = 'stat_type';`

---

### Issue: Incremental mode not working

**Symptom**: Always running full mode (40-60 min)

**Solution**:
1. Check metadata exists: `aws s3 ls s3://$S3_PLAYER_BUCKET/processed_data/ | grep metadata`
2. Check metadata content: `aws s3 cp s3://$S3_PLAYER_BUCKET/processed_data/PRA_metadata.json -`
3. Verify `last_processed_date` is recent
4. Force incremental: `python phase2_5_precalculate_all_percentages.py --mode incremental`

---

## ✅ Completion Checklist

Mark each item when completed:

- [ ] Database migration completed (`stat_type` column added)
- [ ] Phase 2 tested locally
- [ ] Phase 2.5 tested locally (at least one stat type)
- [ ] Phase 3 tested locally (at least one stat type)
- [ ] Phase 4 tested locally
- [ ] Initial full pipeline run completed (Phases 2, 2.5, 3)
- [ ] S3 verified: 4 processed data files
- [ ] S3 verified: 4 metadata files
- [ ] S3 verified: 132 model files
- [ ] Database verified: `stat_type` column exists
- [ ] Database verified: Predictions have all stat types
- [ ] Git committed and pushed to GitHub
- [ ] GitHub Actions workflow reviewed
- [ ] First scheduled run completed successfully
- [ ] Incremental mode verified working
- [ ] Prediction quality validated

---

## 🎉 Success Criteria

Your implementation is successful when:

1. ✅ All 132 models trained and uploaded to S3
2. ✅ All 4 stat types have processed data in S3
3. ✅ Phase 4 generates predictions for all stat types
4. ✅ Database contains predictions with `stat_type` column populated
5. ✅ Incremental mode saves 95% processing time on daily updates
6. ✅ GitHub Actions runs successfully with updated code
7. ✅ Predictions meet quality thresholds across all stat types

---

## 📞 Support

If you encounter issues not covered in this checklist:

1. Check GitHub Issues for similar problems
2. Review pipeline logs in `logs/` directory
3. Check S3 bucket contents for missing files
4. Verify environment variables are set correctly
5. Test each phase individually to isolate the issue

---

**Document Version**: 1.0
**Last Updated**: 2025-12-29
**Generated By**: Claude Code (Opus)
