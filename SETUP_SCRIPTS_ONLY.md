# Setup Guide: Push Scripts Directory Only

This guide shows how to push only the `/scripts` directory to GitHub.

## Method 1: Make Scripts Directory a Standalone Repository

### Step 1: Navigate to Scripts Directory
```bash
cd /Users/bolajioloyede/Documents/FQ_Predictor/nba_betting_env/nba_props/deviation.io/domain/scripts
```

### Step 2: Initialize Git Repository
```bash
# Initialize git in scripts directory
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Logs
logs/
*.log

# Environment variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files (optional - uncomment if you don't want to track data)
# data/
# *.csv
# *.pkl
EOF

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: NBA Props Pipeline"
```

### Step 3: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `nba-props-pipeline` (or your choice)
3. Description: "NBA Player Props Prediction Pipeline with GitHub Actions"
4. Choose **Public** (unlimited free minutes) or **Private** (2,000 free minutes/month)
5. **DO NOT** check "Initialize with README"
6. Click **Create repository**

### Step 4: Push to GitHub
```bash
# Add remote (replace USERNAME and REPO_NAME)
git remote add origin https://github.com/USERNAME/nba-props-pipeline.git

# Push code
git branch -M main
git push -u origin main
```

### Step 5: Verify GitHub Actions Workflow

Your workflow file should now be at:
```
.github/workflows/nba-pipeline.yml
```

Check on GitHub:
- Go to your repository
- Navigate to `.github/workflows/`
- Verify `nba-pipeline.yml` exists

### Step 6: Update Workflow File (Already Done!)

The workflow is already configured to work from the scripts directory root:

```yaml
- name: Install dependencies
  run: pip install -r requirements.txt

- name: Run NBA Pipeline
  run: python github_pipeline_orchestrator.py --phases "${{ steps.phases.outputs.phases }}"
```

No changes needed since scripts is now the root!

### Step 7: Configure GitHub Secrets

Follow the same steps as in GITHUB_ACTIONS_SETUP.md:

1. Go to **Repository → Settings → Secrets and variables → Actions**
2. Add these 7 secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION`
   - `S3_PLAYER_BUCKET`
   - `S3_LINEUP_BUCKET`
   - `S3_MODEL_BUCKET`
   - `NEON_DATABASE_URL`

### Step 8: Test the Pipeline

1. Go to **Actions** tab
2. Click **NBA Props Pipeline**
3. Click **Run workflow**
4. Enter phases: `1,2,4`
5. Click **Run workflow**

---

## Method 2: Push Scripts as Part of Existing Repo (Not Recommended)

If you already have a repo at the parent level, you could:

```bash
cd /Users/bolajioloyede/Documents/FQ_Predictor/nba_betting_env/nba_props/deviation.io/domain

# Add only scripts directory
git add scripts/
git add .github/
git commit -m "Add NBA Props pipeline scripts"
git push
```

But this requires pushing the entire domain folder structure. Method 1 is cleaner.

---

## Updated File Structure

Your GitHub repository will look like this:

```
nba-props-pipeline/
├── .github/
│   └── workflows/
│       └── nba-pipeline.yml
├── .gitignore
├── github_pipeline_orchestrator.py
├── phase1_fetch_data_optimized.py
├── phase2_process_data.py
├── phase2_5_precalculate_all_percentages.py
├── phase3_optimized.py
├── phase4_generate_predictions.py
├── requirements.txt
├── s3_utils.py
├── metadata_utils.py
├── db_utils.py
└── logs/  (excluded by .gitignore)
```

---

## Important Notes

### Data Files
The `.gitignore` excludes logs by default. If you have data files or pickles in the scripts directory, decide if you want to track them:

- **Track data:** Remove `# data/` and `# *.csv` from .gitignore
- **Exclude data:** Keep as-is (data stays in S3, not GitHub)

### Environment Variables
Never commit `.env` files! Always use GitHub Secrets.

### Workflow Path
Since scripts is now the repository root, the workflow runs commands directly:
```yaml
run: python github_pipeline_orchestrator.py
```

Instead of:
```yaml
working-directory: scripts/
run: python github_pipeline_orchestrator.py
```

---

## Quick Commands Summary

```bash
# 1. Navigate to scripts
cd /Users/bolajioloyede/Documents/FQ_Predictor/nba_betting_env/nba_props/deviation.io/domain/scripts

# 2. Initialize git
git init
git add .
git commit -m "Initial commit: NBA Props Pipeline"

# 3. Add remote (replace with your repo URL)
git remote add origin https://github.com/USERNAME/REPO_NAME.git

# 4. Push
git branch -M main
git push -u origin main
```

That's it! Your scripts directory is now a standalone GitHub repository ready for GitHub Actions.
