# AutoDataEngineer: Complete Setup Guide
## From GitHub Upload to Running in Google Colab

---

## PART 1: Push to GitHub (5 Minutes)

### Step 1: Create Repository
1. Go to [github.com/new](https://github.com/new)
2. Name: `AutoDataEngineer`
3. Description: `Self-correcting multi-agent system that autonomously cleans messy data — writes code, runs it, fixes its own bugs`
4. Visibility: **Public**
5. Do NOT check "Add README" (we have one)
6. Click **Create repository**

### Step 2: Upload Files

**Option A — GitHub Web UI:**
1. Click **"uploading an existing file"**
2. Unzip `AutoDataEngineer.zip` on your computer
3. Drag all files and folders in:
   ```
   README.md
   requirements.txt
   .gitignore
   src/__init__.py
   src/agents.py
   notebooks/01_data_ingestion.ipynb
   notebooks/02_agents_pipeline.ipynb
   notebooks/03_quality_report.ipynb
   ```
4. Commit message: `Initial commit: AutoDataEngineer multi-agent system`
5. Click **Commit changes**

**Option B — Git CLI:**
```bash
cd AutoDataEngineer
git init && git add . && git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/AutoDataEngineer.git
git branch -M main && git push -u origin main
```

---

## PART 2: Get OpenAI API Key (3 Minutes)

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create new key → name it `AutoDataEngineer`
3. Copy the `sk-...` key
4. Add $5 credits at billing settings (you'll use ~$2-3)

---

## PART 3: Run in Google Colab

### Recommended: Run Everything in ONE Notebook

This avoids Colab's file persistence issues.

### Step 1: Open a fresh Colab notebook
Go to [colab.research.google.com](https://colab.research.google.com) → **New Notebook**

### Step 2: Cell 1 — Master Setup

```python
# ── MASTER SETUP ──
!pip install openai langgraph langchain-core pandas requests tabulate -q

import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

# Download agents.py from GitHub (replace YOUR_USERNAME)
!mkdir -p src data
!wget -q https://raw.githubusercontent.com/YOUR_USERNAME/AutoDataEngineer/main/src/__init__.py -O src/__init__.py
!wget -q https://raw.githubusercontent.com/YOUR_USERNAME/AutoDataEngineer/main/src/agents.py -O src/agents.py

import sys
sys.path.insert(0, 'src')
print("Setup complete")
```

**Set up your API key securely:**
1. Click the **🔑 key icon** in left sidebar
2. Add new secret: name = `OPENAI_API_KEY`, value = your `sk-...` key
3. Toggle **Notebook access** ON

### Step 3: Cell 2 — Pull Messy Data (Notebook 01)

```python
import json
import random
import requests
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Pull from Open Food Facts API
def fetch_openfoodfacts(num_pages=5, page_size=50):
    all_products = []
    base_url = "https://world.openfoodfacts.org/cgi/search.pl"
    for page in range(1, num_pages + 1):
        print(f"  Fetching page {page}/{num_pages}...")
        try:
            resp = requests.get(base_url, params={
                "action": "process", "json": "true",
                "page": page, "page_size": page_size,
                "fields": "product_name,brands,categories,quantity,serving_size,"
                          "energy_100g,fat_100g,sugars_100g,proteins_100g,"
                          "nutrition_grade_fr,countries,last_modified_datetime,code,packaging",
            }, timeout=15)
            all_products.extend(resp.json().get("products", []))
        except Exception as e:
            print(f"  Page {page} failed: {e}")
    return all_products

print("Pulling data from Open Food Facts API...")
products = fetch_openfoodfacts()

if len(products) >= 50:
    df_raw = pd.DataFrame(products)
    print(f"Got {len(df_raw)} products from API")
else:
    # Fallback: generate synthetic messy data
    print("API unavailable, using synthetic data...")
    # (Copy the generate_messy_data function from notebook 01)
    # ... or open 01_data_ingestion.ipynb from GitHub and run Cell 3

KEEP_COLUMNS = [
    "product_name", "brands", "categories", "quantity", "serving_size",
    "energy_100g", "fat_100g", "sugars_100g", "proteins_100g",
    "nutrition_grade_fr", "countries", "last_modified_datetime", "code", "packaging"
]
available_cols = [c for c in KEEP_COLUMNS if c in df_raw.columns]
df_raw = df_raw[available_cols].copy()
df_raw = df_raw.replace({"": np.nan, " ": np.nan})

df_raw.to_csv(DATA_DIR / "raw_data.csv", index=False)
print(f"Saved: {len(df_raw)} rows x {len(df_raw.columns)} columns")
```

### Step 4: Cell 3 — Profile Data + Generate Contract

```python
from agents import ProfilerAgent

profiler = ProfilerAgent()
profile_result = profiler.profile(df_raw)

# Show results
report = profile_result.get("profiling_report", {})
print(f"Issues found: {report.get('issues_found', 0)}")
for issue in report.get("issues", []):
    print(f"  [{issue.get('severity','?').upper()}] {issue.get('column','?')}: {issue.get('issue','')}")

print("\n-- Data Contract --")
for col, rules in profile_result.get("data_contract", {}).items():
    if not col.startswith("__"):
        print(f"  {col}: type={rules.get('expected_type','?')}, action={rules.get('cleaning_action','none')}")

with open(DATA_DIR / "profile_result.json", "w") as f:
    json.dump(profile_result, f, indent=2, default=str)
```

### Step 5: Cell 4 — Run the Self-Correcting Pipeline

```python
import time
from io import StringIO
from agents import CoderAgent, QAAgent
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

coder = CoderAgent()
qa_agent = QAAgent()

# (Copy the full LangGraph pipeline code from notebook 02, cells 4-5)
# Or open 02_agents_pipeline.ipynb from GitHub and copy cells 4+5

# ... pipeline code here ...

# Run it
start_time = time.time()
result = app.invoke({...})  # See notebook 02 for full invocation
elapsed = time.time() - start_time
```

**For the cleanest experience:** Open each `.ipynb` directly from GitHub in Colab, just make sure to run the master setup cell first in each one.

### Step 6: Cell 5 — Quality Report

```python
# (Copy from notebook 03)
# Loads raw_data.csv and cleaned_data.csv, generates before/after report
```

---

## Running Individual Notebooks from GitHub

If you prefer running notebooks separately:

### Open any notebook directly:
```
https://colab.research.google.com/github/YOUR_USERNAME/AutoDataEngineer/blob/main/notebooks/01_data_ingestion.ipynb
```

### For each notebook, add this as the FIRST cell:
```python
!pip install openai langgraph langchain-core pandas requests tabulate -q
import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

!mkdir -p src data
!wget -q https://raw.githubusercontent.com/YOUR_USERNAME/AutoDataEngineer/main/src/__init__.py -O src/__init__.py
!wget -q https://raw.githubusercontent.com/YOUR_USERNAME/AutoDataEngineer/main/src/agents.py -O src/agents.py
import sys; sys.path.insert(0, 'src')
```

### For Notebook 02, also upload the data:
```python
# If data/ files don't exist from Notebook 01, re-run NB01 first
# Or upload raw_data.csv manually via Colab's file browser
```

### Run order: 01 → 02 → 03 (always in same session)

---

## What to Screenshot for LinkedIn

After running the pipeline, grab these:

1. **The profiling output** — shows the Data Contract and issues found
2. **The processing log** — shows each agent working + retry attempts
3. **The generated Python code** — AI-written cleaning script (the wow factor)
4. **The before/after comparison** — nulls fixed, duplicates removed, quality score
5. **The self-correcting loop log** — error memory + strategy escalation in action

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: agents` | Run the `wget` setup cell first |
| `FileNotFoundError: raw_data.csv` | Run Notebook 01 in the same session |
| API timeout on Open Food Facts | Code auto-falls back to synthetic data |
| `exec()` code fails all 3 attempts | Expected sometimes — shows retry cap working! Re-run the pipeline cell |
| `JSONDecodeError` in ProfilerAgent | Re-run the cell — GPT occasionally returns bad JSON |

---

## Quick Reference

| Notebook | Creates | Time | Cost |
|----------|---------|------|------|
| 01 — Data Ingestion | `raw_data.csv`, `raw_meta.json` | ~2 min | Free |
| 02 — Pipeline | `cleaned_data.csv`, `profile_result.json` | ~3-5 min | ~$1-2 |
| 03 — Quality Report | `quality_report.json`, `quality_report.md` | ~1 min | Free |
| **Total** | | **~6-8 min** | **Under $3** |
