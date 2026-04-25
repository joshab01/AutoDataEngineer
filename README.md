# AutoDataEngineer

**A self-correcting multi-agent system that autonomously cleans messy data — it writes its own code, runs it, and fixes its own bugs.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#quick-start)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Problem

Data teams spend **40% of their time cleaning data**. Not analyzing. Not modeling. Cleaning. Missing values, inconsistent formats, mixed units, duplicated rows — the same grunt work on every new dataset.

Traditional cleaning is rule-based: write an `if` statement for every possible mess. But real data breaks in ways you can't predict. "500g" and "0.5 kg" and "500 grams" in the same column. Dates as "Jan 5, 2024" and "2024-01-05" and "01/05/24" in the same field.

You can't write rules for every mess. But you can build an agent that looks at the mess, understands the intent, and writes the fix itself.

---

## What AutoDataEngineer Does

Feed it a messy dataset. It profiles the problems, writes Python code to clean it, runs the code, and if the code fails — it reads the error, rewrites the code, and tries again. Autonomously.

**Input:** Messy CSV/API data with nulls, mixed formats, duplicates, inconsistent values
**Output:** Clean data + a quality report showing exactly what was fixed
**Key Feature:** The system debugs its own code — no human needed in the loop

---

## Architecture

```
┌────────────────────────┐
│   Messy Data (API/CSV) │
└───────────┬────────────┘
            │
┌───────────▼────────────┐
│   PROFILER AGENT       │
│                        │
│ • Scans every column   │
│ • Finds all issues     │
│ • Generates a          │
│   DATA CONTRACT        │
│   (what "clean" means) │
└───────────┬────────────┘
            │
┌───────────▼────────────────────────────┐
│   CODER AGENT                          │
│                                        │
│ • Reads contract + issues              │
│ • Writes a Python cleaning script      │
│ • On retry: receives full error history│
│ • On attempt 3: STRATEGY ESCALATION    │
│   (tries fundamentally new approach)   │
└───────────┬────────────────────────────┘
            │
┌───────────▼────────────────────────────┐
│   QA AGENT                             │
│                                        │
│ • Executes the code                    │
│ • Level 1: Did it crash?              │
│ • Level 2: Does output match contract?│
│                                        │
│   ┌─────────────────────────┐          │
│   │ SELF-CORRECTING LOOP    │          │
│   │                         │          │
│   │ If fail → send error    │          │
│   │ back to Coder Agent     │          │
│   │ with FULL error history │──→ retry │
│   │ (error memory)          │          │
│   │                         │          │
│   │ Max 3 attempts          │          │
│   └─────────────────────────┘          │
└───────────┬────────────────────────────┘
            │
┌───────────▼────────────┐
│   QUALITY REPORT       │
│                        │
│ • Before vs After      │
│ • Per-column scores    │
│ • What was fixed       │
└────────────────────────┘
```

---

## The Key Concept: Data Contracts

The Profiler Agent doesn't just find errors — it generates a **Data Contract**: a formal specification of what clean data should look like.

```json
{
  "energy_100g": {
    "expected_type": "float",
    "nullable": false,
    "constraints": "must be positive, max 5000",
    "cleaning_action": "convert strings to float, remove units"
  }
}
```

The QA Agent then validates the cleaned output against this contract. This is how production data engineering teams work — and showing you know it sets this project apart.

---

## The Interesting Bug: The Polite Infinite Loop

The Coder and QA Agents got stuck passing the same problem back and forth. Agent 2 wrote code that failed. Agent 3 sent the error back. Agent 2 rewrote with the same logical mistake. Failed again. Rewrote. Same mistake.

**The fix had three parts:**

1. **Retry cap** — max 3 attempts, then flag for human review
2. **Error memory** — on each retry, ALL previous errors are appended to the prompt, so the Coder sees "attempt 1 failed because X, attempt 2 failed because Y"
3. **Strategy escalation** — on attempt 3, the prompt changes from "fix this code" to "the previous approach isn't working, use a fundamentally different method"

See: [`src/agents.py` → `CoderAgent.generate_code()`](src/agents.py)

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Agent orchestration | LangGraph | Conditional edges for the retry loop |
| Code generation | GPT-4o-mini | Writes Python cleaning scripts |
| Code execution | Python `exec()` | Runs generated code in isolated namespace |
| Data source | Open Food Facts API | Real messy data, free, no API key |
| Validation | Data Contracts | Schema-based output validation |
| Data handling | pandas | Standard data manipulation |

---

## Quick Start

### Prerequisites
- Python 3.10+
- [OpenAI API key](https://platform.openai.com/api-keys) (total cost: ~$2-3)

### Run in Google Colab

Add this to the first cell:
```python
!pip install openai langgraph langchain-core pandas requests tabulate -q
import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

# Download agents.py from GitHub
!mkdir -p src data
!wget -q https://raw.githubusercontent.com/YOUR_USERNAME/AutoDataEngineer/main/src/__init__.py -O src/__init__.py
!wget -q https://raw.githubusercontent.com/YOUR_USERNAME/AutoDataEngineer/main/src/agents.py -O src/agents.py
import sys; sys.path.insert(0, 'src')
```

| # | Notebook | What It Does | Time |
|---|----------|-------------|------|
| 01 | [01_data_ingestion](notebooks/01_data_ingestion.ipynb) | Pull real messy data from API | ~2 min |
| 02 | [02_agents_pipeline](notebooks/02_agents_pipeline.ipynb) | **Full self-correcting pipeline** | ~3-5 min |
| 03 | [03_quality_report](notebooks/03_quality_report.ipynb) | Before/after quality analysis | ~1 min |

### Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/AutoDataEngineer.git
cd AutoDataEngineer
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
cd notebooks
python 01_data_ingestion.py
python 02_agents_pipeline.py
python 03_quality_report.py
```

---

## Project Structure

```
AutoDataEngineer/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   └── agents.py             # ProfilerAgent, CoderAgent (with error memory),
│                              # QAAgent (with contract validation)
├── notebooks/
│   ├── 01_data_ingestion.py   # Pull + explore messy data
│   ├── 02_agents_pipeline.py  # Full LangGraph pipeline
│   └── 03_quality_report.py   # Before/after analysis
└── data/                      # Generated at runtime
    ├── raw_data.csv
    ├── cleaned_data.csv
    ├── profile_result.json
    ├── quality_report.json
    └── quality_report.md
```

---

## Sample Output

```
PIPELINE STATISTICS
  Total attempts:        2
  Errors encountered:    1
  Contract violations:   0
  Validation passed:     True
  Nulls fixed:           47
  Duplicates removed:    8
  Rows: 200 → 184
  Quality score:         89.2/100
```

---

## What I'd Add for Production

- **Multiple data source connectors** (Snowflake, BigQuery, S3)
- **Sandboxed code execution** (Docker container instead of exec())
- **Version-controlled Data Contracts** stored alongside the data
- **Feedback loop** — human corrections improve future code generation
- **Scheduling** — run as a nightly data quality pipeline

---

## Cost

| Item | Cost |
|------|------|
| Profiling (NB 01) | Free (API call) |
| Pipeline (NB 02) | ~$1-2 |
| Quality Report (NB 03) | Free (local computation) |
| **Total** | **Under $3** |

---

## License

MIT

---

*Built as part of my Build in Public series. Second project showcasing autonomous AI agents — this time with self-correcting code generation.*
