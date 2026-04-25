"""
src/agents.py — AutoDataEngineer Agent Definitions

Three core agents:
  1. ProfilerAgent — Analyzes messy data + generates a Data Contract
  2. CoderAgent   — Writes Python cleaning code (with error memory on retries)
  3. QAAgent      — Executes code + validates output against the Data Contract

The self-correcting loop: CoderAgent writes → QAAgent runs → if fail → error
goes back to CoderAgent with full error history + strategy escalation.
"""

import json
import logging
import traceback
import pandas as pd
from io import StringIO
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("AutoDE")

client = OpenAI()
MODEL = "gpt-4o-mini"


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 1: Profiler Agent
# ═════════════════════════════════════════════════════════════════════════════

class ProfilerAgent:
    """
    Analyzes raw messy data and produces:
      1. A profiling report (what's wrong with the data)
      2. A Data Contract (what "clean" looks like)

    The Data Contract is the key concept — it defines expected types,
    constraints, and rules that the QA Agent validates against.
    """

    def profile(self, df: pd.DataFrame) -> dict:
        """
        Profile a DataFrame and generate a Data Contract.

        Returns dict with:
          - profiling_report: human-readable analysis of issues
          - data_contract: machine-readable spec of what clean data looks like
          - column_issues: per-column breakdown of problems found
        """
        logger.info(f"ProfilerAgent: Analyzing {len(df)} rows x {len(df.columns)} columns...")

        # ── Step 1: Automated statistical profiling ──
        stats = self._compute_stats(df)

        # ── Step 2: LLM-powered analysis (catches semantic issues) ──
        sample_csv = df.head(20).to_csv(index=False)
        stats_text = json.dumps(stats, indent=2, default=str)

        prompt = f"""You are a senior data engineer. Analyze this dataset and produce TWO things:

1. PROFILING REPORT — List every data quality issue you find:
   - Null/missing values (which columns, how many)
   - Inconsistent formats (dates, units, casing)
   - Duplicates or near-duplicates
   - Invalid values (negative prices, future dates for past events, etc.)
   - Mixed types in single columns
   - Any other anomalies

2. DATA CONTRACT — For each column, define what CLEAN data should look like:
   - expected_type: string/float/int/date/boolean
   - nullable: true/false
   - format: specific format if applicable (e.g., "ISO-8601" for dates, "grams" for weight)
   - constraints: any rules (e.g., "must be positive", "max 5000", "must be lowercase")
   - cleaning_action: what needs to be done ("standardize units to grams", "parse mixed date formats to ISO-8601", "fill nulls with 'Unknown'")

COLUMN STATISTICS:
{stats_text}

SAMPLE DATA (first 20 rows):
{sample_csv}

Return ONLY a JSON object with this structure:
{{
  "profiling_report": {{
    "total_rows": <int>,
    "total_columns": <int>,
    "issues_found": <int>,
    "summary": "<2-3 sentence overview>",
    "issues": [
      {{"column": "<name>", "issue": "<description>", "severity": "high/medium/low", "affected_rows": <int or estimate>}}
    ]
  }},
  "data_contract": {{
    "<column_name>": {{
      "expected_type": "<type>",
      "nullable": <bool>,
      "format": "<format or null>",
      "constraints": "<rules or null>",
      "cleaning_action": "<what to do>"
    }}
  }}
}}

No markdown fences. No extra text."""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4000,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("ProfilerAgent: Failed to parse LLM output, using stats-only profile")
            result = {
                "profiling_report": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "issues_found": sum(1 for s in stats.values() if s.get("null_count", 0) > 0),
                    "summary": "Auto-generated profile from statistics.",
                    "issues": []
                },
                "data_contract": {}
            }

        result["column_stats"] = stats
        num_issues = len(result.get("profiling_report", {}).get("issues", []))
        logger.info(f"ProfilerAgent: Found {num_issues} issues, contract covers {len(result.get('data_contract', {}))} columns")
        return result

    def _compute_stats(self, df: pd.DataFrame) -> dict:
        """Compute per-column statistics."""
        stats = {}
        for col in df.columns:
            col_stats = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_pct": round(float(df[col].isnull().mean() * 100), 1),
                "unique_count": int(df[col].nunique()),
                "total_rows": len(df),
            }
            # Sample values for context
            non_null = df[col].dropna()
            if len(non_null) > 0:
                col_stats["sample_values"] = [str(v) for v in non_null.sample(min(5, len(non_null))).tolist()]
            if df[col].dtype in ["float64", "int64"]:
                col_stats["min"] = float(df[col].min()) if not df[col].isnull().all() else None
                col_stats["max"] = float(df[col].max()) if not df[col].isnull().all() else None
            stats[col] = col_stats

        # Check for duplicate rows
        stats["__duplicates__"] = {
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_pct": round(float(df.duplicated().mean() * 100), 1),
        }
        return stats


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 2: Coder Agent
# ═════════════════════════════════════════════════════════════════════════════

class CoderAgent:
    """
    Writes Python cleaning code based on the profiling report + data contract.

    On retries:
      - Receives full error history (error memory)
      - On attempt 3+: gets "strategy escalation" prompt to try a
        fundamentally different approach instead of tweaking the same code
    """

    def generate_code(
        self,
        profiling_result: dict,
        sample_csv: str,
        attempt: int = 1,
        error_history: list[dict] | None = None,
    ) -> str:
        """
        Generate Python code to clean the dataset.

        Args:
            profiling_result: output from ProfilerAgent
            sample_csv: CSV string of sample data
            attempt: current attempt number (1-indexed)
            error_history: list of {"attempt": N, "code": "...", "error": "..."} from prior failures

        Returns:
            Python code as a string
        """
        logger.info(f"CoderAgent: Generating cleaning code (attempt {attempt})...")

        contract_text = json.dumps(profiling_result.get("data_contract", {}), indent=2)
        issues_text = json.dumps(profiling_result.get("profiling_report", {}).get("issues", []), indent=2)

        # ── Build the prompt based on attempt number ──

        base_instructions = f"""You are a Python data engineer. Write a complete Python script that
cleans a messy CSV dataset loaded as a pandas DataFrame.

DATA CONTRACT (what clean data should look like):
{contract_text}

ISSUES FOUND IN THE DATA:
{issues_text}

SAMPLE OF THE RAW DATA:
{sample_csv}

STRICT RULES FOR YOUR CODE:
- The input variable is `df` (a pandas DataFrame already loaded)
- The output must be a cleaned DataFrame assigned to `df_cleaned`
- Also create a dict called `cleaning_log` that tracks what you did:
  cleaning_log = {{"actions": [], "rows_before": len(df), "rows_after": 0}}
  Append a string to cleaning_log["actions"] for each cleaning step
  Set cleaning_log["rows_after"] = len(df_cleaned) at the end
- Use ONLY pandas and standard library (no external packages)
- Handle errors gracefully — use try/except for operations that might fail
- Do NOT read or write files — work only with the `df` variable
- Do NOT print anything — results go in df_cleaned and cleaning_log
- Do NOT import pandas — it's already imported as pd"""

        if attempt == 1:
            prompt = base_instructions + "\n\nWrite the complete Python code. Return ONLY the code. No markdown fences. No explanations."

        elif attempt == 2 and error_history:
            # Error memory — show what failed before
            error_context = "\n\n--- PREVIOUS ATTEMPTS THAT FAILED ---\n"
            for prev in error_history:
                error_context += f"\nAttempt {prev['attempt']}:\n"
                error_context += f"Error: {prev['error'][:500]}\n"
                error_context += f"Code that failed:\n{prev['code'][:800]}\n"

            prompt = base_instructions + error_context + """

The previous code failed. Fix the specific errors shown above.
Return ONLY the corrected code. No markdown fences."""

        else:
            # ═══ STRATEGY ESCALATION (attempt 3+) ═══════════════════════
            error_context = "\n\n--- ALL PREVIOUS FAILURES ---\n"
            for prev in (error_history or []):
                error_context += f"\nAttempt {prev['attempt']}: {prev['error'][:300]}\n"

            prompt = base_instructions + error_context + """

IMPORTANT: The previous approaches are NOT WORKING. Do NOT try variations of the same logic.

Use a FUNDAMENTALLY DIFFERENT strategy:
- If previous code used .apply(), try vectorized operations instead
- If previous code tried type conversion directly, try string cleaning first
- If previous code had complex logic, simplify — do the minimum viable cleaning
- When in doubt, use the safest approach: coerce errors to NaN, drop what can't be fixed

Return ONLY the code. No markdown fences."""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3 if attempt == 1 else 0.5,  # Higher temp on retries for diversity
            max_tokens=3000,
        )

        code = response.choices[0].message.content.strip()

        # Clean markdown fences if present
        if code.startswith("```"):
            code = code.split("\n", 1)[1]
            if "```" in code:
                code = code.rsplit("```", 1)[0]

        logger.info(f"CoderAgent: Generated {len(code.splitlines())} lines of code")
        return code


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 3: QA Agent
# ═════════════════════════════════════════════════════════════════════════════

class QAAgent:
    """
    Executes cleaning code and validates the output against the Data Contract.

    Two levels of validation:
      1. Execution — did the code run without crashing?
      2. Contract — does the cleaned data match the Data Contract?
    """

    MAX_RETRIES = 3

    def execute_and_validate(
        self,
        code: str,
        df: pd.DataFrame,
        data_contract: dict,
    ) -> dict:
        """
        Execute cleaning code and validate against the contract.

        Returns dict with:
          - success: bool
          - df_cleaned: DataFrame or None
          - cleaning_log: dict or None
          - execution_error: str or None
          - contract_violations: list of violation dicts
          - validation_passed: bool
        """
        logger.info("QAAgent: Executing cleaning code...")

        result = {
            "success": False,
            "df_cleaned": None,
            "cleaning_log": None,
            "execution_error": None,
            "contract_violations": [],
            "validation_passed": False,
        }

        # ── Step 1: Execute the code ──
        try:
            # Create isolated namespace with the input DataFrame
            namespace = {"df": df.copy(), "pd": pd}

            exec(code, namespace)

            # Check outputs exist
            if "df_cleaned" not in namespace:
                result["execution_error"] = "Code ran but did not create 'df_cleaned' variable"
                logger.error("QAAgent: Missing df_cleaned in output")
                return result

            df_cleaned = namespace["df_cleaned"]
            cleaning_log = namespace.get("cleaning_log", {"actions": ["No log created"], "rows_before": len(df), "rows_after": len(df_cleaned)})

            result["df_cleaned"] = df_cleaned
            result["cleaning_log"] = cleaning_log
            result["success"] = True
            logger.info(f"QAAgent: Code executed — {len(df)} rows → {len(df_cleaned)} rows")

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result["execution_error"] = error_msg
            logger.error(f"QAAgent: Execution failed — {type(e).__name__}: {str(e)}")
            return result

        # ── Step 2: Validate against Data Contract ──
        if data_contract and result["success"]:
            violations = self._validate_contract(df_cleaned, data_contract)
            result["contract_violations"] = violations
            result["validation_passed"] = len(violations) == 0

            if violations:
                logger.warning(f"QAAgent: {len(violations)} contract violations found")
            else:
                logger.info("QAAgent: All contract validations passed")

        return result

    def _validate_contract(self, df: pd.DataFrame, contract: dict) -> list[dict]:
        """Check cleaned DataFrame against the Data Contract."""
        violations = []

        for col_name, rules in contract.items():
            if col_name.startswith("__"):
                continue  # Skip metadata keys

            if col_name not in df.columns:
                violations.append({
                    "column": col_name,
                    "rule": "column_exists",
                    "message": f"Column '{col_name}' missing from cleaned data",
                    "severity": "high",
                })
                continue

            col = df[col_name]

            # Check nullable constraint
            if not rules.get("nullable", True):
                null_count = int(col.isnull().sum())
                if null_count > 0:
                    violations.append({
                        "column": col_name,
                        "rule": "not_nullable",
                        "message": f"{null_count} null values remain (contract says not nullable)",
                        "severity": "high",
                        "count": null_count,
                    })

            # Check expected type
            expected_type = rules.get("expected_type", "")
            if expected_type in ("float", "int", "numeric"):
                non_null = col.dropna()
                if len(non_null) > 0:
                    try:
                        pd.to_numeric(non_null, errors="raise")
                    except (ValueError, TypeError):
                        bad_count = 0
                        for v in non_null:
                            try:
                                float(v)
                            except (ValueError, TypeError):
                                bad_count += 1
                        if bad_count > 0:
                            violations.append({
                                "column": col_name,
                                "rule": "expected_type",
                                "message": f"{bad_count} values cannot be converted to {expected_type}",
                                "severity": "medium",
                                "count": bad_count,
                            })

        # Check for remaining duplicates
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            violations.append({
                "column": "__all__",
                "rule": "no_duplicates",
                "message": f"{dup_count} duplicate rows remain",
                "severity": "medium",
                "count": dup_count,
            })

        return violations
